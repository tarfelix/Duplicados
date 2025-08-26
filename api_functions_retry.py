# -*- coding: utf-8 -*-
"""
Módulo Cliente HTTP com Resiliência (Otimizado)
===============================================

Este módulo fornece uma classe, HttpClientRetry, para interagir com a API de
cancelamento de atividades. Ele implementa funcionalidades essenciais para
garantir a robustez das operações.

Melhorias nesta versão:
- O Rate Limiter é desativado durante o modo de teste (Dry Run) para acelerar
  a validação e os testes.
- Adicionada concorrência opcional para lidar com múltiplas chamadas de forma
  mais eficiente.
"""
import requests
import time
import logging
from typing import Dict, Any, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

class HttpClientRetry:
    def __init__(self,
                 base_url: str,
                 entity_id: int,
                 token: str,
                 calls_per_second: float = 3.0,
                 max_attempts: int = 3,
                 timeout: int = 15,
                 dry_run: bool = False):
        """
        Inicializa o cliente HTTP.

        Args:
            base_url (str): A URL base da API.
            entity_id (int): O ID da entidade a ser usado nas chamadas.
            token (str): O token de autorização (Bearer).
            calls_per_second (float): Máximo de chamadas por segundo.
            max_attempts (int): Número máximo de tentativas para cada chamada.
            timeout (int): Timeout em segundos para as requisições.
            dry_run (bool): Se True, não executa as chamadas reais.
        """
        self.base_url = base_url.rstrip("/")
        self.entity_id = entity_id
        self.token = token
        self.timeout = timeout
        self.max_attempts = max_attempts
        self.dry_run = dry_run
        
        # Para o rate limiting
        self.min_interval = 1.0 / calls_per_second if calls_per_second > 0 else 0
        self.last_call_ts = 0

    def _get_headers(self) -> Dict[str, str]:
        """Monta os cabeçalhos padrão para as requisições."""
        return {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.token}"
        }

    def _rate_limit(self):
        """
        Garante que o número de chamadas por segundo não seja excedido.
        Otimização: Ignora o limite em modo dry_run.
        """
        if self.dry_run:
            return
            
        if self.min_interval > 0:
            elapsed = time.time() - self.last_call_ts
            if elapsed < self.min_interval:
                time.sleep(self.min_interval - elapsed)
            self.last_call_ts = time.time()

    def _make_request(self, method: str, endpoint: str, json_data: Optional[Dict] = None) -> Optional[Dict[str, Any]]:
        """
        Realiza uma requisição HTTP com lógica de retry e backoff.
        """
        self._rate_limit()
        
        url = f"{self.base_url}/{endpoint}"
        
        for attempt in range(self.max_attempts):
            try:
                response = requests.request(
                    method=method.upper(),
                    url=url,
                    headers=self._get_headers(),
                    json=json_data,
                    timeout=self.timeout
                )
                response.raise_for_status()
                return response.json()
            
            except requests.exceptions.HTTPError as e:
                if 400 <= e.response.status_code < 500:
                    logging.error(f"Erro de cliente ({e.response.status_code}) na chamada para {url}. Não haverá nova tentativa. Resposta: {e.response.text}")
                    return {"ok": False, "success": False, "error": f"Client Error: {e.response.status_code}", "message": e.response.text}
                logging.warning(f"Erro de servidor ({e.response.status_code}) na tentativa {attempt + 1}/{self.max_attempts} para {url}.")
            
            except requests.exceptions.RequestException as e:
                logging.warning(f"Erro de conexão na tentativa {attempt + 1}/{self.max_attempts} para {url}: {e}")

            if attempt < self.max_attempts - 1:
                wait_time = (2 ** attempt)
                time.sleep(wait_time)

        logging.error(f"Todas as {self.max_attempts} tentativas falharam para a requisição {method} {url}.")
        return None

    def activity_canceled(self, activity_id: str, user_name: str, principal_id: str) -> Dict[str, Any]:
        """
        Envia uma requisição para cancelar uma atividade por duplicidade.

        Args:
            activity_id (str): O ID da atividade a ser cancelada.
            user_name (str): O nome do usuário que está realizando a ação.
            principal_id (str): O ID da atividade principal que motivou o cancelamento.

        Returns:
            dict: A resposta da API ou um dicionário de simulação em caso de dry_run.
        """
        observation_message = (
            f"Cancelado pelo instrumento de verificar duplicado por {user_name}. "
            f"Atividade duplicada da principal ID {principal_id}."
        )

        if self.dry_run:
            logging.info(f"[DRY-RUN] Simulado o cancelamento da atividade ID: {activity_id} com a observação: '{observation_message}'")
            # Simula uma pequena latência de rede
            time.sleep(0.1)
            return {"ok": True, "success": True, "message": "Dry run mode", "code": "200", "activity_id": activity_id}

        endpoint = f"activity/{self.entity_id}/activitycanceledduplicate"
        body = {
            "entity_id": self.entity_id,
            "id": activity_id,
            "activity_type_id": 152,
            "user_name": user_name,
            "observation": observation_message
        }
        
        response = self._make_request(method="PUT", endpoint=endpoint, json_data=body)
        
        if response is None:
            return {"ok": False, "success": False, "error": "Max retries exceeded", "activity_id": activity_id}
            
        if "ok" not in response and "success" not in response:
            response["ok"] = False
            response["success"] = False
        
        response["activity_id"] = activity_id
        return response

    def process_cancellations_concurrently(self, items_to_cancel: list[dict], user_name: str, progress_callback=None, max_workers=4) -> dict:
        """
        Processa uma lista de cancelamentos de forma concorrente.

        Args:
            items_to_cancel (list[dict]): Lista de dicionários, cada um com 'ID a Cancelar' e 'Duplicata do Principal'.
            user_name (str): Nome do usuário para registrar na API.
            progress_callback (callable, optional): Função para reportar o progresso (recebe float de 0.0 a 1.0).
            max_workers (int): Número máximo de threads para as chamadas de API.

        Returns:
            dict: Um resumo dos resultados com 'success', 'failed' e uma lista de 'errors'.
        """
        results = {"success": 0, "failed": 0, "errors": []}
        total_items = len(items_to_cancel)

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(
                    self.activity_canceled,
                    item["ID a Cancelar"],
                    user_name,
                    item["Duplicata do Principal"]
                ): item for item in items_to_cancel
            }

            for i, future in enumerate(as_completed(futures)):
                try:
                    response = future.result()
                    if response and (response.get("ok") or response.get("success")):
                        results["success"] += 1
                    else:
                        results["failed"] += 1
                        results["errors"].append(response)
                except Exception as e:
                    item = futures[future]
                    results["failed"] += 1
                    error_details = {
                        "activity_id": item["ID a Cancelar"],
                        "error": "Exception",
                        "message": str(e)
                    }
                    results["errors"].append(error_details)
                
                if progress_callback:
                    progress_callback((i + 1) / total_items)
        
        return results
