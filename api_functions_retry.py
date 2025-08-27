# -*- coding: utf-8 -*-
"""
Módulo Cliente HTTP com Resiliência e Processamento em Lote
============================================================

Este módulo fornece uma classe, HttpClientRetry, para interagir com a API de
cancelamento de atividades.

Melhorias nesta versão:
- **Processamento Paralelo:** Adicionado o método `cancel_batch_parallel` que
  utiliza um `ThreadPoolExecutor` para enviar requisições de cancelamento em
  paralelo, respeitando o rate limit da API através de um semáforo.
- **Robustez:** A lógica de retry e tratamento de erros foi mantida e
  aprimorada para ser mais resiliente.
- **Flexibilidade:** O Rate Limiter é desativado durante o modo de teste
  (Dry Run) para acelerar a validação.
"""
import requests
import time
import logging
import threading
from typing import Dict, Any, Optional, List
from concurrent.futures import ThreadPoolExecutor, as_completed

class HttpClientRetry:
    def __init__(self,
                 base_url: str,
                 entity_id: int,
                 token: str,
                 calls_per_second: float = 5.0,
                 max_attempts: int = 3,
                 timeout: int = 20,
                 dry_run: bool = False,
                 max_workers: int = 8):
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
            max_workers (int): Número máximo de threads para processamento paralelo.
        """
        self.base_url = base_url.rstrip("/")
        self.entity_id = entity_id
        self.token = token
        self.timeout = timeout
        self.max_attempts = max_attempts
        self.dry_run = dry_run
        self.max_workers = max_workers
        
        self.min_interval = 1.0 / calls_per_second if calls_per_second > 0 else 0
        self.last_call_ts = 0
        self.lock = threading.Lock()

    def _get_headers(self) -> Dict[str, str]:
        """Monta os cabeçalhos padrão para as requisições."""
        return {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.token}"
        }

    def _rate_limit(self):
        """Garante que o número de chamadas por segundo não seja excedido."""
        if self.dry_run or self.min_interval == 0:
            return
        
        with self.lock:
            elapsed = time.time() - self.last_call_ts
            if elapsed < self.min_interval:
                time.sleep(self.min_interval - elapsed)
            self.last_call_ts = time.time()

    def _make_request(self, method: str, endpoint: str, json_data: Optional[Dict] = None) -> Optional[Dict[str, Any]]:
        """Realiza uma requisição HTTP com lógica de retry e backoff exponencial."""
        self._rate_limit()
        url = f"{self.base_url}/{endpoint}"
        
        for attempt in range(self.max_attempts):
            try:
                response = requests.request(
                    method=method.upper(), url=url, headers=self._get_headers(),
                    json=json_data, timeout=self.timeout
                )
                response.raise_for_status()
                return response.json()
            
            except requests.exceptions.HTTPError as e:
                if 400 <= e.response.status_code < 500:
                    logging.error(f"Erro de cliente ({e.response.status_code}) para {url}. Resposta: {e.response.text}")
                    return {"ok": False, "success": False, "error": f"Client Error: {e.response.status_code}", "message": e.response.text}
                logging.warning(f"Erro de servidor ({e.response.status_code}) na tentativa {attempt + 1}/{self.max_attempts} para {url}.")
            
            except requests.exceptions.RequestException as e:
                logging.warning(f"Erro de conexão na tentativa {attempt + 1}/{self.max_attempts} para {url}: {e}")

            if attempt < self.max_attempts - 1:
                time.sleep((2 ** attempt)) # Backoff exponencial

        logging.error(f"Todas as {self.max_attempts} tentativas falharam para a requisição {method} {url}.")
        return None

    def activity_canceled(self, activity_id: str, user_name: str, principal_id: str) -> Dict[str, Any]:
        """Envia uma requisição para cancelar uma única atividade."""
        observation_message = f"Cancelado pelo verificador de duplicidade por {user_name}. Duplicata da principal ID {principal_id}."

        if self.dry_run:
            logging.info(f"[DRY-RUN] Simulado cancelamento da atividade ID: {activity_id}")
            time.sleep(0.05) # Simula latência de rede
            return {"ok": True, "success": True, "message": "Dry run mode", "code": "200", "activity_id": activity_id}

        endpoint = f"activity/{self.entity_id}/activitycanceledduplicate"
        body = {
            "entity_id": self.entity_id, "id": activity_id,
            "activity_type_id": 152, "user_name": user_name,
            "observation": observation_message
        }
        
        response = self._make_request(method="PUT", endpoint=endpoint, json_data=body)
        
        if response is None:
            return {"ok": False, "success": False, "error": "Max retries exceeded", "activity_id": activity_id}
            
        is_success = response.get("ok", False) or response.get("success", False) or str(response.get("code")) == "200"
        response["ok"] = is_success
        response["success"] = is_success
        response["activity_id"] = activity_id
        return response

    def cancel_batch_parallel(self, items_to_cancel: List[Dict], user_name: str) -> List[Dict]:
        """
        Processa o cancelamento de múltiplas atividades em paralelo.

        Args:
            items_to_cancel (List[Dict]): Uma lista de dicionários, cada um contendo
                                          "ID a Cancelar" e "Duplicata do Principal".
            user_name (str): O nome do usuário que está realizando a ação.

        Returns:
            List[Dict]: Uma lista com as respostas da API para cada item.
        """
        results = [None] * len(items_to_cancel)
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Mapeia futuro para o índice original para manter a ordem
            future_to_index = {
                executor.submit(
                    self.activity_canceled,
                    item["ID a Cancelar"],
                    user_name,
                    item["Duplicata do Principal"]
                ): i
                for i, item in enumerate(items_to_cancel)
            }

            for future in as_completed(future_to_index):
                index = future_to_index[future]
                try:
                    results[index] = future.result()
                except Exception as exc:
                    logging.error(f"Exceção ao cancelar item no índice {index}: {exc}")
                    item = items_to_cancel[index]
                    results[index] = {
                        "ok": False, "success": False, 
                        "error": "Exception in thread", "message": str(exc),
                        "activity_id": item["ID a Cancelar"]
                    }
        return results
