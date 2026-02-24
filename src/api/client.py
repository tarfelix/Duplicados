import streamlit as st
import requests
import time
import logging
from typing import Dict, Any, Optional
from src.config import get_secret

def get_api_client(dry_run: bool = False) -> Optional['HttpClientRetry']:
    """Cria uma nova instância do cliente para a API de cancelamento."""
    url = get_secret("api.url_api")
    entity_id = get_secret("api.entity_id")
    token = get_secret("api.token")
    
    if not all([url, entity_id, token]):
        return None
        
    return HttpClientRetry(
        base_url=str(url),
        entity_id=int(entity_id),
        token=str(token),
        calls_per_second=float(get_secret("api_client.calls_per_second", 3.0)),
        max_attempts=int(get_secret("api_client.max_attempts", 3)),
        timeout=int(get_secret("api_client.timeout", 15)),
        dry_run=dry_run
    )

class HttpClientRetry:
    def __init__(self,
                 base_url: str,
                 entity_id: int,
                 token: str,
                 calls_per_second: float = 3.0,
                 max_attempts: int = 3,
                 timeout: int = 15,
                 dry_run: bool = False):
        self.base_url = base_url.rstrip("/")
        self.entity_id = entity_id
        self.token = token
        self.timeout = timeout
        self.max_attempts = max_attempts
        self.dry_run = dry_run
        self.min_interval = 1.0 / calls_per_second if calls_per_second > 0 else 0
        self.last_call_ts = 0

    def _get_headers(self) -> Dict[str, str]:
        return {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.token}"
        }

    def _rate_limit(self):
        if self.min_interval > 0:
            elapsed = time.time() - self.last_call_ts
            if elapsed < self.min_interval:
                time.sleep(self.min_interval - elapsed)
            self.last_call_ts = time.time()

    def _make_request(self, method: str, endpoint: str, json_data: Optional[Dict] = None) -> Optional[Dict[str, Any]]:
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
                try:
                    return response.json()
                except ValueError:
                    return {"ok": False, "success": False, "error": "Invalid JSON", "message": response.text}
            
            except requests.exceptions.HTTPError as e:
                if 400 <= e.response.status_code < 500:
                    return {"ok": False, "success": False, "error": f"Client Error: {e.response.status_code}", "message": e.response.text}
            except requests.exceptions.RequestException:
                pass

            if attempt < self.max_attempts - 1:
                time.sleep(2 ** attempt)
        return None

    def activity_canceled(self, activity_id: str, user_name: str, principal_id: str) -> Dict[str, Any]:
        observation_message = (
            f"Cancelado pelo instrumento de verificar duplicado por {user_name}. "
            f"Atividade duplicada da principal ID {principal_id}."
        )

        if self.dry_run:
            return {"ok": True, "success": True, "message": "Dry run mode", "code": "200"}

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
            return {"ok": False, "success": False, "error": "Max retries exceeded"}
            
        if "ok" not in response and "success" not in response:
            response["ok"] = False
            response["success"] = False
            
        return response
