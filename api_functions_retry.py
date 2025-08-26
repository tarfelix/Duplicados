
# api_functions_retry_opt.py — resilient HTTP client with retry/backoff,
# optional dry_run (ignores rate-limit) and concurrent helpers.
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple, Callable
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests


@dataclass
class RetryConfig:
    max_retries: int = 5
    backoff_factor: float = 0.6  # exponential backoff base
    status_forcelist: Tuple[int, ...] = (429, 500, 502, 503, 504)
    calls_per_second: float = 4.0
    timeout: float = 30.0
    dry_run: bool = False  # when True, don't enforce rate limit (fast simulations)


class HttpClientRetry:
    def __init__(
        self,
        base_url: str,
        headers: Optional[Dict[str, str]] = None,
        retry: Optional[RetryConfig] = None,
        session: Optional[requests.Session] = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.session = session or requests.Session()
        self.headers = headers or {}
        self.retry = retry or RetryConfig()
        self._min_interval = 1.0 / max(0.1, self.retry.calls_per_second)
        self._last_call_ts = 0.0

    def _rate_limit(self) -> None:
        # IMPORTANT: ignore rate limit when dry_run is True
        if self.retry.dry_run:
            return
        now = time.time()
        elapsed = now - self._last_call_ts
        wait = self._min_interval - elapsed
        if wait > 0:
            time.sleep(wait)
        self._last_call_ts = time.time()

    def request(self, method: str, path: str, **kwargs) -> requests.Response:
        url = f"{self.base_url}/{path.lstrip('/')}"
        headers = kwargs.pop("headers", {})
        merged_headers = dict(self.headers)
        merged_headers.update(headers)

        for attempt in range(self.retry.max_retries + 1):
            self._rate_limit()
            try:
                resp = self.session.request(
                    method.upper(), url, headers=merged_headers, timeout=self.retry.timeout, **kwargs
                )
            except requests.RequestException as exc:
                if attempt >= self.retry.max_retries:
                    raise
                time.sleep(self.retry.backoff_factor * (2 ** attempt))
                continue

            if resp.status_code in self.retry.status_forcelist:
                if attempt >= self.retry.max_retries:
                    resp.raise_for_status()
                time.sleep(self.retry.backoff_factor * (2 ** attempt))
                continue

            # ok or other handled errors
            return resp

        # Should not get here
        raise RuntimeError("Exhausted retries with no response returned.")

    # ---- Domain specific helpers (adapt endpoints to your API) ----

    def cancel_activity(self, activity_id: str, reason: str, user: str) -> Dict[str, Any]:
        """
        Cancela uma atividade duplicada.
        Ajuste o path abaixo para o endpoint real da sua API.
        """
        payload = {"reason": reason, "requested_by": user}
        if self.retry.dry_run:
            # Simula resposta rápida
            return {"ok": True, "activity_id": activity_id, "dry_run": True}
        resp = self.request("POST", f"/activities/{activity_id}/cancel", json=payload)
        if resp.status_code >= 400:
            return {"ok": False, "status": resp.status_code, "error": resp.text, "activity_id": activity_id}
        return {"ok": True, "status": resp.status_code, "activity_id": activity_id}

    def set_principal(self, activity_id: str, group_key: str, user: str) -> Dict[str, Any]:
        """
        Define a atividade principal de um grupo.
        """
        payload = {"group_key": group_key, "requested_by": user}
        if self.retry.dry_run:
            return {"ok": True, "activity_id": activity_id, "dry_run": True}
        resp = self.request("POST", f"/activities/{activity_id}/set-primary", json=payload)
        if resp.status_code >= 400:
            return {"ok": False, "status": resp.status_code, "error": resp.text, "activity_id": activity_id}
        return {"ok": True, "status": resp.status_code, "activity_id": activity_id}

    # ---- Concurrency helpers ----

    def process_cancellations_concurrently(
        self,
        items: Iterable[Tuple[str, str]],  # (activity_id, reason)
        requested_by: str,
        update_progress: Optional[Callable[[int, int], None]] = None,
        max_workers: int = 4,
    ) -> List[Dict[str, Any]]:
        """
        Cancela atividades em paralelo. `items` é uma lista de tuplas (activity_id, reason).
        """
        results: List[Dict[str, Any]] = []
        items_list = list(items)
        total = len(items_list)
        if total == 0:
            return results

        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            futs = [
                ex.submit(self.cancel_activity, activity_id, reason, requested_by)
                for activity_id, reason in items_list
            ]
            for i, fut in enumerate(as_completed(futs), 1):
                try:
                    res = fut.result()
                except Exception as exc:  # noqa: BLE001
                    res = {"ok": False, "error": str(exc)}
                results.append(res)
                if update_progress:
                    update_progress(i, total)

        return results
