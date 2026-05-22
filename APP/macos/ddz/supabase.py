from __future__ import annotations

import asyncio
import json
import os
import time
from typing import Any
from urllib import error, parse, request


SUPABASE_REST_URL = os.environ.get(
    "DDZ_SUPABASE_REST_URL",
    "https://nxudrqppkflkqjrmyaqv.supabase.co/rest/v1/",
)
SUPABASE_USERS_KEY = os.environ.get(
    "DDZ_SUPABASE_USERS_KEY",
    "sb_publishable_CBNUUamK7GboW2S1f0SCMA_ISMEvc_X",
)
SUPABASE_PVP_KEY = os.environ.get(
    "DDZ_SUPABASE_PVP_KEY",
    "sb_publishable_oYTuUWq7OfzNrmDKh2iaIQ_55UXh_qq",
)

USERS_TABLE = os.environ.get("DDZ_SUPABASE_USERS_TABLE", "ddz_users")
ROOMS_TABLE = os.environ.get("DDZ_SUPABASE_ROOMS_TABLE", "ddz_pvp_rooms")
EVENTS_TABLE = os.environ.get("DDZ_SUPABASE_EVENTS_TABLE", "ddz_pvp_events")

DEFAULT_MAX_RETRIES = 3
DEFAULT_BASE_DELAY = 0.5
DEFAULT_BACKOFF_FACTOR = 2.0


class SupabaseError(RuntimeError):
    pass


def _is_retryable(exc: Exception) -> bool:
    if isinstance(exc, error.URLError):
        return True
    if isinstance(exc, error.HTTPError):
        return exc.code is None or exc.code >= 500
    if isinstance(exc, TimeoutError):
        return True
    return False


def _should_retry(status_code: int | None) -> bool:
    if status_code is None:
        return True
    return status_code >= 500


class SupabaseClient:
    def __init__(
        self,
        api_key: str,
        base_url: str = SUPABASE_REST_URL,
        max_retries: int = DEFAULT_MAX_RETRIES,
        base_delay: float = DEFAULT_BASE_DELAY,
        backoff_factor: float = DEFAULT_BACKOFF_FACTOR,
    ) -> None:
        self.api_key = api_key
        self.base_url = base_url.rstrip("/") + "/"
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.backoff_factor = backoff_factor
        self._consecutive_failures = 0
        self._circuit_open_until: float = 0.0

    def select(
        self,
        table: str,
        filters: dict[str, str] | None = None,
        order: str | None = None,
        limit: int | None = None,
    ) -> list[dict]:
        params = {"select": "*"}
        if filters:
            params.update(filters)
        if order:
            params["order"] = order
        if limit is not None:
            params["limit"] = str(limit)
        return self._request("GET", table, params=params)

    def insert(self, table: str, payload: dict) -> list[dict]:
        return self._request(
            "POST",
            table,
            payload=payload,
            prefer="return=representation",
        )

    def upsert(self, table: str, payload: dict, conflict: str) -> list[dict]:
        return self._request(
            "POST",
            table,
            params={"on_conflict": conflict},
            payload=payload,
            prefer="resolution=merge-duplicates,return=representation",
        )

    def update(self, table: str, filters: dict[str, str], payload: dict) -> list[dict]:
        return self._request(
            "PATCH",
            table,
            params=filters,
            payload=payload,
            prefer="return=representation",
        )

    def delete(self, table: str, filters: dict[str, str]) -> list[dict]:
        return self._request(
            "DELETE",
            table,
            params=filters,
            prefer="return=representation",
        )

    def _request(
        self,
        method: str,
        table: str,
        params: dict[str, str] | None = None,
        payload: dict | None = None,
        prefer: str | None = None,
    ) -> Any:
        # circuit breaker check
        now = time.monotonic()
        if self._circuit_open_until > now:
            raise SupabaseError(
                f"Supabase circuit breaker open until "
                f"{time.strftime('%H:%M:%S', time.localtime(self._circuit_open_until))}"
            )

        query = parse.urlencode(params or {}, doseq=True, safe="(),.*")
        url = self.base_url + table
        if query:
            url += "?" + query

        body = None
        headers = {
            "apikey": self.api_key,
            "Authorization": f"Bearer {self.api_key}",
            "Accept": "application/json",
        }
        if payload is not None:
            body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            headers["Content-Type"] = "application/json"
        if prefer is not None:
            headers["Prefer"] = prefer

        last_error: Exception | None = None

        for attempt in range(self.max_retries + 1):
            req = request.Request(url, data=body, headers=headers, method=method)
            try:
                with request.urlopen(req, timeout=15) as response:
                    if 200 <= response.status < 300:
                        self._consecutive_failures = 0
                    raw = response.read().decode("utf-8")
            except error.HTTPError as exc:
                raw = exc.read().decode("utf-8", errors="replace")
                if not _should_retry(exc.code):
                    self._consecutive_failures += 1
                    raise SupabaseError(
                        f"Supabase request failed: HTTP {exc.code} {raw[:300]}"
                    ) from exc
                last_error = SupabaseError(
                    f"Supabase request failed: HTTP {exc.code} {raw[:300]}"
                )
            except error.URLError as exc:
                last_error = SupabaseError(
                    f"Cannot connect to Supabase: {exc.reason}"
                )
            except TimeoutError as exc:
                last_error = SupabaseError("Supabase request timed out")

            if last_error is None:
                break

            if attempt < self.max_retries:
                delay = self.base_delay * (self.backoff_factor ** attempt)
                time.sleep(delay)

        if last_error is not None:
            self._consecutive_failures += 1
            if self._consecutive_failures >= 5:
                self._circuit_open_until = time.monotonic() + 30.0
            raise last_error

        if not raw:
            return []
        try:
            return json.loads(raw)
        except json.JSONDecodeError as exc:
            raise SupabaseError(
                f"Supabase returned unparseable data: {raw[:200]}"
            ) from exc


def eq(value: str) -> str:
    return "eq." + value


def neq(value: str) -> str:
    return "neq." + value
