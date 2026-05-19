from __future__ import annotations

import json
import os
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


class SupabaseError(RuntimeError):
    pass


class SupabaseClient:
    def __init__(self, api_key: str, base_url: str = SUPABASE_REST_URL) -> None:
        self.api_key = api_key
        self.base_url = base_url.rstrip("/") + "/"

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

        req = request.Request(url, data=body, headers=headers, method=method)
        try:
            with request.urlopen(req, timeout=15) as response:
                raw = response.read().decode("utf-8")
        except error.HTTPError as exc:
            details = exc.read().decode("utf-8", errors="replace")
            raise SupabaseError(f"Supabase 请求失败: HTTP {exc.code} {details}") from exc
        except error.URLError as exc:
            raise SupabaseError(f"无法连接 Supabase，请检查网络: {exc.reason}") from exc

        if not raw:
            return []
        try:
            return json.loads(raw)
        except json.JSONDecodeError as exc:
            raise SupabaseError(f"Supabase 返回了无法解析的数据: {raw[:200]}") from exc


def eq(value: str) -> str:
    return "eq." + value


def neq(value: str) -> str:
    return "neq." + value
