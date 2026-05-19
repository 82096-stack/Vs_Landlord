from __future__ import annotations

import hashlib
import secrets
from datetime import datetime
from typing import Any

from ddz.accounts import DKLEN, HASH_ITERATIONS, PASSWORD_SCHEME, SALT_BYTES
from ddz.rules import MODE_RULES
from ddz.supabase import EVENTS_TABLE, ROOMS_TABLE, SUPABASE_PVP_KEY, SupabaseClient, eq, neq


ROOM_STATUS_LOBBY = "lobby"
ROOM_STATUS_PLAYING = "playing"
ROOM_STATUS_COMPLETED = "completed"


class PvpManager:
    """Thin persistence layer for PvP rooms.

    DB stores only metadata + final results. Live game state lives
    exclusively in GameRoom memory (ddz/game_room.py).

    Writes use state_version for optimistic concurrency control.
    """

    def __init__(self, supabase_client: SupabaseClient | None = None) -> None:
        self.client = supabase_client or SupabaseClient(SUPABASE_PVP_KEY)

    # ------------------------------------------------------------------
    # Room CRUD
    # ------------------------------------------------------------------

    def create_room(
        self,
        owner_username: str,
        room_name: str,
        password: str,
        mode: str,
        max_rounds: int,
    ) -> tuple[bool, str, dict | None]:
        room_name = room_name.strip()
        if not 2 <= len(room_name) <= 30:
            return False, "房间名称长度需要在 2 到 30 个字符之间。", None
        if mode not in MODE_RULES:
            return False, "斗地主形式不合法。", None
        if not 1 <= max_rounds <= 50:
            return False, "比赛轮数需要在 1 到 50 之间。", None
        if self.get_room(room_name) is not None:
            return False, "房间名已经存在，请换一个名称。", None

        seats = [
            {
                "seat": 0,
                "username": owner_username,
                "ready": True,
                "joined_at": self._now(),
            }
        ]
        scores = {owner_username: 0}
        payload = {
            "room_name": room_name,
            "owner_username": owner_username,
            "password": self._build_password_record(password) if password else None,
            "mode": mode,
            "max_rounds": max_rounds,
            "current_round": 0,
            "status": ROOM_STATUS_LOBBY,
            "seats": seats,
            "scores": scores,
            "current_turn": owner_username,
            "state_version": 1,
            "winner_username": None,
        }
        rows = self.client.insert(ROOMS_TABLE, payload)
        room = rows[0] if rows else payload
        self._record_event(room_name, "room_created", owner_username, {"mode": mode, "max_rounds": max_rounds})
        return True, "房间创建成功。", room

    def list_rooms(self) -> list[dict]:
        return self.client.select(
            ROOMS_TABLE,
            filters={"status": neq("dissolved")},
            order="created_at.desc",
            limit=30,
        )

    def get_room(self, room_name: str) -> dict | None:
        rows = self.client.select(ROOMS_TABLE, filters={"room_name": eq(room_name)}, limit=1)
        return rows[0] if rows else None

    def join_room(self, username: str, room_name: str, password: str) -> tuple[bool, str, dict | None]:
        room = self.get_room(room_name)
        if room is None:
            return False, "房间不存在。", None
        if room["status"] != ROOM_STATUS_LOBBY:
            return False, "该房间已经开始或结束，不能加入。", room
        if room.get("password") and not self._verify_password(password, room["password"]):
            return False, "房间密码错误。", room

        seats = list(room.get("seats") or [])
        if any(seat["username"] == username for seat in seats):
            return True, "你已经在该房间中。", room
        if len(seats) >= MODE_RULES[room["mode"]]["player_count"]:
            return False, "房间已满。", room

        next_seat = self._first_open_seat(seats, MODE_RULES[room["mode"]]["player_count"])
        seats.append({
            "seat": next_seat,
            "username": username,
            "ready": True,
            "joined_at": self._now(),
        })
        seats.sort(key=lambda item: item["seat"])
        scores = dict(room.get("scores") or {})
        scores.setdefault(username, 0)

        version = int(room.get("state_version", 0))
        room = self._patch_room(
            room_name,
            {"seats": seats, "scores": scores, "current_turn": seats[0]["username"]},
            expected_version=version,
        )
        self._record_event(room_name, "player_joined", username, {"seat": next_seat})
        return True, "加入房间成功。", room

    def disband_room(self, username: str, room_name: str) -> tuple[bool, str]:
        room = self.get_room(room_name)
        if room is None:
            return False, "房间不存在或已经被删除。"
        if room["owner_username"] != username:
            return False, "只有房主可以解散房间。"
        self.client.delete(EVENTS_TABLE, {"room_name": eq(room_name)})
        self.client.delete(ROOMS_TABLE, {"room_name": eq(room_name)})
        return True, "房间已解散，房间数据已删除。"

    def start_room(self, username: str, room_name: str) -> tuple[bool, str, dict | None]:
        room = self.get_room(room_name)
        if room is None:
            return False, "房间不存在。", None
        if room["owner_username"] != username:
            return False, "只有房主可以开始比赛。", room
        if room["status"] != ROOM_STATUS_LOBBY:
            return False, "该房间不能重复开始。", room

        seats = sorted(room.get("seats") or [], key=lambda item: item["seat"])
        required = MODE_RULES[room["mode"]]["player_count"]
        if len(seats) != required:
            return False, f"人数不足，需要 {required} 人才能开始。", room
        scores = {seat["username"]: 0 for seat in seats}
        version = int(room.get("state_version", 0))
        room = self._patch_room(
            room_name,
            {
                "status": ROOM_STATUS_PLAYING,
                "current_round": 1,
                "scores": scores,
                "current_turn": seats[0]["username"],
                "winner_username": None,
            },
            expected_version=version,
        )
        self._record_event(room_name, "room_started", username, {"round": 1})
        return True, "比赛开始。", room

    # ------------------------------------------------------------------
    # Final result persistence (only called when game ends)
    # ------------------------------------------------------------------

    def record_final_result(
        self,
        room_name: str,
        username: str,
        landlord_username: str,
        landlord_won: bool,
        multiplier: int,
    ) -> tuple[bool, str, dict | None]:
        """Persist final round result. Only owner can call this after a round ends."""
        room = self.get_room(room_name)
        if room is None:
            return False, "房间不存在。", None
        if room["owner_username"] != username:
            return False, "只有房主可以录入本轮结算。", room
        if room["status"] != ROOM_STATUS_PLAYING:
            return False, "房间不在比赛中。", room
        multiplier = max(1, multiplier)
        seats = sorted(room.get("seats") or [], key=lambda item: item["seat"])
        usernames = [seat["username"] for seat in seats]
        if landlord_username not in usernames:
            return False, "地主用户名不在房间中。", room

        scores = dict(room.get("scores") or {})
        farmer_count = len(usernames) - 1
        for seat_username in usernames:
            scores.setdefault(seat_username, 0)
            if seat_username == landlord_username:
                scores[seat_username] += farmer_count * multiplier if landlord_won else -farmer_count * multiplier
            else:
                scores[seat_username] += -multiplier if landlord_won else multiplier

        current_round = int(room["current_round"])
        max_rounds = int(room["max_rounds"])
        completed = current_round >= max_rounds
        winner_username = max(scores, key=lambda item: (scores[item], item)) if completed else None
        next_round = current_round if completed else current_round + 1
        version = int(room.get("state_version", 0))

        room = self._patch_room(
            room_name,
            {
                "status": ROOM_STATUS_COMPLETED if completed else ROOM_STATUS_PLAYING,
                "current_round": next_round,
                "scores": scores,
                "current_turn": usernames[0],
                "winner_username": winner_username,
            },
            expected_version=version,
        )
        self._record_event(
            room_name,
            "round_finished",
            username,
            {
                "round": current_round,
                "landlord": landlord_username,
                "landlord_won": landlord_won,
                "multiplier": multiplier,
                "scores": scores,
                "winner": winner_username,
            },
        )
        message = f"比赛结束，胜利者是 {winner_username}。" if completed else f"第 {current_round} 轮已结算。"
        return True, message, room

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _patch_room(self, room_name: str, payload: dict, expected_version: int | None = None) -> dict:
        if expected_version is not None:
            payload["state_version"] = expected_version + 1
            filters = {
                "room_name": eq(room_name),
                "state_version": eq(str(expected_version)),
            }
        else:
            filters = {"room_name": eq(room_name)}
        rows = self.client.update(ROOMS_TABLE, filters, payload)
        if not rows:
            # version mismatch — re-read and return current state
            return self.get_room(room_name) or {}
        return rows[0]

    def _record_event(self, room_name: str, event_type: str, actor_username: str, payload: dict) -> None:
        self.client.insert(
            EVENTS_TABLE,
            {
                "room_name": room_name,
                "event_type": event_type,
                "actor_username": actor_username,
                "payload": payload,
            },
        )

    @staticmethod
    def _first_open_seat(seats: list[dict], player_count: int) -> int:
        used = {int(seat["seat"]) for seat in seats}
        for seat in range(player_count):
            if seat not in used:
                return seat
        return len(seats)

    @staticmethod
    def _now() -> str:
        return datetime.now().isoformat(timespec="seconds")

    @staticmethod
    def _build_password_record(password: str) -> dict:
        salt = secrets.token_hex(SALT_BYTES)
        return {
            "scheme": PASSWORD_SCHEME,
            "iterations": HASH_ITERATIONS,
            "dklen": DKLEN,
            "salt": salt,
            "hash": PvpManager._hash_password(password, salt, HASH_ITERATIONS, DKLEN),
        }

    @staticmethod
    def _verify_password(password: str, record: dict) -> bool:
        expected = PvpManager._hash_password(
            password,
            record["salt"],
            int(record.get("iterations", HASH_ITERATIONS)),
            int(record.get("dklen", DKLEN)),
        )
        return secrets.compare_digest(expected, record["hash"])

    @staticmethod
    def _hash_password(password: str, salt: str, iterations: int, dklen: int) -> str:
        hashed = hashlib.pbkdf2_hmac(
            "sha256",
            password.encode("utf-8"),
            bytes.fromhex(salt),
            iterations,
            dklen=dklen,
        )
        return hashed.hex()
