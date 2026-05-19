from __future__ import annotations

import asyncio
import hashlib
import secrets
import time
from typing import Any

from fastapi import WebSocket

from ddz.game_room import GameRoom, SeatInfo, _generate_room_id


RECOVERY_GRACE_SECONDS = 300  # 5 minutes
TOKEN_SECRET_LENGTH = 32


class ConnectionManager:
    def __init__(self) -> None:
        self.rooms: dict[str, GameRoom] = {}
        # ws_id -> (room_id, seat_index)
        self._ws_map: dict[int, tuple[str, int]] = {}
        # user_id -> ws_id (for disconnection tracking)
        self._user_map: dict[str, int] = {}
        # (room_id, seat_index) -> recovery token info
        self._recovery_tokens: dict[str, dict] = {}
        self._token_secret = secrets.token_hex(TOKEN_SECRET_LENGTH)

    # ------------------------------------------------------------------
    # Room management
    # ------------------------------------------------------------------

    def create_room(self, mode: str, host_username: str) -> GameRoom:
        room_id = _generate_room_id()
        while room_id in self.rooms:
            room_id = _generate_room_id()
        room = GameRoom(room_id, mode, host_username)
        self.rooms[room_id] = room
        return room

    def get_room(self, room_id: str) -> GameRoom | None:
        return self.rooms.get(room_id)

    def remove_room(self, room_id: str) -> None:
        room = self.rooms.pop(room_id, None)
        if room is None:
            return
        self._ws_map = {
            wid: (rid, sid)
            for wid, (rid, sid) in self._ws_map.items()
            if rid != room_id
        }
        self._user_map = {
            uid: wid
            for uid, wid in self._user_map.items()
            if wid in self._ws_map
        }
        # Clean up recovery tokens for this room
        self._recovery_tokens = {
            k: v for k, v in self._recovery_tokens.items()
            if v.get("room_id") != room_id
        }

    async def cleanup_stale_rooms(self) -> None:
        now = time.time()
        stale = []
        for room_id, room in self.rooms.items():
            has_live_human = any(
                s.is_human and s.connected
                for s in room.seats.values()
            )
            if not has_live_human:
                stale.append(room_id)
        for room_id in stale:
            self.remove_room(room_id)

    # ------------------------------------------------------------------
    # WS registration
    # ------------------------------------------------------------------

    def register_ws(self, ws: WebSocket, room_id: str, seat_index: int, username: str) -> None:
        ws_id = id(ws)
        self._ws_map[ws_id] = (room_id, seat_index)
        self._user_map[username] = ws_id

    def unregister_ws(self, ws: WebSocket) -> tuple[str, int] | None:
        ws_id = id(ws)
        info = self._ws_map.pop(ws_id, None)
        if info is None:
            return None
        room_id, _ = info
        # Clean user map
        for uid, wid in list(self._user_map.items()):
            if wid == ws_id:
                del self._user_map[uid]
                break
        return info

    def get_ws_info(self, ws: WebSocket) -> tuple[str, int] | None:
        return self._ws_map.get(id(ws))

    def get_user_ws_id(self, username: str) -> int | None:
        return self._user_map.get(username)

    # ------------------------------------------------------------------
    # Reconnection
    # ------------------------------------------------------------------

    def generate_recovery_token(self, room_id: str, seat_index: int, username: str) -> str:
        nonce = secrets.token_hex(16)
        payload = f"{room_id}:{seat_index}:{username}:{nonce}:{int(time.time())}"
        signature = hashlib.sha256(
            f"{payload}:{self._token_secret}".encode()
        ).hexdigest()[:16]
        token = f"{payload}:{signature}"

        self._recovery_tokens[token] = {
            "room_id": room_id,
            "seat_index": seat_index,
            "username": username,
            "created_at": time.time(),
        }
        return token

    def validate_recovery_token(self, token: str) -> dict | None:
        stored = self._recovery_tokens.get(token)
        if stored is None:
            return None

        age = time.time() - stored["created_at"]
        if age > RECOVERY_GRACE_SECONDS:
            del self._recovery_tokens[token]
            return None

        # Verify signature
        parts = token.rsplit(":", 1)
        if len(parts) != 2:
            return None
        payload, signature = parts
        expected = hashlib.sha256(
            f"{payload}:{self._token_secret}".encode()
        ).hexdigest()[:16]
        if not secrets.compare_digest(signature, expected):
            return None

        return stored

    async def reconnect_player(self, token: str, new_ws: WebSocket) -> dict | None:
        info = self.validate_recovery_token(token)
        if info is None:
            return None

        room_id = info["room_id"]
        seat_index = info["seat_index"]
        username = info["username"]

        room = self.get_room(room_id)
        if room is None:
            return None

        seat = room.seats.get(seat_index)
        if seat is None or seat.username != username:
            return None

        # Detach old WS if present
        if seat.ws is not None:
            old_ws_id = id(seat.ws)
            self._ws_map.pop(old_ws_id, None)

        # Attach new WS
        seat.ws = new_ws
        seat.connected = True
        self._ws_map[id(new_ws)] = (room_id, seat_index)
        self._user_map[username] = id(new_ws)

        # Start sender task
        room._start_sender(seat_index)

        # Send full state snapshot
        snapshot = room.full_state_snapshot(for_seat=seat_index)

        # Clean up token
        del self._recovery_tokens[token]

        return snapshot

    # ------------------------------------------------------------------
    # Disconnect handling
    # ------------------------------------------------------------------

    async def handle_disconnect(self, ws: WebSocket) -> tuple[str, int, str] | None:
        info = self.unregister_ws(ws)
        if info is None:
            return None

        room_id, seat_index = info
        room = self.get_room(room_id)
        if room is None:
            return None

        seat = room.seats.get(seat_index)
        if seat is None:
            return None

        username = seat.username
        seat.connected = False
        seat.ws = None

        # Stop sender task
        await room._stop_sender(seat_index)

        # Generate recovery token
        recovery_token = self.generate_recovery_token(room_id, seat_index, username)

        # Notify other players
        await room._broadcast("player_disconnected", {
            "seat": seat_index,
            "username": username,
        })

        # Check if all humans disconnected
        human_left = any(
            s.is_human and s.connected
            for s in room.seats.values()
        )
        if not human_left and room.state == "playing":
            await room._broadcast("all_disconnected", {
                "message": "所有玩家已断开，房间将在5分钟后清理。",
            })

        return room_id, seat_index, recovery_token
