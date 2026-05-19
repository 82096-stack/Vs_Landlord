from __future__ import annotations

import hashlib
import json
import os
import secrets
from datetime import datetime
from pathlib import Path
from typing import Any

from ddz.supabase import SUPABASE_USERS_KEY, USERS_TABLE, SupabaseClient, SupabaseError


DEFAULT_ACCOUNT_FILE = Path(__file__).resolve().parent.parent / "data" / "users.json"
PASSWORD_SCHEME = "pbkdf2_sha256"
PROFILE_SCHEME = "pbkdf2_sha256_profile_v2"
HASH_ITERATIONS = 600_000
LEGACY_HASH_ITERATIONS = 200_000
PROFILE_HASH_ITERATIONS = 60_000
SALT_BYTES = 16
DKLEN = 32
STARTING_RATING = 1200
RANKED_MIN_RATING = 200
DAILY_REPLENISH_LIMIT = 2
SUPABASE_PROFILE_SECRET = os.environ.get(
    "DDZ_PROFILE_SECRET",
    "ddz-supabase-profile-integrity-v1",
)
PROFILE_DEFAULTS = {
    "created_at": "",
    "wins": 0,
    "losses": 0,
    "games": 0,
    "rating": STARTING_RATING,
    "ranked_wins": 0,
    "ranked_losses": 0,
    "ranked_games": 0,
    "casual_wins": 0,
    "casual_losses": 0,
    "casual_games": 0,
    "daily_replenish_date": "",
    "daily_replenish_used": 0,
}


class AccountManager:
    def __init__(
        self,
        storage_path: Path | None = None,
        supabase_client: SupabaseClient | None = None,
    ) -> None:
        self.storage_path = storage_path
        self.supabase_client = supabase_client
        self.use_supabase = storage_path is None
        self.profile_key_path = None if storage_path is None else self.storage_path.with_name(".profile_integrity_key")
        if self.use_supabase:
            self.supabase_client = self.supabase_client or SupabaseClient(SUPABASE_USERS_KEY)
        else:
            self.storage_path.parent.mkdir(parents=True, exist_ok=True)
        if not self.use_supabase and not self.storage_path.exists():
            self._save({"users": {}})

    def register(self, username: str, password: str) -> tuple[bool, str]:
        username = username.strip()
        if not 3 <= len(username) <= 20:
            return False, "用户名长度需要在 3 到 20 个字符之间。"
        if not password or len(password) < 6:
            return False, "密码至少需要 6 位。"

        data = self._load()
        if username in data["users"]:
            return False, "该用户名已存在。"

        profile = self._default_profile()
        profile["created_at"] = datetime.now().isoformat(timespec="seconds")
        user_record = {
            "password": self._build_password_record(password),
            "profile": self._protect_profile(profile),
        }
        if self.use_supabase:
            assert self.supabase_client is not None
            try:
                self.supabase_client.insert(
                    USERS_TABLE,
                    {
                        "username": username,
                        "password": user_record["password"],
                        "profile": user_record["profile"],
                    },
                )
            except SupabaseError as exc:
                if "duplicate" in str(exc).lower() or "23505" in str(exc):
                    return False, "该用户名已存在。"
                raise
            return True, "注册成功。"

        data["users"][username] = user_record
        self._save(data)
        return True, "注册成功。"

    def authenticate(self, username: str, password: str) -> tuple[bool, str]:
        data = self._load()
        user = data["users"].get(username.strip())
        if not user:
            return False, "用户名不存在。"

        password_record = self._extract_password_record(user)
        password_hash = self._hash_password(
            password=password,
            salt=password_record["salt"],
            iterations=password_record["iterations"],
            dklen=password_record["dklen"],
        )
        if not secrets.compare_digest(password_hash, password_record["hash"]):
            return False, "密码错误。"

        changed = False
        if "password" not in user:
            user["password"] = password_record
            user.pop("salt", None)
            user.pop("password_hash", None)
            changed = True

        _, profile_changed = self._extract_profile(user)
        changed = changed or profile_changed
        if changed:
            self._save(data)
        return True, "登录成功。"

    def record_result(
        self,
        username: str,
        won: bool,
        match_type: str = "casual",
        rating_delta: int = 0,
    ) -> None:
        data = self._load()
        user = data["users"].get(username)
        if not user:
            return

        profile, _ = self._extract_profile(user)
        profile["games"] += 1
        if won:
            profile["wins"] += 1
        else:
            profile["losses"] += 1

        if match_type == "ranked":
            profile["ranked_games"] += 1
            profile["rating"] += rating_delta
            if won:
                profile["ranked_wins"] += 1
            else:
                profile["ranked_losses"] += 1
        else:
            profile["casual_games"] += 1
            if won:
                profile["casual_wins"] += 1
            else:
                profile["casual_losses"] += 1

        user["profile"] = self._protect_profile(profile)
        self._save(data)

    def prepare_ranked_entry(self, username: str) -> tuple[bool, str, dict | None]:
        data = self._load()
        user = data["users"].get(username)
        if not user:
            return False, "当前账号数据不存在。", None

        profile, _ = self._extract_profile(user)
        today = datetime.now().date().isoformat()
        if profile["daily_replenish_date"] != today:
            profile["daily_replenish_date"] = today
            profile["daily_replenish_used"] = 0

        if profile["rating"] >= RANKED_MIN_RATING:
            user["profile"] = self._protect_profile(profile)
            self._save(data)
            return True, "积分满足要求，可以进入积分赛。", self._compose_stats(username, profile)

        if profile["daily_replenish_used"] < DAILY_REPLENISH_LIMIT:
            profile["daily_replenish_used"] += 1
            profile["daily_replenish_date"] = today
            profile["rating"] = STARTING_RATING
            user["profile"] = self._protect_profile(profile)
            self._save(data)
            return (
                True,
                f"当前积分低于 {RANKED_MIN_RATING}，已自动补分到 {STARTING_RATING}。"
                f" 今日已补分 {profile['daily_replenish_used']}/{DAILY_REPLENISH_LIMIT} 次。",
                self._compose_stats(username, profile),
            )

        return (
            False,
            f"当前积分低于 {RANKED_MIN_RATING}，且今日补分次数已用完。请明天再试。",
            self._compose_stats(username, profile),
        )

    def list_users(self) -> list[dict]:
        data = self._load()
        users = []
        changed = False
        for username, user in data["users"].items():
            profile, profile_changed = self._extract_profile(user)
            changed = changed or profile_changed
            users.append(self._compose_stats(username, profile))
        if changed:
            self._save(data)
        users.sort(key=lambda item: (-item["rating"], -item["wins"], -item["win_rate"], item["username"]))
        return users

    def get_user_stats(self, username: str) -> dict | None:
        users = self.list_users()
        for rank, user in enumerate(users, start=1):
            if user["username"] == username:
                user_with_rank = dict(user)
                user_with_rank["rank"] = rank
                return user_with_rank
        return None

    def _load(self) -> dict:
        if self.use_supabase:
            assert self.supabase_client is not None
            rows = self.supabase_client.select(USERS_TABLE, order="username.asc")
            users = {}
            for row in rows:
                username = row["username"]
                users[username] = {
                    "password": row["password"],
                    "profile": row["profile"],
                }
            return {"users": users}

        assert self.storage_path is not None
        with self.storage_path.open("r", encoding="utf-8") as file:
            return json.load(file)

    def _save(self, data: dict) -> None:
        if self.use_supabase:
            assert self.supabase_client is not None
            for username, user in data["users"].items():
                self.supabase_client.upsert(
                    USERS_TABLE,
                    {
                        "username": username,
                        "password": user["password"],
                        "profile": user["profile"],
                    },
                    "username",
                )
            return

        assert self.storage_path is not None
        with self.storage_path.open("w", encoding="utf-8") as file:
            json.dump(data, file, ensure_ascii=False, indent=2)

    def _build_password_record(self, password: str) -> dict:
        salt = secrets.token_hex(SALT_BYTES)
        return {
            "scheme": PASSWORD_SCHEME,
            "iterations": HASH_ITERATIONS,
            "dklen": DKLEN,
            "salt": salt,
            "hash": self._hash_password(
                password=password,
                salt=salt,
                iterations=HASH_ITERATIONS,
                dklen=DKLEN,
            ),
        }

    @staticmethod
    def _extract_password_record(user: dict) -> dict:
        if "password" in user:
            record = user["password"]
            return {
                "scheme": record.get("scheme", PASSWORD_SCHEME),
                "iterations": int(record.get("iterations", HASH_ITERATIONS)),
                "dklen": int(record.get("dklen", DKLEN)),
                "salt": record["salt"],
                "hash": record["hash"],
            }
        return {
            "scheme": PASSWORD_SCHEME,
            "iterations": LEGACY_HASH_ITERATIONS,
            "dklen": DKLEN,
            "salt": user["salt"],
            "hash": user["password_hash"],
        }

    def _extract_profile(self, user: dict) -> tuple[dict, bool]:
        changed = False
        integrity_failed = False
        profile = self._default_profile()

        if "profile" in user and isinstance(user["profile"], dict):
            stored_profile = user["profile"]
            for field, default in PROFILE_DEFAULTS.items():
                stored_value = stored_profile.get(field)
                if isinstance(stored_value, dict) and {"value", "salt", "hash"} <= set(stored_value):
                    if self._verify_protected_value(stored_value):
                        profile[field] = stored_value["value"]
                        if stored_value.get("scheme") != PROFILE_SCHEME:
                            changed = True
                    else:
                        profile[field] = self._fallback_profile_value(field, default)
                        integrity_failed = True
                        changed = True
                elif stored_value is not None:
                    profile[field] = stored_value
                    changed = True
                else:
                    profile[field] = self._fallback_profile_value(field, default)
                    changed = True
        else:
            for field, default in PROFILE_DEFAULTS.items():
                if field in user:
                    profile[field] = user[field]
                else:
                    profile[field] = self._fallback_profile_value(field, default)
                changed = True

        if not profile["created_at"]:
            profile["created_at"] = datetime.now().isoformat(timespec="seconds")
            changed = True

        if integrity_failed:
            today = datetime.now().date().isoformat()
            profile["rating"] = 0
            profile["daily_replenish_date"] = today
            profile["daily_replenish_used"] = DAILY_REPLENISH_LIMIT

        if changed:
            user["profile"] = self._protect_profile(profile)
            for field in PROFILE_DEFAULTS:
                user.pop(field, None)
        return profile, changed

    def _protect_profile(self, profile: dict) -> dict:
        protected: dict[str, dict] = {}
        for field, value in profile.items():
            protected[field] = self._protect_value(value)
        return protected

    def _protect_value(self, value: Any) -> dict:
        salt = secrets.token_hex(SALT_BYTES)
        canonical = self._serialize_value(value)
        return {
            "scheme": PROFILE_SCHEME,
            "iterations": PROFILE_HASH_ITERATIONS,
            "dklen": DKLEN,
            "value": value,
            "salt": salt,
            "hash": self._hash_profile_value(canonical, salt, PROFILE_HASH_ITERATIONS, DKLEN),
        }

    def _verify_protected_value(self, record: dict) -> bool:
        iterations = int(record.get("iterations", PROFILE_HASH_ITERATIONS))
        dklen = int(record.get("dklen", DKLEN))
        if record.get("scheme") == PROFILE_SCHEME:
            expected = self._hash_profile_value(
                self._serialize_value(record["value"]),
                record["salt"],
                iterations,
                dklen,
            )
        else:
            expected = self._hash_password(
                self._serialize_value(record["value"]),
                record["salt"],
                iterations,
                dklen,
            )
        return secrets.compare_digest(expected, record["hash"])

    def _compose_stats(self, username: str, profile: dict) -> dict:
        games = int(profile["games"])
        wins = int(profile["wins"])
        ranked_games = int(profile["ranked_games"])
        casual_games = int(profile["casual_games"])
        return {
            "username": username,
            "games": games,
            "wins": wins,
            "losses": int(profile["losses"]),
            "win_rate": 0.0 if games == 0 else wins / games,
            "created_at": profile["created_at"],
            "rating": int(profile["rating"]),
            "ranked_games": ranked_games,
            "ranked_wins": int(profile["ranked_wins"]),
            "ranked_losses": int(profile["ranked_losses"]),
            "ranked_win_rate": 0.0 if ranked_games == 0 else int(profile["ranked_wins"]) / ranked_games,
            "casual_games": casual_games,
            "casual_wins": int(profile["casual_wins"]),
            "casual_losses": int(profile["casual_losses"]),
            "casual_win_rate": 0.0 if casual_games == 0 else int(profile["casual_wins"]) / casual_games,
            "daily_replenish_date": profile["daily_replenish_date"],
            "daily_replenish_used": int(profile["daily_replenish_used"]),
        }

    @staticmethod
    def _default_profile() -> dict:
        return dict(PROFILE_DEFAULTS)

    @staticmethod
    def _fallback_profile_value(field: str, default: Any) -> Any:
        if field == "created_at":
            return datetime.now().isoformat(timespec="seconds")
        return default

    @staticmethod
    def _serialize_value(value: Any) -> str:
        return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))

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

    def _hash_profile_value(self, canonical_value: str, salt: str, iterations: int, dklen: int) -> str:
        payload = f"{self._profile_secret()}:{canonical_value}"
        return self._hash_password(payload, salt, iterations, dklen)

    def _profile_secret(self) -> str:
        if self.use_supabase:
            return SUPABASE_PROFILE_SECRET
        assert self.profile_key_path is not None
        if not self.profile_key_path.exists():
            self.profile_key_path.write_text(secrets.token_hex(32), encoding="utf-8")
        secret = self.profile_key_path.read_text(encoding="utf-8").strip()
        if not secret:
            secret = secrets.token_hex(32)
            self.profile_key_path.write_text(secret, encoding="utf-8")
        return secret
