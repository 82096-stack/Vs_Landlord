from __future__ import annotations

import asyncio
import json
import os
import random
import string
from collections import Counter
from dataclasses import dataclass, field
from typing import Any

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse

# ============================================================
# Constants
# ============================================================

RANK_MAP = {
    "3": 3, "4": 4, "5": 5, "6": 6, "7": 7, "8": 8, "9": 9,
    "10": 10, "J": 11, "Q": 12, "K": 13, "A": 14, "2": 15,
}
RANK_LABEL = {v: k for k, v in RANK_MAP.items()}
RANK_LABEL[16] = "SJ"
RANK_LABEL[17] = "BJ"

SUIT_ORDER = {"C": 0, "D": 1, "H": 2, "S": 3, "J": 4}
SUITS = ["C", "D", "H", "S"]
RANK_STRS = ["3", "4", "5", "6", "7", "8", "9", "10", "J", "Q", "K", "A", "2"]

# ============================================================
# Card Helpers
# ============================================================

def parse_card(s: str) -> dict | None:
    """Parse a card string like '3H', 'KS', 'SJ', 'BJ' into {rank, suit, str}."""
    if not s or not isinstance(s, str):
        return None
    s = s.strip()
    if s == "SJ":
        return {"rank": 16, "suit": "J", "str": "SJ"}
    if s == "BJ":
        return {"rank": 17, "suit": "J", "str": "BJ"}
    if len(s) < 2:
        return None
    suit = s[-1].upper()
    if suit not in SUIT_ORDER or suit == "J":
        return None
    rank_str = s[:-1].upper()
    rank = RANK_MAP.get(rank_str)
    if rank is None:
        return None
    return {"rank": rank, "suit": suit, "str": s.upper()}


def card_sort_key(card_str: str) -> tuple:
    c = parse_card(card_str)
    if c is None:
        return (99, 99)
    return (c["rank"], SUIT_ORDER.get(c["suit"], 99))


def build_deck() -> list[str]:
    deck: list[str] = []
    for rank_str in RANK_STRS:
        for suit in SUITS:
            deck.append(f"{rank_str}{suit}")
    deck.append("SJ")
    deck.append("BJ")
    return deck


def deal_cards(deck: list[str]) -> tuple[list[list[str]], list[str]]:
    hands: list[list[str]] = [[], [], []]
    for i in range(51):
        hands[i % 3].append(deck[i])
    bottom = deck[51:54]
    for h in hands:
        h.sort(key=card_sort_key)
    return hands, bottom


# ============================================================
# Combo System
# ============================================================

def _is_consecutive(ranks: list[int]) -> bool:
    return all(ranks[i] + 1 == ranks[i + 1] for i in range(len(ranks) - 1))


def _rank_with_count(counts: dict[int, int], n: int) -> int | None:
    for r, c in counts.items():
        if c == n:
            return r
    return None


def identify_combo(card_strs: list[str]) -> dict | None:
    """Identify combo type from a list of card strings. Returns combo dict or None."""
    if not card_strs:
        return None

    cards = [parse_card(s) for s in card_strs]
    if any(c is None for c in cards):
        return None

    cards.sort(key=lambda c: (c["rank"], SUIT_ORDER.get(c["suit"], 99)))
    counts: dict[int, int] = Counter(c["rank"] for c in cards)
    total = len(cards)
    ranks = sorted(counts.keys())

    # Rocket: SJ + BJ
    if total == 2 and counts.get(16, 0) == 1 and counts.get(17, 0) == 1:
        return {"type": "rocket", "main_rank": 17, "cards": [c["str"] for c in cards]}

    # Bomb: 4+ of same rank
    if len(counts) == 1 and total >= 4:
        return {"type": "bomb", "main_rank": ranks[0], "bomb_size": total,
                "cards": [c["str"] for c in cards]}

    # Four + 2 singles (6 cards)
    if total == 6:
        fr = _rank_with_count(counts, 4)
        if fr is not None:
            return {"type": "four_two_single", "main_rank": fr,
                    "cards": [c["str"] for c in cards]}

    # Four + 2 pairs (8 cards)
    if total == 8:
        fr = _rank_with_count(counts, 4)
        if fr is not None:
            rest = {r: c for r, c in counts.items() if r != fr}
            if len(rest) == 2 and all(c == 2 for c in rest.values()):
                return {"type": "four_two_pair", "main_rank": fr,
                        "cards": [c["str"] for c in cards]}

    # Airplane (trio_straight with attachments)
    trio_ranks = sorted(r for r, c in counts.items() if c >= 3 and r < 15)

    for seq_len in range(len(trio_ranks), 1, -1):
        for start in range(len(trio_ranks) - seq_len + 1):
            seq = trio_ranks[start:start + seq_len]
            if not _is_consecutive(seq):
                continue

            # Airplane + pairs (seq_len * 3 + seq_len * 2 cards)
            if total == seq_len * 5:
                rem = dict(counts)
                for r in seq:
                    rem[r] -= 3
                rem = {r: c for r, c in rem.items() if c > 0}
                if len(rem) == seq_len and all(c == 2 for c in rem.values()):
                    return {"type": "airplane_pair", "main_rank": seq[-1],
                            "seq_len": seq_len, "cards": [c["str"] for c in cards]}

            # Airplane + singles (seq_len * 3 + seq_len * 1 cards)
            if total == seq_len * 4:
                rem = dict(counts)
                for r in seq:
                    rem[r] -= 3
                rem = {r: c for r, c in rem.items() if c > 0}
                if len(rem) == seq_len and all(c == 1 for c in rem.values()):
                    return {"type": "airplane_single", "main_rank": seq[-1],
                            "seq_len": seq_len, "cards": [c["str"] for c in cards]}

    # Straight: 5+ consecutive singles (3-A only)
    if total >= 5 and len(counts) == total:
        if all(r < 15 for r in ranks) and _is_consecutive(ranks):
            return {"type": "straight", "main_rank": ranks[-1], "seq_len": len(ranks),
                    "cards": [c["str"] for c in cards]}

    # Pair straight: 3+ consecutive pairs (3-A only)
    if total >= 6 and total % 2 == 0:
        if all(c == 2 for c in counts.values()):
            if all(r < 15 for r in ranks) and _is_consecutive(ranks) and len(ranks) >= 3:
                return {"type": "pair_straight", "main_rank": ranks[-1],
                        "seq_len": len(ranks), "cards": [c["str"] for c in cards]}

    # Trio straight (without attachments): 2+ consecutive trios
    if total >= 6 and total % 3 == 0:
        if all(c == 3 for c in counts.values()):
            if all(r < 15 for r in ranks) and _is_consecutive(ranks) and len(ranks) >= 2:
                return {"type": "trio_straight", "main_rank": ranks[-1],
                        "seq_len": len(ranks), "cards": [c["str"] for c in cards]}

    # Trio + 1 (4 cards)
    if total == 4:
        tr = _rank_with_count(counts, 3)
        if tr is not None:
            return {"type": "trio_single", "main_rank": tr,
                    "cards": [c["str"] for c in cards]}

    # Trio + 2 (5 cards)
    if total == 5:
        tr = _rank_with_count(counts, 3)
        pr = _rank_with_count(counts, 2)
        if tr is not None and pr is not None and tr != pr:
            return {"type": "trio_pair", "main_rank": tr,
                    "cards": [c["str"] for c in cards]}

    # Trio (3 cards)
    if total == 3 and len(counts) == 1:
        return {"type": "trio", "main_rank": ranks[0],
                "cards": [c["str"] for c in cards]}

    # Pair (2 cards)
    if total == 2 and len(counts) == 1:
        return {"type": "pair", "main_rank": ranks[0],
                "cards": [c["str"] for c in cards]}

    # Single (1 card)
    if total == 1:
        return {"type": "single", "main_rank": ranks[0],
                "cards": [c["str"] for c in cards]}

    return None


def combo_pattern_key(combo: dict) -> tuple:
    t = combo["type"]
    if t == "bomb":
        return (t, combo.get("bomb_size", 4))
    return (t, combo.get("seq_len", 0))


def can_beat(combo: dict, target: dict | None) -> bool:
    if target is None:
        return True
    if target["type"] == "rocket":
        return False
    if combo["type"] == "rocket":
        return True
    if combo["type"] == "bomb":
        if target["type"] != "bomb":
            return True
        cb = combo.get("bomb_size", 4)
        tb = target.get("bomb_size", 4)
        if cb != tb:
            return cb > tb
        return combo["main_rank"] > target["main_rank"]
    if target["type"] == "bomb":
        return False
    if combo_pattern_key(combo) != combo_pattern_key(target):
        return False
    return combo["main_rank"] > target["main_rank"]


# ============================================================
# Game Room
# ============================================================

def _generate_room_id() -> str:
    return "".join(random.choices(string.ascii_uppercase + string.digits, k=6))


@dataclass
class SeatInfo:
    username: str = ""
    ws: WebSocket | None = None
    connected: bool = True
    is_ai: bool = False


class GameRoom:

    def __init__(self, room_id: str, host_username: str) -> None:
        self.room_id = room_id
        self.host_username = host_username
        self.max_players = 3
        self.seats: dict[int, SeatInfo] = {}
        self.state = "waiting"  # waiting | bidding | playing | finished

        # Game state — matches spec: players, hands, current_turn, last_play
        self.hands: dict[int, list[str]] = {}
        self.current_turn: int = 0
        self.last_play: dict | None = None  # {"seat": N, "combo": {...}}
        self.last_player_idx: int | None = None
        self.landlord_seat: int | None = None
        self.bottom_cards: list[str] = []
        self.bid_scores: dict[int, int] = {}
        self.pass_count: int = 0

        self._response_events: dict[int, asyncio.Event] = {}
        self._responses: dict[int, Any] = {}
        self._game_task: asyncio.Task | None = None

    # ---- seat management ----

    def add_player(self, username: str) -> int | None:
        if len(self.seats) >= self.max_players:
            return None
        for i in range(self.max_players):
            if i not in self.seats:
                self.seats[i] = SeatInfo(username=username)
                return i
        return None

    def remove_player(self, username: str) -> int | None:
        for idx, seat in list(self.seats.items()):
            if seat.username == username:
                del self.seats[idx]
                return idx
        return None

    def all_seats_filled(self) -> bool:
        return len(self.seats) == self.max_players

    def fill_with_ai(self) -> None:
        for i in range(self.max_players):
            if i not in self.seats:
                self.seats[i] = SeatInfo(username=f"AI-{i + 1}", is_ai=True)

    # ---- async helpers ----

    async def _ask_player(self, seat_idx: int, message: dict, timeout: float = 120.0) -> Any:
        seat = self.seats.get(seat_idx)
        if seat is None or seat.is_ai or not seat.connected or seat.ws is None:
            return None

        existing = self._responses.pop(seat_idx, None)
        if existing is not None:
            return existing

        event = asyncio.Event()
        self._response_events[seat_idx] = event
        try:
            await seat.ws.send_json(message)
            await asyncio.wait_for(event.wait(), timeout=timeout)
            return self._responses.pop(seat_idx, None)
        except asyncio.TimeoutError:
            return None
        finally:
            self._response_events.pop(seat_idx, None)

    def handle_response(self, seat_idx: int, data: Any) -> None:
        self._responses[seat_idx] = data
        event = self._response_events.get(seat_idx)
        if event:
            event.set()

    async def _broadcast(self, message: dict, exclude: int | None = None) -> None:
        for idx, seat in list(self.seats.items()):
            if idx == exclude:
                continue
            if seat.ws and not seat.is_ai and seat.connected:
                try:
                    await seat.ws.send_json(message)
                except Exception:
                    seat.connected = False

    async def _send_to(self, seat_idx: int, message: dict) -> None:
        seat = self.seats.get(seat_idx)
        if seat and seat.ws and not seat.is_ai and seat.connected:
            try:
                await seat.ws.send_json(message)
            except Exception:
                seat.connected = False

    # ---- player info ----

    def _seat_player_info(self, idx: int) -> dict:
        seat = self.seats.get(idx)
        if seat is None:
            return {"seat": idx, "username": None, "connected": False,
                    "hand_size": 0, "role": "farmer"}
        hand_size = len(self.hands.get(idx, []))
        role = "landlord" if idx == self.landlord_seat else "farmer"
        return {"seat": idx, "username": seat.username, "connected": seat.connected,
                "hand_size": hand_size, "role": role, "is_ai": seat.is_ai}

    def public_room_state(self) -> dict:
        return {
            "type": "room_state",
            "room_id": self.room_id,
            "state": self.state,
            "players": [self._seat_player_info(i) for i in range(self.max_players)],
            "host_username": self.host_username,
            "current_turn": self.current_turn if self.state == "playing" else None,
            "landlord_seat": self.landlord_seat,
        }

    # ---- game flow ----

    async def start_game(self) -> None:
        if self.state != "waiting":
            return
        self.state = "bidding"
        self.fill_with_ai()
        self._game_task = asyncio.create_task(self._run_game())

    async def _run_game(self) -> None:
        try:
            # Deal
            deck = build_deck()
            random.shuffle(deck)
            hands, bottom = deal_cards(deck)
            for i in range(3):
                self.hands[i] = hands[i]
            self.bottom_cards = bottom

            await self._broadcast({
                "type": "game_started",
                "bottom_cards_count": len(bottom),
                "players": [self._seat_player_info(i) for i in range(3)],
            })

            # Send hands to human players
            for i in range(3):
                seat = self.seats.get(i)
                if seat and not seat.is_ai:
                    await self._send_to(i, {
                        "type": "your_cards",
                        "cards": self.hands[i],
                        "hand_size": len(self.hands[i]),
                    })

            # Bidding phase
            landlord = await self._bidding_phase()
            if landlord is None:
                # No one bid — assign randomly
                landlord = random.randrange(3)
                self.bid_scores[landlord] = 1

            self.landlord_seat = landlord
            self.hands[landlord].extend(self.bottom_cards)
            self.hands[landlord].sort(key=card_sort_key)

            await self._broadcast({
                "type": "landlord_assigned",
                "seat": landlord,
                "username": self.seats[landlord].username,
                "bottom_cards": self.bottom_cards,
                "bid_scores": self.bid_scores,
            })

            # Send updated hand to landlord
            seat = self.seats.get(landlord)
            if seat and not seat.is_ai:
                await self._send_to(landlord, {
                    "type": "your_cards",
                    "cards": self.hands[landlord],
                    "hand_size": len(self.hands[landlord]),
                    "bottom_cards_added": self.bottom_cards,
                })

            # Playing phase
            self.state = "playing"
            winner = await self._playing_phase()

            # End phase
            self.state = "finished"
            winner_seat = self.seats.get(winner)
            await self._broadcast({
                "type": "game_over",
                "winner_seat": winner,
                "winner_name": winner_seat.username if winner_seat else f"Player {winner}",
                "winner_role": "landlord" if winner == self.landlord_seat else "farmer",
                "landlord_seat": self.landlord_seat,
                "landlord_won": winner == self.landlord_seat,
                "final_hands": {
                    str(i): self.hands.get(i, []) for i in range(3)
                },
            })

        except Exception as e:
            await self._broadcast({"type": "error", "message": f"Game error: {e}"})
            import traceback
            traceback.print_exc()
        finally:
            self.state = "finished"
            await self._broadcast(self.public_room_state())

    # ---- bidding ----

    async def _bidding_phase(self) -> int | None:
        # Determine first bidder (seat 0 goes first, or random)
        first = random.randrange(3)
        highest_bid = 0
        highest_seat: int | None = None

        await self._broadcast({"type": "phase", "phase": "bidding", "first_seat": first})

        for offset in range(3):
            seat_idx = (first + offset) % 3
            seat = self.seats[seat_idx]

            await self._broadcast({
                "type": "bid_turn",
                "seat": seat_idx,
                "username": seat.username,
                "highest_bid": highest_bid,
            })

            if seat.is_ai:
                # Simple AI: bid if hand has strong cards
                hand = self.hands[seat_idx]
                bid = self._ai_choose_bid(hand, highest_bid)
            else:
                allowed = [0] + list(range(highest_bid + 1, 4))
                resp = await self._ask_player(seat_idx, {
                    "type": "ask_bid",
                    "highest_bid": highest_bid,
                    "allowed_bids": allowed,
                })
                bid = 0
                if resp and isinstance(resp.get("score"), int):
                    bid = resp["score"]
                    if bid not in allowed:
                        bid = 0

            self.bid_scores[seat_idx] = bid

            await self._broadcast({
                "type": "bid_result",
                "seat": seat_idx,
                "username": seat.username,
                "score": bid,
            })

            if bid > highest_bid:
                highest_bid = bid
                highest_seat = seat_idx

            if highest_bid == 3:
                break

        return highest_seat if highest_bid > 0 else None

    @staticmethod
    def _ai_choose_bid(hand: list[str], current_highest: int) -> int:
        # Simple heuristic: count high-value cards
        score = 0
        for c in hand:
            parsed = parse_card(c)
            if parsed is None:
                continue
            r = parsed["rank"]
            if r == 17:
                score += 3
            elif r == 16:
                score += 2
            elif r == 15:
                score += 1
            elif r == 14:
                score += 0.5

        if score >= 5 and current_highest < 3:
            return min(3, current_highest + 1)
        if score >= 3 and current_highest < 2:
            return min(2, current_highest + 1)
        if score >= 2 and current_highest < 1:
            return 1
        return 0

    # ---- playing ----

    async def _playing_phase(self) -> int:
        assert self.landlord_seat is not None
        self.current_turn = self.landlord_seat
        self.last_play = None
        self.last_player_idx = None

        while True:
            # Check if all others passed — leader plays again
            if self.last_play is not None and self.current_turn == self.last_player_idx:
                self.last_play = None
                await self._broadcast({
                    "type": "new_round",
                    "leader_seat": self.current_turn,
                    "username": self.seats[self.current_turn].username,
                })

            seat = self.seats[self.current_turn]
            is_opening = self.last_play is None

            # Broadcast turn info
            await self._broadcast({
                "type": "turn_update",
                "current_turn": self.current_turn,
                "username": seat.username,
                "is_opening": is_opening,
                "last_play": self._last_play_dict(),
            })

            if seat.is_ai:
                action, play_cards = self._ai_choose_play(
                    self.current_turn, self.last_play
                )
            else:
                resp = await self._ask_player(self.current_turn, {
                    "type": "your_turn",
                    "is_opening": is_opening,
                    "last_play": self._last_play_dict(),
                    "hand": self.hands[self.current_turn],
                    "can_pass": not is_opening,
                })
                if resp is None:
                    action = "pass"
                    play_cards = []
                else:
                    action = resp.get("type", resp.get("action", "pass"))
                    play_cards = resp.get("cards", [])

            if action == "pass":
                if is_opening:
                    # Cannot pass when opening — force play single lowest
                    hand = sorted(self.hands[self.current_turn], key=card_sort_key)
                    if hand:
                        play_cards = [hand[0]]
                        action = "play_card"
                    else:
                        # Should not happen
                        self.current_turn = (self.current_turn + 1) % 3
                        continue
                else:
                    await self._broadcast({
                        "type": "player_passed",
                        "seat": self.current_turn,
                        "username": seat.username,
                    })
                    self.current_turn = (self.current_turn + 1) % 3
                    continue

            if action in ("play_card", "play"):
                result = self._process_play(self.current_turn, play_cards)
                if result.get("error"):
                    await self._send_to(self.current_turn, {
                        "type": "error",
                        "message": result["error"],
                    })
                    continue

                # Play accepted
                await self._broadcast({
                    "type": "play_result",
                    "seat": self.current_turn,
                    "username": seat.username,
                    "cards_played": play_cards,
                    "combo": result["combo"],
                    "remaining": len(self.hands[self.current_turn]),
                })

                # Update game state
                self.last_play = {
                    "seat": self.current_turn,
                    "combo": result["combo"],
                    "cards": play_cards,
                }
                self.last_player_idx = self.current_turn

                # Check win
                if not self.hands[self.current_turn]:
                    return self.current_turn

            self.current_turn = (self.current_turn + 1) % 3

    def _process_play(self, seat_idx: int, card_strs: list[str]) -> dict:
        # Validate cards are in hand
        hand = self.hands[seat_idx]
        hand_set = set(hand)
        if not all(c in hand_set for c in card_strs):
            return {"error": "手中没有这些牌"}

        # Check for duplicate selection
        selected_counts = Counter(card_strs)
        hand_counts = Counter(hand)
        for c, count in selected_counts.items():
            if hand_counts.get(c, 0) < count:
                return {"error": f"手牌中 {c} 数量不足"}

        # Identify combo
        combo = identify_combo(card_strs)
        if combo is None:
            return {"error": "无效牌型，请重新选择"}

        # Validate against last play
        if self.last_play is not None:
            target_combo = self.last_play["combo"]
            if not can_beat(combo, target_combo):
                return {"error": "这组牌压不过当前牌型"}

        # Remove cards from hand
        for c in card_strs:
            hand.remove(c)
        hand.sort(key=card_sort_key)

        return {"combo": combo}

    def _last_play_dict(self) -> dict | None:
        if self.last_play is None:
            return None
        return {
            "seat": self.last_play["seat"],
            "combo_type": self.last_play["combo"]["type"],
            "main_rank": self.last_play["combo"]["main_rank"],
            "cards_count": len(self.last_play["cards"]),
            "combo": self.last_play["combo"],
        }

    # ---- AI play ----

    def _ai_choose_play(self, seat_idx: int, last_play: dict | None
                        ) -> tuple[str, list[str]]:
        hand = self.hands[seat_idx]
        if last_play is None:
            # Opening: play lowest single
            hand_sorted = sorted(hand, key=card_sort_key)
            return ("play_card", [hand_sorted[0]])

        target_combo = last_play["combo"]

        # Generate candidate plays from hand
        candidates = self._generate_plays(hand)
        for combo in candidates:
            if can_beat(combo, target_combo):
                return ("play_card", combo["cards"])

        return ("pass", [])

    def _generate_plays(self, hand: list[str]) -> list[dict]:
        results: list[dict] = []
        counts = Counter(parse_card(c)["rank"] for c in hand if parse_card(c))
        grouped: dict[int, list[str]] = {}
        for c in hand:
            parsed = parse_card(c)
            if parsed is None:
                continue
            grouped.setdefault(parsed["rank"], []).append(c)

        # Singles
        for rank, cards in grouped.items():
            results.append({"type": "single", "main_rank": rank, "cards": [cards[0]]})

        # Pairs
        for rank, cards in grouped.items():
            if len(cards) >= 2:
                results.append({"type": "pair", "main_rank": rank, "cards": cards[:2]})

        # Trios
        for rank, cards in grouped.items():
            if len(cards) >= 3:
                results.append({"type": "trio", "main_rank": rank, "cards": cards[:3]})

        # Bombs
        for rank, cards in grouped.items():
            if len(cards) >= 4:
                for size in range(4, len(cards) + 1):
                    results.append({"type": "bomb", "main_rank": rank,
                                    "bomb_size": size, "cards": cards[:size]})

        # Rocket
        if 16 in grouped and 17 in grouped:
            results.append({"type": "rocket", "main_rank": 17,
                            "cards": [grouped[16][0], grouped[17][0]]})

        return sorted(results, key=lambda c: (c["type"] == "rocket",
                                               c["type"] == "bomb",
                                               c["main_rank"]))


# ============================================================
# Connection Manager
# ============================================================

class ConnectionManager:

    def __init__(self) -> None:
        self.rooms: dict[str, GameRoom] = {}
        self.ws_to_room: dict[int, tuple[str, int]] = {}  # id(ws) -> (room_id, seat)

    def create_room(self, host_username: str) -> GameRoom:
        room_id = _generate_room_id()
        while room_id in self.rooms:
            room_id = _generate_room_id()
        room = GameRoom(room_id, host_username)
        self.rooms[room_id] = room
        return room

    def get_room(self, room_id: str) -> GameRoom | None:
        return self.rooms.get(room_id)

    def remove_room(self, room_id: str) -> None:
        self.rooms.pop(room_id, None)
        self.ws_to_room = {
            wid: (rid, sid)
            for wid, (rid, sid) in self.ws_to_room.items()
            if rid != room_id
        }

    def register_ws(self, ws: WebSocket, room_id: str, seat: int) -> None:
        self.ws_to_room[id(ws)] = (room_id, seat)

    def unregister_ws(self, ws: WebSocket) -> tuple[str, int] | None:
        return self.ws_to_room.pop(id(ws), None)


# ============================================================
# FastAPI Application
# ============================================================

app = FastAPI(title="DouDiZhu WebSocket Server")

manager = ConnectionManager()

INDEX_HTML = """<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>斗地主</title>
<style>
*{box-sizing:border-box;margin:0;padding:0}
body{font-family:system-ui,sans-serif;background:#0d1b2a;color:#e0e1dd;min-height:100vh}
.container{max-width:900px;margin:0 auto;padding:20px}
h1{text-align:center;color:#f4a261;margin:20px 0}
.panel{background:#1b2838;border:1px solid #2a3a4a;border-radius:12px;padding:16px;margin:10px 0}
.panel h3{color:#e9c46a;margin-bottom:10px}
input,button{font-size:1rem;padding:8px 12px;border-radius:6px;border:1px solid #3a4a5a;background:#16212e;color:#e0e1dd;margin:4px}
button{background:#f4a261;color:#0d1b2a;font-weight:bold;cursor:pointer}
button:hover{background:#e76f51}
button:disabled{background:#555;cursor:not-allowed}
.row{display:flex;gap:6px;flex-wrap:wrap}
.row>*{flex:1;min-width:120px}
#log{background:#0a1118;border:1px solid #2a3a4a;border-radius:8px;padding:12px;height:280px;overflow-y:auto;font-family:monospace;font-size:.85rem;white-space:pre-wrap}
#hand{min-height:50px;background:#0a1118;border:1px solid #2a3a4a;border-radius:8px;padding:10px;font-family:monospace}
.card-btn{display:inline-block;background:#f4a261;color:#0d1b2a;padding:5px 8px;margin:3px;border-radius:5px;cursor:pointer;font-weight:bold;font-size:.85rem;border:2px solid transparent;user-select:none}
.card-btn.selected{border-color:#e76f51;background:#e76f51;color:#fff}
</style>
</head>
<body>
<div class="container">
<h1>斗地主 WebSocket</h1>
<div class="panel">
<h3>连接</h3>
<div class="row">
<input id="username" placeholder="昵称" maxlength="20">
<button onclick="createRoom()">创建房间</button>
<input id="joinRoomId" placeholder="房间号" maxlength="6">
<button onclick="joinRoom()">加入房间</button>
</div>
<button id="startBtn" onclick="startGame()" style="display:none;background:#2ecc71">开始游戏</button>
<div id="roomInfo" style="margin-top:8px;color:#e9c46a"></div>
</div>
<div class="panel" id="gamePanel" style="display:none">
<h3>游戏面板</h3>
<div id="gameState" style="color:#9aa5b1;margin-bottom:8px"></div>
<div id="handContainer">
<label>你的手牌 <span id="cardCount">0</span> 张</label>
<div id="hand"></div>
</div>
<div class="row" style="margin-top:8px">
<button id="playBtn" onclick="playCards()" disabled>出牌</button>
<button id="passBtn" onclick="send({type:'pass'})" disabled>过牌</button>
</div>
<div id="bidPanel" style="display:none;margin-top:8px">
<label>叫地主</label>
<div class="row" id="bidButtons"></div>
</div>
</div>
<div class="panel">
<h3>日志</h3>
<div id="log"></div>
</div>
</div>
<script>
let ws=null,roomId=null,mySeat=null,myCards=[],selected=new Set();
function L(m){const e=document.getElementById('log');const t=new Date().toLocaleTimeString();e.textContent+=`[${t}] ${m}\\n`;e.scrollTop=e.scrollHeight}
function connect(){const p=location.protocol==='https:'?'wss:':'ws:';ws=new WebSocket(`${p}//${location.host}/ws`);ws.onopen=()=>L('已连接');ws.onclose=()=>L('已断开');ws.onmessage=e=>{handle(JSON.parse(e.data))}}
function send(d){if(ws&&ws.readyState===WebSocket.OPEN)ws.send(JSON.stringify(d))}
function createRoom(){const u=document.getElementById('username').value.trim();if(!u){L('请输入昵称');return}send({type:'create_room',username:u})}
function joinRoom(){const u=document.getElementById('username').value.trim();const rid=document.getElementById('joinRoomId').value.trim().toUpperCase();if(!u||!rid){L('请输入昵称和房间号');return}send({type:'join_room',room_id:rid,username:u})}
function startGame(){send({type:'start_game'})}
function playCards(){if(selected.size===0)return;const cards=Array.from(selected).map(i=>myCards[i]);send({type:'play_card',cards});selected.clear();document.querySelectorAll('.card-btn.selected').forEach(b=>b.classList.remove('selected'))}
function handle(msg){
L(`收到: ${msg.type||msg.action||'?'}`);
switch(msg.type){
case'room_created':roomId=msg.room_id;document.getElementById('roomInfo').innerHTML=`房间 <b>${roomId}</b> 已创建`;document.getElementById('gamePanel').style.display='block';break;
case'room_joined':roomId=msg.room_id;mySeat=msg.seat;document.getElementById('roomInfo').innerHTML=`已加入 <b>${roomId}</b> 座位${mySeat+1}`;document.getElementById('gamePanel').style.display='block';break;
case'room_state':updateRoom(msg);break;
case'game_started':L('游戏开始!');document.getElementById('gameState').textContent='游戏中';break;
case'your_cards':myCards=msg.cards||[];renderHand();document.getElementById('cardCount').textContent=myCards.length;break;
case'ask_bid':showBid(msg.allowed_bids||[0,1,2,3]);break;
case'bid_turn':break;
case'bid_result':L(`座位${msg.seat} ${msg.username}: ${msg.score===0?'不叫':msg.score+'分'}`);break;
case'landlord_assigned':L(`座位${msg.seat} ${msg.username} 成为地主`);renderHand();break;
case'turn_update':L(`轮到 座位${msg.current_turn} ${msg.username}`);break;
case'your_turn':L(msg.is_opening?'请出牌（新回合）':`请出牌，压过 ${msg.last_play?.combo_type||'?'}`);document.getElementById('playBtn').disabled=false;document.getElementById('passBtn').disabled=msg.is_opening;renderHand();break;
case'play_result':L(`座位${msg.seat} 出牌 ${msg.combo.type} 剩${msg.remaining}张`);document.getElementById('playBtn').disabled=true;document.getElementById('passBtn').disabled=true;break;
case'player_passed':L(`座位${msg.seat} ${msg.username} 过牌`);break;
case'game_over':L(`游戏结束! 胜者: ${msg.winner_name} (${msg.winner_role})`);document.getElementById('playBtn').disabled=true;document.getElementById('passBtn').disabled=true;break;
case'error':L(`错误: ${msg.message}`);break;
}
}
function updateRoom(msg){
const ps=(msg.players||[]).map(p=>`座位${p.seat}: ${p.username||'空'} ${p.role==='landlord'?'[地主]':''} ${p.hand_size}张`).join(' | ');
L(`房间 ${msg.room_id} ${msg.state}\\n${ps}`);
if(msg.state==='waiting'&&roomId){
const filled=(msg.players||[]).filter(p=>p.username).length>=3;
document.getElementById('startBtn').style.display=filled?'inline-block':'none';
}
}
function renderHand(){const c=document.getElementById('hand');selected.clear();c.innerHTML=myCards.map((s,i)=>`<span class="card-btn" onclick="toggle(${i})" id="c-${i}">${s}</span>`).join('')}
function toggle(i){const b=document.getElementById('c-'+i);if(selected.has(i)){selected.delete(i);b.classList.remove('selected')}else{selected.add(i);b.classList.add('selected')}}
function showBid(allowed){const p=document.getElementById('bidPanel');const btns=document.getElementById('bidButtons');p.style.display='block';btns.innerHTML=allowed.map(b=>'<button onclick="send({type:\'bid\',score:'+b+'});document.getElementById(\'bidPanel\').style.display=\'none\'">'+(b===0?'不叫':b+'分')+'</button>').join('')}
connect();
</script>
</body>
</html>"""


@app.get("/", response_class=HTMLResponse)
async def root():
    return INDEX_HTML


@app.get("/health")
async def health():
    return {"status": "ok", "rooms": len(manager.rooms)}


@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()
    current_room_id: str | None = None
    current_seat: int | None = None

    try:
        while True:
            raw = await ws.receive_text()
            try:
                data = json.loads(raw)
            except json.JSONDecodeError:
                await ws.send_json({"type": "error", "message": "Invalid JSON"})
                continue

            msg_type = data.get("type", data.get("action", ""))

            # ---- room management ----

            if msg_type == "create_room":
                username = (data.get("username") or "player").strip()[:30]
                room = manager.create_room(username)
                seat = room.add_player(username)
                if seat is None:
                    await ws.send_json({"type": "error", "message": "Failed to join room"})
                    continue
                current_room_id = room.room_id
                current_seat = seat
                room.seats[seat].ws = ws
                manager.register_ws(ws, room.room_id, seat)
                await ws.send_json({
                    "type": "room_created",
                    "room_id": room.room_id,
                    "seat": seat,
                })
                await room._broadcast(room.public_room_state())

            elif msg_type == "join_room":
                room_id = data.get("room_id", "").strip().upper()
                username = (data.get("username") or "player").strip()[:30]
                room = manager.get_room(room_id)
                if room is None:
                    await ws.send_json({"type": "error", "message": "房间不存在"})
                    continue
                if room.state != "waiting":
                    await ws.send_json({"type": "error", "message": "游戏已开始"})
                    continue
                seat = room.add_player(username)
                if seat is None:
                    await ws.send_json({"type": "error", "message": "房间已满"})
                    continue
                current_room_id = room_id
                current_seat = seat
                room.seats[seat].ws = ws
                manager.register_ws(ws, room_id, seat)
                await ws.send_json({
                    "type": "room_joined",
                    "room_id": room_id,
                    "seat": seat,
                })
                await room._broadcast(room.public_room_state())

            elif msg_type == "start_game":
                room = manager.get_room(current_room_id) if current_room_id else None
                if room is None:
                    await ws.send_json({"type": "error", "message": "未在房间中"})
                    continue
                if room.host_username != room.seats.get(current_seat, SeatInfo()).username:
                    await ws.send_json({"type": "error", "message": "只有房主可以开始"})
                    continue
                if not room.all_seats_filled():
                    room.fill_with_ai()
                # Verify minimum 1 human
                if not any(s.is_ai is False and s.connected for s in room.seats.values()):
                    await ws.send_json({"type": "error", "message": "至少需要一名真人玩家"})
                    continue
                await room.start_game()

            elif msg_type == "leave_room":
                if current_room_id:
                    room = manager.get_room(current_room_id)
                    if room and current_seat is not None:
                        room.remove_player(room.seats[current_seat].username)
                        manager.unregister_ws(ws)
                        await room._broadcast(room.public_room_state())
                        human_count = sum(1 for s in room.seats.values()
                                         if not s.is_ai and s.connected)
                        if human_count == 0:
                            manager.remove_room(current_room_id)
                current_room_id = None
                current_seat = None

            # ---- game actions ----

            elif msg_type in ("bid", "play_card", "pass"):
                room = manager.get_room(current_room_id) if current_room_id else None
                if room is None or current_seat is None:
                    await ws.send_json({"type": "error", "message": "未在游戏中"})
                    continue

                if msg_type == "bid":
                    room.handle_response(current_seat, {
                        "score": data.get("score", 0),
                    })
                elif msg_type == "pass":
                    room.handle_response(current_seat, {
                        "type": "pass",
                        "cards": [],
                    })
                elif msg_type == "play_card":
                    room.handle_response(current_seat, {
                        "type": "play_card",
                        "cards": data.get("cards", []),
                    })

            else:
                await ws.send_json({"type": "error", "message": f"未知操作: {msg_type}"})

    except WebSocketDisconnect:
        pass
    except Exception:
        import traceback
        traceback.print_exc()
    finally:
        # Cleanup on disconnect
        if current_room_id is not None and current_seat is not None:
            info = manager.unregister_ws(ws)
            if info:
                room = manager.get_room(current_room_id)
                if room and current_seat in room.seats:
                    room.seats[current_seat].connected = False
                    room.seats[current_seat].ws = None
                    try:
                        await room._broadcast({
                            "type": "player_disconnected",
                            "seat": current_seat,
                            "username": room.seats[current_seat].username,
                        })
                    except Exception:
                        pass


# ============================================================
# Entry Point
# ============================================================

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", "8000"))
    uvicorn.run("main:app", host="0.0.0.0", port=port, log_level="info")
