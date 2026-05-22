from __future__ import annotations

import asyncio
import random
import string
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from fastapi import WebSocket

from ddz.ai import RuleBasedAI, TableKnowledge
from ddz.models import Card, Combo, Player, card_sort_key, format_cards
from ddz.rules import (
    MODE_RULES,
    can_beat,
    choose_marked_card,
    deal_cards,
    identify_combo,
)
from ddz.settlement import PVP_BASE_SCORE, build_settlement, score_deltas


def _generate_room_id() -> str:
    return "".join(random.choices(string.ascii_uppercase + string.digits, k=6))


def _card_dict(card: Card) -> dict:
    return {
        "rank": card.rank,
        "suit": card.suit,
        "deck_id": card.deck_id,
        "serial": card.serial,
        "label": card.label,
    }


def _combo_dict(combo: Combo) -> dict:
    return {
        "kind": combo.kind,
        "display_name": combo.display_name,
        "main_rank": combo.main_rank,
        "sequence_length": combo.sequence_length,
        "total_cards": combo.total_cards,
        "bomb_size": combo.bomb_size,
        "cards": [_card_dict(c) for c in combo.cards],
        "description": combo.describe(),
    }


def _hand_summary(hand: list[Card]) -> list[dict]:
    return [_card_dict(c) for c in hand]


@dataclass
class SeatInfo:
    username: str
    ws: WebSocket | None = None
    is_human: bool = True
    player: Player | None = None
    connected: bool = True
    send_queue: asyncio.Queue[dict | None] | None = None
    sender_task: asyncio.Task | None = None


class GameRoom:
    def __init__(self, room_id: str, mode: str, host_username: str):
        self.room_id = room_id
        self.mode = mode
        self.host_username = host_username
        self.rules = MODE_RULES[mode]
        self.max_players = self.rules["player_count"]

        self.seats: dict[int, SeatInfo] = {}
        self.state = "waiting"  # waiting | playing | finished
        self.game_task: asyncio.Task | None = None
        self.observers: list[WebSocket] = []

        self._ai_engine = RuleBasedAI()
        self._response_events: dict[int, asyncio.Event] = {}
        self._responses: dict[int, Any] = {}

        self.landlord_index: int | None = None
        self.bombs_played: int = 0
        self.marked_card: Card | None = None
        self.bottom_cards: list[Card] = []
        self.knowledge: TableKnowledge | None = None
        self.current_turn: int | None = None
        self.last_combo: Combo | None = None
        self.last_player_index: int | None = None
        self.highest_bid: int = 0
        self.effective_bid: int = 0
        self.redeal_count: int = 0
        self.report_multiplier: int = 0
        self.play_counts: list[int] = [0] * self.max_players
        self.base_score: int = PVP_BASE_SCORE
        self.match_kind: str = "casual"
        self.round_finished_callback = None

    # ------------------------------------------------------------------
    # Unified message envelope
    # ------------------------------------------------------------------

    def _envelope(self, type_: str, payload: dict, request_id: str | None = None) -> dict:
        return {
            "type": type_,
            "room_id": self.room_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "request_id": request_id,
            "payload": payload,
        }

    # ------------------------------------------------------------------
    # Event queue (non-blocking send)
    # ------------------------------------------------------------------

    def _start_sender(self, seat_index: int) -> None:
        seat = self.seats.get(seat_index)
        if seat is None or seat.ws is None:
            return
        if seat.sender_task is not None and not seat.sender_task.done():
            return

        seat.send_queue = asyncio.Queue(maxsize=256)
        seat.sender_task = asyncio.create_task(self._sender_loop(seat_index))

    async def _sender_loop(self, seat_index: int) -> None:
        seat = self.seats.get(seat_index)
        if seat is None or seat.send_queue is None:
            return
        queue = seat.send_queue
        ws = seat.ws
        try:
            while True:
                msg = await queue.get()
                if msg is None:  # sentinel to stop
                    break
                if ws is not None:
                    try:
                        await ws.send_json(msg)
                    except Exception:
                        seat.connected = False
                        break
        except asyncio.CancelledError:
            pass
        except Exception:
            seat.connected = False

    async def _stop_sender(self, seat_index: int) -> None:
        seat = self.seats.get(seat_index)
        if seat is None:
            return
        if seat.send_queue is not None:
            try:
                seat.send_queue.put_nowait(None)  # sentinel
            except asyncio.QueueFull:
                pass
        if seat.sender_task is not None and not seat.sender_task.done():
            seat.sender_task.cancel()
            try:
                await seat.sender_task
            except asyncio.CancelledError:
                pass
        seat.send_queue = None
        seat.sender_task = None

    async def _broadcast(self, type_: str, payload: dict, exclude: int | None = None, request_id: str | None = None) -> None:
        envelope = self._envelope(type_, payload, request_id)
        for observer in list(self.observers):
            try:
                await observer.send_json(envelope)
            except Exception:
                self.observers.remove(observer)
        for idx, seat in list(self.seats.items()):
            if idx == exclude:
                continue
            if seat.send_queue is not None and seat.is_human:
                try:
                    seat.send_queue.put_nowait(envelope)
                except asyncio.QueueFull:
                    pass

    async def _send_to(self, seat_index: int, type_: str, payload: dict, request_id: str | None = None) -> None:
        seat = self.seats.get(seat_index)
        if seat is None or seat.send_queue is None or not seat.is_human:
            return
        envelope = self._envelope(type_, payload, request_id)
        try:
            seat.send_queue.put_nowait(envelope)
        except asyncio.QueueFull:
            pass

    # ------------------------------------------------------------------
    # Request / response with request_id passthrough
    # ------------------------------------------------------------------

    async def _ask_player(self, seat_index: int, type_: str, payload: dict, timeout: float = 120.0, request_id: str | None = None) -> Any:
        seat = self.seats.get(seat_index)
        if seat is None or not seat.is_human:
            return None
        if seat.ws is None or not seat.connected:
            return None

        # Create event FIRST so handle_response always finds it.
        # Then check for any response that arrived before the event was created.
        event = asyncio.Event()
        self._response_events[seat_index] = event

        existing = self._responses.pop(seat_index, None)
        if existing is not None:
            event.set()

        await self._send_to(seat_index, type_, payload, request_id)

        try:
            await asyncio.wait_for(event.wait(), timeout=timeout)
            return self._responses.pop(seat_index, None)
        except asyncio.TimeoutError:
            return None
        finally:
            self._response_events.pop(seat_index, None)

    def handle_response(self, seat_index: int, data: Any) -> None:
        self._responses[seat_index] = data
        event = self._response_events.get(seat_index)
        if event:
            event.set()

    def add_observer(self, ws: WebSocket) -> None:
        if ws not in self.observers:
            self.observers.append(ws)

    # ------------------------------------------------------------------
    # Seat management
    # ------------------------------------------------------------------

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

    def fill_with_ai(self) -> None:
        for i in range(self.max_players):
            if i not in self.seats:
                name = f"AI-{i + 1}"
                self.seats[i] = SeatInfo(username=name, is_human=False)
                self.seats[i].player = Player(name=name, is_human=False)

    def all_seats_filled(self) -> bool:
        return len(self.seats) == self.max_players

    def human_count(self) -> int:
        return sum(1 for s in self.seats.values() if s.is_human)

    # ------------------------------------------------------------------
    # State snapshot
    # ------------------------------------------------------------------

    def full_state_snapshot(self, for_seat: int | None = None) -> dict:
        players_state = []
        for i in range(self.max_players):
            seat = self.seats.get(i)
            p = seat.player if seat else None
            entry = {
                "seat": i,
                "username": seat.username if seat else None,
                "is_human": seat.is_human if seat else False,
                "connected": seat.connected if seat else False,
                "hand_size": p.hand_size if p else 0,
                "role": p.role if p else "farmer",
                "bid_score": p.bid_score if p else 0,
                "revealed": p.revealed if p else False,
                "announced": p.announced if p else False,
            }
            # Show hand cards only to the owner
            if for_seat is not None and i == for_seat and p is not None:
                entry["hand"] = _hand_summary(p.hand)
            players_state.append(entry)

        return self._envelope("state_snapshot", {
            "room_id": self.room_id,
            "mode": self.mode,
            "mode_label": self.rules["label"],
            "state": self.state,
            "max_players": self.max_players,
            "host_username": self.host_username,
            "landlord_index": self.landlord_index,
            "current_turn": self.current_turn,
            "last_combo": _combo_dict(self.last_combo) if self.last_combo else None,
            "last_player_index": self.last_player_index,
            "bombs_played": self.bombs_played,
            "marked_card": self.marked_card.label if self.marked_card else None,
            "bottom_cards": _hand_summary(self.bottom_cards),
            "players": players_state,
        })

    def public_room_state(self) -> dict:
        players = []
        for i in range(self.max_players):
            seat = self.seats.get(i)
            if seat:
                p = seat.player
                players.append({
                    "seat": i,
                    "username": seat.username,
                    "is_human": seat.is_human,
                    "connected": seat.connected,
                    "hand_size": p.hand_size if p else 0,
                    "role": p.role if p else "farmer",
                })
            else:
                players.append({
                    "seat": i, "username": None, "is_human": False,
                    "connected": False, "hand_size": 0, "role": "farmer",
                })
        return self._envelope("room_state", {
            "mode": self.mode,
            "mode_label": self.rules["label"],
            "state": self.state,
            "max_players": self.max_players,
            "players": players,
            "host_username": self.host_username,
        })

    # ------------------------------------------------------------------
    # Game entry
    # ------------------------------------------------------------------

    async def start_game(self) -> None:
        self.state = "playing"
        self.fill_with_ai()

        players = []
        for i in range(self.max_players):
            seat = self.seats[i]
            if seat.player is None:
                seat.player = Player(name=seat.username, is_human=seat.is_human)
            players.append(seat.player)

        self.knowledge = TableKnowledge(self.mode, self.max_players)

        await self._broadcast("game_starting", {
            "mode": self.mode,
            "mode_label": self.rules["label"],
            "players": [
                {"seat": i, "username": s.username, "is_human": s.is_human}
                for i, s in sorted(self.seats.items())
            ],
        })

        self.game_task = asyncio.create_task(self._run_game())

    async def _run_game(self) -> None:
        try:
            players = [self.seats[i].player for i in range(self.max_players)]

            for p in players:
                p.role = "farmer"
                p.bid_score = 0
                p.bid_participated = False
                p.bombs_used = 0
                p.revealed = False
                p.announced = False
                p.report_level = 0

            self.bombs_played = 0
            self.landlord_index = None
            self.highest_bid = 0
            self.effective_bid = 0
            self.redeal_count = 0
            self.report_multiplier = 0
            self.play_counts = [0] * self.max_players

            dealer_round = 1
            while dealer_round <= 5:
                await self._deal_phase(players)
                if await self._bidding_phase(players):
                    break
                dealer_round += 1
                self.redeal_count += 1
                await self._broadcast("redeal", {
                    "message": f"无人叫地主，重新发牌 (第{dealer_round - 1}次)。",
                    "message_key": "redeal",
                    "params": {"round": dealer_round - 1},
                    "description": f"No one bid for landlord, redealing (attempt {dealer_round - 1}).",
                })

            if self.landlord_index is None:
                await self._broadcast("error", {
                    "message": "多次发牌后仍无人叫地主，游戏结束。",
                    "message_key": "no_landlord_after_redeals",
                    "description": "After multiple redeals, no one bid for landlord. Game aborted.",
                })
                self.state = "finished"
                return

            landlord = players[self.landlord_index]
            await self._reveal_phase(players, landlord)
            if self.mode == "extended":
                await self._report_phase(players)

            winner_idx = await self._playing_phase(players)
            await self._end_phase(players, winner_idx)
        except Exception as e:
            await self._broadcast("error", {"message": f"Game error: {e}"})
            import traceback
            traceback.print_exc()
        finally:
            self.state = "finished"
            await self._broadcast("room_state", self.public_room_state()["payload"])

    # ------------------------------------------------------------------
    # Deal phase
    # ------------------------------------------------------------------

    async def _deal_phase(self, players: list[Player]) -> None:
        self.marked_card = choose_marked_card(self.mode)
        self.bottom_cards = deal_cards(players, self.mode, self.marked_card)
        self.knowledge = TableKnowledge(self.mode, len(players))

        marker_holder = None
        for i, p in enumerate(players):
            if any(c.serial == self.marked_card.serial for c in p.hand):
                marker_holder = i
                break

        await self._broadcast("cards_dealt", {
            "marked_card": self.marked_card.label if self.marked_card else "",
            "marker_holder_seat": marker_holder,
            "marker_holder_name": players[marker_holder].name if marker_holder is not None else "",
            "bottom_count": len(self.bottom_cards),
            "description": f"Cards dealt. Marked card: {self.marked_card.label if self.marked_card else '?'}. {players[marker_holder].name if marker_holder is not None else '?'} holds the marked card and will bid first.",
        })

        for i, seat in self.seats.items():
            if seat.is_human and seat.player:
                await self._send_to(i, "your_hand", {
                    "cards": _hand_summary(seat.player.hand),
                })

    # ------------------------------------------------------------------
    # Bidding phase
    # ------------------------------------------------------------------

    async def _bidding_phase(self, players: list[Player]) -> bool:
        if self.mode == "classic":
            return await self._classic_bidding(players)
        return await self._extended_bidding(players)

    async def _extended_bidding(self, players: list[Player]) -> bool:
        marker_idx = 0
        if self.marked_card:
            for i, p in enumerate(players):
                if any(c.serial == self.marked_card.serial for c in p.hand):
                    marker_idx = i
                    break

        highest_bid = 0
        highest_idx: int | None = None

        for offset in range(len(players)):
            idx = (marker_idx + offset) % len(players)
            player = players[idx]
            seat = self.seats[idx]

            await self._broadcast("bidding_turn", {
                "seat": idx,
                "player_name": player.name,
                "highest_bid": highest_bid,
                "description": f"{player.name}'s turn to bid (current highest: {highest_bid}).",
            })

            if seat.is_human:
                allowed = [0] + list(range(highest_bid + 1, 4))
                resp = await self._ask_player(idx, "ask_bid", {
                    "highest_bid": highest_bid,
                    "allowed_bids": allowed,
                    "hand": _hand_summary(player.hand),
                })
                if resp is None or "bid" not in resp:
                    bid = 0
                else:
                    bid = int(resp["bid"])
                    if bid not in allowed:
                        bid = 0
            else:
                self.knowledge.init_beliefs(players, player.hand)
                bid = self._ai_engine.choose_bid(player.hand, self.mode, highest_bid)

            player.bid_score = bid
            player.bid_participated = True

            bid_desc = f"{player.name} bids {bid}." if bid > 0 else f"{player.name} passes on bidding."
            if bid > highest_bid:
                bid_desc += f" New highest bid: {bid}."
            await self._broadcast("bid_result", {
                "seat": idx,
                "player_name": player.name,
                "bid": bid,
                "description": bid_desc,
                "highest_bid": max(highest_bid, bid),
            })

            if bid > highest_bid:
                highest_bid = bid
                highest_idx = idx

            if highest_bid == 3:
                break

        if highest_idx is None or highest_bid == 0:
            await self._broadcast("no_bidder", {
                "message": "无人叫地主。",
                "message_key": "no_bidder",
            })
            return False

        self.highest_bid = highest_bid
        self.landlord_index = highest_idx
        await self._assign_landlord(players)
        return True

    async def _classic_bidding(self, players: list[Player]) -> bool:
        marker_idx = 0
        if self.marked_card:
            for i, p in enumerate(players):
                if any(c.serial == self.marked_card.serial for c in p.hand):
                    marker_idx = i
                    break

        turn_order = [(marker_idx + o) % len(players) for o in range(len(players))]
        candidate_idx: int | None = None
        desires = [0] * len(players)

        for order_pos, idx in enumerate(turn_order):
            player = players[idx]
            seat = self.seats[idx]

            await self._broadcast("bidding_turn", {
                "seat": idx,
                "player_name": player.name,
                "phase": "call",
                "description": f"{player.name}'s turn to call landlord.",
            })

            if seat.is_human:
                resp = await self._ask_player(idx, "ask_call", {
                    "hand": _hand_summary(player.hand),
                })
                wants = resp and resp.get("call", False)
            else:
                self.knowledge.init_beliefs(players, player.hand)
                wants = self._ai_engine.choose_bid(player.hand, self.mode, 0) > 0
                desires[idx] = self._ai_engine.preview_bid_strength(player.hand, self.mode)

            player.bid_score = 1 if wants else 0
            player.bid_participated = True

            call_desc = f"{player.name} calls landlord." if wants else f"{player.name} does not call."
            await self._broadcast("call_result", {
                "seat": idx,
                "player_name": player.name,
                "call": wants,
                "description": call_desc,
            })

            if wants:
                candidate_idx = idx
                remaining = turn_order[order_pos + 1:]
                break

        if candidate_idx is None:
            await self._broadcast("no_bidder", {
                "message": "无人叫地主。",
                "message_key": "no_bidder",
            })
            return False

        last_robber_idx: int | None = None
        for idx in remaining:
            player = players[idx]
            seat = self.seats[idx]
            current_bid = max(p.bid_score for p in players)

            await self._broadcast("bidding_turn", {
                "seat": idx,
                "player_name": player.name,
                "phase": "rob",
                "highest_bid": current_bid,
                "description": f"{player.name}'s turn to rob landlord (current bid: {current_bid}).",
            })

            if seat.is_human:
                resp = await self._ask_player(idx, "ask_rob", {
                    "highest_bid": current_bid,
                    "hand": _hand_summary(player.hand),
                })
                rob = resp is not None and resp.get("rob", False)
            else:
                if desires[idx] == 0:
                    desires[idx] = self._ai_engine.preview_bid_strength(player.hand, self.mode)
                rob = self._ai_engine.choose_rob(player.hand, self.mode, desires[idx], current_bid)

            player.bid_participated = True
            rob_desc = f"{player.name} robs the landlord." if rob else f"{player.name} does not rob."
            await self._broadcast("rob_result", {
                "seat": idx,
                "player_name": player.name,
                "rob": rob,
                "description": rob_desc,
            })

            if rob:
                candidate_idx = idx
                last_robber_idx = idx
                player.bid_score = max(player.bid_score, 2)

        if last_robber_idx is not None:
            caller_idx = turn_order[0]
            caller = players[caller_idx]
            current_bid = max(p.bid_score for p in players)
            seat = self.seats[caller_idx]

            await self._broadcast("bidding_turn", {
                "seat": caller_idx,
                "player_name": caller.name,
                "phase": "final_rob",
                "highest_bid": current_bid,
            })

            if seat.is_human:
                resp = await self._ask_player(caller_idx, "ask_rob", {
                    "highest_bid": current_bid,
                    "final": True,
                    "hand": _hand_summary(caller.hand),
                })
                final_rob = resp is not None and resp.get("rob", False)
            else:
                final_rob = self._ai_engine.choose_rob(caller.hand, self.mode, desires[caller_idx], current_bid)

            await self._broadcast("rob_result", {
                "seat": caller_idx,
                "player_name": caller.name,
                "rob": final_rob,
                "final": True,
            })

            if final_rob:
                candidate_idx = caller_idx
                caller.bid_score = max(caller.bid_score, 2)
            else:
                candidate_idx = last_robber_idx

        if candidate_idx is None:
            return False

        self.landlord_index = candidate_idx
        players[candidate_idx].bid_score = max(players[candidate_idx].bid_score, 1)
        self.highest_bid = max(1, max(player.bid_score for player in players))
        await self._assign_landlord(players)
        return True

    async def _assign_landlord(self, players: list[Player]) -> None:
        assert self.landlord_index is not None
        landlord = players[self.landlord_index]
        landlord.role = "landlord"
        landlord.hand.extend(self.bottom_cards)
        landlord.sort_hand()
        self.effective_bid = max(1, self.highest_bid)
        self.knowledge.set_landlord(self.landlord_index, self.bottom_cards)

        await self._broadcast("landlord_assigned", {
            "seat": self.landlord_index,
            "player_name": landlord.name,
            "bottom_cards": _hand_summary(self.bottom_cards),
            "description": f"{landlord.name} becomes the landlord and receives {len(self.bottom_cards)} bottom cards: {format_cards(self.bottom_cards)}.",
        })

        for i, seat in self.seats.items():
            if seat.is_human and seat.player and i == self.landlord_index:
                await self._send_to(i, "your_hand", {
                    "cards": _hand_summary(seat.player.hand),
                })

    # ------------------------------------------------------------------
    # Reveal & Report
    # ------------------------------------------------------------------

    async def _reveal_phase(self, players: list[Player], landlord: Player) -> None:
        seat = self.seats[self.landlord_index]
        if seat.is_human:
            resp = await self._ask_player(self.landlord_index, "ask_reveal", {
                "hand": _hand_summary(landlord.hand),
            })
            reveal = resp is not None and resp.get("reveal", False)
        else:
            reveal = self._ai_engine.choose_reveal(landlord.hand, self.mode, "landlord")

        landlord.revealed = reveal
        if reveal and self.mode == "extended":
            self.effective_bid = max(self.effective_bid, 4)
        await self._broadcast("reveal_result", {
            "seat": self.landlord_index,
            "player_name": landlord.name,
            "reveal": reveal,
        })

    async def _report_phase(self, players: list[Player]) -> None:
        for idx, player in enumerate(players):
            report_level = self._detect_report(player.hand)
            if report_level <= 0:
                continue
            seat = self.seats[idx]
            if seat.is_human:
                resp = await self._ask_player(idx, "ask_report", {
                    "report_level": report_level,
                    "hand": _hand_summary(player.hand),
                })
                announce = resp is not None and resp.get("report", False)
            else:
                announce = self._ai_engine.choose_report(player.hand, self.mode, player.role, report_level)

            if announce:
                player.announced = True
                player.report_level = report_level
                self.report_multiplier += report_level
                label = "双报道" if report_level >= 2 else "报道"
                await self._broadcast("report_result", {
                    "seat": idx,
                    "player_name": player.name,
                    "report": True,
                    "report_label": label,
                })

    @staticmethod
    def _detect_report(hand: list[Card]) -> int:
        counts = Counter(c.rank for c in hand)
        if any(count >= 8 for count in counts.values()):
            return 2
        if counts.get(16, 0) >= 1 and counts.get(17, 0) >= 1:
            return 1
        if any(count >= 7 for count in counts.values()):
            return 1
        return 0

    # ------------------------------------------------------------------
    # Playing phase
    # ------------------------------------------------------------------

    async def _playing_phase(self, players: list[Player]) -> int:
        assert self.landlord_index is not None
        self.current_turn = self.landlord_index
        self.last_combo = None
        self.last_player_index = None

        while True:
            if self.last_combo is not None and self.current_turn == self.last_player_index:
                await self._broadcast("new_round", {
                    "leader_seat": self.current_turn,
                    "leader_name": players[self.current_turn].name,
                    "description": f"Everyone else passed. {players[self.current_turn].name} starts a new trick.",
                })
                self.last_combo = None

            player = players[self.current_turn]
            seat = self.seats[self.current_turn]
            opened_round = self.last_combo is None
            has_human = any(s.is_human for s in self.seats.values())

            # Notify other players whose turn it is
            turn_desc = f"It's {player.name}'s turn."
            if opened_round:
                turn_desc += " New trick — free play."
            elif self.last_combo is not None:
                last_name = players[self.last_player_index].name if self.last_player_index is not None else "?"
                turn_desc += f" Must beat {last_name}'s {self.last_combo.describe()}."
            await self._broadcast("play_turn", {
                "seat": self.current_turn,
                "player_name": player.name,
                "is_opening": opened_round,
                "last_combo": _combo_dict(self.last_combo) if self.last_combo else None,
                "last_player_seat": self.last_player_index,
                "last_player_name": players[self.last_player_index].name if self.last_player_index is not None else None,
                "combo_display": self.last_combo.describe() if self.last_combo else None,
                "description": turn_desc,
            }, exclude=self.current_turn)

            if seat.is_human:
                resp = await self._ask_player(self.current_turn, "ask_play", {
                    "is_opening": opened_round,
                    "last_combo": _combo_dict(self.last_combo) if self.last_combo else None,
                    "last_player_name": players[self.last_player_index].name if self.last_player_index is not None else None,
                    "hand": _hand_summary(player.hand),
                    "can_pass": not opened_round,
                    "combo_display": self.last_combo.describe() if self.last_combo else None,
                })

                action = resp.get("action") if resp else "pass"
            else:
                if self.knowledge is None:
                    self.knowledge = TableKnowledge(self.mode, len(players))
                if not has_human:
                    self.knowledge.init_beliefs(players, player.hand)
                chosen = self._ai_engine.choose_play(
                    player, players, self.last_combo, self.last_player_index, self.knowledge,
                )
                action = "play" if chosen else "pass"
                if chosen:
                    serial_to_idx = {c.serial: i for i, c in enumerate(player.hand)}
                    resp = {"cards": [serial_to_idx[c.serial] for c in chosen if c.serial in serial_to_idx]}
                else:
                    resp = {}

            if action == "pass":
                if not opened_round:
                    pass_desc = f"{player.name} passes. ({player.hand_size} cards remaining)"
                    await self._broadcast("play_action", {
                        "seat": self.current_turn,
                        "player_name": player.name,
                        "action": "pass",
                        "remaining_count": player.hand_size,
                        "current_turn": (self.current_turn + 1) % len(players),
                        "description": pass_desc,
                    })
                    self.current_turn = (self.current_turn + 1) % len(players)
                    continue

            indices = resp.get("cards", []) if resp else []
            if not isinstance(indices, (list, tuple)) or not indices:
                if not opened_round:
                    pass_desc = f"{player.name} passes. ({player.hand_size} cards remaining)"
                    await self._broadcast("play_action", {
                        "seat": self.current_turn,
                        "player_name": player.name,
                        "action": "pass",
                        "remaining_count": player.hand_size,
                        "current_turn": (self.current_turn + 1) % len(players),
                        "description": pass_desc,
                    })
                    self.current_turn = (self.current_turn + 1) % len(players)
                    continue

            selected = [player.hand[i] for i in indices if 0 <= i < len(player.hand)]
            if not selected:
                self.current_turn = (self.current_turn + 1) % len(players)
                continue

            combo = identify_combo(selected, self.mode)
            if combo is None:
                await self._send_to(self.current_turn, "error", {
                    "message": "无效牌型，请重新选择。",
                    "message_key": "invalid_combo",
                })
                continue

            if self.last_combo is not None and not can_beat(combo, self.last_combo):
                await self._send_to(self.current_turn, "error", {
                    "message": "这组牌压不过当前牌型。",
                    "message_key": "cannot_beat",
                })
                continue

            if not self._check_bomb_limit(player, combo):
                await self._send_to(self.current_turn, "error", {
                    "message": "炸弹/王炸次数已用完。",
                    "message_key": "bomb_limit_exceeded",
                })
                continue

            # Remove cards and update state BEFORE broadcasting
            selected_serials = {c.serial for c in selected}
            player.hand = [c for c in player.hand if c.serial not in selected_serials]
            player.sort_hand()

            if combo.kind in {"bomb", "rocket"}:
                self.bombs_played += 1
                player.bombs_used += 1
            self.play_counts[self.current_turn] += 1

            if self.knowledge:
                self.knowledge.record_play(self.current_turn, combo, opened_round)

            self.last_combo = combo
            self.last_player_index = self.current_turn

            next_turn = (self.current_turn + 1) % len(players)

            play_desc = f"{player.name} plays {combo.describe()}: {format_cards(selected)}. ({player.hand_size} cards left)"
            # Enriched play_action — complete info for UI rendering
            await self._broadcast("play_action", {
                "seat": self.current_turn,
                "player_name": player.name,
                "action": "play",
                "combo": _combo_dict(combo),
                "cards_played": [_card_dict(c) for c in selected],
                "remaining_count": player.hand_size,
                "current_turn": next_turn,
                "last_combo": _combo_dict(self.last_combo),
                "combo_display": combo.describe(),
                "description": play_desc,
            })

            if not player.hand:
                await self._broadcast("player_empty", {
                    "seat": self.current_turn,
                    "player_name": player.name,
                })
                return self.current_turn

            self.current_turn = next_turn

    def _check_bomb_limit(self, player: Player, combo: Combo) -> bool:
        if self.mode != "extended":
            return True
        if combo.kind not in {"bomb", "rocket"}:
            return True
        if player.role == "landlord":
            return True
        if not player.bid_participated:
            return True
        limit = 2 if player.bid_score >= 2 else 1
        return player.bombs_used < limit

    # ------------------------------------------------------------------
    # End phase
    # ------------------------------------------------------------------

    async def _end_phase(self, players: list[Player], winner_idx: int) -> None:
        winner = players[winner_idx]
        landlord_won = winner.role == "landlord"
        settlement = build_settlement(
            players=players,
            winner=winner,
            landlord_index=self.landlord_index,
            base_score=self.base_score,
            highest_bid=self.highest_bid,
            effective_bid=self.effective_bid,
            bombs_played=self.bombs_played,
            redeal_count=self.redeal_count,
            report_multiplier=self.report_multiplier,
            marked_card=self.marked_card,
            play_counts=self.play_counts,
            score_enabled=self.match_kind != "casual_no_score",
        )
        deltas = score_deltas(players, settlement, self.landlord_index, landlord_won)
        all_hands = {}
        for i, p in enumerate(players):
            all_hands[self.seats[i].username] = _hand_summary(p.hand)

        await self._broadcast("game_over", {
            "winner_seat": winner_idx,
            "winner_name": winner.name,
            "winner_role": winner.role,
            "landlord_seat": self.landlord_index,
            "landlord_name": players[self.landlord_index].name if self.landlord_index is not None else "",
            "landlord_won": landlord_won,
            "bombs_played": self.bombs_played,
            "highest_bid": self.highest_bid,
            "settlement": settlement,
            "score_deltas": deltas,
            "final_hands": all_hands,
            "description": f"Game over! {winner.name} ({winner.role}) wins. Landlord {'won' if landlord_won else 'lost'}. Final scores: {deltas}.",
        })
        if self.round_finished_callback is not None:
            await self.round_finished_callback(self, players, winner_idx, settlement, deltas)
        self.state = "finished"
