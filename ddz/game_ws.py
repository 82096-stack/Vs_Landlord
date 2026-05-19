from __future__ import annotations

import asyncio
import random
import secrets
import string
from collections import Counter
from dataclasses import dataclass, field
from typing import Any

from fastapi import WebSocket

from ddz.ai import RuleBasedAI, TableKnowledge
from ddz.models import Card, Combo, Player, card_sort_key, format_cards
from ddz.rules import (
    MODE_RULES,
    can_beat,
    choose_marked_card,
    deal_cards,
    generate_candidate_plays,
    identify_combo,
)


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


def _all_hands(players: list[Player]) -> list[list[dict]]:
    return [_hand_summary(p.hand) for p in players]


def _generate_room_id() -> str:
    return "".join(random.choices(string.ascii_uppercase + string.digits, k=6))


@dataclass
class SeatInfo:
    username: str
    ws: WebSocket | None = None
    is_human: bool = True
    ai: Any = None
    player: Player | None = None
    connected: bool = True


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

        self._ai_engine = RuleBasedAI()
        self._response_events: dict[int, asyncio.Event] = {}
        self._responses: dict[int, Any] = {}

        self.landlord_index: int | None = None
        self.bombs_played: int = 0
        self.marked_card: Card | None = None
        self.bottom_cards: list[Card] = []
        self.knowledge: TableKnowledge | None = None

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

    # ---- async helpers ----

    async def _ask_player(self, seat_index: int, message: dict, timeout: float = 120.0) -> Any:
        seat = self.seats.get(seat_index)
        if seat is None or not seat.is_human:
            return None

        if seat.ws is None or not seat.connected:
            return None

        # check if a response already arrived before the event was set up
        existing = self._responses.get(seat_index)
        if existing is not None:
            self._responses.pop(seat_index, None)
            return existing

        event = asyncio.Event()
        self._response_events[seat_index] = event

        try:
            await seat.ws.send_json(message)
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

    async def _broadcast(self, message: dict, exclude: int | None = None) -> None:
        for idx, seat in list(self.seats.items()):
            if idx == exclude:
                continue
            if seat.ws and seat.is_human:
                try:
                    await seat.ws.send_json(message)
                except Exception:
                    seat.connected = False

    async def _send_to(self, seat_index: int, message: dict) -> None:
        seat = self.seats.get(seat_index)
        if seat and seat.ws and seat.is_human:
            try:
                await seat.ws.send_json(message)
            except Exception:
                seat.connected = False

    # ---- public state ----

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
                players.append({"seat": i, "username": None, "is_human": False, "connected": False, "hand_size": 0, "role": "farmer"})
        return {
            "type": "room_state",
            "room_id": self.room_id,
            "mode": self.mode,
            "mode_label": self.rules["label"],
            "state": self.state,
            "max_players": self.max_players,
            "players": players,
            "host_username": self.host_username,
        }

    # ---- game entry ----

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

        await self._broadcast({"type": "game_starting", "mode": self.mode, "mode_label": self.rules["label"], "players": [{"seat": i, "username": s.username, "is_human": s.is_human} for i, s in sorted(self.seats.items())]})

        self.game_task = asyncio.create_task(self._run_game())

    async def _run_game(self) -> None:
        try:
            players = [self.seats[i].player for i in range(self.max_players)]

            # reset
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

            # re-deal loop: try up to 5 times if no one bids
            dealer_round = 1
            while dealer_round <= 5:
                await self._deal_phase(players)
                if await self._bidding_phase(players):
                    break
                dealer_round += 1
                await self._broadcast({"type": "redeal", "message": f"无人叫地主，重新发牌 (第{dealer_round - 1}次)。"})

            if self.landlord_index is None:
                await self._broadcast({"type": "error", "message": "多次发牌后仍无人叫地主，游戏结束。"})
                self.state = "finished"
                return

            landlord = players[self.landlord_index]
            await self._reveal_phase(players, landlord)
            if self.mode == "extended":
                await self._report_phase(players)

            winner = await self._playing_phase(players)
            await self._end_phase(players, winner)
        except Exception as e:
            await self._broadcast({"type": "error", "message": f"Game error: {e}"})
            import traceback
            traceback.print_exc()
        finally:
            self.state = "finished"
            await self._broadcast(self.public_room_state())

    # ---- deal phase ----

    async def _deal_phase(self, players: list[Player]) -> None:
        self.marked_card = choose_marked_card(self.mode)
        self.bottom_cards = deal_cards(players, self.mode, self.marked_card)
        self.knowledge = TableKnowledge(self.mode, len(players))

        # find marker holder
        marker_holder = None
        for i, p in enumerate(players):
            if any(c.serial == self.marked_card.serial for c in p.hand):
                marker_holder = i
                break

        await self._broadcast({
            "type": "cards_dealt",
            "marked_card": self.marked_card.label if self.marked_card else "",
            "marker_holder_seat": marker_holder,
            "marker_holder_name": players[marker_holder].name if marker_holder is not None else "",
            "bottom_count": len(self.bottom_cards),
        })

        for i, seat in self.seats.items():
            if seat.is_human and seat.player:
                await self._send_to(i, {
                    "type": "your_hand",
                    "cards": _hand_summary(seat.player.hand),
                })

    # ---- bidding phase ----

    async def _bidding_phase(self, players: list[Player]) -> bool:
        if self.mode == "classic":
            return await self._classic_bidding(players)
        return await self._extended_bidding(players)

    async def _extended_bidding(self, players: list[Player]) -> bool:
        # find marker holder as starting point
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

            await self._broadcast({
                "type": "bidding_turn",
                "seat": idx,
                "player_name": player.name,
                "highest_bid": highest_bid,
            })

            if seat.is_human:
                allowed = [0] + list(range(highest_bid + 1, 4))
                resp = await self._ask_player(idx, {
                    "type": "ask_bid",
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

            await self._broadcast({
                "type": "bid_result",
                "seat": idx,
                "player_name": player.name,
                "bid": bid,
            })

            if bid > highest_bid:
                highest_bid = bid
                highest_idx = idx

            if highest_bid == 3:
                break

        if highest_idx is None or highest_bid == 0:
            await self._broadcast({"type": "no_bidder", "message": "无人叫地主。"})
            return False

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

        # call phase
        for order_pos, idx in enumerate(turn_order):
            player = players[idx]
            seat = self.seats[idx]

            await self._broadcast({
                "type": "bidding_turn",
                "seat": idx,
                "player_name": player.name,
                "phase": "call",
            })

            if seat.is_human:
                resp = await self._ask_player(idx, {
                    "type": "ask_call",
                    "hand": _hand_summary(player.hand),
                })
                wants = resp and resp.get("call", False)
            else:
                self.knowledge.init_beliefs(players, player.hand)
                wants = self._ai_engine.choose_bid(player.hand, self.mode, 0) > 0
                desires[idx] = self._ai_engine.preview_bid_strength(player.hand, self.mode)

            player.bid_score = 1 if wants else 0
            player.bid_participated = True

            await self._broadcast({
                "type": "call_result",
                "seat": idx,
                "player_name": player.name,
                "call": wants,
            })

            if wants:
                candidate_idx = idx
                remaining = turn_order[order_pos + 1:]
                break

        if candidate_idx is None:
            await self._broadcast({"type": "no_bidder", "message": "无人叫地主。"})
            return False

        # rob phase
        last_robber_idx: int | None = None
        for idx in remaining:
            player = players[idx]
            seat = self.seats[idx]
            current_bid = max(p.bid_score for p in players)

            await self._broadcast({
                "type": "bidding_turn",
                "seat": idx,
                "player_name": player.name,
                "phase": "rob",
                "highest_bid": current_bid,
            })

            if seat.is_human:
                resp = await self._ask_player(idx, {
                    "type": "ask_rob",
                    "highest_bid": current_bid,
                    "hand": _hand_summary(player.hand),
                })
                rob = resp is not None and resp.get("rob", False)
            else:
                if desires[idx] == 0:
                    desires[idx] = self._ai_engine.preview_bid_strength(player.hand, self.mode)
                rob = self._ai_engine.choose_rob(player.hand, self.mode, desires[idx], current_bid)

            player.bid_participated = True
            await self._broadcast({
                "type": "rob_result",
                "seat": idx,
                "player_name": player.name,
                "rob": rob,
            })

            if rob:
                candidate_idx = idx
                last_robber_idx = idx
                player.bid_score = max(player.bid_score, 2)

        # final rob: original caller gets a second chance if someone else robbed
        if last_robber_idx is not None:
            caller_idx = turn_order[0]  # original caller
            caller = players[caller_idx]
            current_bid = max(p.bid_score for p in players)
            seat = self.seats[caller_idx]

            await self._broadcast({
                "type": "bidding_turn",
                "seat": caller_idx,
                "player_name": caller.name,
                "phase": "final_rob",
                "highest_bid": current_bid,
            })

            if seat.is_human:
                resp = await self._ask_player(caller_idx, {
                    "type": "ask_rob",
                    "highest_bid": current_bid,
                    "final": True,
                    "hand": _hand_summary(caller.hand),
                })
                final_rob = resp is not None and resp.get("rob", False)
            else:
                final_rob = self._ai_engine.choose_rob(caller.hand, self.mode, desires[caller_idx], current_bid)

            await self._broadcast({
                "type": "rob_result",
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
        await self._assign_landlord(players)
        return True

    async def _assign_landlord(self, players: list[Player]) -> None:
        assert self.landlord_index is not None
        landlord = players[self.landlord_index]
        landlord.role = "landlord"
        landlord.hand.extend(self.bottom_cards)
        landlord.sort_hand()
        self.knowledge.set_landlord(self.landlord_index, self.bottom_cards)

        await self._broadcast({
            "type": "landlord_assigned",
            "seat": self.landlord_index,
            "player_name": landlord.name,
            "bottom_cards": _hand_summary(self.bottom_cards),
        })

        # send updated hand to landlord
        for i, seat in self.seats.items():
            if seat.is_human and seat.player and i == self.landlord_index:
                await self._send_to(i, {
                    "type": "your_hand",
                    "cards": _hand_summary(seat.player.hand),
                })

    # ---- reveal ----

    async def _reveal_phase(self, players: list[Player], landlord: Player) -> None:
        seat = self.seats[self.landlord_index]
        if seat.is_human:
            resp = await self._ask_player(self.landlord_index, {
                "type": "ask_reveal",
                "hand": _hand_summary(landlord.hand),
            })
            reveal = resp is not None and resp.get("reveal", False)
        else:
            reveal = self._ai_engine.choose_reveal(landlord.hand, self.mode, "landlord")

        landlord.revealed = reveal
        await self._broadcast({
            "type": "reveal_result",
            "seat": self.landlord_index,
            "player_name": landlord.name,
            "reveal": reveal,
        })

    # ---- report ----

    async def _report_phase(self, players: list[Player]) -> None:
        for idx, player in enumerate(players):
            report_level = self._detect_report(player.hand)
            if report_level <= 0:
                continue
            seat = self.seats[idx]
            if seat.is_human:
                resp = await self._ask_player(idx, {
                    "type": "ask_report",
                    "report_level": report_level,
                    "hand": _hand_summary(player.hand),
                })
                announce = resp is not None and resp.get("report", False)
            else:
                announce = self._ai_engine.choose_report(player.hand, self.mode, player.role, report_level)

            if announce:
                player.announced = True
                player.report_level = report_level
                label = "双报道" if report_level >= 2 else "报道"
                await self._broadcast({
                    "type": "report_result",
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

    # ---- playing phase ----

    async def _playing_phase(self, players: list[Player]) -> int:
        assert self.landlord_index is not None
        current_idx = self.landlord_index
        last_combo: Combo | None = None
        last_player_idx: int | None = None

        while True:
            # if everyone passed, last player leads
            if last_combo is not None and current_idx == last_player_idx:
                await self._broadcast({
                    "type": "new_round",
                    "leader_seat": current_idx,
                    "leader_name": players[current_idx].name,
                })
                last_combo = None

            player = players[current_idx]
            seat = self.seats[current_idx]
            opened_round = last_combo is None
            has_human = any(s.is_human for s in self.seats.values())

            await self._broadcast({
                "type": "play_turn",
                "seat": current_idx,
                "player_name": player.name,
                "is_opening": opened_round,
                "last_combo": _combo_dict(last_combo) if last_combo else None,
                "last_player_seat": last_player_idx,
                "last_player_name": players[last_player_idx].name if last_player_idx is not None else None,
            }, exclude=current_idx)

            if seat.is_human:
                resp = await self._ask_player(current_idx, {
                    "type": "ask_play",
                    "is_opening": opened_round,
                    "last_combo": _combo_dict(last_combo) if last_combo else None,
                    "last_player_name": players[last_player_idx].name if last_player_idx is not None else None,
                    "hand": _hand_summary(player.hand),
                    "can_pass": not opened_round,
                })

                action = resp.get("action") if resp else "pass"
            else:
                if self.knowledge is None:
                    self.knowledge = TableKnowledge(self.mode, len(players))
                if not has_human:
                    self.knowledge.init_beliefs(players, player.hand)
                chosen = self._ai_engine.choose_play(
                    player, players, last_combo, last_player_idx, self.knowledge,
                )
                action = "play" if chosen else "pass"
                if chosen:
                    serial_to_idx = {c.serial: i for i, c in enumerate(player.hand)}
                    resp = {"cards": [serial_to_idx[c.serial] for c in chosen if c.serial in serial_to_idx]}
                else:
                    resp = {}

            if action == "pass":
                if not opened_round:
                    await self._broadcast({
                        "type": "play_action",
                        "seat": current_idx,
                        "player_name": player.name,
                        "action": "pass",
                    })
                    current_idx = (current_idx + 1) % len(players)
                    continue

            # parse selected cards
            indices = resp.get("cards", []) if resp else []
            if not isinstance(indices, (list, tuple)) or not indices:
                # invalid play treated as pass
                if not opened_round:
                    await self._broadcast({
                        "type": "play_action",
                        "seat": current_idx,
                        "player_name": player.name,
                        "action": "pass",
                    })
                    current_idx = (current_idx + 1) % len(players)
                    continue

            selected = [player.hand[i] for i in indices if 0 <= i < len(player.hand)]
            if not selected:
                current_idx = (current_idx + 1) % len(players)
                continue

            combo = identify_combo(selected, self.mode)
            if combo is None:
                await self._send_to(current_idx, {
                    "type": "error",
                    "message": "无效牌型，请重新选择。",
                })
                continue

            if last_combo is not None and not can_beat(combo, last_combo):
                await self._send_to(current_idx, {
                    "type": "error",
                    "message": "这组牌压不过当前牌型。",
                })
                continue

            if not self._check_bomb_limit(player, combo):
                await self._send_to(current_idx, {
                    "type": "error",
                    "message": "炸弹/王炸次数已用完。",
                })
                continue

            # remove cards
            selected_serials = {c.serial for c in selected}
            player.hand = [c for c in player.hand if c.serial not in selected_serials]
            player.sort_hand()

            if combo.kind in {"bomb", "rocket"}:
                self.bombs_played += 1
                player.bombs_used += 1

            if self.knowledge:
                self.knowledge.record_play(current_idx, combo, opened_round)

            await self._broadcast({
                "type": "play_action",
                "seat": current_idx,
                "player_name": player.name,
                "action": "play",
                "combo": _combo_dict(combo),
                "cards_played": [_card_dict(c) for c in selected],
                "remaining_count": player.hand_size,
            })

            last_combo = combo
            last_player_idx = current_idx

            if not player.hand:
                await self._broadcast({
                    "type": "player_empty",
                    "seat": current_idx,
                    "player_name": player.name,
                })
                return current_idx

            current_idx = (current_idx + 1) % len(players)

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

    # ---- end ----

    async def _end_phase(self, players: list[Player], winner_idx: int) -> None:
        winner = players[winner_idx]
        landlord_won = winner.role == "landlord"
        all_hands = {}
        for i, p in enumerate(players):
            all_hands[self.seats[i].username] = _hand_summary(p.hand)

        await self._broadcast({
            "type": "game_over",
            "winner_seat": winner_idx,
            "winner_name": winner.name,
            "winner_role": winner.role,
            "landlord_seat": self.landlord_index,
            "landlord_name": players[self.landlord_index].name if self.landlord_index is not None else "",
            "landlord_won": landlord_won,
            "bombs_played": self.bombs_played,
            "final_hands": all_hands,
        })
        self.state = "finished"


class ConnectionManager:
    def __init__(self) -> None:
        self.rooms: dict[str, GameRoom] = {}
        self.ws_to_room: dict[int, tuple[str, int]] = {}  # ws_id -> (room_id, seat_index)

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
        self.rooms.pop(room_id, None)
        # clean up ws mappings
        self.ws_to_room = {
            wid: (rid, sid) for wid, (rid, sid) in self.ws_to_room.items() if rid != room_id
        }

    def register_ws(self, ws: WebSocket, room_id: str, seat_index: int) -> None:
        self.ws_to_room[id(ws)] = (room_id, seat_index)

    def unregister_ws(self, ws: WebSocket) -> tuple[str, int] | None:
        return self.ws_to_room.pop(id(ws), None)

    async def handle_disconnect(self, ws: WebSocket) -> None:
        info = self.unregister_ws(ws)
        if info is None:
            return
        room_id, seat_index = info
        room = self.get_room(room_id)
        if room is None:
            return

        seat = room.seats.get(seat_index)
        if seat:
            seat.connected = False
            seat.ws = None

        human_left = any(s.is_human and s.connected for s in room.seats.values())

        await room._broadcast({
            "type": "player_disconnected",
            "seat": seat_index,
            "username": seat.username if seat else "unknown",
        })

        if not human_left and room.state == "playing":
            await room._broadcast({
                "type": "all_disconnected",
                "message": "所有玩家已断开，房间将在5分钟后清理。",
            })
