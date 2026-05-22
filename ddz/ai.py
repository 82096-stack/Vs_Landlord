from __future__ import annotations

import itertools
import math
import random
import time
from collections import Counter
from dataclasses import dataclass, field
from functools import lru_cache

from ddz.models import Card, Combo, Player
from ddz.rules import MODE_RULES, can_beat, generate_candidate_plays, identify_combo, same_team


RANK_SEQUENCE = tuple(range(3, 18))
HIGH_CONTROL_RANKS = (17, 16, 15, 14)
PAIR_LANE_KINDS = {"pair", "pair_straight"}
SINGLE_LANE_KINDS = {"single"}
TRIO_LANE_KINDS = {
    "trio",
    "trio_single",
    "trio_pair",
    "trio_straight",
    "airplane_single",
    "airplane_pair",
}
PAIR_ATTACHMENT_KINDS = {"trio_pair", "airplane_pair", "four_two_pair"}
SINGLE_ATTACHMENT_KINDS = {"trio_single", "airplane_single", "four_two_single"}
STRUCTURED_KINDS = {
    "straight",
    "pair_straight",
    "trio_straight",
    "airplane_single",
    "airplane_pair",
    "trio_pair",
    "trio_single",
    "four_two_single",
    "four_two_pair",
}

GAME_PHASE_EARLY = "early"
GAME_PHASE_MID = "mid"
GAME_PHASE_LATE = "late"
GAME_PHASE_ENDGAME = "endgame"


@dataclass
class PlayerProfile:
    preferred_families: Counter[str] = field(default_factory=Counter)
    exact_kinds: Counter[str] = field(default_factory=Counter)
    last_open_kind: str | None = None
    last_open_family: str | None = None
    pair_scarcity: float = 0.0
    low_pair_scarcity: float = 0.0
    single_scarcity: float = 0.0
    revealed_bombs: int = 0
    high_card_signals: int = 0
    played_high_singles: list[int] = field(default_factory=list)
    played_high_pairs: list[int] = field(default_factory=list)
    consistency_score: float = 0.0

    def preferred_kind(self) -> str | None:
        if self.last_open_kind:
            return self.last_open_kind
        if not self.exact_kinds:
            return None
        return self.exact_kinds.most_common(1)[0][0]

    def preferred_family(self) -> str | None:
        if self.last_open_family:
            return self.last_open_family
        if not self.preferred_families:
            return None
        return self.preferred_families.most_common(1)[0][0]


@dataclass(frozen=True)
class TurnRecord:
    player_index: int
    combo_kind: str
    combo_family: str
    opened_round: bool


@dataclass(frozen=True)
class PlayContext:
    is_landlord: bool
    current_index: int
    next_index: int
    next_player_is_teammate: bool
    next_player_is_landlord: bool
    last_player_is_landlord: bool
    last_player_is_teammate: bool
    opponent_min_cards: int
    teammate_min_cards: int
    last_player_cards: int
    urgent_block: bool
    endgame: bool
    seat_offset_from_landlord: int
    is_landlord_upstream: bool
    is_landlord_downstream: bool
    is_opposite_landlord: bool
    control_margin: float
    control_cards: int
    opponent_bomb_threat: float
    next_teammate_kind: str | None
    next_teammate_family: str | None
    next_opponent_pair_scarcity: float
    next_opponent_low_pair_scarcity: float
    next_opponent_single_scarcity: float
    game_phase: str = GAME_PHASE_MID
    opponent_bombs_remaining: float = 0.0


@dataclass(frozen=True)
class CounterPressure:
    hold_probability: float
    next_opponent_pressure: float
    any_opponent_pressure: float
    bomb_pressure: float
    teammate_follow_probability: float


class TableKnowledge:

    def __init__(self, mode: str, player_count: int) -> None:
        self.mode = mode
        self.player_count = player_count
        self.deck_count = MODE_RULES[mode]["deck_count"]
        self.landlord_index: int | None = None
        self.bottom_cards: list[Card] = []
        self.played_rank_counts: Counter[int] = Counter()
        self.player_played_rank_counts: list[Counter[int]] = [Counter() for _ in range(player_count)]
        self.profiles: list[PlayerProfile] = [PlayerProfile() for _ in range(player_count)]
        self.history: list[TurnRecord] = []

        self._player_known_serials: list[set[int]] = [set() for _ in range(player_count)]
        self._player_impossible_serials: list[set[int]] = [set() for _ in range(player_count)]
        self._all_serial_to_card: dict[int, Card] = {}
        self._rank_to_serials: dict[int, list[int]] = {}
        self._total_bombs_played: int = 0
        self._initialized_beliefs: bool = False

    def set_landlord(self, landlord_index: int, bottom_cards: list[Card]) -> None:
        self.landlord_index = landlord_index
        self.bottom_cards = list(bottom_cards)

    def init_beliefs(self, players: list[Player], own_hand: list[Card]) -> None:
        if self._initialized_beliefs:
            return
        self._initialized_beliefs = True

        deck = self._build_full_deck()
        for card in deck:
            self._all_serial_to_card[card.serial] = card
        for card in deck:
            self._rank_to_serials.setdefault(card.rank, []).append(card.serial)

        own_serials = {card.serial for card in own_hand}
        bottom_serials = {card.serial for card in self.bottom_cards}

        for idx in range(self.player_count):
            self._player_impossible_serials[idx] = own_serials.copy()
            if idx != self.landlord_index and self.landlord_index is not None:
                self._player_impossible_serials[idx] |= bottom_serials
            if players[idx].revealed:
                for card in players[idx].hand:
                    self._player_known_serials[idx].add(card.serial)

    def record_play(self, player_index: int, combo: Combo, opened_round: bool) -> None:
        rank_counts = Counter(card.rank for card in combo.cards)
        self.played_rank_counts.update(rank_counts)
        self.player_played_rank_counts[player_index].update(rank_counts)

        if self._initialized_beliefs:
            for card in combo.cards:
                self._player_known_serials[player_index].add(card.serial)
                for other_idx in range(self.player_count):
                    if other_idx != player_index:
                        self._player_impossible_serials[other_idx].add(card.serial)

        family = combo_to_family(combo.kind)
        profile = self.profiles[player_index]
        profile.exact_kinds[combo.kind] += 1
        profile.preferred_families[family] += 1
        if opened_round:
            profile.last_open_kind = combo.kind
            profile.last_open_family = family

        if combo.kind in {"bomb", "rocket"}:
            profile.revealed_bombs += 1
            self._total_bombs_played += 1
            if opened_round:
                profile.consistency_score += 1.0

        if combo.kind == "single" and combo.main_rank >= 14 and opened_round:
            profile.high_card_signals += 1
            profile.played_high_singles.append(combo.main_rank)
        if combo.kind == "pair" and combo.main_rank >= 13 and opened_round:
            profile.played_high_pairs.append(combo.main_rank)

        if opened_round and combo.kind == profile.last_open_kind:
            profile.consistency_score += 0.6

        self._update_scarcity_inference(profile, combo)
        self.history.append(TurnRecord(player_index, combo.kind, family, opened_round))

    def total_rank_count(self, rank: int) -> int:
        if rank >= 16:
            return self.deck_count
        return 4 * self.deck_count

    def unplayed_rank_count(self, rank: int) -> int:
        return self.total_rank_count(rank) - self.played_rank_counts[rank]

    def remaining_outside_player(self, rank: int, hand: list[Card]) -> int:
        own_count = sum(1 for card in hand if card.rank == rank)
        return max(0, self.unplayed_rank_count(rank) - own_count)

    def opponent_rank_count_belief(
        self,
        target_index: int,
        rank: int,
        hand: list[Card],
    ) -> float:
        if not self._initialized_beliefs:
            remaining = self.remaining_outside_player(rank, hand)
            total_unknown = max(1, sum(self.unplayed_rank_count(r) for r in RANK_SEQUENCE) - len(hand))
            return remaining * 0.33
        known = sum(
            1 for serial in self._player_known_serials[target_index]
            if self._all_serial_to_card.get(serial, Card(rank=0, suit="C", deck_id=0, serial=0)).rank == rank
        )
        impossible_in_target = sum(
            1 for serial in self._player_impossible_serials[target_index]
            if self._all_serial_to_card.get(serial, Card(rank=0, suit="C", deck_id=0, serial=0)).rank == rank
        )
        total_rank = self.total_rank_count(rank)
        possible_for_target = total_rank - impossible_in_target
        return max(0.0, float(possible_for_target - known) * 0.5)

    def opponent_has_bomb_rank_probability(
        self,
        target_index: int,
        rank: int,
        hand: list[Card],
        players: list[Player],
    ) -> float:
        if rank >= 16:
            return 0.0
        if players[target_index].hand_size < 4:
            return 0.0
        if players[target_index].revealed:
            counts = Counter(card.rank for card in players[target_index].hand)
            return 1.0 if counts.get(rank, 0) >= 4 else 0.0

        played_by_target = self.player_played_rank_counts[target_index][rank]
        if self.mode == "classic" and played_by_target > 0:
            return 0.0

        remaining = self.remaining_outside_player(rank, hand)
        if remaining < 4:
            return 0.0

        share = players[target_index].hand_size / max(1, self._cards_outside_player(hand))
        return combinations_probability_weight(remaining, 4, share) * 0.8

    def sample_opponent_hand(
        self,
        target_index: int,
        players: list[Player],
        own_hand: list[Card],
    ) -> list[Card]:
        target = players[target_index]
        if target.revealed:
            return list(target.hand)

        needed = target.hand_size
        known_cards: list[Card] = []
        for serial in self._player_known_serials[target_index]:
            card = self._all_serial_to_card.get(serial)
            if card:
                known_cards.append(card)

        if len(known_cards) >= needed:
            return sorted(known_cards[:needed], key=lambda c: (c.rank, c.suit))

        pool: list[Card] = []
        own_serials = {c.serial for c in own_hand}
        for card in self._all_serial_to_card.values():
            if card.serial in own_serials:
                continue
            if card.serial in self._player_impossible_serials[target_index]:
                continue
            if card.serial in self._player_known_serials[target_index]:
                continue
            pool.append(card)

        still_need = needed - len(known_cards)
        sampled = random.sample(pool, min(still_need, len(pool)))
        return sorted(known_cards + sampled, key=lambda c: (c.rank, c.suit))

    def known_opponent_serials(self, target_index: int) -> set[int]:
        return self._player_known_serials[target_index].copy()

    def impossible_opponent_serials(self, target_index: int) -> set[int]:
        return self._player_impossible_serials[target_index].copy()

    def all_card_pool(self) -> list[Card]:
        return list(self._all_serial_to_card.values())

    def total_bombs_played_count(self) -> int:
        return self._total_bombs_played

    def estimate_opponent_bombs_remaining(
        self,
        current_index: int,
        players: list[Player],
        hand: list[Card],
    ) -> float:
        count = 0.0
        for idx, player in enumerate(players):
            if idx == current_index or same_team(players[current_index], player):
                continue
            if player.revealed:
                counts = Counter(c.rank for c in player.hand)
                count += sum(1 for c in counts.values() if c >= 4)
                continue
            for rank in range(3, 16):
                prob = self.opponent_has_bomb_rank_probability(idx, rank, hand, players)
                if prob > 0.3:
                    count += prob
        if self.remaining_outside_player(16, hand) > 0 and self.remaining_outside_player(17, hand) > 0:
            for idx, player in enumerate(players):
                if idx == current_index or same_team(players[current_index], player):
                    continue
                if player.hand_size >= 2:
                    count += 0.4
                    break
        return count

    def game_phase(self, players: list[Player], hand: list[Card]) -> str:
        max_hand = max(p.hand_size for p in players)
        if max_hand <= 6:
            return GAME_PHASE_ENDGAME
        total_played = sum(self.played_rank_counts.values())
        total_cards = 54 * self.deck_count
        ratio = total_played / total_cards
        if ratio < 0.3:
            return GAME_PHASE_EARLY
        if ratio < 0.65:
            return GAME_PHASE_MID
        return GAME_PHASE_LATE

    def control_margin(self, hand: list[Card]) -> float:
        own_counts = Counter(card.rank for card in hand)
        margin = 0.0
        higher_remaining_cache: dict[int, int] = {}

        for rank in HIGH_CONTROL_RANKS:
            my_count = own_counts.get(rank, 0)
            if my_count == 0:
                continue
            if rank not in higher_remaining_cache:
                higher_remaining_cache[rank] = sum(
                    self.remaining_outside_player(higher_rank, hand)
                    for higher_rank in HIGH_CONTROL_RANKS
                    if higher_rank > rank
                )
            higher_remaining = higher_remaining_cache[rank]
            same_remaining = self.remaining_outside_player(rank, hand)
            if higher_remaining == 0:
                margin += my_count * 2.6
                if same_remaining == 0:
                    margin += my_count * 1.4
            else:
                margin += max(0.0, my_count * 1.5 - higher_remaining * 0.8)
        return margin

    def control_cards(self, hand: list[Card]) -> int:
        return sum(1 for card in hand if card.rank in HIGH_CONTROL_RANKS)

    def possible_opponent_bomb_ranks(
        self,
        current_index: int,
        players: list[Player],
        hand: list[Card],
    ) -> list[int]:
        ranks: list[int] = []
        for rank in range(3, 16):
            remaining = self.remaining_outside_player(rank, hand)
            if remaining < 4:
                continue

            for index, opponent in enumerate(players):
                if index == current_index or same_team(players[current_index], opponent):
                    continue
                if opponent.hand_size < 4:
                    continue
                if self.mode == "classic" and self.player_played_rank_counts[index][rank] > 0:
                    continue
                ranks.append(rank)
                break
        return ranks

    def possible_opponent_rocket(
        self,
        current_index: int,
        players: list[Player],
        hand: list[Card],
    ) -> bool:
        small_remaining = self.remaining_outside_player(16, hand)
        big_remaining = self.remaining_outside_player(17, hand)
        if small_remaining == 0 or big_remaining == 0:
            return False
        for index, opponent in enumerate(players):
            if index == current_index or same_team(players[current_index], opponent):
                continue
            if opponent.hand_size >= 2:
                return True
        return False

    def opponent_bomb_threat(
        self,
        current_index: int,
        players: list[Player],
        hand: list[Card],
    ) -> float:
        ranks = self.possible_opponent_bomb_ranks(current_index, players, hand)
        threat = float(len(ranks))
        if self.possible_opponent_rocket(current_index, players, hand):
            threat += 1.5
        return threat

    def counter_pressure(
        self,
        current_index: int,
        combo: Combo,
        players: list[Player],
        hand: list[Card],
    ) -> CounterPressure:
        player = players[current_index]
        ordered_indexes = [
            (current_index + offset) % len(players)
            for offset in range(1, len(players))
        ]

        next_opponent_pressure = 0.0
        bomb_pressure = 0.0
        teammate_follow_probability = 0.0
        opponent_pressures: list[float] = []

        for turn_order, target_index in enumerate(ordered_indexes, start=1):
            target_player = players[target_index]
            if same_team(player, target_player):
                teammate_follow_probability = max(
                    teammate_follow_probability,
                    self._estimate_teammate_follow_probability(target_index, combo, target_player),
                )
                continue

            same_shape = self._estimate_player_same_shape_counter_probability(
                current_index,
                target_index,
                combo,
                players,
                hand,
            )
            explosive = 0.0
            if combo.kind != "rocket":
                explosive = self._estimate_player_bomb_pressure(
                    current_index,
                    target_index,
                    combo,
                    players,
                    hand,
                )

            total = clamp01(1.0 - (1.0 - same_shape) * (1.0 - explosive))
            if turn_order == 1:
                next_opponent_pressure = total
            bomb_pressure = max(bomb_pressure, explosive)
            opponent_pressures.append(total)

        any_opponent_pressure = 1.0
        for pressure in opponent_pressures:
            any_opponent_pressure *= 1.0 - pressure
        any_opponent_pressure = clamp01(1.0 - any_opponent_pressure)
        hold_probability = clamp01(1.0 - any_opponent_pressure)

        return CounterPressure(
            hold_probability=hold_probability,
            next_opponent_pressure=next_opponent_pressure,
            any_opponent_pressure=any_opponent_pressure,
            bomb_pressure=bomb_pressure,
            teammate_follow_probability=teammate_follow_probability,
        )

    def profile(self, player_index: int) -> PlayerProfile:
        return self.profiles[player_index]

    def _update_scarcity_inference(self, profile: PlayerProfile, combo: Combo) -> None:
        main_ranks = combo_main_ranks(combo)
        attachment_pairs = [
            rank
            for rank, count in Counter(card.rank for card in combo.cards).items()
            if rank not in main_ranks and count >= 2
        ]
        attachment_singles = [
            rank
            for rank, count in Counter(card.rank for card in combo.cards).items()
            if rank not in main_ranks and count == 1
        ]

        if combo.kind in PAIR_ATTACHMENT_KINDS and attachment_pairs:
            if min(attachment_pairs) >= 12:
                profile.low_pair_scarcity += 1.3
            if sum(attachment_pairs) / len(attachment_pairs) >= 13:
                profile.pair_scarcity += 1.0
        elif combo.kind in PAIR_LANE_KINDS:
            profile.pair_scarcity = max(0.0, profile.pair_scarcity - 0.5)
            profile.low_pair_scarcity = max(0.0, profile.low_pair_scarcity - 0.5)

        if combo.kind in SINGLE_ATTACHMENT_KINDS and attachment_singles:
            if min(attachment_singles) >= 14:
                profile.single_scarcity += 1.1
        elif combo.kind in SINGLE_LANE_KINDS or combo.kind == "straight":
            profile.single_scarcity = max(0.0, profile.single_scarcity - 0.35)

    def exact_counting_feasible(self, players: list[Player], own_index: int) -> bool:
        total_unknown = sum(p.hand_size for i, p in enumerate(players) if i != own_index and not p.revealed)
        return total_unknown <= 10 and self._initialized_beliefs

    def deduce_opponent_rank_bounds(
        self,
        target_index: int,
        own_hand: list[Card],
        players: list[Player],
    ) -> dict[int, tuple[int, int]]:
        target = players[target_index]
        bounds: dict[int, tuple[int, int]] = {}
        own_counts = Counter(c.rank for c in own_hand)
        if target.revealed:
            target_counts = Counter(c.rank for c in target.hand)
            for rank in RANK_SEQUENCE:
                bounds[rank] = (target_counts.get(rank, 0), target_counts.get(rank, 0))
            return bounds
        for rank in RANK_SEQUENCE:
            total = self.total_rank_count(rank)
            played = self.played_rank_counts[rank]
            own = own_counts.get(rank, 0)
            min_possible = 0
            known_in_target = 0
            if self._initialized_beliefs:
                known_in_target = sum(
                    1 for serial in self._player_known_serials[target_index]
                    if self._all_serial_to_card.get(serial, Card(rank=0, suit="C", deck_id=0, serial=0)).rank == rank
                )
                impossible_in_target = sum(
                    1 for serial in self._player_impossible_serials[target_index]
                    if self._all_serial_to_card.get(serial, Card(rank=0, suit="C", deck_id=0, serial=0)).rank == rank
                )
                max_possible = total - impossible_in_target - own
                min_possible = known_in_target
            else:
                remaining = total - played - own
                other_players = sum(
                    1 for i, p in enumerate(players) if i != target_index and i < len(players)
                )
                max_possible = min(remaining, target.hand_size)
            bounds[rank] = (min_possible, max_possible)
        return bounds

    def enumerate_exact_worlds(
        self,
        own_index: int,
        players: list[Player],
        own_hand: list[Card],
        max_worlds: int = 200,
    ) -> list[dict[int, list[Card]]]:
        if not self._initialized_beliefs:
            return []
        unknown_indices = [
            i for i, p in enumerate(players)
            if i != own_index and not p.revealed and p.hand
        ]
        if not unknown_indices:
            return [{}]
        revealed_world = {}
        for i, p in enumerate(players):
            if i != own_index and p.revealed and p.hand:
                revealed_world[i] = list(p.hand)
        pool = []
        own_serials = {c.serial for c in own_hand}
        bottom_serials = {c.serial for c in (self.bottom_cards or [])}
        for card in self._all_serial_to_card.values():
            if card.serial in own_serials:
                continue
            if card.serial in bottom_serials and self.landlord_index not in unknown_indices:
                continue
            pool.append(card)
        worlds = []
        seen = set()
        attempts = 0
        max_attempts = max_worlds * 3
        target_sizes = {i: players[i].hand_size for i in unknown_indices}
        while len(worlds) < max_worlds and attempts < max_attempts:
            attempts += 1
            random.shuffle(pool)
            world = dict(revealed_world)
            offset = 0
            valid = True
            for idx in unknown_indices:
                needed = target_sizes[idx]
                assigned = pool[offset:offset + needed]
                offset += needed
                if not self._is_hand_consistent(idx, assigned, players):
                    valid = False
                    break
                world[idx] = sorted(assigned, key=lambda c: (c.rank, c.suit))
            if not valid:
                continue
            sig = tuple(
                (idx, tuple(c.serial for c in sorted(world.get(idx, []), key=lambda x: x.serial)))
                for idx in sorted(world)
            )
            if sig not in seen:
                seen.add(sig)
                worlds.append(world)
        return worlds

    def _is_hand_consistent(
        self,
        player_index: int,
        hand: list[Card],
        players: list[Player],
    ) -> bool:
        if len(hand) != players[player_index].hand_size:
            return False
        for card in hand:
            if card.serial in self._player_impossible_serials[player_index]:
                return False
        if players[player_index].revealed:
            revealed_serials = {c.serial for c in players[player_index].hand}
            if not revealed_serials.issubset({c.serial for c in hand}):
                return False
        return True

    def _cards_outside_player(self, hand: list[Card]) -> int:
        return max(1, sum(self.unplayed_rank_count(rank) for rank in RANK_SEQUENCE) - len(hand))

    def _player_share(
        self,
        current_index: int,
        target_index: int,
        players: list[Player],
        hand: list[Card],
    ) -> float:
        total_unknown = self._cards_outside_player(hand)
        share = players[target_index].hand_size / total_unknown
        if share <= 0:
            return 0.0

        player = players[current_index]
        target = players[target_index]
        profile = self.profile(target_index)
        if not same_team(player, target):
            if target.role == "landlord":
                share *= 1.06
            share *= 1.0 + min(0.16, profile.revealed_bombs * 0.05)
        return min(0.98, share)

    def _lane_capacity_factor(self, target_index: int, family: str) -> float:
        profile = self.profile(target_index)
        factor = 1.0 + min(0.25, profile.preferred_families[family] * 0.06)

        if family == "pair":
            factor *= max(0.45, 1.0 - profile.pair_scarcity * 0.14 - profile.low_pair_scarcity * 0.08)
        elif family == "single":
            factor *= max(0.5, 1.0 - profile.single_scarcity * 0.16)
        elif family == "trio":
            factor *= max(0.65, 1.0 - profile.single_scarcity * 0.05)
        return factor

    def _estimate_teammate_follow_probability(
        self,
        target_index: int,
        combo: Combo,
        target_player: Player,
    ) -> float:
        profile = self.profile(target_index)
        family = combo_to_family(combo.kind)
        base = 0.18 + min(0.38, target_player.hand_size / 18.0)
        if profile.last_open_kind == combo.kind:
            base += 0.26
        elif profile.last_open_family == family:
            base += 0.18
        base += min(0.16, profile.preferred_families[family] * 0.05)
        base += min(0.10, profile.consistency_score * 0.04)
        return clamp01(base)

    def _estimate_player_same_shape_counter_probability(
        self,
        current_index: int,
        target_index: int,
        combo: Combo,
        players: list[Player],
        hand: list[Card],
    ) -> float:
        target = players[target_index]
        if target.revealed:
            return self._exact_revealed_counter_probability(combo, target)
        if combo.kind == "rocket":
            return 0.0
        if target.hand_size < combo.total_cards:
            return 0.0
        if combo.kind == "single":
            return self._single_counter_probability(current_index, target_index, combo, players, hand)
        if combo.kind == "pair":
            return self._multi_of_kind_counter_probability(
                current_index,
                target_index,
                combo,
                players,
                hand,
                required_count=2,
                family="pair",
            )
        if combo.kind in {"trio", "trio_single", "trio_pair"}:
            probability = self._multi_of_kind_counter_probability(
                current_index,
                target_index,
                combo,
                players,
                hand,
                required_count=3,
                family="trio",
            )
            if combo.kind == "trio_single":
                probability *= self._attachment_capacity(target_index, target, "single")
            elif combo.kind == "trio_pair":
                probability *= self._attachment_capacity(target_index, target, "pair")
            return clamp01(probability)
        if combo.kind == "straight":
            return self._sequence_counter_probability(
                current_index,
                target_index,
                combo,
                players,
                hand,
                required_count=1,
                family="single",
            )
        if combo.kind == "pair_straight":
            return self._sequence_counter_probability(
                current_index,
                target_index,
                combo,
                players,
                hand,
                required_count=2,
                family="pair",
            )
        if combo.kind == "trio_straight":
            return self._sequence_counter_probability(
                current_index,
                target_index,
                combo,
                players,
                hand,
                required_count=3,
                family="trio",
            )
        if combo.kind in {"airplane_single", "airplane_pair"}:
            probability = self._sequence_counter_probability(
                current_index,
                target_index,
                combo,
                players,
                hand,
                required_count=3,
                family="trio",
            )
            probability *= self._attachment_capacity(
                target_index,
                target,
                "pair" if combo.kind == "airplane_pair" else "single",
                amount=combo.sequence_length,
            )
            return clamp01(probability)
        if combo.kind in {"four_two_single", "four_two_pair"}:
            probability = self._multi_of_kind_counter_probability(
                current_index,
                target_index,
                combo,
                players,
                hand,
                required_count=4,
                family="trio",
            )
            probability *= self._attachment_capacity(
                target_index,
                target,
                "pair" if combo.kind == "four_two_pair" else "single",
                amount=2,
            )
            return clamp01(probability)
        if combo.kind == "bomb":
            return self._better_bomb_probability(current_index, target_index, combo, players, hand)
        return 0.0

    def _single_counter_probability(
        self,
        current_index: int,
        target_index: int,
        combo: Combo,
        players: list[Player],
        hand: list[Card],
    ) -> float:
        better_cards = sum(
            self.remaining_outside_player(rank, hand)
            for rank in range(combo.main_rank + 1, 18)
        )
        share = self._player_share(current_index, target_index, players, hand)
        lane = self._lane_capacity_factor(target_index, "single")
        expected = better_cards * share * lane
        return probability_from_expected(expected, 0.85)

    def _multi_of_kind_counter_probability(
        self,
        current_index: int,
        target_index: int,
        combo: Combo,
        players: list[Player],
        hand: list[Card],
        required_count: int,
        family: str,
    ) -> float:
        share = self._player_share(current_index, target_index, players, hand)
        lane = self._lane_capacity_factor(target_index, family)
        weight = 0.0

        for rank in range(combo.main_rank + 1, 18):
            available = self.remaining_outside_player(rank, hand)
            if available < required_count:
                continue
            weight += combinations_probability_weight(available, required_count, share)

        return probability_from_expected(weight * lane, 1.05)

    def _sequence_counter_probability(
        self,
        current_index: int,
        target_index: int,
        combo: Combo,
        players: list[Player],
        hand: list[Card],
        required_count: int,
        family: str,
    ) -> float:
        share = self._player_share(current_index, target_index, players, hand)
        lane = self._lane_capacity_factor(target_index, family)
        sequence_length = combo.sequence_length
        weight = 0.0
        max_end = 14

        for end_rank in range(combo.main_rank + 1, max_end + 1):
            start_rank = end_rank - sequence_length + 1
            if start_rank < 3:
                continue
            term = 1.0
            for rank in range(start_rank, end_rank + 1):
                available = self.remaining_outside_player(rank, hand)
                if available < required_count:
                    term = 0.0
                    break
                term *= per_rank_hold_factor(available, required_count, share)
            weight += term

        size_factor = min(1.2, players[target_index].hand_size / max(combo.total_cards, 1))
        return probability_from_expected(weight * lane * size_factor, 1.2)

    def _attachment_capacity(
        self,
        target_index: int,
        target_player: Player,
        lane: str,
        amount: int = 1,
    ) -> float:
        profile = self.profile(target_index)
        if lane == "pair":
            scarcity = profile.pair_scarcity * 0.18 + profile.low_pair_scarcity * 0.12
            base = 0.92 - scarcity
        else:
            scarcity = profile.single_scarcity * 0.18
            base = 0.94 - scarcity
        size_factor = min(1.0, target_player.hand_size / max(5, amount * 2))
        return clamp01(base * size_factor)

    def _estimate_player_bomb_pressure(
        self,
        current_index: int,
        target_index: int,
        combo: Combo,
        players: list[Player],
        hand: list[Card],
    ) -> float:
        if players[target_index].revealed:
            return self._exact_revealed_bomb_pressure(combo, players[target_index])
        if combo.kind == "bomb":
            return self._better_bomb_probability(current_index, target_index, combo, players, hand)
        if combo.kind == "rocket":
            return 0.0

        share = self._player_share(current_index, target_index, players, hand)
        weight = 0.0
        for rank in self.possible_opponent_bomb_ranks(current_index, players, hand):
            if self.mode == "classic" and self.player_played_rank_counts[target_index][rank] > 0:
                continue
            available = self.remaining_outside_player(rank, hand)
            if available < 4:
                continue
            weight += combinations_probability_weight(available, 4, share)

        if self.remaining_outside_player(16, hand) > 0 and self.remaining_outside_player(17, hand) > 0:
            if players[target_index].hand_size >= 2:
                weight += share * 0.7

        profile = self.profile(target_index)
        weight *= 1.0 + min(0.3, profile.revealed_bombs * 0.08)
        return probability_from_expected(weight, 0.95)

    def _better_bomb_probability(
        self,
        current_index: int,
        target_index: int,
        combo: Combo,
        players: list[Player],
        hand: list[Card],
    ) -> float:
        if combo.kind == "rocket":
            return 0.0

        share = self._player_share(current_index, target_index, players, hand)
        weight = 0.0

        if combo.kind == "bomb":
            for rank in range(combo.main_rank + 1, 18):
                available = self.remaining_outside_player(rank, hand)
                if rank < 16 and available >= combo.bomb_size:
                    weight += combinations_probability_weight(available, combo.bomb_size, share)
            if self.remaining_outside_player(16, hand) > 0 and self.remaining_outside_player(17, hand) > 0:
                weight += share * 0.8
        else:
            weight = self._estimate_player_bomb_pressure(current_index, target_index, combo, players, hand)
            return clamp01(weight)

        return probability_from_expected(weight, 1.0)

    def _exact_revealed_counter_probability(self, combo: Combo, target: Player) -> float:
        if combo.kind == "rocket":
            return 0.0
        for candidate in generate_candidate_plays(target.hand, self.mode):
            if can_beat(candidate, combo):
                if combo.kind in {"bomb", "rocket"} and candidate.kind not in {"bomb", "rocket"}:
                    continue
                return 1.0
        return 0.0

    def _exact_revealed_bomb_pressure(self, combo: Combo, target: Player) -> float:
        if combo.kind == "rocket":
            return 0.0
        for candidate in generate_candidate_plays(target.hand, self.mode):
            if candidate.kind not in {"bomb", "rocket"}:
                continue
            if can_beat(candidate, combo):
                return 1.0
        return 0.0

    def _build_full_deck(self) -> list[Card]:
        from ddz.rules import build_deck
        return build_deck(self.mode)


@dataclass
class MCTSNode:
    state_key: tuple
    visits: int = 0
    total_value: float = 0.0
    children: dict[tuple, "MCTSNode"] = field(default_factory=dict)
    untried_actions: list[tuple] = field(default_factory=list)
    parent: "MCTSNode | None" = None

    @property
    def value(self) -> float:
        if self.visits == 0:
            return 0.0
        return self.total_value / self.visits

    def ucb1(self, parent_visits: int, exploration: float) -> float:
        if self.visits == 0:
            return float("inf")
        return self.value + exploration * math.sqrt(math.log(parent_visits) / self.visits)

    def best_child(self, exploration: float) -> "MCTSNode | None":
        if not self.children:
            return None
        return max(self.children.values(), key=lambda child: child.ucb1(self.visits, exploration))

    def most_visited_child(self) -> "MCTSNode | None":
        if not self.children:
            return None
        return max(self.children.values(), key=lambda child: child.visits)


class EndgameSolver:

    def __init__(self, max_worlds: int = 200, max_depth: int = 30, time_limit: float = 2.0) -> None:
        self.max_worlds = max_worlds
        self.max_depth = max_depth
        self.time_limit = time_limit
        self._transposition: dict[tuple, tuple[float, int]] = {}
        self._our_player_index: int = -1
        self._teammates: set[int] = set()

    def should_activate(self, players: list[Player], own_hand: list[Card]) -> bool:
        total_unknown = sum(p.hand_size for p in players if p.hand) - len(own_hand)
        max_opponent = max(
            (p.hand_size for p in players if p.hand and p.hand != own_hand),
            default=0,
        )
        min_any = min(
            (p.hand_size for p in players if p.hand),
            default=99,
        )
        if total_unknown <= 12 and max_opponent <= 7:
            return True
        # Broader activation: any player very close to winning
        if min_any <= 4:
            return True
        if total_unknown <= 18 and max_opponent <= 8:
            return True
        return False

    def solve(
        self,
        player: Player,
        players: list[Player],
        last_combo: Combo | None,
        last_player_index: int | None,
        knowledge: TableKnowledge,
        mode: str,
        ai: object,
    ) -> list[Card] | None:
        current_index = next(i for i, p in enumerate(players) if p is player)
        own_hand = list(player.hand)
        self._our_player_index = current_index
        self._teammates = {i for i, p in enumerate(players) if same_team(player, p)}
        self._transposition.clear()

        worlds = knowledge.enumerate_exact_worlds(current_index, players, own_hand, self.max_worlds)
        if not worlds:
            worlds = self._sample_worlds(players, current_index, own_hand, knowledge)
        if not worlds:
            return None

        valid_plays = self._get_valid_plays(own_hand, last_combo, last_player_index, mode)
        if not valid_plays:
            return None

        start_time = time.time()
        best_play = None
        best_score = -float("inf")

        for max_d in range(4, self.max_depth + 1, 4):
            if time.time() - start_time > self.time_limit:
                break
            self._transposition.clear()
            depth_best = None
            depth_best_score = -float("inf")
            for play_cards in valid_plays:
                if time.time() - start_time > self.time_limit:
                    break
                total_score = 0.0
                worlds_scored = 0
                for world in worlds:
                    if worlds_scored >= len(worlds):
                        break
                    result = self._alpha_beta_minimax(
                        world=world,
                        players=players,
                        current_index=current_index,
                        last_combo=last_combo,
                        last_player_index=last_player_index,
                        our_play=play_cards,
                        depth=0,
                        alpha=-float("inf"),
                        beta=float("inf"),
                        is_our_branch=True,
                        mode=mode,
                        max_depth=max_d,
                    )
                    total_score += result
                    worlds_scored += 1
                avg_score = total_score / max(worlds_scored, 1)
                if avg_score > depth_best_score:
                    depth_best_score = avg_score
                    depth_best = play_cards
            if depth_best is not None and depth_best_score > best_score:
                best_score = depth_best_score
                best_play = depth_best
            if best_score >= 0.95:
                break

        return best_play

    def _sample_worlds(
        self,
        players: list[Player],
        current_index: int,
        own_hand: list[Card],
        knowledge: TableKnowledge,
    ) -> list[dict[int, list[Card]]]:
        worlds = []
        opponent_indices = [i for i, p in enumerate(players) if i != current_index and p.hand]
        if not opponent_indices:
            return [{}]
        revealed_world = {}
        unknown = []
        for idx in opponent_indices:
            if players[idx].revealed:
                revealed_world[idx] = list(players[idx].hand)
            else:
                unknown.append(idx)
        if not unknown:
            return [revealed_world]
        seen = set()
        for _ in range(min(self.max_worlds, 60)):
            world = dict(revealed_world)
            for idx in unknown:
                world[idx] = knowledge.sample_opponent_hand(idx, players, own_hand)
            sig = tuple(
                (idx, tuple(c.serial for c in sorted(world.get(idx, []), key=lambda x: x.serial)))
                for idx in sorted(world)
            )
            if sig not in seen:
                seen.add(sig)
                worlds.append(world)
        return worlds

    def _get_valid_plays(
        self,
        hand: list[Card],
        last_combo: Combo | None,
        last_player_index: int | None,
        mode: str,
    ) -> list[list[Card] | None]:
        candidates = generate_candidate_plays(hand, mode)
        valid: list[list[Card] | None] = []
        if last_combo is None or last_player_index is None:
            valid.extend(combo.cards for combo in candidates)
        else:
            for combo in candidates:
                if can_beat(combo, last_combo):
                    valid.append(combo.cards)
            valid.append(None)
        return valid

    def _alpha_beta_minimax(
        self,
        world: dict[int, list[Card]],
        players: list[Player],
        current_index: int,
        last_combo: Combo | None,
        last_player_index: int | None,
        our_play: list[Card] | None,
        depth: int,
        alpha: float,
        beta: float,
        is_our_branch: bool,
        mode: str,
        max_depth: int = 30,
    ) -> float:
        if depth > max_depth:
            return 0.0

        current_hand = world.get(current_index, [])
        if not current_hand and current_index not in world:
            return 0.0

        if depth == 0 and our_play is not None:
            result, new_world, new_lc, new_li = self._apply_play(
                world, current_index, our_play, last_combo, last_player_index, mode
            )
            if result == "win":
                return 1.0
            if result == "invalid":
                return -1.0
            next_idx = (current_index + 1) % len(players)
            return self._alpha_beta_minimax(
                world=new_world, players=players, current_index=next_idx,
                last_combo=new_lc, last_player_index=new_li,
                our_play=None, depth=depth + 1, alpha=alpha, beta=beta,
                is_our_branch=False, mode=mode, max_depth=max_depth,
            )

        if depth == 0 and our_play is None:
            return 0.0

        current_hand = world.get(current_index, [])
        if not current_hand:
            if current_index in self._teammates:
                return 1.0
            return -1.0

        state_key = self._make_state_key(world, current_index, last_combo, last_player_index)
        if state_key in self._transposition:
            cached_value, cached_depth = self._transposition[state_key]
            if cached_depth >= max_depth - depth:
                return cached_value

        valid_plays = self._get_valid_plays(current_hand, last_combo, last_player_index, mode)
        if not valid_plays:
            return 0.0

        is_our_player = current_index in self._teammates
        if is_our_player:
            value = -float("inf")
            for play in valid_plays:
                result, new_world, new_lc, new_li = self._apply_play(
                    world, current_index, play, last_combo, last_player_index, mode
                )
                if result == "win":
                    value = 1.0
                    break
                if result == "invalid":
                    continue
                next_idx = (current_index + 1) % len(players)
                child_val = self._alpha_beta_minimax(
                    world=new_world, players=players, current_index=next_idx,
                    last_combo=new_lc, last_player_index=new_li,
                    our_play=None, depth=depth + 1, alpha=alpha, beta=beta,
                    is_our_branch=False, mode=mode, max_depth=max_depth,
                )
                value = max(value, child_val)
                alpha = max(alpha, value)
                if alpha >= beta:
                    break
            self._transposition[state_key] = (value, max_depth - depth)
            return value
        else:
            value = float("inf")
            for play in valid_plays:
                result, new_world, new_lc, new_li = self._apply_play(
                    world, current_index, play, last_combo, last_player_index, mode
                )
                if result == "win":
                    value = -1.0
                    break
                if result == "invalid":
                    continue
                next_idx = (current_index + 1) % len(players)
                child_val = self._alpha_beta_minimax(
                    world=new_world, players=players, current_index=next_idx,
                    last_combo=new_lc, last_player_index=new_li,
                    our_play=None, depth=depth + 1, alpha=alpha, beta=beta,
                    is_our_branch=False, mode=mode, max_depth=max_depth,
                )
                value = min(value, child_val)
                beta = min(beta, value)
                if alpha >= beta:
                    break
            self._transposition[state_key] = (value, max_depth - depth)
            return value

    def _apply_play(
        self,
        world: dict[int, list[Card]],
        player_index: int,
        play: list[Card] | None,
        last_combo: Combo | None,
        last_player_index: int | None,
        mode: str,
    ) -> tuple[str, dict[int, list[Card]], Combo | None, int | None]:
        if play is None:
            if last_combo is None:
                return ("invalid", world, last_combo, last_player_index)
            return ("pass", world, last_combo, last_player_index)

        combo = identify_combo(play, mode)
        if combo is None:
            return ("invalid", world, last_combo, last_player_index)

        if last_combo is not None and not can_beat(combo, last_combo):
            return ("invalid", world, last_combo, last_player_index)

        new_world = dict(world)
        old_hand = list(world.get(player_index, []))
        play_serials = {c.serial for c in play}
        new_hand = [c for c in old_hand if c.serial not in play_serials]
        if len(new_hand) == len(old_hand):
            return ("invalid", world, last_combo, last_player_index)

        new_world[player_index] = new_hand
        if not new_hand:
            return ("win", new_world, combo, player_index)

        return ("ok", new_world, combo, player_index)

    @staticmethod
    def _make_state_key(
        world: dict[int, list[Card]],
        current_index: int,
        last_combo: Combo | None,
        last_player_index: int | None,
    ) -> tuple:
        hands = tuple(
            (idx, tuple(c.serial for c in sorted(hand, key=lambda x: x.serial)))
            for idx, hand in sorted(world.items())
        )
        lc_key = (
            last_combo.kind,
            last_combo.main_rank,
            last_combo.sequence_length,
            last_combo.bomb_size,
            tuple(sorted(c.rank for c in last_combo.cards)),
        ) if last_combo else None
        return (hands, current_index, lc_key, last_player_index)


class MCTSEngine:

    def __init__(self, exploration_weight: float = 1.8, time_limit: float = 2.0) -> None:
        self.exploration_weight = exploration_weight
        self.time_limit = time_limit
        self._root: MCTSNode | None = None
        self._player_count: int = 0

    def should_use_mcts(
        self,
        context: PlayContext,
        last_combo: Combo | None,
        players: list[Player],
        hand: list[Card],
    ) -> bool:
        if context.endgame:
            return True
        if context.game_phase == GAME_PHASE_ENDGAME:
            return True
        if last_combo and last_combo.kind not in {"bomb", "rocket"} and not context.urgent_block:
            if context.opponent_bomb_threat >= 1.0:
                return True
        if context.is_landlord and context.control_margin < 1.0:
            return True
        if not context.is_landlord and context.opponent_min_cards <= 4:
            return True
        if context.opponent_min_cards <= 5:
            return True
        if context.next_opponent_pair_scarcity > 0.6:
            return True
        return False

    def search(
        self,
        player: Player,
        players: list[Player],
        last_combo: Combo | None,
        last_player_index: int | None,
        knowledge: TableKnowledge,
        mode: str,
        num_iterations: int = 600,
    ) -> list[Card] | None:
        current_index = next(i for i, p in enumerate(players) if p is player)
        own_hand = list(player.hand)
        self._player_count = len(players)

        valid_actions = self._get_actions(own_hand, last_combo, last_player_index, mode)
        if not valid_actions:
            return None

        initial_state = self._encode_state(
            own_hand, players, current_index, last_combo, last_player_index
        )

        self._root = MCTSNode(state_key=initial_state)
        for action_cards in valid_actions:
            action_key = self._encode_action(action_cards)
            child = MCTSNode(state_key=action_key, parent=self._root)
            self._root.children[action_key] = child
            child.untried_actions = []

        start_time = time.time()
        iteration = 0

        while iteration < num_iterations and time.time() - start_time < self.time_limit:
            sampled_world = {}
            for idx, p in enumerate(players):
                if idx != current_index and p.hand:
                    sampled_world[idx] = knowledge.sample_opponent_hand(idx, players, own_hand)
            sampled_world[current_index] = list(own_hand)

            state = dict(sampled_world)
            sim_last_combo = last_combo
            sim_last_player_index = last_player_index
            node = self._root

            if node.children:
                node = self._select(node)
                if node is not self._root:
                    chosen_action = self._decode_action(node.state_key)
                    result, state, sim_last_combo, sim_last_player_index = self._apply_sim_action(
                        state, current_index, chosen_action, sim_last_combo,
                        sim_last_player_index, mode
                    )
                    if result == "win":
                        self._backpropagate(node, 1.0)
                        iteration += 1
                        continue
                    if result == "invalid":
                        self._backpropagate(node, -1.0)
                        iteration += 1
                        continue

            reward = self._rollout(
                state, current_index, sim_last_combo, sim_last_player_index,
                players, mode, knowledge
            )
            self._backpropagate(node, reward)
            iteration += 1

        if self._root is None or not self._root.children:
            return None

        best_child = self._root.most_visited_child()
        if best_child is None:
            return None
        return self._decode_action(best_child.state_key)

    def _select(self, node: MCTSNode) -> MCTSNode:
        current = node
        while current.children:
            current_sorted = sorted(
                current.children.values(),
                key=lambda c: c.ucb1(current.visits, self.exploration_weight),
                reverse=True,
            )
            current = current_sorted[0]
        return current

    def _backpropagate(self, node: MCTSNode, reward: float) -> None:
        current: MCTSNode | None = node
        while current is not None:
            current.visits += 1
            current.total_value += reward
            current = current.parent

    def _get_actions(
        self,
        hand: list[Card],
        last_combo: Combo | None,
        last_player_index: int | None,
        mode: str,
    ) -> list[list[Card] | None]:
        candidates = generate_candidate_plays(hand, mode)
        actions: list[list[Card] | None] = []
        if last_combo is None or last_player_index is None:
            actions.extend(combo.cards for combo in candidates)
        else:
            for combo in candidates:
                if can_beat(combo, last_combo):
                    actions.append(combo.cards)
            actions.append(None)
        actions.sort(key=lambda a: (
            a is None,
            a is not None and identify_combo(a, mode) is not None and identify_combo(a, mode).kind in {"bomb", "rocket"},
        ))
        if len(actions) > 25:
            scored_actions = []
            for action in actions:
                if action is None:
                    scored_actions.append((0, action))
                else:
                    combo = identify_combo(action, mode)
                    score = combo.total_cards if combo else 0
                    if combo and combo.kind in STRUCTURED_KINDS:
                        score += 10
                    scored_actions.append((-score, action))
            scored_actions.sort(key=lambda x: x[0])
            actions = [a for _, a in scored_actions[:25]]
            has_none = any(a is None for a in actions)
            has_none_in_original = any(a is None for _, a in scored_actions)
            if not has_none and has_none_in_original:
                actions.append(None)
        return actions

    def _apply_sim_action(
        self,
        world_state: dict[int, list[Card]],
        current_index: int,
        action_cards: list[Card] | None,
        last_combo: Combo | None,
        last_player_index: int | None,
        mode: str,
    ) -> tuple[str, dict[int, list[Card]], Combo | None, int | None]:
        if action_cards is None:
            if last_combo is None:
                return ("invalid", world_state, last_combo, last_player_index)
            return ("pass", world_state, last_combo, last_player_index)
        combo = identify_combo(action_cards, mode)
        if combo is None:
            return ("invalid", world_state, last_combo, last_player_index)
        if last_combo is not None and not can_beat(combo, last_combo):
            return ("invalid", world_state, last_combo, last_player_index)
        new_state = dict(world_state)
        old_hand = list(world_state.get(current_index, []))
        play_serials = {c.serial for c in action_cards}
        new_hand = [c for c in old_hand if c.serial not in play_serials]
        if len(new_hand) == len(old_hand):
            return ("invalid", world_state, last_combo, last_player_index)
        new_state[current_index] = new_hand
        if not new_hand:
            return ("win", new_state, combo, current_index)
        return ("ok", new_state, combo, current_index)

    def _rollout(
        self,
        world_state: dict[int, list[Card]],
        current_index: int,
        last_combo: Combo | None,
        last_player_index: int | None,
        players: list[Player],
        mode: str,
        knowledge: TableKnowledge,
    ) -> float:
        sim_index = (current_index + 1) % self._player_count
        steps = 0
        max_steps = 60
        sim_last_combo = last_combo
        sim_last_player = last_player_index
        our_index = current_index

        while steps < max_steps:
            hand = world_state.get(sim_index, [])
            if not hand:
                if sim_index == our_index or (sim_index in self._get_teammates(players, our_index)):
                    return 1.0
                return -1.0
            candidates = generate_candidate_plays(hand, mode)
            if sim_last_combo is not None and sim_last_player != sim_index:
                valid = [c for c in candidates if can_beat(c, sim_last_combo)]
                if valid:
                    chosen = self._heuristic_pick(valid, hand, mode, responding=True)
                else:
                    sim_index = (sim_index + 1) % self._player_count
                    steps += 1
                    continue
            else:
                if sim_last_combo is not None and sim_last_player == sim_index:
                    sim_last_combo = None
                    sim_last_player = None
                chosen = self._heuristic_pick(candidates, hand, mode, responding=False)
            play_serials = {c.serial for c in chosen.cards}
            world_state[sim_index] = [c for c in hand if c.serial not in play_serials]
            if not world_state[sim_index]:
                if sim_index == our_index or (sim_index in self._get_teammates(players, our_index)):
                    return 1.0
                return -1.0
            sim_last_combo = chosen
            sim_last_player = sim_index
            sim_index = (sim_index + 1) % self._player_count
            steps += 1
        return 0.0

    @staticmethod
    def _get_teammates(players: list[Player], our_index: int) -> set[int]:
        our_player = players[our_index]
        return {i for i, p in enumerate(players) if same_team(our_player, p)}

    @staticmethod
    def _heuristic_pick(
        candidates: list[Combo], hand: list[Card], mode: str, responding: bool
    ) -> Combo:
        if not candidates:
            sorted_hand = sorted(hand, key=lambda c: (c.rank, c.suit))
            return Combo("single", [sorted_hand[0]], main_rank=sorted_hand[0].rank)
        structured = [c for c in candidates if c.kind in STRUCTURED_KINDS]
        non_bomb = [c for c in candidates if c.kind not in {"bomb", "rocket"}]
        if structured:
            candidates = structured
        elif non_bomb:
            candidates = non_bomb
        if responding:
            candidates.sort(key=lambda c: (c.main_rank, -c.total_cards))
            return candidates[0]
        else:
            candidates.sort(key=lambda c: (-c.total_cards, c.main_rank))
            return candidates[0]

    @staticmethod
    def _encode_state(
        own_hand: list[Card],
        players: list[Player],
        current_index: int,
        last_combo: Combo | None,
        last_player_index: int | None,
    ) -> tuple:
        hand_sig = tuple(sorted(c.serial for c in own_hand))
        opponent_sizes = tuple(
            p.hand_size for i, p in enumerate(players) if i != current_index
        )
        lc_key = (
            last_combo.kind, last_combo.main_rank, last_combo.sequence_length,
            last_combo.bomb_size, tuple(sorted(c.rank for c in last_combo.cards)),
        ) if last_combo else None
        return (hand_sig, opponent_sizes, current_index, lc_key, last_player_index)

    @staticmethod
    def _encode_action(action: list[Card] | None) -> tuple:
        if action is None:
            return ("pass",)
        return tuple((c.rank, c.serial) for c in sorted(action, key=lambda x: x.serial))

    @staticmethod
    def _decode_action(action_key: tuple) -> list[Card] | None:
        if action_key == ("pass",):
            return None
        cards = []
        for rank, serial in action_key:
            cards.append(Card(rank=rank, suit="C", deck_id=0, serial=serial))
        return cards


class BeamSearchEngine:
    """Breadth-first beam search for structured hand play sequences.

    Better than MCTS for hands with many straights, airplanes, and other
    structured combinations where the search space is more limited.
    """

    def __init__(self, beam_width: int = 8, max_depth: int = 5, time_limit: float = 1.2) -> None:
        self.beam_width = beam_width
        self.max_depth = max_depth
        self.time_limit = time_limit

    def search(
        self,
        player: Player,
        players: list[Player],
        last_combo: Combo | None,
        last_player_index: int | None,
        knowledge: TableKnowledge,
        mode: str,
    ) -> list[Card] | None:
        own_hand = list(player.hand)
        current_index = next(i for i, p in enumerate(players) if p is player)

        valid_plays = self._get_valid_plays(own_hand, last_combo, last_player_index, mode)
        if not valid_plays:
            return None

        # Score each initial play
        scored = []
        for play in valid_plays:
            combo = identify_combo(play, mode)
            if combo is None:
                continue
            remaining = [c for c in own_hand if c not in play]
            base_score = self._evaluate_position(
                remaining, players, current_index, combo, knowledge, mode
            )
            scored.append((base_score, play, combo, remaining))

        if not scored:
            return None

        # Beam search: keep top beam_width candidates
        scored.sort(key=lambda x: x[0], reverse=True)
        beam = scored[:self.beam_width]

        start_time = time.time()
        for depth in range(1, self.max_depth):
            if time.time() - start_time > self.time_limit:
                break
            next_beam = []
            for _, _, _, remaining in beam:
                sub_plays = generate_candidate_plays(remaining, mode)
                for sub_play in sub_plays:
                    sub_combo = identify_combo(sub_play, mode)
                    if sub_combo is None:
                        continue
                    sub_remaining = [c for c in remaining if c not in sub_play]
                    sub_score = self._evaluate_position(
                        sub_remaining, players, current_index, sub_combo, knowledge, mode
                    )
                    next_beam.append((sub_score, sub_play, sub_combo, sub_remaining))
            if not next_beam:
                break
            next_beam.sort(key=lambda x: x[0], reverse=True)
            beam = next_beam[:self.beam_width]

        return beam[0][1] if beam else scored[0][1]

    @staticmethod
    def _get_valid_plays(
        hand: list[Card],
        last_combo: Combo | None,
        last_player_index: int | None,
        mode: str,
    ) -> list[list[Card]]:
        candidates = generate_candidate_plays(hand, mode)
        if last_combo is None or last_player_index is None:
            return [c.cards for c in candidates]
        return [c.cards for c in candidates if can_beat(c, last_combo)]

    @staticmethod
    def _evaluate_position(
        remaining: list[Card],
        players: list[Player],
        current_index: int,
        last_combo: Combo,
        knowledge: TableKnowledge,
        mode: str,
    ) -> float:
        score = 0.0
        n = len(remaining)

        # Empty hand = win
        if n == 0:
            return 1000.0

        # Prefer fewer remaining cards
        score -= n * 3.0

        # Prefer keeping bombs for last
        bomb_count = sum(1 for c in remaining if knowledge.is_control_card(c))
        score += bomb_count * 5.0

        # Prefer structured combos in remaining
        remaining_combos = list(generate_candidate_plays(remaining, mode))
        if remaining_combos:
            score -= max(0, len(remaining_combos) - 3) * 2.0

        # Penalize leaving singles
        singles = sum(1 for c in remaining if knowledge.is_control_card(c))
        score -= (n - singles) * 0.5

        return score


class OpponentModel:
    """Tracks and models opponent behavior patterns to adjust AI decisions.

    Models aggressiveness, bluff frequency, bomb usage timing, and
    response patterns to better predict opponent actions.
    """

    def __init__(self) -> None:
        # Per-seat tracking
        self._aggressiveness: dict[int, float] = {}
        self._bomb_usage: dict[int, int] = {}
        self._pass_count: dict[int, int] = {}
        self._play_count: dict[int, int] = {}
        self._high_card_frequency: dict[int, float] = {}
        self._last_play_ranks: dict[int, list[int]] = {}

    def record_play(self, seat_index: int, combo: Combo | None, is_pass: bool) -> None:
        if is_pass or combo is None:
            self._pass_count[seat_index] = self._pass_count.get(seat_index, 0) + 1
            return

        self._play_count[seat_index] = self._play_count.get(seat_index, 0) + 1

        if combo.kind in {"bomb", "rocket"}:
            self._bomb_usage[seat_index] = self._bomb_usage.get(seat_index, 0) + 1

        # Track high card frequency
        total = self._play_count.get(seat_index, 0) + self._pass_count.get(seat_index, 0)
        if total > 0:
            ranks = self._last_play_ranks.setdefault(seat_index, [])
            ranks.append(combo.main_rank)
            if len(ranks) > 10:
                ranks.pop(0)
            high_ranks = sum(1 for r in ranks if r >= 15)
            self._high_card_frequency[seat_index] = high_ranks / len(ranks) if ranks else 0.3

        # Aggressiveness: ratio of plays to total actions
        if total > 3:
            self._aggressiveness[seat_index] = (
                self._play_count[seat_index] / total
            )

    def aggressiveness(self, seat_index: int) -> float:
        return self._aggressiveness.get(seat_index, 0.5)

    def bomb_likelihood(self, seat_index: int) -> float:
        plays = self._play_count.get(seat_index, 0)
        if plays == 0:
            return 0.15
        return min(0.6, self._bomb_usage.get(seat_index, 0) / plays)

    def high_card_tendency(self, seat_index: int) -> float:
        return self._high_card_frequency.get(seat_index, 0.3)

    def pass_rate(self, seat_index: int) -> float:
        total = self._play_count.get(seat_index, 0) + self._pass_count.get(seat_index, 0)
        if total == 0:
            return 0.5
        return self._pass_count.get(seat_index, 0) / total

    def adjust_threshold(self, base_threshold: float, seat_index: int, is_landlord: bool) -> float:
        """Adjust a scoring threshold based on opponent behavior."""
        agg = self.aggressiveness(seat_index)
        bomb_lik = self.bomb_likelihood(seat_index)
        pass_rate = self.pass_rate(seat_index)

        adjusted = base_threshold
        # Aggressive opponents: lower threshold (play more)
        if agg > 0.65:
            adjusted -= 6.0
        elif agg < 0.35:
            adjusted += 4.0

        # Bomb-prone opponents: raise threshold for non-bomb plays
        if bomb_lik > 0.3 and not is_landlord:
            adjusted += 3.0

        # High pass rate opponents: lower threshold (they will likely pass)
        if pass_rate > 0.6:
            adjusted -= 5.0

        return adjusted


class RuleBasedAI:

    COMBO_PRIORITY = {
        "airplane_pair": 0,
        "airplane_single": 1,
        "trio_straight": 2,
        "pair_straight": 3,
        "straight": 4,
        "trio_pair": 5,
        "trio_single": 6,
        "trio": 7,
        "pair": 8,
        "single": 9,
        "four_two_pair": 10,
        "four_two_single": 11,
        "bomb": 12,
        "rocket": 13,
    }

    def __init__(self) -> None:
        self.endgame_solver = EndgameSolver()
        self.mcts_engine = MCTSEngine()
        self.beam_engine = BeamSearchEngine()
        self.opponent_model = OpponentModel()

    def choose_bid(self, hand: list[Card], mode: str, current_high: int) -> int:
        strength = self._estimate_landlord_strength(hand, mode)
        farmer_dominance = self._estimate_farmer_dominance(hand, mode)
        control_count = self._control_card_count(hand)
        thresholds = {
            "classic": [15.0, 21.0, 27.0],
            "extended": [21.0, 28.0, 35.0],
        }[mode]

        bid = self._project_bid_level(strength, thresholds)

        if current_high == 0 and bid >= 2:
            if (
                farmer_dominance >= strength * 0.94
                and control_count >= (4 if mode == "classic" else 5)
                and strength < thresholds[-1] + 4.0
            ):
                bid = 0 if mode == "classic" else max(0, bid - 1)
        elif current_high == 0 and bid == 0:
            if farmer_dominance >= thresholds[0] + 1.0 and control_count >= (4 if mode == "classic" else 5):
                bid = 1

        if bid <= current_high:
            return 0
        return bid

    def preview_bid_strength(self, hand: list[Card], mode: str) -> int:
        thresholds = {
            "classic": [15.0, 21.0, 27.0],
            "extended": [21.0, 28.0, 35.0],
        }[mode]
        return self._project_bid_level(self._estimate_landlord_strength(hand, mode), thresholds)

    def choose_rob(
        self,
        hand: list[Card],
        mode: str,
        self_desire: int,
        current_landlord_bid: int,
    ) -> bool:
        strength = self._estimate_landlord_strength(hand, mode)
        farmer_dominance = self._estimate_farmer_dominance(hand, mode)
        strong_enough = strength >= (22.0 if mode == "classic" else 30.0)
        clear_upgrade = self_desire >= current_landlord_bid + 1
        if farmer_dominance >= strength * 0.98 and self._control_card_count(hand) >= 5 and strength < 28.0:
            return False
        return strong_enough or clear_upgrade

    def choose_reveal(self, hand: list[Card], mode: str, role: str) -> bool:
        strength = self._estimate_landlord_strength(hand, mode)
        turns = self._estimate_turns(hand, mode)
        control_count = self._control_card_count(hand)
        if role == "landlord":
            return strength >= (30.0 if mode == "classic" else 38.0) or (turns <= 5 and control_count >= 4)
        return turns <= 4 and control_count >= 5

    def choose_report(self, hand: list[Card], mode: str, role: str, report_level: int) -> bool:
        if mode != "extended" or report_level <= 0:
            return False
        if report_level >= 2:
            return True

        turns = self._estimate_turns(hand, mode)
        control_count = self._control_card_count(hand)
        strength = self._estimate_landlord_strength(hand, mode)
        if role == "landlord":
            return strength >= 34.0 or (turns <= 5 and control_count >= 5)
        return turns <= 5 and control_count >= 5

    def choose_play(
        self,
        player: Player,
        players: list[Player],
        last_combo: Combo | None,
        last_player_index: int | None,
        knowledge: TableKnowledge | None = None,
    ) -> list[Card] | None:
        table = knowledge or TableKnowledge(self._guess_mode(players), len(players))
        context = self._build_context(player, players, last_player_index, table)

        table.init_beliefs(players, player.hand)

        candidates = [
            combo
            for combo in generate_candidate_plays(player.hand, table.mode)
            if self._is_combo_allowed(player, combo, table.mode)
        ]

        if not candidates:
            return None

        if not last_combo or last_player_index is None:
            return self._choose_opening_play(player.hand, candidates, context, table, players).cards

        last_player = players[last_player_index]
        if same_team(player, last_player):
            takeover = self._choose_teammate_takeover(
                player.hand,
                candidates,
                last_combo,
                context,
                table,
                players,
            )
            return takeover.cards if takeover is not None else None

        responses = [combo for combo in candidates if can_beat(combo, last_combo)]
        if not responses:
            return None

        non_bomb_responses = [c for c in responses if c.kind not in {"bomb", "rocket"}]
        has_regular_response = len(non_bomb_responses) > 0

        # Landlord with few cards: always take a winning play
        if player.role == "landlord" and player.hand_size <= 4:
            winning = [c for c in responses if self._leaves_no_cards(player.hand, c)]
            if winning:
                # Prefer non-bomb winning plays
                non_bomb_winning = [c for c in winning if c.kind not in {"bomb", "rocket"}]
                chosen = non_bomb_winning[0] if non_bomb_winning else winning[0]
                return chosen.cards
            # If can't win but only has few cards, prefer non-bomb responses
            if non_bomb_responses:
                scored_responses = [
                    (self._score_response_play(player.hand, c, last_combo, context, table, players), c)
                    for c in non_bomb_responses
                ]
                best = max(scored_responses, key=lambda x: (x[0], -x[1].main_rank))
                return best[1].cards

        same_kind_regular = any(
            c for c in responses
            if c.kind == last_combo.kind and c.kind not in {"bomb", "rocket"}
        )
        if self.endgame_solver.should_activate(players, player.hand):
            if not same_kind_regular:
                endgame_result = self.endgame_solver.solve(
                    player, players, last_combo, last_player_index, table, table.mode, self
                )
                if endgame_result is not None:
                    return endgame_result

        # Try beam search for structured hands (many straights/planes)
        structured_count = sum(
            1 for c in candidates if c.kind in {"straight", "pair_straight", "trio_straight",
                                                 "airplane_single", "airplane_pair"}
        )
        if structured_count >= 3 and has_regular_response:
            beam_result = self.beam_engine.search(
                player, players, last_combo, last_player_index, table, table.mode
            )
            if beam_result is not None:
                return beam_result

        multiple_kinds = len({c.kind for c in responses if c.kind not in {"bomb", "rocket"}}) > 1
        if self.mcts_engine.should_use_mcts(context, last_combo, players, player.hand):
            if multiple_kinds or not same_kind_regular:
                mcts_result = self.mcts_engine.search(
                    player, players, last_combo, last_player_index, table, table.mode
                )
                if mcts_result is not None:
                    return mcts_result

        chosen, best_score = self._choose_response_play(
            player.hand,
            responses,
            last_combo,
            context,
            table,
            players,
        )
        if self._should_pass_response(player.hand, chosen, best_score, last_combo, context):
            return None
        return chosen.cards

    @staticmethod
    def _is_combo_allowed(player: Player, combo: Combo, mode: str) -> bool:
        if mode != "extended":
            return True
        if combo.kind not in {"bomb", "rocket"}:
            return True
        if player.role == "landlord":
            return True
        if not player.bid_participated:
            return True
        limit = 2 if player.bid_score >= 2 else 1
        return player.bombs_used < limit

    def _choose_opening_play(
        self,
        hand: list[Card],
        candidates: list[Combo],
        context: PlayContext,
        knowledge: TableKnowledge,
        players: list[Player],
    ) -> Combo:
        finishing = [combo for combo in candidates if self._leaves_no_cards(hand, combo)]
        if finishing:
            return max(
                finishing,
                key=lambda combo: (
                    self._score_opening_play(hand, combo, context, knowledge, players),
                    combo.total_cards,
                ),
            )

        handoff = self._choose_teammate_handoff(candidates, context)
        if handoff is not None:
            return handoff

        if (
            context.game_phase == GAME_PHASE_ENDGAME
            and self.mcts_engine.should_use_mcts(context, None, players, hand)
            and not (context.next_player_is_landlord and context.opponent_min_cards <= 2)
            and not (not context.next_player_is_teammate and context.opponent_min_cards <= 1)
        ):
            mcts_result = self.mcts_engine.search(
                players[context.current_index],
                players,
                None,
                None,
                knowledge,
                knowledge.mode,
                num_iterations=150,
            )
            if mcts_result is not None:
                mcts_combo = identify_combo(mcts_result, knowledge.mode)
                if mcts_combo is not None:
                    return mcts_combo

        has_structured_option = any(
            combo.kind not in {"single", "pair", "bomb", "rocket"} for combo in candidates
        )
        scored = [
            (
                self._score_opening_play(hand, combo, context, knowledge, players)
                - self._opening_fragment_penalty(combo, has_structured_option),
                combo,
            )
            for combo in candidates
        ]
        return max(scored, key=lambda item: (item[0], -item[1].main_rank, item[1].total_cards))[1]

    @staticmethod
    def _choose_teammate_handoff(candidates: list[Combo], context: PlayContext) -> Combo | None:
        if not context.next_player_is_teammate:
            return None
        if context.teammate_min_cards not in (1, 2):
            return None
        if context.opponent_min_cards < context.teammate_min_cards:
            return None

        desired_kind = "single" if context.teammate_min_cards == 1 else "pair"
        lane_candidates = [
            combo
            for combo in candidates
            if combo.kind == desired_kind and combo.main_rank < 16
        ]
        if not lane_candidates:
            return None

        low_lane = [combo for combo in lane_candidates if combo.main_rank <= 13]
        if low_lane:
            lane_candidates = low_lane
        return min(lane_candidates, key=lambda combo: combo.main_rank)

    def _choose_response_play(
        self,
        hand: list[Card],
        candidates: list[Combo],
        last_combo: Combo,
        context: PlayContext,
        knowledge: TableKnowledge,
        players: list[Player],
    ) -> tuple[Combo, float]:
        scored = [
            (self._score_response_play(hand, combo, last_combo, context, knowledge, players), combo)
            for combo in candidates
        ]
        best_score, best_combo = max(
            scored,
            key=lambda item: (
                item[0],
                item[1].kind not in {"bomb", "rocket"},
                -item[1].main_rank,
            ),
        )
        return best_combo, best_score

    def _choose_teammate_takeover(
        self,
        hand: list[Card],
        candidates: list[Combo],
        last_combo: Combo,
        context: PlayContext,
        knowledge: TableKnowledge,
        players: list[Player],
    ) -> Combo | None:
        winning = [
            combo
            for combo in candidates
            if can_beat(combo, last_combo) and self._leaves_no_cards(hand, combo)
        ]
        if winning:
            return max(
                winning,
                key=lambda combo: self._score_response_play(
                    hand,
                    combo,
                    last_combo,
                    context,
                    knowledge,
                    players,
                ),
            )

        if context.last_player_cards <= 2:
            return None

        responses = [
            combo
            for combo in candidates
            if can_beat(combo, last_combo) and combo.kind not in {"bomb", "rocket"}
        ]
        if not responses:
            return None

        if context.game_phase == GAME_PHASE_ENDGAME:
            mcts_result = self.mcts_engine.search(
                players[context.current_index],
                players,
                last_combo,
                next(
                    i for i, p in enumerate(players)
                    if p is not players[context.current_index] and same_team(p, players[context.current_index])
                ),
                knowledge,
                knowledge.mode,
                num_iterations=150,
            )
            if mcts_result is not None:
                return identify_combo(mcts_result, knowledge.mode)

        chosen, best_score = self._choose_response_play(
            hand,
            responses,
            last_combo,
            context,
            knowledge,
            players,
        )
        if self._should_take_teammate_lead(
            hand,
            chosen,
            best_score,
            last_combo,
            context,
            knowledge,
            players,
        ):
            return chosen
        return None

    def _build_context(
        self,
        player: Player,
        players: list[Player],
        last_player_index: int | None,
        knowledge: TableKnowledge,
    ) -> PlayContext:
        current_index = next(index for index, candidate in enumerate(players) if candidate is player)
        next_index = (current_index + 1) % len(players)
        next_player = players[next_index]

        opponent_sizes = [
            candidate.hand_size
            for candidate in players
            if candidate is not player and not same_team(player, candidate)
        ]
        teammate_sizes = [
            candidate.hand_size
            for candidate in players
            if candidate is not player and same_team(player, candidate)
        ]

        last_player_cards = 99 if last_player_index is None else players[last_player_index].hand_size
        opponent_min_cards = min(opponent_sizes) if opponent_sizes else 99
        teammate_min_cards = min(teammate_sizes) if teammate_sizes else 99
        urgent_block = last_player_cards <= 2 or opponent_min_cards <= 1
        endgame = player.hand_size <= 6 or opponent_min_cards <= 4 or teammate_min_cards <= 3

        landlord_index = knowledge.landlord_index
        if landlord_index is None:
            landlord_index = next(
                (index for index, candidate in enumerate(players) if candidate.role == "landlord"),
                0,
            )
        seat_offset = (current_index - landlord_index) % len(players)

        next_teammate_kind = None
        next_teammate_family = None
        next_opponent_pair_scarcity = 0.0
        next_opponent_low_pair_scarcity = 0.0
        next_opponent_single_scarcity = 0.0

        if same_team(player, next_player):
            profile = knowledge.profile(next_index)
            next_teammate_kind = profile.preferred_kind()
            next_teammate_family = profile.preferred_family()
        else:
            profile = knowledge.profile(next_index)
            next_opponent_pair_scarcity = profile.pair_scarcity
            next_opponent_low_pair_scarcity = profile.low_pair_scarcity
            next_opponent_single_scarcity = profile.single_scarcity

        phase = knowledge.game_phase(players, player.hand)
        bombs_remaining = knowledge.estimate_opponent_bombs_remaining(current_index, players, player.hand)

        return PlayContext(
            is_landlord=player.role == "landlord",
            current_index=current_index,
            next_index=next_index,
            next_player_is_teammate=same_team(player, next_player),
            next_player_is_landlord=next_player.role == "landlord",
            last_player_is_landlord=last_player_index is not None and players[last_player_index].role == "landlord",
            last_player_is_teammate=(
                last_player_index is not None and same_team(player, players[last_player_index])
            ),
            opponent_min_cards=opponent_min_cards,
            teammate_min_cards=teammate_min_cards,
            last_player_cards=last_player_cards,
            urgent_block=urgent_block,
            endgame=endgame,
            seat_offset_from_landlord=seat_offset,
            is_landlord_upstream=player.role != "landlord" and next_player.role == "landlord",
            is_landlord_downstream=player.role != "landlord" and seat_offset == 1,
            is_opposite_landlord=player.role != "landlord" and len(players) == 4 and seat_offset == 2,
            control_margin=knowledge.control_margin(player.hand),
            control_cards=knowledge.control_cards(player.hand),
            opponent_bomb_threat=knowledge.opponent_bomb_threat(current_index, players, player.hand),
            next_teammate_kind=next_teammate_kind,
            next_teammate_family=next_teammate_family,
            next_opponent_pair_scarcity=next_opponent_pair_scarcity,
            next_opponent_low_pair_scarcity=next_opponent_low_pair_scarcity,
            next_opponent_single_scarcity=next_opponent_single_scarcity,
            game_phase=phase,
            opponent_bombs_remaining=bombs_remaining,
        )

    def _score_opening_play(
        self,
        hand: list[Card],
        combo: Combo,
        context: PlayContext,
        knowledge: TableKnowledge,
        players: list[Player],
    ) -> float:
        mode = knowledge.mode
        remaining_signature = self._remaining_signature(hand, combo)
        current_turns = self._estimate_turns(hand, mode)
        remaining_score = self._score_signature(remaining_signature, mode)
        turns_after = self._estimate_turns_for_signature(remaining_signature, mode)
        pressure = knowledge.counter_pressure(context.current_index, combo, players, hand)

        score = remaining_score
        score += self._combo_structure_bonus(combo)
        score += self._tempo_bonus(combo, context, opening=True)
        score += self._finish_bonus(turns_after, remaining_signature, combo, context)
        score += self._support_bonus(combo, context)
        score += self._pressure_bonus(combo, context)
        score += self._control_strategy_bonus(combo, context, opening=True)
        score += self._lead_security_bonus(combo, context, pressure, opening=True)
        score += self._turn_optimization_bonus(current_turns, turns_after, combo, context)
        score += self._reentry_bonus(hand, combo, context, mode)
        score += self._endgame_lane_bonus(combo, context, opening=True)
        score -= self._breakup_penalty(hand, combo)
        score -= self._bomb_preservation_penalty(hand, combo, context)
        score -= self._explosive_penalty(combo, context, turns_after, opening=True, pressure=pressure)
        score -= self._bomb_risk_penalty(combo, context, opening=True, pressure=pressure)
        score -= self._high_structure_reserve_penalty(combo, context, current_turns, turns_after)
        score += self._game_phase_opening_adjustment(combo, context)

        if combo.kind == "single":
            score += max(0, 11 - combo.main_rank) * 0.7
        elif combo.kind == "pair":
            score += max(0, 11 - combo.main_rank) * 0.45

        if context.is_opposite_landlord and combo.kind in STRUCTURED_KINDS:
            score += 6.5
        if context.is_landlord_upstream and combo.kind in {"single", "pair"} and combo.main_rank <= 10:
            score -= 15.0
        if context.next_player_is_landlord and combo.kind in {"single", "pair"} and combo.main_rank <= 9:
            score -= 7.5
        if combo.kind in {"single", "pair"} and combo.main_rank >= 14 and not context.next_player_is_landlord:
            score -= 6.0 + context.control_margin * 0.6

        return score

    def _score_response_play(
        self,
        hand: list[Card],
        combo: Combo,
        last_combo: Combo,
        context: PlayContext,
        knowledge: TableKnowledge,
        players: list[Player],
    ) -> float:
        mode = knowledge.mode
        remaining_signature = self._remaining_signature(hand, combo)
        current_turns = self._estimate_turns(hand, mode)
        remaining_score = self._score_signature(remaining_signature, mode)
        turns_after = self._estimate_turns_for_signature(remaining_signature, mode)
        pressure = knowledge.counter_pressure(context.current_index, combo, players, hand)

        score = remaining_score
        score += self._finish_bonus(turns_after, remaining_signature, combo, context)
        score += self._tempo_bonus(combo, context, opening=False)
        score += self._support_bonus(combo, context)
        score += self._pressure_bonus(combo, context)
        score += self._control_strategy_bonus(combo, context, opening=False)
        score += self._lead_security_bonus(combo, context, pressure, opening=False)
        score += self._turn_optimization_bonus(current_turns, turns_after, combo, context)
        score += self._reentry_bonus(hand, combo, context, mode)
        score += self._endgame_lane_bonus(combo, context, opening=False)
        score -= self._breakup_penalty(hand, combo)
        score -= self._bomb_preservation_penalty(hand, combo, context)
        score -= self._explosive_penalty(combo, context, turns_after, opening=False, pressure=pressure)
        score -= self._bomb_risk_penalty(combo, context, opening=False, pressure=pressure)
        score -= self._overkill_penalty(combo, last_combo, context)

        if combo.kind == last_combo.kind and combo.pattern_key() == last_combo.pattern_key():
            score += 8.0
            if combo.main_rank - last_combo.main_rank <= 2:
                score += 4.0
        if context.next_player_is_teammate:
            score += 6.0
        if context.urgent_block:
            score += 11.0
        if context.last_player_cards <= 2:
            score += 16.0
        if context.is_landlord:
            score += 20.0
            if combo.kind in {"single", "pair", "trio"}:
                score += 5.5
        if context.last_player_is_landlord and context.is_landlord_upstream:
            score += 11.0
        if context.last_player_is_landlord and context.is_opposite_landlord:
            score += 8.0
        if context.last_player_is_landlord and context.is_landlord_downstream:
            if combo.kind == last_combo.kind and combo.pattern_key() == last_combo.pattern_key():
                score += 4.5
                if combo.main_rank - last_combo.main_rank <= 2:
                    score += 4.5
            else:
                score -= 6.0
        if combo.kind in {"single", "pair"} and combo.main_rank >= 15 and not context.urgent_block:
            score -= 12.0

        score += self._game_phase_response_adjustment(combo, last_combo, context)

        return score

    def _game_phase_opening_adjustment(self, combo: Combo, context: PlayContext) -> float:
        if context.game_phase == GAME_PHASE_EARLY:
            if combo.kind in STRUCTURED_KINDS:
                return 5.0
            if combo.kind in {"bomb", "rocket"}:
                return -10.0
        elif context.game_phase == GAME_PHASE_LATE:
            if combo.kind in {"bomb", "rocket"} and context.control_margin < 2.0:
                return 6.0
            if combo.kind in {"single", "pair"} and combo.main_rank <= 10:
                return -4.0
        elif context.game_phase == GAME_PHASE_ENDGAME:
            if combo.kind in {"single", "pair", "trio"}:
                return 3.0
        return 0.0

    def _game_phase_response_adjustment(
        self, combo: Combo, last_combo: Combo, context: PlayContext
    ) -> float:
        if context.game_phase == GAME_PHASE_LATE and combo.kind in {"bomb", "rocket"}:
            if context.opponent_min_cards <= 4:
                return 8.0
        if context.game_phase == GAME_PHASE_ENDGAME and context.urgent_block:
            return 5.0
        return 0.0

    def _should_pass_response(
        self,
        hand: list[Card],
        combo: Combo,
        score: float,
        last_combo: Combo,
        context: PlayContext,
    ) -> bool:
        if self._leaves_no_cards(hand, combo):
            return False
        if context.urgent_block or context.endgame:
            return False
        if combo.kind in {"bomb", "rocket"} and last_combo.kind not in {"bomb", "rocket"}:
            return True
        if context.is_landlord:
            if combo.kind not in {"bomb", "rocket"}:
                return False
            if last_combo.kind in {"bomb", "rocket"}:
                return score < 24.0
            return not context.urgent_block and not context.endgame

        # Farmer: when teammate has 1-2 cards, pass aggressively to let teammate win
        if context.teammate_min_cards <= 2:
            return True

        threshold = 44.0
        if last_combo.kind in {"single", "pair", "trio"}:
            threshold = 36.0
        if context.next_player_is_teammate:
            threshold += 6.0 if context.last_player_is_landlord else 12.0
        if context.last_player_is_landlord and context.is_landlord_downstream:
            if combo.kind == last_combo.kind and combo.pattern_key() == last_combo.pattern_key():
                threshold -= 4.0
                if combo.main_rank - last_combo.main_rank <= 2:
                    threshold -= 6.0
            else:
                threshold += 8.0
        if context.last_player_is_landlord and context.is_opposite_landlord:
            threshold -= 10.0
        if context.last_player_is_landlord and context.is_landlord_upstream:
            threshold -= 14.0
        if context.opponent_bomb_threat == 0 and combo.kind not in {"bomb", "rocket"}:
            threshold -= 2.0

        if combo.kind not in {"bomb", "rocket"} and context.opponent_bombs_remaining >= 1.5:
            threshold -= 4.0

        if context.game_phase == GAME_PHASE_LATE and context.opponent_min_cards <= 4:
            threshold -= 6.0

        return score < threshold

    def _should_take_teammate_lead(
        self,
        hand: list[Card],
        combo: Combo,
        score: float,
        last_combo: Combo,
        context: PlayContext,
        knowledge: TableKnowledge,
        players: list[Player],
    ) -> bool:
        if combo.kind in {"bomb", "rocket"}:
            return False
        if combo.pattern_key() != last_combo.pattern_key():
            return False

        turns_after = self._estimate_turns_for_signature(
            self._remaining_signature(hand, combo),
            knowledge.mode,
        )
        if turns_after <= 1:
            return True

        pressure = knowledge.counter_pressure(context.current_index, combo, players, hand)
        small_cover = combo.main_rank - last_combo.main_rank <= 2

        if context.is_landlord_downstream and small_cover and turns_after <= 2:
            return True
        if pressure.hold_probability >= 0.62 and small_cover and turns_after <= 2:
            return True
        if score >= 40.0 and small_cover and turns_after <= 2:
            return True

        if context.game_phase == GAME_PHASE_LATE and context.last_player_cards <= 5:
            if small_cover and turns_after <= 3:
                return True

        return False

    def _estimate_landlord_strength(self, hand: list[Card], mode: str) -> float:
        signature = self._hand_to_signature(hand)
        counts = Counter(card.rank for card in hand)
        turns = self._estimate_turns(hand, mode)
        candidates = generate_candidate_plays(hand, mode)

        bombs = sum(1 for count in counts.values() if count >= 4)
        trios = sum(1 for count in counts.values() if count >= 3)
        pairs = sum(1 for count in counts.values() if count >= 2)

        longest_straight = max(
            (combo.total_cards for combo in candidates if combo.kind == "straight"),
            default=0,
        )
        longest_pair_straight = max(
            (combo.sequence_length for combo in candidates if combo.kind == "pair_straight"),
            default=0,
        )
        longest_airplane = max(
            (combo.sequence_length for combo in candidates if combo.kind.startswith("airplane")),
            default=0,
        )

        non_overlapping_combos = self._count_non_overlapping_combos(hand, mode)
        flexibility = min(1.5, len(candidates) / 20.0)

        score = 0.0
        score += self._control_score_from_signature(signature) * 1.5
        score += bombs * 5.5
        score += trios * 1.5
        score += pairs * 0.3
        score += longest_straight * 0.7
        score += longest_pair_straight * 1.2
        score += longest_airplane * 2.8
        score += max(0, 10 - turns) * 2.25
        score += non_overlapping_combos * 2.0
        score += flexibility * 3.0
        if counts.get(16, 0) >= 1 and counts.get(17, 0) >= 1:
            score += 4.0

        high_scattered = sum(1 for rank, cnt in counts.items() if cnt == 1 and rank >= 13)
        score -= high_scattered * 1.8

        bomb_quality = sum(
            1.0 for rank, cnt in counts.items()
            if cnt >= 4 and rank >= 10
        )
        score += bomb_quality * 1.5

        return score

    def _estimate_farmer_dominance(self, hand: list[Card], mode: str) -> float:
        signature = self._hand_to_signature(hand)
        turns = self._estimate_turns(hand, mode)
        control = self._control_score_from_signature(signature)
        candidates = generate_candidate_plays(hand, mode)
        low_pressure = sum(
            1
            for combo in candidates
            if combo.kind in {"single", "pair", "trio"} and combo.main_rank <= 10
        )
        structure = sum(1 for combo in candidates if combo.kind in STRUCTURED_KINDS)
        bombs = sum(1 for combo in candidates if combo.kind == "bomb")
        return control * 1.4 + max(0, 8 - turns) * 2.2 + low_pressure * 0.8 + structure * 0.6 + bombs * 1.0

    @staticmethod
    def _project_bid_level(strength: float, thresholds: list[float]) -> int:
        bid = 0
        for level, threshold in enumerate(thresholds, start=1):
            if strength >= threshold:
                bid = level
        return bid

    @staticmethod
    def _control_card_count(hand: list[Card]) -> int:
        return sum(1 for card in hand if card.rank >= 14)

    @staticmethod
    def _count_non_overlapping_combos(hand: list[Card], mode: str) -> int:
        remaining = list(hand)
        count = 0
        used_ranks: set[int] = set()
        candidates = generate_candidate_plays(remaining, mode)
        candidates.sort(key=lambda c: (-c.total_cards, c.kind not in {"bomb", "rocket"}))

        for combo in candidates:
            combo_ranks = {c.rank for c in combo.cards}
            if combo_ranks & used_ranks:
                continue
            used_ranks |= combo_ranks
            count += 1

        return count

    def _combo_structure_bonus(self, combo: Combo) -> float:
        if combo.kind == "airplane_pair":
            return 28.0 + combo.sequence_length * 3.5
        if combo.kind == "airplane_single":
            return 23.0 + combo.sequence_length * 3.0
        if combo.kind == "trio_straight":
            return 20.0 + combo.sequence_length * 2.8
        if combo.kind == "pair_straight":
            return 18.0 + combo.sequence_length * 2.2
        if combo.kind == "straight":
            return 13.0 + combo.sequence_length * 1.7
        if combo.kind == "trio_pair":
            return 11.0
        if combo.kind == "trio_single":
            return 8.0
        if combo.kind == "trio":
            return 5.5
        if combo.kind == "pair":
            return 2.5
        if combo.kind == "single":
            return 1.0
        if combo.kind == "four_two_pair":
            return 6.0
        if combo.kind == "four_two_single":
            return 3.0
        return -4.0

    @staticmethod
    def _opening_fragment_penalty(combo: Combo, has_structured_option: bool) -> float:
        if not has_structured_option:
            return 0.0
        if combo.kind == "single":
            return 10.0
        if combo.kind == "pair":
            return 4.0
        return 0.0

    def _tempo_bonus(self, combo: Combo, context: PlayContext, opening: bool) -> float:
        score = 0.0
        if opening:
            if context.opponent_min_cards <= 2:
                score += combo.total_cards * 0.85
                if combo.kind in {"single", "pair"}:
                    score += combo.main_rank * 0.65
            else:
                score += combo.total_cards * 0.35
        else:
            if context.next_player_is_teammate:
                score += 4.0
            if combo.kind not in {"bomb", "rocket"}:
                score += 3.0
        return score

    def _high_structure_reserve_penalty(
        self,
        combo: Combo,
        context: PlayContext,
        current_turns: int,
        turns_after: int,
    ) -> float:
        high_structure_kinds = {
            "straight",
            "pair_straight",
            "trio_straight",
            "airplane_single",
            "airplane_pair",
        }
        if combo.kind not in high_structure_kinds:
            return 0.0
        if turns_after <= 1:
            return 0.0

        chasing_spring = (
            current_turns <= 4
            and turns_after <= 3
            and context.control_cards >= 4
            and context.control_margin >= 3.0
        )
        if chasing_spring:
            return 0.0

        high_cards_used = sum(1 for card in combo.cards if card.rank >= 11)
        control_cards_used = sum(1 for card in combo.cards if card.rank >= 14)
        penalty = 0.0

        if combo.main_rank >= 13:
            penalty += (combo.main_rank - 12) * 10.0
        elif combo.main_rank == 12:
            penalty += 4.0

        if high_cards_used >= 3:
            penalty += (high_cards_used - 2) * 4.0
        if control_cards_used:
            control_tax = 4.0 if context.control_margin >= 3.0 else 8.0
            penalty += control_cards_used * control_tax
        if combo.total_cards >= 8 and combo.main_rank >= 12:
            penalty += 8.0

        if context.is_landlord:
            penalty *= 0.9
        if context.opponent_min_cards <= 3:
            penalty *= 0.45
        return penalty

    def _finish_bonus(
        self,
        turns_after: int,
        remaining_signature: tuple[int, ...],
        combo: Combo,
        context: PlayContext,
    ) -> float:
        remaining_cards = sum(remaining_signature)
        if remaining_cards == 0:
            return 400.0
        if turns_after == 1:
            return 130.0
        if turns_after == 2 and (context.endgame or remaining_cards <= 8):
            return 55.0
        if combo.total_cards >= remaining_cards:
            return 12.0
        return 0.0

    def _turn_optimization_bonus(
        self,
        current_turns: int,
        turns_after: int,
        combo: Combo,
        context: PlayContext,
    ) -> float:
        projected_turns = 1 + turns_after
        delta = current_turns - projected_turns
        if delta > 0:
            bonus = delta * 16.0
            if context.is_opposite_landlord and combo.kind in STRUCTURED_KINDS:
                bonus += 5.0
            return bonus
        if delta == 0:
            return 4.0 if combo.kind in STRUCTURED_KINDS else 1.5
        penalty = abs(delta) * 22.0
        if combo.kind in {"bomb", "rocket"}:
            penalty += 12.0
        return -penalty

    def _reentry_bonus(
        self,
        hand: list[Card],
        combo: Combo,
        context: PlayContext,
        mode: str,
    ) -> float:
        if combo.kind not in {"single", "pair", "trio"}:
            return 0.0
        candidates = generate_candidate_plays(hand, mode)
        same_lane_higher = [
            candidate
            for candidate in candidates
            if candidate.kind == combo.kind and candidate.main_rank > combo.main_rank
        ]
        if not same_lane_higher:
            return 0.0

        smallest_higher = min(candidate.main_rank for candidate in same_lane_higher)
        bonus = max(0.0, 15.0 - (smallest_higher - combo.main_rank) * 1.8)
        if context.is_landlord:
            bonus *= 1.2
        if context.is_opposite_landlord:
            bonus *= 1.35
        if context.is_landlord_downstream and context.last_player_is_landlord:
            bonus *= 1.25
        return bonus

    def _bomb_preservation_penalty(self, hand: list[Card], combo: Combo, context: PlayContext) -> float:
        if combo.kind in {"bomb", "rocket"}:
            return 0.0
        total_counts = Counter(card.rank for card in hand)
        used_counts = Counter(card.rank for card in combo.cards)
        penalty = 0.0
        has_control = context.control_cards >= 3 and context.control_margin >= 2.5

        for rank, used in used_counts.items():
            before = total_counts[rank]
            if before >= 4 and used < before:
                penalty += 26.0
                if not has_control:
                    penalty += 18.0
                elif rank >= 14:
                    penalty -= 8.0

        if context.game_phase == GAME_PHASE_LATE and has_control:
            penalty *= 0.7

        return max(0.0, penalty)

    def _endgame_lane_bonus(self, combo: Combo, context: PlayContext, opening: bool) -> float:
        score = 0.0
        if context.opponent_min_cards <= 2 and combo.kind not in {"bomb", "rocket"}:
            if combo.total_cards == context.opponent_min_cards:
                penalty = 30.0 if opening else 13.0
                if context.next_player_is_landlord:
                    penalty += 8.0
                if combo.kind in {"single", "pair"} and combo.main_rank >= 15:
                    penalty *= 0.35
                # Stronger penalty when opponent is about to win
                if context.opponent_min_cards <= 1 and opening:
                    penalty *= 1.6
                score -= penalty
            elif combo.total_cards > context.opponent_min_cards:
                score += 8.0 if opening else 4.0

        if context.next_player_is_teammate and context.teammate_min_cards <= 2:
            if combo.total_cards == context.teammate_min_cards:
                score += 48.0 if opening else 30.0
                if context.teammate_min_cards == 1 and combo.kind == "single":
                    score += 28.0  # boosted from 14
                    # Extra bonus for low singles (easier for teammate to beat)
                    if combo.main_rank <= 10:
                        score += 10.0
                elif context.teammate_min_cards == 2 and combo.kind == "pair":
                    score += 20.0  # boosted from 14
            elif combo.total_cards > context.teammate_min_cards and combo.kind in STRUCTURED_KINDS:
                # Much stronger penalty for blocking teammate's win with a complex combo
                penalty = 40.0 if opening else 20.0  # increased from 18/8
                if context.teammate_min_cards == 1:
                    penalty += 30.0  # never block teammate with 1 card
                score -= penalty
        return score

    def _support_bonus(self, combo: Combo, context: PlayContext) -> float:
        if not context.next_player_is_teammate:
            return 0.0

        scale = 1.25 if context.teammate_min_cards <= 8 else 0.9
        if context.is_opposite_landlord:
            scale *= 1.35
        elif context.is_landlord_downstream:
            scale *= 1.2
        elif context.is_landlord_upstream:
            scale *= 1.15
        kind = context.next_teammate_kind
        family = context.next_teammate_family
        bonus = 0.0

        if kind == "pair":
            if combo.kind == "pair":
                bonus += 13.0
            elif combo.kind == "pair_straight":
                bonus += 7.0
            elif combo.kind in PAIR_ATTACHMENT_KINDS:
                bonus -= 9.0
        elif kind == "single":
            if combo.kind == "single":
                bonus += 12.0
            elif combo.kind == "straight":
                bonus += 5.0
            elif combo.kind in SINGLE_ATTACHMENT_KINDS:
                bonus -= 7.5
        elif kind == "pair_straight":
            if combo.kind == "pair_straight":
                bonus += 11.0
            elif combo.kind == "pair":
                bonus += 5.0
        elif kind == "straight":
            if combo.kind == "straight":
                bonus += 9.0
        elif kind and combo.kind == kind:
            bonus += 7.0
        elif family and combo_to_family(combo.kind) == family:
            bonus += 5.0

        if context.teammate_min_cards <= 4:
            bonus *= 1.3

        return bonus * scale

    def _pressure_bonus(self, combo: Combo, context: PlayContext) -> float:
        if context.next_player_is_teammate:
            return 0.0

        bonus = 0.0
        if combo.kind in PAIR_LANE_KINDS:
            bonus += context.next_opponent_pair_scarcity * 4.2
            bonus += context.next_opponent_low_pair_scarcity * 2.6
            if combo.main_rank <= 11:
                bonus += context.next_opponent_low_pair_scarcity * 1.5
        if combo.kind in SINGLE_LANE_KINDS or combo.kind == "straight":
            bonus += context.next_opponent_single_scarcity * 3.8
            if combo.main_rank <= 10:
                bonus += context.next_opponent_single_scarcity * 1.2
        return bonus

    def _control_strategy_bonus(self, combo: Combo, context: PlayContext, opening: bool) -> float:
        score = context.control_margin * (1.4 if opening else 0.9)
        if combo.kind in STRUCTURED_KINDS and context.control_margin >= 3.0:
            score += 4.0
        if combo.kind in {"single", "pair"} and combo.main_rank >= 14:
            if context.next_player_is_landlord or context.urgent_block:
                score += 2.5
            else:
                score -= 6.0
        if combo.kind in {"single", "pair"} and combo.main_rank <= 10 and context.next_player_is_landlord:
            score -= max(0.0, 7.0 - context.control_margin * 1.2)
        if context.is_opposite_landlord and combo.kind in STRUCTURED_KINDS:
            score += 5.0
        return score

    def _lead_security_bonus(
        self,
        combo: Combo,
        context: PlayContext,
        pressure: CounterPressure,
        opening: bool,
    ) -> float:
        score = pressure.hold_probability * (16.0 if opening else 13.5)
        score += pressure.teammate_follow_probability * (5.0 if context.next_player_is_teammate else 2.0)

        if context.next_player_is_landlord:
            score -= pressure.next_opponent_pressure * (16.0 if combo.kind in {"single", "pair"} else 10.0)
        elif not context.next_player_is_teammate:
            score -= pressure.next_opponent_pressure * 7.0

        if combo.kind in STRUCTURED_KINDS:
            score -= pressure.any_opponent_pressure * (6.5 if opening else 4.0)
        if combo.kind in {"single", "pair"} and combo.main_rank >= 14:
            score += pressure.hold_probability * 4.5
        return score

    def _overkill_penalty(self, combo: Combo, last_combo: Combo, context: PlayContext) -> float:
        if combo.kind in {"bomb", "rocket"}:
            base = 12.0 if context.urgent_block else 24.0
            if context.game_phase == GAME_PHASE_LATE and context.opponent_min_cards <= 4:
                base *= 0.5
            return base
        if combo.kind != last_combo.kind or combo.pattern_key() != last_combo.pattern_key():
            return 10.0

        delta = combo.main_rank - last_combo.main_rank
        penalty = max(0.0, (delta - 1) * 2.4)
        if context.last_player_is_landlord and context.is_landlord_downstream:
            penalty += delta * 1.5
        return penalty

    def _bomb_risk_penalty(
        self,
        combo: Combo,
        context: PlayContext,
        opening: bool,
        pressure: CounterPressure,
    ) -> float:
        if combo.kind in {"bomb", "rocket"}:
            return 0.0
        if pressure.bomb_pressure <= 0 and context.opponent_bomb_threat <= 0:
            return 0.0
        threat = max(context.opponent_bomb_threat * 0.08, pressure.bomb_pressure)
        if combo.kind in {"straight", "pair_straight", "trio_straight", "airplane_single", "airplane_pair"}:
            return min(14.0, threat * (15.0 if opening else 10.5))
        if combo.total_cards >= 6:
            return min(7.0, threat * 7.5)
        return 0.0

    def _explosive_penalty(
        self,
        combo: Combo,
        context: PlayContext,
        turns_after: int,
        opening: bool,
        pressure: CounterPressure,
    ) -> float:
        if combo.kind not in {"bomb", "rocket"}:
            return 0.0
        if turns_after <= 1:
            return 0.0

        penalty = 22.0
        if opening and not context.endgame:
            penalty += 8.0
        if not context.urgent_block and not context.endgame:
            penalty += 8.0
        if pressure.any_opponent_pressure < 0.35:
            penalty += 3.0
        if context.opponent_bomb_threat > 0:
            penalty += 2.0

        if context.game_phase == GAME_PHASE_LATE:
            penalty *= 0.65
        if context.opponent_min_cards <= 4:
            penalty *= 0.5
        if context.urgent_block:
            penalty *= 0.35

        own_bombs_left = sum(
            1 for rank, count in Counter(c.rank for c in combo.cards).items()
            if count >= 4
        )
        if own_bombs_left > 1:
            penalty *= 0.7

        return penalty

    def _breakup_penalty(self, hand: list[Card], combo: Combo) -> float:
        total_counts = Counter(card.rank for card in hand)
        used_counts = Counter(card.rank for card in combo.cards)
        penalty = 0.0

        for rank, used in used_counts.items():
            before = total_counts[rank]
            if used < before:
                if before >= 4:
                    penalty += 16.0
                elif before == 3:
                    penalty += 11.0
                elif before == 2:
                    penalty += 7.0
            if combo.kind in {"single", "pair"} and before >= 3:
                penalty += 5.0
            if rank >= 16 and combo.kind != "rocket":
                penalty += used * 7.0
            elif rank == 15 and combo.kind not in {"bomb", "rocket"}:
                penalty += used * 4.0
            elif rank >= 13 and combo.kind == "single":
                penalty += 2.5

        return penalty

    def _estimate_turns(self, hand: list[Card], mode: str) -> int:
        return self._estimate_turns_for_signature(self._hand_to_signature(hand), mode)

    @lru_cache(maxsize=60000)
    def _estimate_turns_for_signature(self, signature: tuple[int, ...], mode: str) -> int:
        total_cards = sum(signature)
        if total_cards == 0:
            return 0

        if total_cards >= 22:
            return self._greedy_turn_estimate(signature, mode)

        hand = self._signature_to_hand(signature)
        if identify_combo(hand, mode) is not None:
            return 1

        if total_cards >= 17:
            best = total_cards
            for combo in self._search_candidates(hand, total_cards, mode)[:10]:
                next_signature = self._subtract_combo(signature, combo)
                estimate = 1 + self._greedy_turn_estimate(next_signature, mode)
                if estimate < best:
                    best = estimate
            return best

        candidates = self._search_candidates(hand, total_cards, mode)
        best = total_cards

        for combo in candidates:
            next_signature = self._subtract_combo(signature, combo)
            if total_cards <= 12:
                estimate = 1 + self._estimate_turns_two_ply(next_signature, mode)
            else:
                estimate = 1 + self._estimate_turns_for_signature(next_signature, mode)
            if estimate < best:
                best = estimate
            if best <= 2:
                break

        return best

    def _estimate_turns_two_ply(self, signature: tuple[int, ...], mode: str) -> int:
        total = sum(signature)
        if total == 0:
            return 0

        hand = self._signature_to_hand(signature)
        if identify_combo(hand, mode) is not None:
            return 1

        candidates = self._search_candidates(hand, total, mode)
        best = total
        for combo in candidates[:8]:
            next_sig = self._subtract_combo(signature, combo)
            estimate = 1 + self._greedy_turn_estimate(next_sig, mode)
            if estimate < best:
                best = estimate
            if best <= 1:
                break
        return best

    @lru_cache(maxsize=60000)
    def _score_signature(self, signature: tuple[int, ...], mode: str) -> float:
        total_cards = sum(signature)
        if total_cards == 0:
            return 240.0

        hand = self._signature_to_hand(signature)
        counts = Counter(card.rank for card in hand)
        candidates = generate_candidate_plays(hand, mode)

        singles = sum(1 for count in counts.values() if count == 1)
        pairs = sum(1 for count in counts.values() if count == 2)
        trios = sum(1 for count in counts.values() if count == 3)
        bombs = sum(1 for count in counts.values() if count >= 4)
        high_singles = sum(1 for rank, count in counts.items() if count == 1 and rank >= 13)
        turns = self._estimate_turns_for_signature(signature, mode)

        longest_straight = max(
            (combo.sequence_length for combo in candidates if combo.kind == "straight"),
            default=0,
        )
        longest_pair_straight = max(
            (combo.sequence_length for combo in candidates if combo.kind == "pair_straight"),
            default=0,
        )
        longest_trio_straight = max(
            (combo.sequence_length for combo in candidates if combo.kind == "trio_straight"),
            default=0,
        )
        longest_airplane = max(
            (combo.sequence_length for combo in candidates if combo.kind.startswith("airplane")),
            default=0,
        )

        score = 120.0
        score -= turns * 24.0
        score -= singles * 4.0
        score -= high_singles * 3.2
        score += pairs * 2.5
        score += trios * 4.0
        score += bombs * 9.5
        score += longest_straight * 1.4
        score += longest_pair_straight * 2.1
        score += longest_trio_straight * 2.4
        score += longest_airplane * 4.0
        score += self._control_score_from_signature(signature)
        return score

    @lru_cache(maxsize=60000)
    def _greedy_turn_estimate(self, signature: tuple[int, ...], mode: str) -> int:
        total_cards = sum(signature)
        if total_cards == 0:
            return 0

        current = signature
        turns = 0
        while sum(current) > 0:
            hand = self._signature_to_hand(current)
            if identify_combo(hand, mode) is not None:
                turns += 1
                break
            combo = self._search_candidates(hand, sum(current), mode)[0]
            current = self._subtract_combo(current, combo)
            turns += 1
        return turns

    def _search_candidates(self, hand: list[Card], total_cards: int, mode: str) -> list[Combo]:
        candidates = generate_candidate_plays(hand, mode)
        limit = 10
        if total_cards <= 12:
            limit = 30
        elif total_cards <= 18:
            limit = 18

        prioritized = sorted(
            candidates,
            key=lambda combo: (
                combo.kind in {"bomb", "rocket"},
                self.COMBO_PRIORITY.get(combo.kind, 99),
                -combo.total_cards,
                combo.main_rank,
                combo.bomb_size,
            ),
        )

        if len(prioritized) <= limit:
            return prioritized

        result = prioritized[:limit]
        explosive_in_result = any(c.kind in {"bomb", "rocket"} for c in result)
        if not explosive_in_result:
            explosive = next((combo for combo in prioritized if combo.kind in {"bomb", "rocket"}), None)
            if explosive is not None:
                result.append(explosive)
        return result

    def _remaining_signature(self, hand: list[Card], combo: Combo) -> tuple[int, ...]:
        signature = self._hand_to_signature(hand)
        return self._subtract_combo(signature, combo)

    def _subtract_combo(self, signature: tuple[int, ...], combo: Combo) -> tuple[int, ...]:
        rank_index = {rank: index for index, rank in enumerate(RANK_SEQUENCE)}
        values = list(signature)
        for rank, count in Counter(card.rank for card in combo.cards).items():
            values[rank_index[rank]] -= count
        return tuple(values)

    def _leaves_no_cards(self, hand: list[Card], combo: Combo) -> bool:
        return sum(self._remaining_signature(hand, combo)) == 0

    @staticmethod
    def _hand_to_signature(hand: list[Card]) -> tuple[int, ...]:
        counts = Counter(card.rank for card in hand)
        return tuple(counts.get(rank, 0) for rank in RANK_SEQUENCE)

    @staticmethod
    def _signature_to_hand(signature: tuple[int, ...]) -> list[Card]:
        hand: list[Card] = []
        serial = 0
        suits = ["C", "D", "H", "S"]
        for rank, count in zip(RANK_SEQUENCE, signature):
            for index in range(count):
                serial += 1
                suit = "J" if rank >= 16 else suits[index % len(suits)]
                deck_id = 1 + index // len(suits)
                hand.append(Card(rank=rank, suit=suit, deck_id=deck_id, serial=serial))
        return hand

    @staticmethod
    def _control_score_from_signature(signature: tuple[int, ...]) -> float:
        score = 0.0
        counts = dict(zip(RANK_SEQUENCE, signature))
        for rank, count in counts.items():
            if rank == 17:
                score += count * 4.6
            elif rank == 16:
                score += count * 4.0
            elif rank == 15:
                score += count * 2.8
            elif rank == 14:
                score += count * 1.3
            elif rank == 13:
                score += count * 0.8
        return score

    @staticmethod
    def _guess_mode(players: list[Player]) -> str:
        return "classic" if len(players) == 3 else "extended"


def combo_to_family(kind: str) -> str:
    if kind in PAIR_LANE_KINDS:
        return "pair"
    if kind in SINGLE_LANE_KINDS:
        return "single"
    if kind in {"straight"}:
        return "straight"
    if kind in TRIO_LANE_KINDS:
        return "trio"
    if kind in {"four_two_single", "four_two_pair"}:
        return "four"
    return "bomb"


def combo_main_ranks(combo: Combo) -> set[int]:
    if combo.kind in {"single", "pair", "trio", "bomb"}:
        return {combo.main_rank}
    if combo.kind in {"trio_single", "trio_pair"}:
        return {combo.main_rank}
    if combo.kind in {"straight", "pair_straight", "trio_straight", "airplane_single", "airplane_pair"}:
        return set(range(combo.main_rank - combo.sequence_length + 1, combo.main_rank + 1))
    if combo.kind in {"four_two_single", "four_two_pair"}:
        return {combo.main_rank}
    if combo.kind == "rocket":
        return {16, 17}
    return {combo.main_rank}


def combinations_probability_weight(available: int, required: int, share: float) -> float:
    if available < required or share <= 0:
        return 0.0
    return math.comb(available, required) * (share**required)


def per_rank_hold_factor(available: int, required: int, share: float) -> float:
    return min(1.0, combinations_probability_weight(available, required, share) * required * 0.9)


def probability_from_expected(expected: float, factor: float) -> float:
    if expected <= 0:
        return 0.0
    return clamp01(1.0 - math.exp(-expected * factor))


def clamp01(value: float) -> float:
    return max(0.0, min(1.0, value))
