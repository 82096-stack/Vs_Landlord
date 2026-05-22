from __future__ import annotations

import random
from collections import Counter
from typing import Iterable

from ddz.models import Card, Combo, Player, card_sort_key


MODE_RULES = {
    "classic": {
        "label": "二打一",
        "player_count": 3,
        "deck_count": 1,
        "bottom_count": 3,
        "hand_size": 17,
    },
    "extended": {
        "label": "三打一",
        "player_count": 4,
        "deck_count": 2,
        "bottom_count": 8,
        "hand_size": 25,
    },
}

SUITS = ["C", "D", "H", "S"]


def build_deck(mode: str) -> list[Card]:
    rules = MODE_RULES[mode]
    serial = 0
    deck: list[Card] = []
    for deck_id in range(1, rules["deck_count"] + 1):
        for suit in SUITS:
            for rank in range(3, 16):
                serial += 1
                deck.append(Card(rank=rank, suit=suit, deck_id=deck_id, serial=serial))
        for rank in (16, 17):
            serial += 1
            deck.append(Card(rank=rank, suit="J", deck_id=deck_id, serial=serial))
    return deck


def deal_cards(players: list[Player], mode: str, marked_card: Card | None = None) -> list[Card]:
    rules = MODE_RULES[mode]
    deck = build_deck(mode)
    random.shuffle(deck)
    if marked_card is not None:
        _move_marked_card_out_of_bottom(deck, marked_card, rules["hand_size"] * len(players))
    for player in players:
        player.hand = []
        player.role = "farmer"
    hand_size = rules["hand_size"]
    for index in range(hand_size):
        for offset, player in enumerate(players):
            player.hand.append(deck[index * len(players) + offset])
    bottom_cards = deck[hand_size * len(players) :]
    for player in players:
        player.sort_hand()
    return sorted(bottom_cards, key=card_sort_key)


def choose_marked_card(mode: str) -> Card:
    return random.choice(build_deck(mode))


def identify_combo(cards: Iterable[Card], mode: str = "classic") -> Combo | None:
    sorted_cards = sorted(cards, key=card_sort_key)
    if not sorted_cards:
        return None

    counts = Counter(card.rank for card in sorted_cards)
    ranks = sorted(counts)
    total = len(sorted_cards)

    if _is_rocket(counts, total):
        return Combo("rocket", sorted_cards, main_rank=17)

    if len(counts) == 1 and total >= 4:
        return Combo("bomb", sorted_cards, main_rank=ranks[0], bomb_size=total)

    if total == 1:
        return Combo("single", sorted_cards, main_rank=ranks[0])

    if total == 2 and len(counts) == 1:
        return Combo("pair", sorted_cards, main_rank=ranks[0])

    if total == 3 and len(counts) == 1:
        return Combo("trio", sorted_cards, main_rank=ranks[0])

    if total == 4:
        trio_rank = _rank_with_count(counts, 3)
        if trio_rank is not None:
            return Combo("trio_single", sorted_cards, main_rank=trio_rank)

    if total == 5:
        trio_rank = _rank_with_count(counts, 3)
        pair_rank = _rank_with_count(counts, 2)
        if trio_rank is not None and pair_rank is not None and trio_rank != pair_rank:
            return Combo("trio_pair", sorted_cards, main_rank=trio_rank)

    straight = _detect_straight(counts, total, mode)
    if straight:
        return Combo("straight", sorted_cards, main_rank=straight[-1], sequence_length=len(straight))

    pair_straight = _detect_pair_straight(counts, total)
    if pair_straight:
        return Combo(
            "pair_straight",
            sorted_cards,
            main_rank=pair_straight[-1],
            sequence_length=len(pair_straight),
        )

    trio_straight = _detect_trio_straight(counts, total)
    if trio_straight:
        return Combo(
            "trio_straight",
            sorted_cards,
            main_rank=trio_straight[-1],
            sequence_length=len(trio_straight),
        )

    airplane_single = _detect_airplane(counts, total, with_pairs=False)
    if airplane_single:
        return Combo(
            "airplane_single",
            sorted_cards,
            main_rank=airplane_single[-1],
            sequence_length=len(airplane_single),
        )

    airplane_pair = _detect_airplane(counts, total, with_pairs=True)
    if airplane_pair:
        return Combo(
            "airplane_pair",
            sorted_cards,
            main_rank=airplane_pair[-1],
            sequence_length=len(airplane_pair),
        )

    four_two_single = _rank_with_count(counts, 4)
    if total == 6 and four_two_single is not None:
        if sum(counts.values()) - counts[four_two_single] == 2:
            return Combo("four_two_single", sorted_cards, main_rank=four_two_single)

    if total == 8 and four_two_single is not None:
        rest = {rank: count for rank, count in counts.items() if rank != four_two_single}
        if len(rest) == 2 and all(count == 2 for count in rest.values()):
            return Combo("four_two_pair", sorted_cards, main_rank=four_two_single)

    return None


def can_beat(candidate: Combo, target: Combo | None) -> bool:
    if target is None:
        return True
    if target.kind == "rocket":
        return False
    if candidate.kind == "rocket":
        return True
    if candidate.kind == "bomb":
        if target.kind != "bomb":
            return True
        if candidate.bomb_size != target.bomb_size:
            return candidate.bomb_size > target.bomb_size
        return candidate.main_rank > target.main_rank
    if target.kind == "bomb":
        return False
    if candidate.pattern_key() != target.pattern_key():
        return False
    return candidate.main_rank > target.main_rank


def generate_candidate_plays(hand: list[Card], mode: str = "classic") -> list[Combo]:
    grouped = _group_by_rank(hand)
    combos: list[Combo] = []

    for rank in sorted(grouped):
        combos.append(Combo("single", grouped[rank][:1], main_rank=rank))
        if len(grouped[rank]) >= 2:
            combos.append(Combo("pair", grouped[rank][:2], main_rank=rank))
        if len(grouped[rank]) >= 3:
            combos.append(Combo("trio", grouped[rank][:3], main_rank=rank))
        if len(grouped[rank]) >= 4:
            for size in range(4, len(grouped[rank]) + 1):
                combos.append(
                    Combo(
                        "bomb",
                        grouped[rank][:size],
                        main_rank=rank,
                        bomb_size=size,
                    )
                )

    if 16 in grouped and 17 in grouped:
        combos.append(Combo("rocket", [grouped[16][0], grouped[17][0]], main_rank=17))
    if len(grouped.get(16, [])) >= 2 and len(grouped.get(17, [])) >= 2:
        combos.append(
            Combo(
                "rocket",
                [grouped[16][0], grouped[16][1], grouped[17][0], grouped[17][1]],
                main_rank=17,
            )
        )

    combos.extend(_generate_straight_candidates(grouped, mode))
    combos.extend(_generate_pair_straight_candidates(grouped))
    combos.extend(_generate_trio_straight_candidates(grouped))
    combos.extend(_generate_trio_attachment_candidates(grouped))
    combos.extend(_generate_four_attachment_candidates(grouped))
    combos.extend(_generate_airplane_candidates(grouped, with_pairs=False))
    combos.extend(_generate_airplane_candidates(grouped, with_pairs=True))
    return _dedupe_combos(combos)


def same_team(first: Player, second: Player) -> bool:
    if first.role == "landlord" or second.role == "landlord":
        return first.role == second.role
    return True


def _move_marked_card_out_of_bottom(deck: list[Card], marked_card: Card, deal_region_size: int) -> None:
    marker_index = next(
        (index for index, card in enumerate(deck) if card.serial == marked_card.serial),
        None,
    )
    if marker_index is None or marker_index < deal_region_size:
        return
    swap_index = random.randrange(deal_region_size)
    deck[marker_index], deck[swap_index] = deck[swap_index], deck[marker_index]


def _group_by_rank(hand: list[Card]) -> dict[int, list[Card]]:
    grouped: dict[int, list[Card]] = {}
    for card in sorted(hand, key=card_sort_key):
        grouped.setdefault(card.rank, []).append(card)
    return grouped


def _dedupe_combos(combos: list[Combo]) -> list[Combo]:
    deduped: list[Combo] = []
    seen: set[tuple] = set()
    for combo in combos:
        signature = (
            combo.kind,
            combo.main_rank,
            combo.sequence_length,
            combo.bomb_size,
            tuple(sorted(card.rank for card in combo.cards)),
        )
        if signature in seen:
            continue
        seen.add(signature)
        deduped.append(combo)
    return deduped


def _is_rocket(counts: Counter[int], total: int) -> bool:
    if set(counts) != {16, 17}:
        return False
    if total == 2 and counts[16] == 1 and counts[17] == 1:
        return True
    return total == 4 and counts[16] == 2 and counts[17] == 2


def _rank_with_count(counts: Counter[int], target_count: int) -> int | None:
    for rank, count in counts.items():
        if count == target_count:
            return rank
    return None


def _detect_straight(counts: Counter[int], total: int, mode: str) -> list[int] | None:
    if total < 5 or len(counts) != total:
        return None
    ranks = sorted(counts)
    if ranks[-1] < 15 and _is_consecutive(ranks):
        return ranks
    if mode == "extended":
        low_order_ranks = _extended_low_sequence_ranks(ranks)
        if low_order_ranks is not None:
            return low_order_ranks
    return None


def _detect_pair_straight(counts: Counter[int], total: int) -> list[int] | None:
    if total < 6 or total % 2 != 0:
        return None
    if any(count != 2 for count in counts.values()):
        return None
    ranks = sorted(counts)
    if ranks[-1] >= 15 or len(ranks) < 3:
        return None
    if _is_consecutive(ranks):
        return ranks
    return None


def _detect_trio_straight(counts: Counter[int], total: int) -> list[int] | None:
    if total < 6 or total % 3 != 0:
        return None
    if any(count != 3 for count in counts.values()):
        return None
    ranks = sorted(counts)
    if ranks[-1] >= 15 or len(ranks) < 2:
        return None
    if _is_consecutive(ranks):
        return ranks
    return None


def _detect_airplane(
    counts: Counter[int],
    total: int,
    with_pairs: bool,
) -> list[int] | None:
    unit = 5 if with_pairs else 4
    min_total = 10 if with_pairs else 8
    if total < min_total or total % unit != 0:
        return None
    sequence_length = total // unit
    trio_ranks = sorted(rank for rank, count in counts.items() if count >= 3 and rank < 15)
    for candidate in _all_consecutive_slices(trio_ranks, sequence_length):
        if len(candidate) != sequence_length:
            continue
        remaining = counts.copy()
        for rank in candidate:
            remaining[rank] -= 3
            if remaining[rank] > 0:
                break
        else:
            attachments = {rank: count for rank, count in remaining.items() if count > 0}
            if len(attachments) != sequence_length:
                continue
            if with_pairs and all(count == 2 for count in attachments.values()):
                return candidate
            if not with_pairs and all(count == 1 for count in attachments.values()):
                return candidate
    return None


def _generate_straight_candidates(grouped: dict[int, list[Card]], mode: str) -> list[Combo]:
    ranks = sorted(rank for rank in grouped if rank < 15)
    combos: list[Combo] = []
    for run in _split_consecutive_runs(ranks):
        if len(run) < 5:
            continue
        for length in range(5, len(run) + 1):
            for slice_ranks in _all_consecutive_slices(run, length):
                cards = [grouped[rank][0] for rank in slice_ranks]
                combos.append(
                    Combo(
                        "straight",
                        cards,
                        main_rank=slice_ranks[-1],
                        sequence_length=length,
                    )
                )
    if mode == "extended":
        combos.extend(_generate_extended_low_straights(grouped))
    return combos


def _generate_pair_straight_candidates(grouped: dict[int, list[Card]]) -> list[Combo]:
    ranks = sorted(rank for rank, cards in grouped.items() if len(cards) >= 2 and rank < 15)
    combos: list[Combo] = []
    for run in _split_consecutive_runs(ranks):
        if len(run) < 3:
            continue
        for length in range(3, len(run) + 1):
            for slice_ranks in _all_consecutive_slices(run, length):
                cards: list[Card] = []
                for rank in slice_ranks:
                    cards.extend(grouped[rank][:2])
                combos.append(
                    Combo(
                        "pair_straight",
                        cards,
                        main_rank=slice_ranks[-1],
                        sequence_length=length,
                    )
                )
    return combos


def _generate_trio_straight_candidates(grouped: dict[int, list[Card]]) -> list[Combo]:
    ranks = sorted(rank for rank, cards in grouped.items() if len(cards) >= 3 and rank < 15)
    combos: list[Combo] = []
    for run in _split_consecutive_runs(ranks):
        if len(run) < 2:
            continue
        for length in range(2, len(run) + 1):
            for slice_ranks in _all_consecutive_slices(run, length):
                cards: list[Card] = []
                for rank in slice_ranks:
                    cards.extend(grouped[rank][:3])
                combos.append(
                    Combo(
                        "trio_straight",
                        cards,
                        main_rank=slice_ranks[-1],
                        sequence_length=length,
                    )
                )
    return combos


def _generate_trio_attachment_candidates(grouped: dict[int, list[Card]]) -> list[Combo]:
    combos: list[Combo] = []
    trio_ranks = sorted(rank for rank, cards in grouped.items() if len(cards) >= 3)
    for trio_rank in trio_ranks:
        trio_cards = grouped[trio_rank][:3]
        for single_cards in _single_attachment_variants(grouped, excluded_ranks={trio_rank}, needed=1):
            combos.append(
                Combo(
                    "trio_single",
                    trio_cards + single_cards,
                    main_rank=trio_rank,
                )
            )
        for pair_ranks in _rank_attachment_variants(
            grouped,
            excluded_ranks={trio_rank},
            needed=1,
            minimum_count=2,
        ):
            combos.append(
                Combo(
                    "trio_pair",
                    trio_cards + grouped[pair_ranks[0]][:2],
                    main_rank=trio_rank,
                )
            )
    return combos


def _generate_four_attachment_candidates(grouped: dict[int, list[Card]]) -> list[Combo]:
    combos: list[Combo] = []
    bomb_ranks = sorted(rank for rank, cards in grouped.items() if len(cards) >= 4)
    for bomb_rank in bomb_ranks:
        bomb_cards = grouped[bomb_rank][:4]
        for single_cards in _single_attachment_variants(grouped, excluded_ranks={bomb_rank}, needed=2):
            combos.append(
                Combo(
                    "four_two_single",
                    bomb_cards + single_cards,
                    main_rank=bomb_rank,
                )
            )
        for pair_ranks in _rank_attachment_variants(
            grouped,
            excluded_ranks={bomb_rank},
            needed=2,
            minimum_count=2,
        ):
            cards = bomb_cards[:]
            for pair_rank in pair_ranks:
                cards += grouped[pair_rank][:2]
            combos.append(Combo("four_two_pair", cards, main_rank=bomb_rank))
    return combos


def _generate_airplane_candidates(
    grouped: dict[int, list[Card]],
    with_pairs: bool,
) -> list[Combo]:
    trio_ranks = sorted(rank for rank, cards in grouped.items() if len(cards) >= 3 and rank < 15)
    combos: list[Combo] = []
    for run in _split_consecutive_runs(trio_ranks):
        if len(run) < 2:
            continue
        for length in range(2, len(run) + 1):
            for main_ranks in _all_consecutive_slices(run, length):
                if with_pairs:
                    attachments = _rank_attachment_variants(
                        grouped,
                        excluded_ranks=set(main_ranks),
                        needed=length,
                        minimum_count=2,
                    )
                else:
                    attachments = _rank_attachment_variants(
                        grouped,
                        excluded_ranks=set(main_ranks),
                        needed=length,
                        minimum_count=1,
                    )

                for attachment_ranks in attachments:
                    cards: list[Card] = []
                    for rank in main_ranks:
                        cards.extend(grouped[rank][:3])
                    for rank in attachment_ranks:
                        cards.extend(grouped[rank][: 2 if with_pairs else 1])
                    combos.append(
                        Combo(
                            "airplane_pair" if with_pairs else "airplane_single",
                            cards,
                            main_rank=main_ranks[-1],
                            sequence_length=length,
                        )
                    )
    return combos


def _split_consecutive_runs(ranks: list[int]) -> list[list[int]]:
    if not ranks:
        return []
    runs: list[list[int]] = []
    current = [ranks[0]]
    for rank in ranks[1:]:
        if rank == current[-1] + 1:
            current.append(rank)
        else:
            runs.append(current)
            current = [rank]
    runs.append(current)
    return runs


def _all_consecutive_slices(ranks: list[int], length: int) -> list[list[int]]:
    if length <= 0 or len(ranks) < length:
        return []
    slices: list[list[int]] = []
    for start in range(0, len(ranks) - length + 1):
        candidate = ranks[start : start + length]
        if _is_consecutive(candidate):
            slices.append(candidate)
    return slices


def _is_consecutive(ranks: list[int]) -> bool:
    return all(ranks[index] + 1 == ranks[index + 1] for index in range(len(ranks) - 1))


def _rank_attachment_variants(
    grouped: dict[int, list[Card]],
    excluded_ranks: set[int],
    needed: int,
    minimum_count: int,
) -> list[list[int]]:
    eligible = [
        rank
        for rank, cards in grouped.items()
        if rank not in excluded_ranks and len(cards) >= minimum_count
    ]
    if len(eligible) < needed:
        return []

    orders = [
        sorted(eligible, key=lambda rank: (len(grouped[rank]), rank)),
        sorted(eligible, key=lambda rank: (rank >= 15, len(grouped[rank]), rank)),
        sorted(eligible, key=lambda rank: (len(grouped[rank]) > minimum_count, -rank)),
        sorted(eligible, key=lambda rank: (-len(grouped[rank]), rank)),
    ]

    variants: list[list[int]] = []
    seen: set[tuple[int, ...]] = set()
    for ordered in orders:
        selected = ordered[:needed]
        key = tuple(sorted(selected))
        if len(selected) == needed and key not in seen:
            variants.append(selected)
            seen.add(key)
    return variants


def _single_attachment_variants(
    grouped: dict[int, list[Card]],
    excluded_ranks: set[int],
    needed: int,
) -> list[list[Card]]:
    cards = [
        card
        for rank, group in grouped.items()
        if rank not in excluded_ranks
        for card in group
    ]
    if len(cards) < needed:
        return []

    orders = [
        sorted(cards, key=lambda card: (len(grouped[card.rank]), card.rank, card_sort_key(card))),
        sorted(cards, key=lambda card: (card.rank >= 15, len(grouped[card.rank]), card.rank, card_sort_key(card))),
        sorted(cards, key=lambda card: (len(grouped[card.rank]) > 1, -card.rank, card_sort_key(card))),
    ]

    variants: list[list[Card]] = []
    seen: set[tuple[int, ...]] = set()
    for ordered in orders:
        selected = ordered[:needed]
        key = tuple(sorted(card.serial for card in selected))
        if len(selected) == needed and key not in seen:
            variants.append(selected)
            seen.add(key)
    return variants


def _extended_low_sequence_ranks(ranks: list[int]) -> list[int] | None:
    if len(ranks) < 5:
        return None
    translated = sorted(_extended_low_rank_value(rank) for rank in ranks)
    if len(set(translated)) != len(translated):
        return None
    if _is_consecutive(translated):
        return sorted(ranks, key=_extended_low_rank_value)
    return None


def _generate_extended_low_straights(grouped: dict[int, list[Card]]) -> list[Combo]:
    low_order = [14, 15] + list(range(3, 14))
    eligible = [rank for rank in low_order if rank in grouped]
    combos: list[Combo] = []
    for run in _split_custom_runs(eligible, _extended_low_rank_value):
        if len(run) < 5:
            continue
        for length in range(5, len(run) + 1):
            for start in range(0, len(run) - length + 1):
                slice_ranks = run[start : start + length]
                if not _is_custom_consecutive(slice_ranks, _extended_low_rank_value):
                    continue
                cards = [grouped[rank][0] for rank in slice_ranks]
                combos.append(
                    Combo(
                        "straight",
                        cards,
                        main_rank=max(slice_ranks, key=_extended_low_rank_value),
                        sequence_length=length,
                    )
                )
    return combos


def _split_custom_runs(ranks: list[int], value_fn) -> list[list[int]]:
    if not ranks:
        return []
    runs: list[list[int]] = []
    current = [ranks[0]]
    for rank in ranks[1:]:
        if value_fn(rank) == value_fn(current[-1]) + 1:
            current.append(rank)
        else:
            runs.append(current)
            current = [rank]
    runs.append(current)
    return runs


def _is_custom_consecutive(ranks: list[int], value_fn) -> bool:
    values = [value_fn(rank) for rank in ranks]
    return all(values[index] + 1 == values[index + 1] for index in range(len(values) - 1))


def _extended_low_rank_value(rank: int) -> int:
    if rank == 14:
        return 1
    if rank == 15:
        return 2
    return rank
