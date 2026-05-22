from __future__ import annotations

from ddz.models import Card, Player


LOCAL_RANKED_BASE_SCORE = 50
PVP_BASE_SCORE = 1


def spring_multipliers(
    players: list[Player],
    landlord_index: int | None,
    play_counts: list[int],
    winner: Player,
) -> tuple[int, int]:
    if landlord_index is None or not play_counts:
        return 0, 0
    landlord_played = play_counts[landlord_index]
    farmer_played = sum(
        count
        for index, count in enumerate(play_counts)
        if index != landlord_index
    )
    if winner.role == "landlord" and farmer_played == 0:
        return 1, 0
    if winner.role != "landlord" and landlord_played == 1:
        return 0, 1
    return 0, 0


def build_settlement(
    *,
    players: list[Player],
    winner: Player,
    landlord_index: int | None,
    base_score: int,
    highest_bid: int,
    effective_bid: int = 0,
    bombs_played: int = 0,
    redeal_count: int = 0,
    report_multiplier: int = 0,
    marked_card: Card | None = None,
    play_counts: list[int] | None = None,
    score_enabled: bool = True,
) -> dict:
    bid_value = max(1, effective_bid or highest_bid or 1)
    bomb_multiplier = bombs_played
    reveal_multiplier = 1 if any(player.revealed for player in players) else 0
    marker_multiplier = 1 if marked_card is not None and marked_card.rank >= 16 else 0
    spring_multiplier, reverse_spring_multiplier = spring_multipliers(
        players,
        landlord_index,
        play_counts or [0] * len(players),
        winner,
    )
    total_multiplier = (
        bomb_multiplier
        + reveal_multiplier
        + redeal_count
        + report_multiplier
        + marker_multiplier
        + spring_multiplier
        + reverse_spring_multiplier
    )
    factor = 2**total_multiplier
    total_score = base_score * bid_value * factor if score_enabled else 0
    return {
        "base_score": base_score,
        "bid_value": bid_value,
        "bomb_multiplier": bomb_multiplier,
        "reveal_multiplier": reveal_multiplier,
        "redeal_multiplier": redeal_count,
        "report_multiplier": report_multiplier,
        "marker_multiplier": marker_multiplier,
        "spring_multiplier": spring_multiplier,
        "reverse_spring_multiplier": reverse_spring_multiplier,
        "multiplier_factor": factor,
        "total_score": total_score,
        "winner_side": winner.role,
    }


def score_deltas(players: list[Player], settlement: dict, landlord_index: int | None, landlord_won: bool) -> dict[str, int]:
    if landlord_index is None:
        return {}
    total = int(settlement["total_score"])
    landlord_name = players[landlord_index].name
    farmer_count = len(players) - 1
    deltas: dict[str, int] = {}
    for player in players:
        if player.name == landlord_name:
            deltas[player.name] = farmer_count * total if landlord_won else -farmer_count * total
        else:
            deltas[player.name] = -total if landlord_won else total
    return deltas
