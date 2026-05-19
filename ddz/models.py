from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


RANK_LABELS = {
    3: "3",
    4: "4",
    5: "5",
    6: "6",
    7: "7",
    8: "8",
    9: "9",
    10: "10",
    11: "J",
    12: "Q",
    13: "K",
    14: "A",
    15: "2",
    16: "小王",
    17: "大王",
}

SUIT_ORDER = {"C": 0, "D": 1, "H": 2, "S": 3, "J": 4}


@dataclass(frozen=True)
class Card:
    rank: int
    suit: str
    deck_id: int
    serial: int

    @property
    def label(self) -> str:
        return RANK_LABELS[self.rank]


@dataclass
class Combo:
    kind: str
    cards: list[Card]
    main_rank: int
    sequence_length: int = 1
    total_cards: int = 0
    bomb_size: int = 0

    def __post_init__(self) -> None:
        if not self.total_cards:
            self.total_cards = len(self.cards)
        if self.kind == "bomb" and not self.bomb_size:
            self.bomb_size = len(self.cards)

    def pattern_key(self) -> tuple:
        if self.kind == "bomb":
            return (self.kind, self.bomb_size)
        return (self.kind, self.sequence_length, self.total_cards)

    @property
    def display_name(self) -> str:
        names = {
            "single": "单张",
            "pair": "对子",
            "trio": "三张",
            "trio_single": "三带一",
            "trio_pair": "三带二",
            "straight": "顺子",
            "pair_straight": "连对",
            "trio_straight": "飞机不带",
            "airplane_single": "飞机带单",
            "airplane_pair": "飞机带对",
            "four_two_single": "四带二",
            "four_two_pair": "四带两对",
            "bomb": "炸弹",
            "rocket": "王炸",
        }
        return names.get(self.kind, self.kind)

    def describe(self) -> str:
        if self.kind == "bomb":
            return f"{self.display_name}({self.bomb_size}张)"
        if self.sequence_length > 1:
            return f"{self.display_name}({self.sequence_length}组)"
        return self.display_name


@dataclass
class Player:
    name: str
    is_human: bool
    account_username: Optional[str] = None
    hand: list[Card] = field(default_factory=list)
    role: str = "farmer"
    bid_score: int = 0
    bid_participated: bool = False
    bombs_used: int = 0
    revealed: bool = False
    announced: bool = False
    report_level: int = 0

    def sort_hand(self) -> None:
        self.hand.sort(key=card_sort_key)

    @property
    def hand_size(self) -> int:
        return len(self.hand)


def card_sort_key(card: Card) -> tuple[int, int, int, int]:
    return (card.rank, SUIT_ORDER.get(card.suit, 99), card.deck_id, card.serial)


def format_cards(cards: list[Card]) -> str:
    return "，".join(card.label for card in sorted(cards, key=card_sort_key))
