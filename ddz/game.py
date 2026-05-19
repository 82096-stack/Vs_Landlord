from __future__ import annotations

from collections import Counter
from typing import Iterable

from ddz.ai import RuleBasedAI, TableKnowledge
from ddz.models import Card, Player, format_cards
from ddz.rules import MODE_RULES, can_beat, choose_marked_card, deal_cards, identify_combo


def clear_screen() -> None:
    print("\033[2J\033[H", end="")


class GameSession:
    def __init__(
        self,
        mode: str,
        players: list[Player],
        match_type: str = "casual",
        god_view: bool = False,
    ) -> None:
        self.mode = mode
        self.rules = MODE_RULES[mode]
        self.players = players[:]
        self.ai = RuleBasedAI()
        self.knowledge = TableKnowledge(mode, len(self.players))
        self.match_type = match_type
        self.god_view = god_view
        self.bottom_cards: list[Card] = []
        self.marked_card: Card | None = None
        self.marker_holder_index: int | None = None
        self.landlord_index: int | None = None
        self.bombs_played = 0
        self.highest_bid = 0
        self.redeal_count = 0
        self.effective_bid = 0
        self.report_multiplier = 0
        self.play_counts = [0] * len(self.players)

    def run(self) -> dict:
        round_count = 1
        while True:
            self._reset_round_state()
            self.marked_card = choose_marked_card(self.mode)
            print(f"\n本轮标记牌号码: {self.marked_card.label}")
            self.bottom_cards = deal_cards(self.players, self.mode, self.marked_card)
            self.knowledge = TableKnowledge(self.mode, len(self.players))
            self.marker_holder_index = self._find_marker_holder(self.marked_card)
            marker_holder = self.players[self.marker_holder_index]
            print(f"标记牌被发给: {marker_holder.name}，由该玩家先选择叫地主。")
            print(f"\n第 {round_count} 次发牌完成，开始叫地主。")
            if self._bidding_phase():
                break
            self.redeal_count += 1
            round_count += 1

        winner = self._playing_phase()
        landlord = self.players[self.landlord_index] if self.landlord_index is not None else None
        landlord_won = winner.role == "landlord"
        settlement = self._build_settlement(winner)
        return {
            "winner_name": winner.name,
            "winner_role": winner.role,
            "landlord_name": landlord.name if landlord else "",
            "landlord_won": landlord_won,
            "bombs_played": self.bombs_played,
            "highest_bid": self.highest_bid,
            "redeal_count": self.redeal_count,
            "marked_card": self.marked_card.label if self.marked_card else "",
            "marker_holder_name": self.players[self.marker_holder_index].name if self.marker_holder_index is not None else "",
            "settlement": settlement,
        }

    def _bidding_phase(self) -> bool:
        if self.mode == "classic":
            return self._classic_bidding_phase()
        return self._extended_bidding_phase()

    def _extended_bidding_phase(self) -> bool:
        highest_bid = 0
        highest_index: int | None = None

        for index in self._turn_order_from(self._bidding_start_index()):
            player = self.players[index]
            if player.is_human:
                self._prepare_human_view(player, phase="叫地主")
                self._print_public_table()
                print(f"你的手牌: {self._render_hand(player.hand)}")
                bid = self._prompt_human_bid(player, highest_bid)
            else:
                bid = self.ai.choose_bid(player.hand, self.mode, highest_bid)
                print(f"{player.name} 选择叫分: {bid}")
            player.bid_score = bid
            player.bid_participated = True

            if bid > highest_bid:
                highest_bid = bid
                highest_index = index
            if highest_bid == 3:
                break

        if highest_index is None:
            print("本轮无人叫地主，重新发牌。")
            return False

        self.highest_bid = highest_bid
        self._assign_landlord(highest_index)
        return True

    def _classic_bidding_phase(self) -> bool:
        desires = [0] * len(self.players)
        call_order = self._turn_order_from(self._bidding_start_index())
        candidate_index: int | None = None
        remaining_rob_order: list[int] = []

        for order_position, index in enumerate(call_order):
            player = self.players[index]
            if player.is_human:
                self._prepare_human_view(player, phase="叫地主")
                self._print_public_table()
                print(f"你的手牌: {self._render_hand(player.hand)}")
                wants_landlord = self._prompt_human_call(player)
            else:
                wants_landlord = self.ai.choose_bid(player.hand, self.mode, 0) > 0
                print(f"{player.name} 选择{'叫地主' if wants_landlord else '不叫'}。")
            desire = self.ai.preview_bid_strength(player.hand, self.mode)
            player.bid_score = 1 if wants_landlord else 0
            player.bid_participated = True
            desires[index] = desire if wants_landlord else 0
            if wants_landlord:
                candidate_index = index
                remaining_rob_order = call_order[order_position + 1 :]
                break

        if candidate_index is None:
            print("本轮无人叫地主，重新发牌。")
            return False

        caller = self.players[candidate_index]
        caller_index = candidate_index
        last_robber_index: int | None = None
        self.highest_bid = max(1, self.players[candidate_index].bid_score)
        print(f"{caller.name} 先叫地主，进入抢地主环节。")

        for index in remaining_rob_order:
            player = self.players[index]
            if player.is_human:
                self._prepare_human_view(player, phase="抢地主")
                self._print_public_table()
                print(f"你的手牌: {self._render_hand(player.hand)}")
                rob = self._prompt_human_rob(player, self.highest_bid)
            else:
                desires[index] = self.ai.preview_bid_strength(player.hand, self.mode)
                rob = self.ai.choose_rob(player.hand, self.mode, desires[index], self.highest_bid)
                print(f"{player.name} 选择{'抢地主' if rob else '不抢'}。")
            player.bid_participated = True
            if rob:
                candidate_index = index
                last_robber_index = index
                self.players[index].bid_score = max(self.players[index].bid_score, 2)
                self.highest_bid = max(self.highest_bid, self.players[index].bid_score)

        if last_robber_index is not None:
            final_rob = self._ask_classic_final_rob(caller_index, desires[caller_index])
            if final_rob:
                candidate_index = caller_index
                self.players[caller_index].bid_score = max(self.players[caller_index].bid_score, 2)
                self.highest_bid = max(self.highest_bid, self.players[caller_index].bid_score)
            else:
                candidate_index = last_robber_index

        self.highest_bid = max(1, max(player.bid_score for player in self.players))
        self._assign_landlord(candidate_index)
        return True

    def _ask_classic_final_rob(self, caller_index: int, caller_desire: int) -> bool:
        caller = self.players[caller_index]
        print(f"有人抢地主，回到 {caller.name} 决定是否抢回地主。")
        if caller.is_human:
            self._prepare_human_view(caller, phase="抢地主")
            self._print_public_table()
            print(f"你的手牌: {self._render_hand(caller.hand)}")
            final_rob = self._prompt_human_rob(caller, self.highest_bid)
        else:
            final_rob = self.ai.choose_rob(caller.hand, self.mode, caller_desire, self.highest_bid)
            print(f"{caller.name} 选择{'抢地主' if final_rob else '不抢'}。")
        return final_rob

    def _assign_landlord(self, highest_index: int) -> None:
        self.landlord_index = highest_index
        landlord = self.players[highest_index]
        landlord.role = "landlord"
        landlord.hand.extend(self.bottom_cards)
        landlord.sort_hand()
        self.effective_bid = max(1, self.highest_bid)
        self.knowledge.set_landlord(highest_index, self.bottom_cards)
        print(f"{landlord.name} 成为地主，获得底牌: {format_cards(self.bottom_cards)}")
        self._reveal_phase(landlord)
        self._report_phase()

    def _playing_phase(self) -> Player:
        assert self.landlord_index is not None

        current_index = self.landlord_index
        last_combo = None
        last_player_index: int | None = None

        while True:
            if last_combo is not None and current_index == last_player_index:
                print(f"\n其余玩家全部过牌，由 {self.players[current_index].name} 重新领出。")
                last_combo = None

            player = self.players[current_index]
            opened_round = last_combo is None
            if player.is_human:
                self._prepare_human_view(player, phase="出牌")
                selected = self._prompt_human_play(player, last_combo, last_player_index)
            else:
                if self.god_view:
                    print(f"\n[上帝视角] {player.name} 当前手牌: {format_cards(player.hand)}")
                selected = self.ai.choose_play(
                    player,
                    self.players,
                    last_combo,
                    last_player_index,
                    self.knowledge,
                )

            if selected is None:
                print(f"{player.name} 选择过牌。")
            else:
                combo = identify_combo(selected, self.mode)
                if combo is None:
                    raise RuntimeError(f"AI 出了非法牌: {format_cards(selected)}")
                if not self._is_combo_allowed(player, combo):
                    raise RuntimeError(f"{player.name} 选择了当前规则不允许的牌: {format_cards(selected)}")
                self._remove_cards(player, selected)
                if combo.kind in {"bomb", "rocket"}:
                    self.bombs_played += 1
                    player.bombs_used += 1
                self.play_counts[current_index] += 1
                self.knowledge.record_play(current_index, combo, opened_round)
                last_combo = combo
                last_player_index = current_index
                print(
                    f"{player.name} 出牌: {combo.describe()} -> {format_cards(selected)}"
                    f" | 剩余 {player.hand_size} 张"
                )
                if not player.hand:
                    self._print_match_result(player)
                    return player

            current_index = (current_index + 1) % len(self.players)

    def _prompt_human_bid(self, player: Player, highest_bid: int) -> int:
        allowed = [0] + list(range(highest_bid + 1, 4))
        while True:
            raw = input(f"{player.name} 请输入叫分 {allowed}，或输入 h 查看建议: ").strip().lower()
            if raw in {"h", "hint"}:
                suggestion = self.ai.choose_bid(player.hand, self.mode, highest_bid)
                print(f"AI 建议叫分: {suggestion}")
                continue
            if not raw.isdigit():
                print("请输入数字。")
                continue
            bid = int(raw)
            if bid in allowed:
                return bid
            print("叫分不合法，请重新输入。")

    def _prompt_human_call(self, player: Player) -> bool:
        while True:
            raw = input(f"{player.name} 请输入 y 叫地主 / n 不叫，或输入 h 查看建议: ").strip().lower()
            if raw in {"h", "hint"}:
                suggestion = self.ai.preview_bid_strength(player.hand, self.mode) > 0
                print(f"AI 建议: {'叫地主' if suggestion else '不叫'}")
                continue
            if raw in {"y", "yes"}:
                return True
            if raw in {"n", "no"}:
                return False
            print("请输入 y、n 或 h。")

    def _prompt_human_rob(self, player: Player, current_bid: int) -> bool:
        while True:
            raw = input(f"{player.name} 请输入 y 抢地主 / n 不抢，或输入 h 查看建议: ").strip().lower()
            if raw in {"h", "hint"}:
                rob = self.ai.choose_rob(
                    player.hand,
                    self.mode,
                    self.ai.preview_bid_strength(player.hand, self.mode),
                    current_bid,
                )
                print(f"AI 建议: {'抢地主' if rob else '不抢'}")
                continue
            if raw in {"y", "yes"}:
                return True
            if raw in {"n", "no"}:
                return False
            print("请输入 y、n 或 h。")

    def _prompt_human_play(
        self,
        player: Player,
        last_combo,
        last_player_index: int | None,
    ) -> list[Card] | None:
        while True:
            self._print_public_table()
            if last_combo is None:
                print("当前为自由出牌。")
            else:
                last_player = self.players[last_player_index] if last_player_index is not None else None
                owner = last_player.name if last_player else "未知"
                print(f"当前桌面牌型: {owner} 的 {last_combo.describe()}")
                print(f"桌面牌面: {format_cards(last_combo.cards)}")

            print(f"\n你的手牌:\n{self._render_hand(player.hand)}")
            raw = input(
                "\n输入要出的牌序号（空格分隔），输入 p 过牌，或输入 h 查看建议: "
            ).strip().lower()
            if raw in {"p", "pass"}:
                if last_combo is None:
                    print("当前由你领出，不能过牌。")
                    continue
                return None
            if raw in {"h", "hint"}:
                suggestion = self.ai.choose_play(
                    player,
                    self.players,
                    last_combo,
                    last_player_index,
                    self.knowledge,
                )
                if suggestion is None:
                    print("AI 建议: 过牌。")
                else:
                    combo = identify_combo(suggestion, self.mode)
                    if combo is None:
                        print("AI 暂时没有可用建议。")
                    else:
                        print(f"AI 建议: {combo.describe()} -> {format_cards(suggestion)}")
                continue

            chosen_indexes = self._parse_indexes(raw, len(player.hand))
            if not chosen_indexes:
                print("输入格式不正确。")
                continue

            selected = [player.hand[index - 1] for index in chosen_indexes]
            combo = identify_combo(selected, self.mode)
            if combo is None:
                print("这组牌不是合法牌型，请重新选择。")
                continue
            if not self._is_combo_allowed(player, combo):
                print("当前规则下，这手炸弹/王炸次数已经用完。")
                continue
            if last_combo is not None and not can_beat(combo, last_combo):
                print("这组牌压不过当前牌型，请重新选择。")
                continue
            return selected

    def _prepare_human_view(self, player: Player, phase: str) -> None:
        if len([member for member in self.players if member.is_human]) > 1:
            input(f"\n请将屏幕交给 {player.name}，按回车开始{phase}...")
        clear_screen()

    def _print_public_table(self) -> None:
        print(f"模式: {self.rules['label']} | {'积分赛' if self.match_type == 'ranked' else '娱乐赛'}")
        for index, player in enumerate(self.players, start=1):
            role_label = "地主" if player.role == "landlord" else "农民"
            source = "本地" if player.is_human else "AI"
            tags: list[str] = []
            if player.bid_score:
                tags.append(f"叫分/身份:{player.bid_score}")
            if player.revealed:
                tags.append("已摊打")
                if self.mode == "extended" and player.role == "landlord":
                    tags.append("头撩")
            if player.announced:
                tags.append("双报道" if player.report_level >= 2 else "报道")
            if self.mode == "extended" and self.landlord_index is not None and player.role != "landlord":
                tags.append(f"已炸:{player.bombs_used}/{self._bomb_limit_label(player)}")
            suffix = f" | {' '.join(tags)}" if tags else ""
            print(f"{index}. {player.name} [{source}/{role_label}] 剩余 {player.hand_size} 张{suffix}")

    def _print_match_result(self, winner: Player) -> None:
        landlord_name = self.players[self.landlord_index].name if self.landlord_index is not None else "未知"
        print("\n对局结束。")
        if winner.role == "landlord":
            print(f"地主 {winner.name} 获胜。")
        else:
            print(f"农民阵营获胜，首个出完的是 {winner.name}。")
        print(f"地主: {landlord_name}")
        print("最终手牌:")
        for player in self.players:
            remaining = format_cards(player.hand) if player.hand else "已出完"
            print(f"- {player.name}: {remaining}")

    @staticmethod
    def _remove_cards(player: Player, selected: Iterable[Card]) -> None:
        selected_serials = {card.serial for card in selected}
        player.hand = [card for card in player.hand if card.serial not in selected_serials]
        player.sort_hand()

    def _reset_round_state(self) -> None:
        self.landlord_index = None
        self.marked_card = None
        self.marker_holder_index = None
        self.highest_bid = 0
        self.bombs_played = 0
        self.effective_bid = 0
        self.report_multiplier = 0
        self.play_counts = [0] * len(self.players)
        for player in self.players:
            player.bid_score = 0
            player.bid_participated = False
            player.bombs_used = 0
            player.revealed = False
            player.announced = False
            player.report_level = 0

    def _reveal_phase(self, landlord: Player) -> None:
        if landlord.is_human:
            landlord.revealed = self._prompt_human_reveal(landlord)
        else:
            landlord.revealed = self.ai.choose_reveal(landlord.hand, self.mode, landlord.role)
        if landlord.revealed:
            if self.mode == "extended":
                self.effective_bid = max(self.effective_bid, 4)
            print(f"{landlord.name} 选择摊打，亮出手牌: {format_cards(landlord.hand)}")

    def _report_phase(self) -> None:
        if self.mode != "extended":
            return

        for player in self.players:
            report_level = self._detect_report_level(player.hand)
            if report_level <= 0:
                continue

            if player.is_human:
                announce = self._prompt_human_report(player, report_level)
            else:
                announce = self.ai.choose_report(player.hand, self.mode, player.role, report_level)
            if not announce:
                continue

            player.announced = True
            player.report_level = report_level
            self.report_multiplier += report_level
            label = "双报道" if report_level >= 2 else "报道"
            print(f"{player.name} 选择{label}。")

    @staticmethod
    def _detect_report_level(hand: list[Card]) -> int:
        counts = Counter(card.rank for card in hand)
        if any(count >= 8 for count in counts.values()):
            return 2
        if counts.get(16, 0) >= 1 and counts.get(17, 0) >= 1:
            return 1
        if any(count >= 7 for count in counts.values()):
            return 1
        return 0

    def _prompt_human_reveal(self, player: Player) -> bool:
        while True:
            raw = input(f"{player.name} 是否选择摊打（明牌）？输入 y / n，或 h 查看建议: ").strip().lower()
            if raw in {"h", "hint"}:
                reveal = self.ai.choose_reveal(player.hand, self.mode, player.role)
                print(f"AI 建议: {'摊打' if reveal else '不摊打'}")
                continue
            if raw in {"y", "yes"}:
                return True
            if raw in {"n", "no"}:
                return False
            print("请输入 y、n 或 h。")

    def _prompt_human_report(self, player: Player, report_level: int) -> bool:
        label = "双报道" if report_level >= 2 else "报道"
        while True:
            raw = input(f"{player.name} 当前可{label}，输入 y / n，或 h 查看建议: ").strip().lower()
            if raw in {"h", "hint"}:
                announce = self.ai.choose_report(player.hand, self.mode, player.role, report_level)
                print(f"AI 建议: {'选择' + label if announce else '不选择'}")
                continue
            if raw in {"y", "yes"}:
                return True
            if raw in {"n", "no"}:
                return False
            print("请输入 y、n 或 h。")

    def _is_combo_allowed(self, player: Player, combo) -> bool:
        if self.mode != "extended":
            return True
        if combo.kind not in {"bomb", "rocket"}:
            return True
        if player.role == "landlord":
            return True
        if not player.bid_participated:
            return True
        return player.bombs_used < self._bomb_limit(player)

    def _bomb_limit(self, player: Player) -> int:
        if player.role == "landlord":
            return 99
        if not player.bid_participated:
            return 99
        if player.bid_score >= 2:
            return 2
        return 1

    def _bomb_limit_label(self, player: Player) -> str:
        limit = self._bomb_limit(player)
        if limit >= 99:
            return "不限"
        return str(limit)

    def _build_settlement(self, winner: Player) -> dict:
        base_score = 50
        bid_value = max(1, self.effective_bid or self.highest_bid or 1)
        bomb_multiplier = self.bombs_played
        reveal_multiplier = 1 if any(player.revealed for player in self.players) else 0
        redeal_multiplier = self.redeal_count
        report_multiplier = self.report_multiplier
        marker_multiplier = 1 if self.marked_card is not None and self.marked_card.rank >= 16 else 0
        spring_multiplier, reverse_spring_multiplier = self._spring_multipliers(winner)
        total_multiplier = (
            bomb_multiplier
            + reveal_multiplier
            + redeal_multiplier
            + report_multiplier
            + marker_multiplier
            + spring_multiplier
            + reverse_spring_multiplier
        )
        factor = 2**total_multiplier
        ranked_total = base_score * bid_value * factor
        total = 0 if self.match_type == "casual" else ranked_total
        return {
            "base_score": base_score,
            "bid_value": bid_value,
            "bomb_multiplier": bomb_multiplier,
            "reveal_multiplier": reveal_multiplier,
            "redeal_multiplier": redeal_multiplier,
            "report_multiplier": report_multiplier,
            "marker_multiplier": marker_multiplier,
            "spring_multiplier": spring_multiplier,
            "reverse_spring_multiplier": reverse_spring_multiplier,
            "multiplier_factor": factor,
            "total_score": total,
            "winner_side": winner.role,
        }

    def _spring_multipliers(self, winner: Player) -> tuple[int, int]:
        if self.landlord_index is None or not self.play_counts:
            return 0, 0

        landlord_played = self.play_counts[self.landlord_index]
        farmer_played = sum(
            count
            for index, count in enumerate(self.play_counts)
            if index != self.landlord_index
        )
        if winner.role == "landlord" and farmer_played == 0:
            return 1, 0
        if winner.role != "landlord" and landlord_played == 1:
            return 0, 1
        return 0, 0

    def _bidding_start_index(self) -> int:
        if self.marker_holder_index is None:
            return 0
        return self.marker_holder_index

    def _turn_order_from(self, start_index: int) -> list[int]:
        return [(start_index + offset) % len(self.players) for offset in range(len(self.players))]

    def _find_marker_holder(self, marked_card: Card) -> int:
        for index, player in enumerate(self.players):
            if any(card.serial == marked_card.serial for card in player.hand):
                return index
        raise RuntimeError("标记牌没有发到任何玩家手中。")

    @staticmethod
    def _parse_indexes(raw: str, hand_size: int) -> list[int]:
        try:
            indexes = [int(part) for part in raw.split()]
        except ValueError:
            return []
        if not indexes or len(indexes) != len(set(indexes)):
            return []
        if any(index < 1 or index > hand_size for index in indexes):
            return []
        return sorted(indexes)

    @staticmethod
    def _render_hand(hand: list[Card]) -> str:
        parts = [f"{index:>2}:{card.label}" for index, card in enumerate(hand, start=1)]
        lines = ["  ".join(parts[start : start + 8]) for start in range(0, len(parts), 8)]
        return "\n".join(lines)
