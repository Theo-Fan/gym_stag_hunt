import math
from typing import List, Tuple, Optional, Dict

from algo.fixed_policy import manhattan_distance


def manhattan(p: Tuple[int, int], q: Tuple[int, int]) -> int:
    return abs(p[0] - q[0]) + abs(p[1] - q[1])


def infer_strategy_coins_detail(
    trajectory: List[Tuple[int, int]],
    agent_color: str,
    coin_pos: Tuple[int, int],
    coin_color: str,
    agent_history: List[int] = None,
    prev_mode: Optional[int] = None,
    window: int = 4,
    tol: float = 3,
    max_then_dist: int = 5,
    verbose: bool = False,
) -> int:
    n = len(trajectory)
    same_color = (str(agent_color).lower() == str(coin_color).lower())

    if n < 2:
        return prev_mode if prev_mode is not None else 0

    if verbose:
        print(f"trajectory: {trajectory}")
        print(f"coin_pos: {coin_pos}, coin_color: {coin_color}, agent_color: {agent_color}")

    if (not same_color) and trajectory[-1] == coin_pos:
        return 1

    if trajectory[-1] != coin_pos:
        unpicker_dist = manhattan(trajectory[-1], coin_pos)
        if unpicker_dist >= max_then_dist and (not same_color):
            return 0

    start = max(1, n - window)
    delta_distance = 0.0
    for i in range(start, n):
        p_prev, p = trajectory[i - 1], trajectory[i]
        delta_distance += manhattan(p_prev, coin_pos) - manhattan(p, coin_pos)

    if same_color and delta_distance > tol: return 0  #
    if same_color and delta_distance < -tol: return 1
    if (not same_color) and delta_distance > tol: return 1
    if (not same_color) and delta_distance < -tol: return 0

    return prev_mode if prev_mode is not None else 0


def sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def global_strategy_infer_coins(
    trajectory: List[Tuple[int, int]],
    agent_color: str,
    coin_pos: Tuple[int, int],
    coin_color: str,
    agent_history: List[int] = None,
    eta: float = 0.1,
    max_then_dist: int = 4,
    verbose: bool = False,
) -> int:
    n = len(trajectory)
    same_color = (str(agent_color).lower() == str(coin_color).lower())
    last_distance = manhattan(trajectory[0], coin_pos)

    if n < 2:
        return 0

    S_mot = 0.0  # motivation score
    for idx, item in enumerate(trajectory):
        curr_distance = manhattan(item, coin_pos)

        if last_distance == curr_distance:
            S_mot += 0.0

        elif last_distance > curr_distance:
            S_mot += 1.0 if same_color else -1.0

        else:  # last_distance < curr_distance
            S_mot += -1.0 if same_color else 1.0

        last_distance = curr_distance

    if manhattan(trajectory[-1], coin_pos) > max_then_dist:
        S_mot = 0.0

    S_mot = round(S_mot / n, 2)  # normalize by length

    S_res = 0.0  # result score
    if trajectory[-1] == coin_pos:
        S_res += 2.0 if same_color else -2.0
    else:
        S_res += 0.0

    S_total = S_mot + S_res
    prob = round(sigmoid(S_total), 2)

    if verbose:
        print(f"trajectory: {trajectory}")
        print(f"coin_pos: {coin_pos}, coin_color: {coin_color}, agent_color: {agent_color}")
        print(f"S_mot: {S_mot}, S_res: {S_res}, S_total: {S_total}, prob: {prob}")

    if agent_history is not None:
        if prob > 0.5 + eta:
            return 0
        elif prob < 0.5 - eta:
            return 1

        if len(agent_history) < 5:
            return 0

        return 1 if sum(agent_history) > (len(agent_history) / 2) else 0

    else:
        return 0 if prob > 0.5 else 1


def global_strategy_infer_coop_mining(
    agent_id: int,
    trajectory: List[Tuple[int, int]],
    ores_pos: Dict,
    rewards: Dict,
    agent_history: List[int] = None,
    eta: float = 0.1,
    verbose: bool = False,
) -> int:
    n = len(trajectory)
    gold_list: List[Tuple[int, int]] = ores_pos.get("gold_ore", [])
    iron_list: List[Tuple[int, int]] = ores_pos.get("iron_ore", [])

    last_gold_distance = manhattan(trajectory[0], gold_list[0])
    last_iron_distance = manhattan(trajectory[0], iron_list[0])

    if n < 2:
        return 0

    S_mot = 0.0  # motivation score
    for idx, item in enumerate(trajectory):
        curr_gold_distance = manhattan(item, gold_list[0])
        curr_iron_distance = manhattan(item, iron_list[0])

        if last_gold_distance > curr_gold_distance:
            S_mot += 1.0

        elif last_gold_distance < curr_gold_distance:
            S_mot += -1.0

        if last_iron_distance > curr_iron_distance:
            S_mot += -1.0
        elif last_iron_distance < curr_iron_distance:  # far
            S_mot += 1.0

        last_gold_distance = curr_gold_distance
        last_iron_distance = curr_iron_distance

    S_mot = round(S_mot / n, 2)  # normalize by length

    S_res = 0.0  # result score
    if trajectory[-1] == gold_list[0]:
        S_res += 2.0
    elif trajectory[-1] == iron_list[0]:
        S_res -= 2.0

    if rewards["agent_0"] == rewards["agent_1"] and rewards["agent_0"] > 0:
        S_res += 2.0

    if rewards[f"agent_{agent_id}"] > rewards[f"agent_{1 - agent_id}"]:
        S_res -= 2.0

    S_total = S_mot + S_res
    prob = round(sigmoid(S_total), 2)

    if verbose:
        print(f"trajectory: {trajectory}")
        print(f"rewards: {rewards}")
        print(f"ores pos: {ores_pos}")
        print(f"S_mot: {S_mot}, S_res: {S_res}, S_total: {S_total}, prob: {prob}\n")

    if agent_history is not None:
        if prob > 0.5 + eta:
            return 0
        elif prob < 0.5 - eta:
            return 1

        if len(agent_history) < 5:
            return 0

        return 1 if sum(agent_history) > (len(agent_history) / 2) else 0

    else:
        return 0 if prob > 0.5 else 1


def global_strategy_infer_stag_hunt(
    agent_id: int,
    trajectory: List[Tuple[int, int]],
    stag_pos: Tuple[int, int],
    plants_pos: List[Tuple[int, int]],
    rewards: Dict,
    agent_history: List[int] = None,
    eta: float = 0.1,
    verbose: bool = False,
) -> int:
    n = len(trajectory)

    if n < 2:
        return 0

    def min_plants_distance(pos: Tuple[int, int]) -> float:
        if not plants_pos:
            return float("inf")
        return min(manhattan(pos, p) for p in plants_pos)

    S_mot = 0.0  # motivation score

    last_stag_distance = manhattan(trajectory[0], stag_pos)
    last_plants_distance = min_plants_distance(trajectory[0])

    for pos in trajectory[1:]:
        curr_stag_distance = manhattan(pos, stag_pos)
        curr_plants_distance = min_plants_distance(pos)

        if curr_stag_distance < last_stag_distance:
            S_mot += 1.0
        elif curr_stag_distance > last_stag_distance:
            S_mot -= 1.0

        if curr_plants_distance < last_plants_distance:
            S_mot -= 1.0
        elif curr_plants_distance > last_plants_distance:
            S_mot += 1.0

        last_stag_distance = curr_stag_distance
        last_plants_distance = curr_plants_distance
    S_mot = round(S_mot / (n - 1), 2)

    S_res = 0.0  # result score
    if trajectory[-1] == stag_pos:
        S_res += 5.0
    elif trajectory[-1] in plants_pos:
        S_res -= 5.0

    if rewards[0] == rewards[1] and rewards[0] > 3:
        S_res += 5.0

    if (rewards[agent_id] > rewards[1 - agent_id]) or rewards[agent_id] == 1:
        S_res -= 5.0

    S_total = S_mot + S_res
    prob = round(sigmoid(S_total), 2)

    if verbose:
        print(f"id: {agent_id}")
        print(f"\ttrajectory: {trajectory}")
        print(f"\trewards: {rewards}")
        print(f"\tstag_pos: {stag_pos}")
        print(f"\tplants_pos: {plants_pos}")
        print(f"\tS_mot: {S_mot}, S_res: {S_res}, S_total: {S_total}, prob: {prob}\n")

    if agent_history is not None:
        if prob > 0.5 + eta:
            return 0
        elif prob < 0.5 - eta:
            return 1

        if len(agent_history) < 5:
            return 0

        return 1 if sum(agent_history) > (len(agent_history) / 2) else 0

    else:
        return 0 if prob > 0.5 else 1
