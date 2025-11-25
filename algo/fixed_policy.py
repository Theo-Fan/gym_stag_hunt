from typing import Dict, List, Tuple

import numpy as np

LEFT = 0
DOWN = 1
RIGHT = 2
UP = 3
STAND = 4

"""
agent_0: blue
agent_1: red orenge

(x, y):
    x: left -> right
    y: top -> down

(x1, y1) <=> (x2, y2):
    left: x1 < x2
    right: x1 > x2
    up: y1 < y2
    down: y1 > y2
"""


def choice_action(
    agent_policy: str, agent_id: int,
    pos_info: Dict[str, List[Tuple[int, int]]], opp_last: int = 0,
) -> int:
    opp_agent_pos = pos_info['entity_positions'][f"agent_{agent_id}"]
    stag_pos = pos_info['entity_positions']["stag"]
    plants_pos = pos_info['entity_positions']["plants"]

    if agent_policy == "allc":
        return allc_policy(opp_agent_pos, stag_pos)
    elif agent_policy == "alld":
        return alld_policy(opp_agent_pos, plants_pos)
    elif agent_policy == "tft":
        return tft_policy(opp_agent_pos, stag_pos, plants_pos, opp_last)
    elif agent_policy == "random":
        return random_policy(opp_agent_pos, stag_pos, plants_pos)
    else:
        raise ValueError(f"Unknown policy: {agent_policy}")


def manhattan_distance(pos1: Tuple[int, int], pos2: Tuple[int, int]) -> int:
    return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])


def allc_policy(agent_pos: Tuple[int, int], stag_pos: Tuple[int, int]) -> int:
    if agent_pos[0] < stag_pos[0]:
        return RIGHT
    elif agent_pos[0] > stag_pos[0]:
        return LEFT
    elif agent_pos[1] < stag_pos[1]:
        return DOWN
    elif agent_pos[1] > stag_pos[1]:
        return UP
    else:
        return STAND


def alld_policy(agent_pos: Tuple[int, int], plants_pos: List[Tuple[int, int]]) -> int:
    # find the nearest plant
    nearest_plant = min(plants_pos, key=lambda p: manhattan_distance(agent_pos, p))

    if agent_pos[0] < nearest_plant[0]:
        return RIGHT
    elif agent_pos[0] > nearest_plant[0]:
        return LEFT
    elif agent_pos[1] < nearest_plant[1]:
        return DOWN
    elif agent_pos[1] > nearest_plant[1]:
        return UP
    else:
        return STAND


def random_policy(agent_pos: Tuple[int, int],
                  stag_pos: Tuple[int, int], plants_pos: List[Tuple[int, int]]) -> int:
    """
    Random policy: randomly choose policy (cooperate / defect)

    Return:
        - t = 0 -> C
        - t = 1 -> D
    """
    t = np.random.randint(0, 2)
    return allc_policy(agent_pos, stag_pos) if t == 0 else alld_policy(agent_pos, plants_pos)


def tft_policy(agent_pos: Tuple[int, int], stag_pos: Tuple[int, int],
               plants_pos: List[Tuple[int, int]], opp_last: int) -> int:
    """
    TFT: agent_id will take opponment last action

    Args:
        - agent_id: int, an index indicating current agent number
        - agent_pos: Tuple[int, int], each agent position
        - stag_pos: Tuple, stag position
        - plants_pos: List, plants positions
        - opp_last: int, opponent last action
        
    """
    return allc_policy(agent_pos, stag_pos) if opp_last == 0 else alld_policy(agent_pos, plants_pos)
