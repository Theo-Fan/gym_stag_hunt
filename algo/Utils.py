from typing import Any

import numpy as np


def _is_int_pair_list(value) -> bool:
    if not isinstance(value, (list, tuple)):
        return False
    if len(value) != 2:
        return False
    return all(isinstance(v, (int, np.integer)) for v in value)


def convert_coords_to_tuples(obj: Any) -> Any:
    """
    - np.array([x, y])             -> (x, y)
    - [x, y]                        -> (x, y)
    - [np.array([x, y]), ...]       -> [(x, y), ...]
    """
    if isinstance(obj, np.ndarray):
        if obj.ndim == 1 and obj.size == 2 and obj.dtype.kind in ("i", "u"):
            return int(obj[0]), int(obj[1])
        obj = obj.tolist()
        return convert_coords_to_tuples(obj)

    if isinstance(obj, dict):
        return {k: convert_coords_to_tuples(v) for k, v in obj.items()}

    if isinstance(obj, (list, tuple)):
        if _is_int_pair_list(obj):
            return int(obj[0]), int(obj[1])
        converted = [convert_coords_to_tuples(v) for v in obj]
        return type(obj)(converted)

    return obj


######### print

import numbers
from typing import Any
import numpy as np

try:
    import torch

    _HAS_TORCH = True
except Exception:
    _HAS_TORCH = False


def pretty_print_dict(d: dict, indent: int = 0) -> None:
    """
    Pretty prints a nested dictionary with indentation.
    Args:
        d (dict): The dictionary to print.
        indent (int): Current indentation level.
    """
    for key, value in d.items():
        if isinstance(value, dict):
            print(" " * indent + f"{key}:")
            pretty_print_dict(value, indent + 4)
        else:
            print(" " * indent + f"{key}: {value}")


def pretty_print_sorted(d: dict, indent: int = 0) -> None:
    """
    Pretty prints a dictionary with sorted keys and handles nested dictionaries.
    Args:
        d (dict): The dictionary to print.
        indent (int): Current indentation level.
    """
    indent_str = " " * indent
    for key in sorted(d.keys()):
        val = d[key]
        if isinstance(val, dict):
            inner = ", ".join(f"'{k}': {val[k]}" for k in sorted(val.keys()))
            print(f"{indent_str}{key}: {{{inner}}},")
        else:
            print(f"{indent_str}{key}: {val},")


label_w = 26
value_w = 8
pad = "  "
precision = 2


def _is_scalar_number(x: Any) -> bool:
    if isinstance(x, numbers.Number):
        return True
    if _HAS_TORCH and isinstance(x, torch.Tensor) and x.ndim == 0:
        return True
    if isinstance(x, np.ndarray) and x.ndim == 0:
        return True
    return False


def _to_float(x: Any) -> float:
    if _HAS_TORCH and isinstance(x, torch.Tensor):
        return float(x.item() if x.ndim == 0 else x)
    if isinstance(x, np.generic):
        return float(x.item())
    return float(x)


def fmt_value(v: Any, width: int = value_w, prec: int = precision) -> str:
    """
    将 v 转为字符串：
      - 数值标量 -> 保留 prec 位小数
      - 列表/元组/ndarray/tensor -> 元素逐个格式化后用逗号拼接
      - 其他类型 -> str(v)
    最后按 width 做右对齐（width<=0 则不对齐）
    """
    try:
        if _is_scalar_number(v):
            s = f"{_to_float(v):.{prec}f}"
        elif _HAS_TORCH and isinstance(v, torch.Tensor):
            # 非标量 tensor
            s = "[" + ", ".join(f"{_to_float(x):.{prec}f}" for x in v.flatten().tolist()) + "]"
        elif isinstance(v, (list, tuple)):
            s = "[" + ", ".join(
                f"{x}" if isinstance(x, numbers.Number) else str(x)
                for x in v
            ) + "]"
        elif isinstance(v, np.ndarray):
            s = "[" + ", ".join(
                f"{x}" for x in v.flatten()
            ) + "]"
        else:
            s = str(v)
    except Exception:
        # 不可转为 float 的情况，退回到原样字符串
        s = str(v)

    return f"{s:>{width}}" if width and width > 0 else s


def row2(label, v0, v1):
    return f"\t{label:<{label_w}}Agent_0: {fmt_value(v0)}{pad * 3}|{pad * 3}Agent_1: {fmt_value(v1)}"


def row1(label: str, v: Any) -> str:
    return f"\t{label:<{label_w}}{fmt_value(v)}"
