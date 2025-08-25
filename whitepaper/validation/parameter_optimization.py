"""Parameter optimization utilities for SEP validation tests.

This module provides a simple grid search interface to explore
parameter combinations for a given test function. Each test function
should accept keyword arguments for the parameters being optimized and
return a numeric score where lower is better.
"""

from __future__ import annotations

import itertools
from typing import Callable, Dict, Iterable, Tuple, Any


def optimize_parameters(
    test_func: Callable[..., float],
    param_ranges: Dict[str, Iterable[Any]],
) -> Tuple[Dict[str, Any], float]:
    """Run a grid search over ``param_ranges`` using ``test_func``.

    Parameters
    ----------
    test_func:
        Function evaluating a parameter combination. It should return a
        numeric score where lower values indicate better performance.
    param_ranges:
        Mapping of parameter names to iterables of candidate values.

    Returns
    -------
    best_params, best_score:
        The best parameter combination and the associated score.
    """
    best_score = float("inf")
    best_params: Dict[str, Any] | None = None

    keys = list(param_ranges.keys())
    for combo in itertools.product(*(param_ranges[k] for k in keys)):
        params = dict(zip(keys, combo))
        score = test_func(**params)
        if score < best_score:
            best_score = score
            best_params = params

    return best_params or {}, best_score
