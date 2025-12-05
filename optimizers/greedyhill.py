from itertools import combinations
from typing import Any
from dataclasses import dataclass
import numpy as np
import random
import heapq

from model import _calculate_swap_delta
from optim import OptimizerLogger
from optimizers import helper

@dataclass
class GreedyHillParams:
    hill_climbing_iterations: int = 10
    pos_swaps_per_step: int = 4
    col_swaps_per_step: int = 4
    greedy_columns: bool = True
    tolerance: float = 0.0001


def improve_batch_worker(args):
    return improve_batch(*args)

def improve_batch(
    char_at_pos_list: list[tuple[int, ...]],
    initial_score_list: list[float],
    order_1: tuple[tuple[np.ndarray, np.ndarray], ...],
    order_2: tuple[tuple[np.ndarray, np.ndarray], ...],
    order_3: tuple[tuple[np.ndarray, np.ndarray], ...],
    swap_position_pairs: tuple[tuple[int, int], ...],
    pis_at_column: tuple[tuple[int, ...], ...],
    progress_queue: Any,
    population_size: int,
    logger: OptimizerLogger,
    solver_args: dict
) -> dict[tuple[int, ...], float]:
    '''
    optimize the layout using a greedy hill climbing algorithm: checks all possible swaps in random order, and immediately
    accepts any swap that improves the score
    '''
    logger.batch_start()

    params = GreedyHillParams()
    for key, value in solver_args.items():
        if hasattr(params, key):
            setattr(params, key, value)

    selected_population = {}

    for seed_id, char_at_pos, initial_score in zip(range(len(char_at_pos_list)), char_at_pos_list, initial_score_list):
               
        population = helper.improve_layout(
            seed_id,
            char_at_pos,
            initial_score,
            params.tolerance,
            order_1,
            order_2,
            order_3,
            swap_position_pairs,
            pis_at_column,
            params.hill_climbing_iterations,
            0,
            0,
            True,
            params.pos_swaps_per_step,
            params.col_swaps_per_step,
            True,
            logger
        )
        
        final_score = min(population.values())
        
        # merge top N
        selected_population.update(heapq.nlargest(population_size, population.items(), key=lambda x: -x[1]))
        
        # keep the top population_size items in selected_population
        selected_population = dict(heapq.nlargest(population_size, selected_population.items(), key=lambda x: -x[1]))

        progress_queue.put(1)
        logger.run(seed_id, initial_score, final_score)

    logger.batch_end(selected_population)
    logger.save()

    return selected_population
