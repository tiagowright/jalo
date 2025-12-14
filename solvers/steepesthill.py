import numpy as np
import heapq
from dataclasses import dataclass
from typing import Any

from logger import OptimizerLogger
from solvers import helper

@dataclass
class SteepestHillParams:
    pos_swaps_per_step: int = 1
    col_swaps_per_step: int = 1
    pos_swaps_first: bool = False
    tolerance: float = 0.00001


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
    center_char_at_pos: tuple[int, ...] | None,
    max_distance: int | None,
    min_distance: int | None,
    solver_args: dict
) -> dict[tuple[int, ...], float]:
    '''
    optimize the layout using a steepest hill climbing algorithm: checks all possible swaps, and takes the best swap that improves the score,
    then start over checking all possible swaps again, and until no more swaps improve the score
    '''
    logger.batch_start()

    params = SteepestHillParams()
    for key, value in solver_args.items():
        if hasattr(params, key):
            setattr(params, key, value)

    selected_population = {}

    for seed_id, char_at_pos, initial_score in zip(range(len(char_at_pos_list)), char_at_pos_list, initial_score_list):

        population = helper.improve_layout(
            seed_id,
            char_at_pos,
            center_char_at_pos,
            max_distance,
            min_distance,
            initial_score,
            params.tolerance,
            order_1,
            order_2,
            order_3,
            swap_position_pairs,
            pis_at_column,
            1,
            0,
            0,
            False,
            params.pos_swaps_per_step,
            params.col_swaps_per_step,
            params.pos_swaps_first,
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
