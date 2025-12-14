import math
from dataclasses import dataclass
from typing import Any
import numpy as np
import heapq

from logger import OptimizerLogger
from solvers import helper


@dataclass
class AnnealingParams:
    # Default number of iterations of annealing for each seed
    annealing_iterations: int = 10
    
    # Default acceptance probability at start and end of annealing (seems like a good choice at the moment)
    p0: float = 0.675 
    pf: float = 0.01

    tolerance: float = 0.0001
    pos_swaps_per_step: int = 1
    col_swaps_per_step: int = 1
    pos_swaps_first: bool = False


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
    optimize the layout using simulated annealing:
    checks all possible swaps to find the best one, and accepts them immediately if they improve the score,
    or if they don't improve the score, accepts them probabilistically based on the temperature
    '''
    logger.batch_start()

    args = AnnealingParams()
    for key, value in solver_args.items():
        if hasattr(args, key):
            setattr(args, key, value)

    selected_population = {}

    progress_counter = 0
    seed_count = len(char_at_pos_list)
    progress_total = 2 * seed_count

    # two passes: first steepest descent and collect uphill deltas, then annealing
    uphill_deltas = []
    improved_char_at_pos_list = []
    improved_initial_score_list = []
    for seed_id, char_at_pos, initial_score in zip(range(len(char_at_pos_list)), char_at_pos_list, initial_score_list):

        population = helper.improve_layout(
            seed_id,
            char_at_pos,
            center_char_at_pos,
            max_distance,
            min_distance,
            initial_score,
            args.tolerance,
            order_1,
            order_2,
            order_3,
            swap_position_pairs,
            pis_at_column,
            1,
            0,
            0,
            False,
            args.pos_swaps_per_step,
            args.col_swaps_per_step,
            args.pos_swaps_first,
            logger,
            uphill_deltas
        )

        best_char_at_pos = min(population.keys(), key=lambda x: population[x])
        improved_char_at_pos_list.append(best_char_at_pos)
        improved_initial_score_list.append(population[best_char_at_pos])
        
        progress_counter += 1
        helper.report_progress(progress_counter, progress_total, seed_count, progress_queue)
        
    # find typical values for uphill deltas from the end of the first pass
    uphill_deltas.sort()
    idx = int(0.75 * (len(uphill_deltas) - 1))
    Δ_typical = uphill_deltas[idx]

    # Compute starting + ending temperatures based on target accept probabilities
    T0 = -Δ_typical / math.log(args.p0)
    Tf = -Δ_typical / math.log(args.pf)

    # second pass: annealing
    for seed_id, char_at_pos, initial_score in zip(range(len(improved_char_at_pos_list)), improved_char_at_pos_list, improved_initial_score_list):

        population = helper.improve_layout(
            seed_id,
            char_at_pos,
            center_char_at_pos,
            max_distance,
            min_distance,
            initial_score,
            args.tolerance,
            order_1,
            order_2,
            order_3,
            swap_position_pairs,
            pis_at_column,
            args.annealing_iterations - 1,
            T0,
            Tf,
            False,
            args.pos_swaps_per_step,
            args.col_swaps_per_step,
            args.pos_swaps_first,
            logger
        )
        
        final_score = min(population.values())
        
        # merge top N
        selected_population.update(heapq.nlargest(population_size, population.items(), key=lambda x: -x[1]))
        
        # keep the top population_size items in selected_population
        selected_population = dict(heapq.nlargest(population_size, selected_population.items(), key=lambda x: -x[1]))

        progress_counter += 1
        helper.report_progress(progress_counter, progress_total, seed_count, progress_queue)
        
        logger.run(seed_id, initial_score, final_score)

    logger.batch_end(selected_population)
    logger.save()

    return selected_population

