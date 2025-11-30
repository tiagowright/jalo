from itertools import combinations
from typing import Any
from dataclasses import dataclass
import numpy as np
import random
import heapq

from model import _calculate_swap_delta
from optim import OptimizerLogger


@dataclass
class GreedyHillParams:
    hill_climbing_iterations: int = 10
    pos_swaps_per_step: int = 4
    col_swaps_per_step: int = 1
    greedy_columns: bool = True
    tolerance: float = 0.00001



def optimize_batch_worker(args):
    return _optimize_batch(*args)

def _optimize_batch(
    char_at_pos_list: list[tuple[int, ...]],
    initial_score_list: list[float],
    tolerance: float,
    order_1: tuple[tuple[np.ndarray, np.ndarray], ...],
    order_2: tuple[tuple[np.ndarray, np.ndarray], ...],
    order_3: tuple[tuple[np.ndarray, np.ndarray], ...],
    pinned_positions: tuple[int, ...],
    swap_position_pairs: tuple[tuple[int, int], ...],
    pis_at_column: tuple[tuple[int, ...], ...],
    group_of_pis_at_column: tuple[tuple[tuple[int, ...], ...], ...],
    progress_queue: Any,
    population_size: int,
    logger: OptimizerLogger,
    solver_args: dict
) -> dict[tuple[int, ...], float]:
    '''
    optimize the layout using hill climbing

    this is a greedy hill climb algorithm that accepts any swap that improves the score immediately.
    '''
    logger.batch_start()

    params = GreedyHillParams()
    for key, value in solver_args.items():
        if hasattr(params, key):
            setattr(params, key, value)

    selected_population = {}

    for seed_id, char_at_pos, initial_score in zip(range(len(char_at_pos_list)), char_at_pos_list, initial_score_list):
        
        population = _optimize(
            seed_id, 
            char_at_pos, 
            initial_score, 
            tolerance, 
            order_1, 
            order_2, 
            order_3, 
            pinned_positions, 
            swap_position_pairs, 
            pis_at_column,
            group_of_pis_at_column,
            params.hill_climbing_iterations, 
            params,
            logger,
            solver_args
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


def _optimize(
    seed_id: int,
    char_at_pos: tuple[int, ...], 
    initial_score: float,
    tolerance: float,
    order_1: tuple[tuple[np.ndarray, np.ndarray], ...], 
    order_2: tuple[tuple[np.ndarray, np.ndarray], ...], 
    order_3: tuple[tuple[np.ndarray, np.ndarray], ...],
    pinned_positions: tuple[int, ...],
    swap_position_pairs: tuple[tuple[int, int], ...],
    pis_at_column: tuple[tuple[int, ...], ...],
    group_of_pis_at_column: tuple[tuple[tuple[int, ...], ...], ...],
    iterations: int,
    params: GreedyHillParams,
    logger: OptimizerLogger,
    solver_args: dict
) -> dict[tuple[int, ...], float]:
    '''
    optimize the layout using hill climbing

    this internal function is set up for multiprocessing, so all the inputs are considered immutable,
    and the only outputs are returned to the caller.
    '''

    population = {char_at_pos: initial_score}

    # cached_scores = Dict.empty(types.UniTuple(types.int64, len(char_at_pos)), types.float64)
    # cached_scores[char_at_pos] = initial_score
    cached_scores = {char_at_pos: initial_score}


    for i in range(iterations):
        current_char_at_pos = char_at_pos
        current_score = initial_score

        while True:
            score_at_start_of_step = current_score

            for _ in range(params.pos_swaps_per_step):
                prev_score = current_score
                current_score, current_char_at_pos = _position_swapping(
                    current_char_at_pos, 
                    prev_score, 
                    tolerance, 
                    order_1, 
                    order_2, 
                    order_3, 
                    pinned_positions, 
                    swap_position_pairs, 
                    params,
                    population,
                    cached_scores
                )

                if prev_score/current_score < (1.0 + params.tolerance):
                    break

            for _ in range(params.col_swaps_per_step):
                prev_score = current_score
                current_score, current_char_at_pos = _column_swapping(
                    current_char_at_pos, 
                    prev_score, 
                    tolerance, 
                    order_1, 
                    order_2, 
                    order_3, 
                    pinned_positions, 
                    pis_at_column,
                    params,
                    cached_scores
                )
                population[current_char_at_pos] = current_score

                if prev_score/current_score < (1.0 + params.tolerance):
                    break

            logger.event(seed_id, i, current_score)

            if current_score/score_at_start_of_step < (1.0 + params.tolerance):
                # this loop made no progress, so we are done on this branch
                break
        

    return population


def swap_char_at_pos(char_at_pos: tuple[int, ...], i: int, j: int) -> tuple[int, ...]:
    return tuple(
        char_at_pos[i] if k == j else
        char_at_pos[j] if k == i else
        char_at_pos[k]
        for k in range(len(char_at_pos))
    )


def _position_swapping(
    char_at_pos: tuple[int, ...], 
    score: float, 
    tolerance: float,
    order_1: tuple[tuple[np.ndarray, np.ndarray], ...], 
    order_2: tuple[tuple[np.ndarray, np.ndarray], ...], 
    order_3: tuple[tuple[np.ndarray, np.ndarray], ...],
    pinned_positions: tuple[int, ...],
    swap_position_pairs: tuple[tuple[int, int], ...],
    params: GreedyHillParams,
    population: dict[tuple[int, ...], float],
    cached_scores: dict[tuple[int, ...], float]
) -> tuple[float, tuple[int, ...]]:
    '''
    simple hill climbing, checks every possible swap in random order, and immediately
    accepts any swap that improves the score

    if N=len(char_at_pos), then there are N*(N-1)/2 swaps to try.

    this internal function is meant to be called from _optimize in a multiprocessing context,
    and the only mutable input is population, which is updated with new layouts and shared in a single 
    worker process within _optimize loops.
    '''
    tolerance = params.tolerance * score
    original_score = score

    for i, j in random.sample(swap_position_pairs, len(swap_position_pairs)):
        if i in pinned_positions or j in pinned_positions:
            continue

        swapped_char_at_pos = swap_char_at_pos(char_at_pos, i, j)

        if swapped_char_at_pos in cached_scores:
            swapped_score = cached_scores[swapped_char_at_pos]
            delta = swapped_score - score
        else:
            delta = _calculate_swap_delta(order_1, order_2, order_3, char_at_pos, i, j, swapped_char_at_pos)  # pyright: ignore[reportArgumentType]
            cached_scores[swapped_char_at_pos] = score + delta

        # accept the swap if it improves the score
        if delta < -tolerance:
            char_at_pos = swapped_char_at_pos
            score += delta
            population[char_at_pos] = score
    
    return (score, char_at_pos)


def _column_swapping(
    char_at_pos: tuple[int, ...], 
    score: float, 
    tolerance: float,
    order_1: tuple[tuple[np.ndarray, np.ndarray], ...], 
    order_2: tuple[tuple[np.ndarray, np.ndarray], ...], 
    order_3: tuple[tuple[np.ndarray, np.ndarray], ...],
    pinned_positions: tuple[int, ...],
    pis_at_column: tuple[tuple[int, ...], ...],
    params: GreedyHillParams,
    cached_scores: dict[tuple[int, ...], float]
) -> tuple[float, tuple[int, ...]]:

    '''
    simple hill climbing, swapping columns to improve the score the most
    '''
    tolerance = params.tolerance * score
    original_score = score
    best_delta = 0
    best_swap = None

    for col1, col2 in combinations(range(len(pis_at_column)), 2):
        if len(pis_at_column[col1]) <= 2 or len(pis_at_column[col2]) <= 2:
            continue

        if any(pi in pinned_positions for pi in pis_at_column[col1]) or any(pi in pinned_positions for pi in pis_at_column[col2]):
            continue

        if len(pis_at_column[col1]) > len(pis_at_column[col2]):
            col1, col2 = col2, col1
        
        # col1 is shorter than col2 or they are the same len
        random_positions_at_col2 = random.sample(pis_at_column[col2], len(pis_at_column[col1]))

        # compute the score after swapping all positions in col1 and col2    
        col_swapped_char_at_pos = char_at_pos
        delta = 0
        for pi1, pi2 in zip(pis_at_column[col1], random_positions_at_col2):
            next_swap_char_at_pos = swap_char_at_pos(col_swapped_char_at_pos, pi1, pi2)
            if next_swap_char_at_pos in cached_scores:
                delta = cached_scores[next_swap_char_at_pos] - original_score

            else:
                delta += _calculate_swap_delta(order_1, order_2, order_3, col_swapped_char_at_pos, pi1, pi2, next_swap_char_at_pos)  # pyright: ignore[reportArgumentType]
                cached_scores[next_swap_char_at_pos] = original_score + delta

            col_swapped_char_at_pos = next_swap_char_at_pos
            
        if delta < best_delta:
            best_delta = delta
            best_swap = col_swapped_char_at_pos

            if params.greedy_columns and best_delta < -tolerance:
                break
    
    if best_swap is not None and best_delta < -tolerance:
        return (original_score + best_delta, best_swap)
    
    return (original_score, char_at_pos)


