import os
import math
from re import T

from itertools import combinations
from typing import List, Tuple, Optional, Any, Iterable
import numpy as np
import random
import logging
import heapq

from model import _calculate_swap_delta
from optim import OptimizerLogger

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
    positions_at_column: tuple[tuple[int, ...], ...],
    iterations: int,
    progress_queue: Any,
    population_size: int,
    logger: OptimizerLogger
) -> dict[tuple[int, ...], float]:
    '''
    optimize the layout using hill climbing
    '''
    logger.batch_start()

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
            positions_at_column, 
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
    positions_at_column: tuple[tuple[int, ...], ...],
    logger: OptimizerLogger
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


    current_char_at_pos = char_at_pos
    current_score = initial_score
    step = 0

    while True:
        score_at_start_of_step = current_score

        current_score, current_char_at_pos = _position_swapping(
            current_char_at_pos, 
            current_score, 
            tolerance, 
            order_1, 
            order_2, 
            order_3, 
            pinned_positions, 
            swap_position_pairs, 
            population,
            cached_scores
        )
        
        current_score, current_char_at_pos = _column_swapping(
            current_char_at_pos, 
            current_score, 
            tolerance, 
            order_1, 
            order_2, 
            order_3, 
            pinned_positions, 
            positions_at_column,
            cached_scores
        )
        population[current_char_at_pos] = current_score

        logger.event(seed_id, step, current_score)
        step += 1

        delta = current_score - score_at_start_of_step
        if -tolerance < delta:
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
    population: dict[tuple[int, ...], float],
    cached_scores: dict[tuple[int, ...], float]
) -> tuple[float, tuple[int, ...]]:
    '''
    simple hill climbing, checks every possible swap, and 
    accepts the swap that improves the score the most

    if N=len(char_at_pos), then there are N*(N-1)/2 swaps to try.

    this internal function is meant to be called from _optimize in a multiprocessing context,
    and the only mutable input is population, which is updated with new layouts and shared in a single 
    worker process within _optimize loops.
    '''
    original_score = score
    best_delta = 0
    best_swap = None

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

        if delta < best_delta:
            best_delta = delta
            best_swap = swapped_char_at_pos

    # accept the best swap if it improves the score
    if best_swap is not None and best_delta < -tolerance:
        return (original_score + best_delta, best_swap)
    
    return (original_score, char_at_pos)


def _apply_column_swap(
    char_at_pos: tuple[int, ...],
    score: float,
    order_1: tuple[tuple[np.ndarray, np.ndarray], ...],
    order_2: tuple[tuple[np.ndarray, np.ndarray], ...],
    order_3: tuple[tuple[np.ndarray, np.ndarray], ...],
    positions_at_column: tuple[tuple[int, ...], ...],
    cached_scores: dict[tuple[int, ...], float],
    col_a_idx: int,
    col_b_idx: int
) -> tuple[float, tuple[int, ...]]:
    """
    Swap content between two columns (col_a_idx, col_b_idx) by
    swapping a set of positions in those columns, just like
    the original function did for (col1, col2).
    """
    positions_a = positions_at_column[col_a_idx]
    positions_b = positions_at_column[col_b_idx]

    # If either column is too small, do nothing (mirrors original guard).
    if len(positions_a) <= 2 or len(positions_b) <= 2:
        return score, char_at_pos

    # Ensure positions_a is the shorter one (or equal length)
    if len(positions_a) > len(positions_b):
        positions_a, positions_b = positions_b, positions_a

    # we don't have information here about row or layer, so we will align on the sequence of positions and cross fingers
    selected_positions_b = positions_b[:len(positions_a)]

    current_layout = char_at_pos
    current_score = score

    for pi1, pi2 in zip(positions_a, selected_positions_b):
        next_layout = swap_char_at_pos(current_layout, pi1, pi2)

        # Check cache for full-layout score
        cached = cached_scores.get(next_layout)
        if cached is not None:
            current_score = cached
        else:
            # Delta from current_layout -> next_layout
            delta = _calculate_swap_delta(
                order_1, order_2, order_3, # pyright: ignore[reportArgumentType]
                current_layout, pi1, pi2, next_layout  # pyright: ignore[reportArgumentType]
            )
            current_score = current_score + delta
            cached_scores[next_layout] = current_score

        current_layout = next_layout

    return current_score, current_layout

def _column_swapping(
    char_at_pos: tuple[int, ...], 
    score: float, 
    tolerance: float,
    order_1: tuple[tuple[np.ndarray, np.ndarray], ...], 
    order_2: tuple[tuple[np.ndarray, np.ndarray], ...], 
    order_3: tuple[tuple[np.ndarray, np.ndarray], ...],
    pinned_positions: tuple[int, ...],
    positions_at_column: tuple[tuple[int, ...], ...],
    cached_scores: dict[tuple[int, ...], float]
) -> tuple[float, tuple[int, ...]]:

    '''
    consider every possible sequence of columns (permuations), and apply the swap that improves the score the most

    if K is the number of columns, then there are K! permutations to try.

    Note that pinning any position in a column prevents the whole column from being swapped.

    '''
    original_score = score
    best_score = original_score
    best_layout = char_at_pos

    movable_cols: list[int] = []
    for col_idx, col_positions in enumerate(positions_at_column):
        # Skip small columns (since we can swap them in position swapping)
        if len(col_positions) <= 2:
            continue

        # Skip columns that contain pinned positions
        if any(pi in pinned_positions for pi in col_positions):
            continue

        movable_cols.append(col_idx)

    m = len(movable_cols)
    if m < 2:
        # Nothing to do: no column swaps possible
        return original_score, char_at_pos


    current_layout = char_at_pos
    current_score = score

    # Counters for Heap's algorithm
    c = [0] * m
    i = 0

    # walk through the swaps generated by Heap's algorithm.
    while i < m:
        print(f"{i} < {m} c[{i}]: {c[i]} -- {c!r}")
        if c[i] < i:
            # Decide which indices to swap, per Heap's algorithm
            if i % 2 == 0:
                j = 0
            else:
                j = c[i]

            # Map permutation indices -> actual physical column indices
            col_a = movable_cols[j]
            col_b = movable_cols[i]

            # Apply the same column swap to the layout
            current_score, current_layout = _apply_column_swap(
                current_layout, current_score, order_1, order_2, order_3, positions_at_column, cached_scores, col_a, col_b
            )

            # Update the permutation state (movable_cols) as Heap's algorithm requires
            movable_cols[j], movable_cols[i] = movable_cols[i], movable_cols[j]

            # Track best layout seen
            if current_score < best_score:
                best_score = current_score
                best_layout = current_layout

            c[i] += 1
            i = 0
        else:
            c[i] = 0
            i += 1

    if best_score < original_score - tolerance:
        return best_score, best_layout

    return original_score, char_at_pos


