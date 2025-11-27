import math
from dataclasses import dataclass
from itertools import combinations
from typing import Any
import numpy as np
import random
import heapq

from model import _calculate_swap_delta
from optim import OptimizerLogger

# Default number of iterations for simulated annealing optimizer
DEFAULT_OPTIMIZER_ITERATIONS = 10

def optimize_batch_worker(args):
    return _optimize_batch(*args)

def _calibrate_temperature(
    char_at_pos_list: list[tuple[int, ...]],
    initial_score_list: list[float],
    tolerance: float,
    order_1: tuple[tuple[np.ndarray, np.ndarray], ...],
    order_2: tuple[tuple[np.ndarray, np.ndarray], ...],
    order_3: tuple[tuple[np.ndarray, np.ndarray], ...],
    pinned_positions: tuple[int, ...],
    swap_position_pairs: tuple[tuple[int, int], ...],
    positions_at_column: tuple[tuple[int, ...], ...],
    logger: OptimizerLogger,
    solver_args: dict,
    p0: float = 0.675, # 67.5% acceptance probability at start of annealing (seems like a good choice at the moment)
    pf: float = 0.01,
    samples_per_layout: int = 4,
    layouts_to_sample: int = 10,
) -> tuple[float, float]:

    uphill_deltas = []

    layouts_to_sample = min(layouts_to_sample, len(char_at_pos_list))
    samples_per_layout = min(samples_per_layout, len(swap_position_pairs)//2 if len(swap_position_pairs) > 2 else 1)

    unpinned_swap_position_pairs = [pair for pair in swap_position_pairs if pair[0] not in pinned_positions and pair[1] not in pinned_positions]
        
    # for char_at_pos in random.sample(char_at_pos_list, layouts_to_sample):
    # for seed_id, char_at_pos, initial_score in zip(range(len(char_at_pos_list)), char_at_pos_list, initial_score_list):
    for seed_id in random.sample(range(len(char_at_pos_list)), layouts_to_sample):
        char_at_pos = char_at_pos_list[seed_id]
        initial_score = initial_score_list[seed_id]

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
            10,
            1e-6,
            1e-6,
            logger,
            solver_args
        )

        char_at_pos = min(population.keys(), key=lambda x: population[x])
        char_at_pos_uphill_deltas = []

        for i, j in unpinned_swap_position_pairs:
            swapped_char_at_pos = swap_char_at_pos(char_at_pos, i, j)
            delta = _calculate_swap_delta(order_1, order_2, order_3, char_at_pos, i, j, swapped_char_at_pos)  # pyright: ignore[reportArgumentType]
            if 0 < delta:
                char_at_pos_uphill_deltas.append(delta)

        # take the lowest (best) samples_per_layout uphill deltas
        char_at_pos_uphill_deltas.sort()
        uphill_deltas.extend(char_at_pos_uphill_deltas[:samples_per_layout])


    # Use 75th percentile as "typical" uphill cost
    uphill_deltas.sort()
    idx = int(0.75 * (len(uphill_deltas) - 1))
    Δ_typical = uphill_deltas[idx]

    # Compute starting + ending temperatures based on target accept probabilities
    T0 = -Δ_typical / math.log(p0)
    Tf = -Δ_typical / math.log(pf)

    # print(f"Δ_typical: {Δ_typical}: T0: {T0}, Tf: {Tf}")

    return (T0, Tf)   


@dataclass
class AnnealingStats:
    swaps_considered: int = 0
    downhill_swaps_accepted: int = 0
    uphill_swaps_accepted: int = 0
    uphill_swaps_rejected: int = 0

    def __str__(self):
        return "\n".join([
            f"-- Annealing stats: --",
            f"Swaps considered: {self.swaps_considered}",
            f"Downhill swaps accepted: {self.downhill_swaps_accepted}",
            f"Uphill swaps accepted: {self.uphill_swaps_accepted}",
            f"Uphill swaps rejected: {self.uphill_swaps_rejected}",
        ])

    def __repr__(self):
        return f"AnnealingStats(swaps_considered={self.swaps_considered}, downhill_swaps_accepted={self.downhill_swaps_accepted}, uphill_swaps_accepted={self.uphill_swaps_accepted}, uphill_swaps_rejected={self.uphill_swaps_rejected})"

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
    progress_queue: Any,
    population_size: int,
    logger: OptimizerLogger,
    solver_args: dict
) -> dict[tuple[int, ...], float]:
    '''
    optimize the layout using simulated annealing
    '''
    logger.batch_start()

    selected_population = {}

    T0, Tf = _calibrate_temperature(
        char_at_pos_list, 
        initial_score_list, 
        tolerance, 
        order_1, 
        order_2, 
        order_3, 
        pinned_positions, 
        swap_position_pairs, 
        positions_at_column,
        logger,
        solver_args
    )

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
            DEFAULT_OPTIMIZER_ITERATIONS,
            T0,
            Tf,
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
    positions_at_column: tuple[tuple[int, ...], ...],
    iterations: int,
    T0: float,
    Tf: float,
    logger: OptimizerLogger,
    solver_args: dict
) -> dict[tuple[int, ...], float]:
    '''
    optimize the layout using simulated annealing

    this internal function is set up for multiprocessing, so all the inputs are considered immutable,
    and the only outputs are returned to the caller.
    '''

    population = {char_at_pos: initial_score}

    # cached_scores = Dict.empty(types.UniTuple(types.int64, len(char_at_pos)), types.float64)
    # cached_scores[char_at_pos] = initial_score
    cached_scores = {char_at_pos: initial_score}

    # Initialize temperature for simulated annealing
    # Start with a high temperature to allow exploration
    # initial_temperature = abs(initial_score) * 0.1 if initial_score != 0 else 1.0
    initial_temperature = T0
    final_temperature = Tf

    step = 0

    annealing_stats = AnnealingStats()

    current_char_at_pos = char_at_pos
    current_score = initial_score
    for i in range(iterations):

        # Calculate current temperature using exponential cooling schedule
        # Temperature decreases from initial_temperature to final_temperature over iterations
        if iterations > 1:
            cooling_factor = (final_temperature / initial_temperature) ** (1.0 / (iterations - 1))
            temperature = initial_temperature * (cooling_factor ** i)
        else:
            temperature = initial_temperature

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
                cached_scores,
                temperature,
                annealing_stats
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
                cached_scores,
                temperature,
                annealing_stats
            )
            population[current_char_at_pos] = current_score

            logger.event(seed_id, step, current_score)
            step += 1

            delta = current_score - score_at_start_of_step
            if abs(delta) < tolerance:
                # this loop made no progress, so we are done on this branch
                break


    # print(annealing_stats)
    
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
    cached_scores: dict[tuple[int, ...], float],
    temperature: float,
    annealing_stats: AnnealingStats
) -> tuple[float, tuple[int, ...]]:
    '''
    simulated annealing for position swapping, checks every possible swap,
    and accepts swaps probabilistically based on temperature

    if N=len(char_at_pos), then there are N*(N-1)/2 swaps to try. We try all,
    find the best swap, and accept it based on simulated annealing criteria.

    this internal function is meant to be called from _optimize in a multiprocessing context,
    and the only mutable input is population, which is updated with new layouts and shared in a single 
    worker process within _optimize loops.
    '''
    original_score = score
    best_delta = float('inf')
    best_swap = None

    for i, j in swap_position_pairs:
        if i in pinned_positions or j in pinned_positions:
            continue

        swapped_char_at_pos = swap_char_at_pos(char_at_pos, i, j)
        annealing_stats.swaps_considered += 1

        if swapped_char_at_pos in cached_scores:
            swapped_score = cached_scores[swapped_char_at_pos]
            delta = swapped_score - score
        else:
            delta = _calculate_swap_delta(order_1, order_2, order_3, char_at_pos, i, j, swapped_char_at_pos)  # pyright: ignore[reportArgumentType]
            cached_scores[swapped_char_at_pos] = score + delta

        if delta < best_delta:
            best_delta = delta
            best_swap = swapped_char_at_pos

    # Accept swap using simulated annealing criteria
    if best_swap is not None:

        # Always accept if it improves the score (delta < 0)
        # if best_delta < -tolerance:
        if best_delta < 0:
            annealing_stats.downhill_swaps_accepted += 1
            return (original_score + best_delta, best_swap)
        
        # Accept worse swaps probabilistically based on temperature
        elif temperature > 0:

            # Use Metropolis criterion: accept with probability exp(-delta/temperature)
            # Note: best_delta is positive here (worse score), so -best_delta/temperature is negative
            acceptance_probability = math.exp(-best_delta / temperature)
            # print(f"acceptance_probability: {acceptance_probability} for delta: {best_delta} and temperature: {temperature}")

            if random.random() < acceptance_probability:
                annealing_stats.uphill_swaps_accepted += 1
                return (original_score + best_delta, best_swap)
            else:
                annealing_stats.uphill_swaps_rejected += 1
    
    return (original_score, char_at_pos)


def _column_swapping(
    char_at_pos: tuple[int, ...], 
    score: float, 
    tolerance: float,
    order_1: tuple[tuple[np.ndarray, np.ndarray], ...], 
    order_2: tuple[tuple[np.ndarray, np.ndarray], ...], 
    order_3: tuple[tuple[np.ndarray, np.ndarray], ...],
    pinned_positions: tuple[int, ...],
    positions_at_column: tuple[tuple[int, ...], ...],
    cached_scores: dict[tuple[int, ...], float],
    temperature: float,
    annealing_stats: AnnealingStats
) -> tuple[float, tuple[int, ...]]:

    '''
    simulated annealing for column swapping, swapping columns and accepting
    swaps probabilistically based on temperature
    '''
    original_score = score
    best_delta = float('inf')
    best_swap = None

    for col1, col2 in combinations(range(len(positions_at_column)), 2):
        if len(positions_at_column[col1]) <= 2 or len(positions_at_column[col2]) <= 2:
            continue

        if any(pi in pinned_positions for pi in positions_at_column[col1]) or any(pi in pinned_positions for pi in positions_at_column[col2]):
            continue

        if len(positions_at_column[col1]) > len(positions_at_column[col2]):
            col1, col2 = col2, col1
        
        # col1 is shorter than col2 or they are the same len
        random_positions_at_col2 = random.sample(positions_at_column[col2], len(positions_at_column[col1]))

        # compute the score after swapping all positions in col1 and col2    
        col_swapped_char_at_pos = char_at_pos
        delta = 0
        for pi1, pi2 in zip(positions_at_column[col1], random_positions_at_col2):
            next_swap_char_at_pos = swap_char_at_pos(col_swapped_char_at_pos, pi1, pi2)
            if next_swap_char_at_pos in cached_scores:
                delta = cached_scores[next_swap_char_at_pos] - original_score

            else:
                delta += _calculate_swap_delta(order_1, order_2, order_3, col_swapped_char_at_pos, pi1, pi2, next_swap_char_at_pos)  # pyright: ignore[reportArgumentType]
                cached_scores[next_swap_char_at_pos] = original_score + delta

            col_swapped_char_at_pos = next_swap_char_at_pos
            
        annealing_stats.swaps_considered += 1

        if delta < best_delta:
            best_delta = delta
            best_swap = col_swapped_char_at_pos
    
    # Accept swap using simulated annealing criteria
    if best_swap is not None:
        # Always accept if it improves the score (delta < 0)
        if best_delta < 0:
            annealing_stats.downhill_swaps_accepted += 1
            return (original_score + best_delta, best_swap)

        # Accept worse swaps probabilistically based on temperature
        elif temperature > 0:
            # Use Metropolis criterion: accept with probability exp(-delta/temperature)
            # Note: best_delta is positive here (worse score), so -best_delta/temperature is negative
            acceptance_probability = math.exp(-best_delta / temperature)
            # print(f"acceptance_probability: {acceptance_probability} for delta: {best_delta} and temperature: {temperature}")

            if random.random() < acceptance_probability:
                annealing_stats.uphill_swaps_accepted += 1
                return (original_score + best_delta, best_swap)
            else:
                annealing_stats.uphill_swaps_rejected += 1
    
    return (original_score, char_at_pos)

