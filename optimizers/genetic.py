import os
import math
from re import T

from typing import List, Tuple, Optional, Any, Iterable
import numpy as np
import random
import logging
import heapq

from model import KeyboardModel, NgramType, _calculate_swap_delta
from freqdist import FreqDist
from layout import KeyboardLayout
from optim import OptimizerLogger
from optimizers.steepesthill import _optimize, _position_swapping, _column_swapping, swap_char_at_pos


def optimize_batch_worker(args):
    return _optimize_batch(*args)


def _calculate_score(
    char_at_pos: tuple[int, ...],
    order_1: tuple[tuple[np.ndarray, np.ndarray], ...],
    order_2: tuple[tuple[np.ndarray, np.ndarray], ...],
    order_3: tuple[tuple[np.ndarray, np.ndarray], ...],
) -> float:
    '''
    Calculate the full score for a char_at_pos layout using order data.
    '''
    C = np.asarray(char_at_pos, dtype=int)
    score = 0.0
    
    # Calculate order 1 contributions
    for F, V in order_1:
        P = F[C]
        score += float(np.sum(P * V))
    
    # Calculate order 2 contributions
    for F, V in order_2:
        P = F[np.ix_(C, C)]
        score += float(np.sum(P * V))
    
    # Calculate order 3 contributions
    for F, V in order_3:
        P = F[np.ix_(C, C, C)]
        score += float(np.sum(P * V))
    
    return score


def _pmx_crossover(
    parent1: tuple[int, ...],
    parent2: tuple[int, ...],
    start: int,
    end: int
) -> tuple[int, ...]:
    '''
    Partially Mapped Crossover (PMX) for permutations.
    Preserves the permutation property (no repeated elements).
    
    Standard PMX algorithm:
    1. Copy segment [start:end] from parent1 to child
    2. For positions outside segment, try to copy from parent2
    3. If parent2's value conflicts with segment, create mapping and resolve
    '''
    n = len(parent1)
    child = list(parent1)  # Start with copy of parent1
    
    # Build mapping: for positions in segment, map parent2[position] -> parent1[position]
    mapping = {}
    for i in range(start, end):
        p1_val = parent1[i]
        p2_val = parent2[i]
        if p1_val != p2_val:
            # Create bidirectional mapping
            mapping[p2_val] = p1_val
    
    # Fill positions outside segment from parent2
    segment_values = set(child[start:end])
    used_values = set(child[start:end])  # Track all values already in child
    
    for i in range(n):
        if i < start or i >= end:
            p2_val = parent2[i]
            # If this value is already used (in segment), need to map it
            if p2_val in used_values:
                # Follow mapping chain until we get a value not yet used
                val = p2_val
                visited = set()
                while val in used_values and val not in visited:
                    visited.add(val)
                    if val in mapping:
                        val = mapping[val]
                    else:
                        # Mapping exhausted - find an unused value from parent2
                        for j in range(n):
                            candidate = parent2[j]
                            if candidate not in used_values:
                                val = candidate
                                break
                        break
                child[i] = val
                used_values.add(val)
            else:
                child[i] = p2_val
                used_values.add(p2_val)
    
    return tuple(child)


def _report_progress(internal_counter: int, internal_total: int, external_total: int, progress_queue: Any) -> None:
    previous_external_counter = int((internal_counter-1)*external_total//internal_total)
    external_counter = int(internal_counter*external_total//internal_total)

    for _ in range(external_counter - previous_external_counter):
        progress_queue.put(1)


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
    optimize the layout using a genetic algorithm
    
    Uses genetic algorithm with:
    - Population initialized from char_at_pos_list
    - PMX crossover to combine parents while preserving permutation property
    - Local optimization with _optimize (10 iterations per child) to find local minima
    - Total iterations distributed across generations
    '''
    logger.batch_start()

    #
    # this is the number of times we will generate new seeds from parents
    # if there are N = len(char_at_pos_list) seeds provided, then we will
    # iteratively generate GENERATIONS * N children seeds from the parents
    #     
    GENERATIONS = 3

    seed_count = len(char_at_pos_list)
    generations_total = GENERATIONS * seed_count

    # the algorithm will first optimize the seed population, then do generation_total child seeds
    progress_total = seed_count + generations_total
    progress_counter = 0
    

    generation_size = max(len(char_at_pos_list), population_size)

    # Initialize population with scores
    population: dict[tuple[int, ...], float] = {}
    for char_at_pos, initial_score in zip(char_at_pos_list, initial_score_list):
        population[char_at_pos] = initial_score

    # run optimize once on each char_at_pos in char_at_pos_list and accumulate the best population
    best_population: dict[tuple[int, ...], float] = {}
    for char_at_pos, initial_score in zip(char_at_pos_list, initial_score_list):
        child_population = _optimize(
            0, 
            char_at_pos, 
            initial_score, 
            tolerance, order_1, order_2, order_3, pinned_positions, swap_position_pairs, positions_at_column, logger)
        best_population.update(child_population)
        
        progress_counter += 1
        _report_progress(progress_counter, progress_total, seed_count, progress_queue)

    # replace population with the best population_size layouts from best_population
    population = dict(sorted(best_population.items(), key=lambda x: x[1])[:population_size])

    remaining_iterations = generations_total
    generation = 0
    
    # Run genetic algorithm until we exhaust iterations budget
    while remaining_iterations > 0:
        # Select parents (sorted by score - lower is better)
        sorted_population = sorted(population.items(), key=lambda x: x[1])
        parents = [item[0] for item in sorted_population]
        min_score = sorted_population[0][1]
        
        if len(parents) < 2:
            break
        
        # Create a child through crossover
        # pick parents so that better scores are more likely to be selected, noting that score can be negative
        weights = [1.0 / (1.0 + item[1] - min_score) for item in sorted_population]
        
        # make sure p1 != p2
        while True:
            p1, p2 = random.choices(parents, weights=weights, k=2)
            if p1 != p2:
                break

        # Perform PMX crossover
        n = len(p1)
        start = random.randint(0, n - 2)
        end = random.randint(start + 1, n)
        child = _pmx_crossover(p1, p2, start, end)

        # assert that child has maintained pinned_positions
        assert all(child[i] == p1[i] for i in pinned_positions), f"child {child} has not maintained pinned_positions {pinned_positions}"
        assert all(child[i] == p2[i] for i in pinned_positions), f"child {child} has not maintained pinned_positions {pinned_positions}"
        
        # Optional: small mutation (swap two random positions with low probability)
        if random.random() < 0.1:  # 10% mutation rate
            unpinned_indices = [i for i in range(n) if i not in pinned_positions]
            if len(unpinned_indices) >= 2:
                swap_indices = random.sample(unpinned_indices, 2)
                child = swap_char_at_pos(child, swap_indices[0], swap_indices[1])
        
        if child == p1 or child == p2:
            continue
        
        # Calculate initial score for child
        child_initial_score = _calculate_score(child, order_1, order_2, order_3)

        # print(f"{logger.batch_name} generation {generation} population size: {len(population)}, ranging from {sorted_population[0][1]} to {sorted_population[-1][1]}")
        # print(f"{logger.batch_name} generation {generation} parents: {population[p1]} and {population[p2]}, child: {child_initial_score}")
        # print()
        
        child_population = _optimize(
            generation, 
            child, 
            child_initial_score, 
            tolerance, 
            order_1, 
            order_2, 
            order_3,
            pinned_positions, 
            swap_position_pairs, 
            positions_at_column, 
            logger
        )

        logger.run(generation, child_initial_score, min(child_population.values()))
        
        # take the best k from child_population, make sure k <= 10% of len(population)
        k = min(len(child_population), max(1, int(0.01 * generation_size)))

        child_population = dict(sorted(child_population.items(), key=lambda x: x[1])[:k])
        population.update(child_population)
        
        # Keep top generation_size individuals (lower scores are better)
        population = dict(sorted(population.items(), key=lambda x: x[1])[:generation_size])
        
        remaining_iterations -= 1
        generation += 1

        progress_counter += 1
        _report_progress(progress_counter, progress_total, seed_count, progress_queue)
    
    logger.batch_end(population)
    logger.save()

    # take population_size best layouts from population
    return dict(sorted(population.items(), key=lambda x: x[1])[:population_size])



