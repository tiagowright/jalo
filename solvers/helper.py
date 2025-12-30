import math
from itertools import combinations
import numpy as np
import random
from typing import Any
import heapq

from model import _calculate_swap_delta
from repl.logger import OptimizerLogger


def swap_char_at_pos(char_at_pos: tuple[int, ...], i: int, j: int) -> tuple[int, ...]:
    return tuple(
        char_at_pos[i] if k == j else
        char_at_pos[j] if k == i else
        char_at_pos[k]
        for k in range(len(char_at_pos))
    )

def update_char_at_pos(char_at_pos: tuple[int, ...], update_values: tuple[int, ...], update_idxs: tuple[int, ...]) -> tuple[int, ...]:
    updated = list(char_at_pos)
    for i, v in zip(update_idxs, update_values):
        updated[i] = v
    return tuple(updated)


def report_progress(internal_counter: int, internal_total: int, external_total: int, progress_queue: Any) -> None:
    # the Optimizer expects progress_queue.put(1) once for each item in the seed list, but 
    # some algorithms iterate multiple times over all seeds (annealing, genetic). So this
    # function computes the external update progress, using the internal progress counter.
    previous_external_counter = int((internal_counter-1)*external_total//internal_total)
    external_counter = int(internal_counter*external_total//internal_total)

    for _ in range(external_counter - previous_external_counter):
        progress_queue.put(1)

def is_progressing(current_score: float, previous_score: float, tolerance: float) -> bool:
    return previous_score > (1.0 + tolerance) * current_score


def improve_layout(
    seed_id: int,
    char_at_pos: tuple[int, ...], 
    center_char_at_pos: tuple[int, ...] | None,
    max_distance: int | None,
    min_distance: int | None,
    initial_score: float,
    tolerance: float,
    order_1: tuple[tuple[np.ndarray, np.ndarray], ...], 
    order_2: tuple[tuple[np.ndarray, np.ndarray], ...], 
    order_3: tuple[tuple[np.ndarray, np.ndarray], ...],
    swap_position_pairs: tuple[tuple[int, int], ...],
    pis_at_column: tuple[tuple[int, ...], ...],
    iterations: int,
    T0: float,
    Tf: float,
    greedy: bool,
    pos_swaps_per_step: int,
    col_swaps_per_step: int,
    pos_swaps_first: bool,
    logger: OptimizerLogger,
    uphill_deltas: list[float] | None = None,
) -> dict[tuple[int, ...], float]:
    '''
    takes a single layout and optimizes it, returning a population of derived layouts

    3 different strategies are possible depending on the arguments:
    
    a) steepest hill (T0==0 and greedy==False): in each step, identify the swap(s) that improve the score the most, 
    and take them best one. This is deterministic, single pass, and fast, but suffers
    from local minima. Because it is deterministic, multiple iterations are not helpful.

    b) greedy hill (T0==0 and greedy=True): in each step, try swaps in a random order, and take the first swap
    that improves the score. This is stochastic, so multiple iterations on the same initial
    seed are used to find multiple good candidates for optimized layouts.

    c) annealing (T0>0 and Tf>0): in each step, identify the best swap (steepest hill), and if the best swap
    improves the score, take it. If not, then take the swap depending on the temperature,
    using the simulated annealing approach. This is also stochastic, and multiple iterations at
    different temperatures are used to find multiple good canidadates.

    The algorithm: For multiple iterations, alternate between finding
    position swaps and whole column swaps, until no further improvement is made in the score
    '''

    assert not (min_distance and max_distance), "specify either min_distance (`generate`) or max_distance (`improve`), not both"

    if greedy and T0 > 0:
        raise ValueError("greedy and T0 > 0 is not allowed, use either greedy (with T0=0) or annealing (with T0>0)")

    if T0 <= 0:
        T0 = 0

    # population is the output of the optimization, the best layouts
    population = {char_at_pos: initial_score}

    # cache stores all layouts found in this execution and caches the score to avoid expensive compute
    center_char_at_pos = center_char_at_pos or char_at_pos
    cached_scores = {char_at_pos: initial_score}
    cached_distances = {char_at_pos: hamming_distance(char_at_pos, center_char_at_pos)}

    if uphill_deltas is None:
        uphill_deltas = []

    # Initialize temperature for simulated annealing
    is_annealing = (T0 > 0)
    initial_temperature = T0
    final_temperature = Tf if Tf > 0 and T0 > 0 else 1e-6 if T0 > 0 else 0


    if not greedy and not is_annealing:
        iterations = 1

    step = 0
    current_char_at_pos = char_at_pos
    current_score = initial_score
    pos_uphill_delta = float('inf')
    col_uphill_delta = float('inf')

    for i in range(iterations):

        # if greedy, then restart each iteration at the seed
        if greedy:
            current_char_at_pos = char_at_pos
            current_score = initial_score
            pos_uphill_delta = float('inf')
            col_uphill_delta = float('inf')

        # Calculate current temperature using exponential cooling schedule
        # Temperature decreases from initial_temperature to final_temperature over iterations
        if iterations > 1 and initial_temperature > 0:
            cooling_factor = (final_temperature / initial_temperature) ** (1.0 / (iterations - 1))
            temperature = initial_temperature * (cooling_factor ** i)
        else:
            temperature = initial_temperature

        while True:
            score_at_start_of_step = current_score

            if pos_swaps_first or step > 0:
                for _ in range(pos_swaps_per_step):
                    prev_score = current_score
                    current_score, current_char_at_pos, pos_uphill_delta = _position_swapping(
                        current_char_at_pos, 
                        center_char_at_pos,
                        max_distance,
                        current_score, 
                        order_1, 
                        order_2, 
                        order_3, 
                        swap_position_pairs, 
                        cached_scores,
                        cached_distances,
                        temperature,
                        greedy
                    )

                    if not is_progressing(current_score, prev_score, tolerance):
                        break

                    population[current_char_at_pos] = current_score


            for _ in range(col_swaps_per_step):
                prev_score = current_score
                current_score, current_char_at_pos, col_uphill_delta = _column_swapping(
                    current_char_at_pos, 
                    center_char_at_pos,
                    max_distance,
                    current_score, 
                    order_1, 
                    order_2, 
                    order_3, 
                    pis_at_column,
                    cached_scores,
                    cached_distances,
                    temperature,
                    greedy
                )

                if not is_progressing(current_score, prev_score, tolerance):
                    break
                
                population[current_char_at_pos] = current_score

            logger.event(seed_id, step, current_score)
            step += 1

            if not is_progressing(current_score, score_at_start_of_step, tolerance):
                # population[current_char_at_pos] = current_score
                # this loop made no progress, so we are done on this branch
                uphill_deltas.append(min(pos_uphill_delta, col_uphill_delta))
                break
    
    if min_distance is not None and min_distance > 0:
        sorted_population = sorted(population.keys(), key=lambda x: population[x])
        cluster_centers = hamming_distance_cluster_centers(sorted_population, min_distance)
        return {sorted_population[i]: population[sorted_population[i]] for i in cluster_centers}
    
    return population

def _position_swapping(
    char_at_pos: tuple[int, ...], 
    center_char_at_pos: tuple[int, ...],
    max_distance: int | None,
    score: float, 
    order_1: tuple[tuple[np.ndarray, np.ndarray], ...], 
    order_2: tuple[tuple[np.ndarray, np.ndarray], ...], 
    order_3: tuple[tuple[np.ndarray, np.ndarray], ...],
    swap_position_pairs: tuple[tuple[int, int], ...],
    cached_scores: dict[tuple[int, ...], float],
    cached_distances: dict[tuple[int, ...], int],
    temperature: float,
    greedy: bool
) -> tuple[float, tuple[int, ...], float]:
    '''
    check all possible single position swaps

    if N=len(char_at_pos), then there are N*(N-1)/2 swaps to try. 

    a) steepesthill (temperature == 0 and greedy == False), then
    accept only the best possible swap, and only if it improves
    the score

    b) greedyhill (temperature == 0 and greedy == True), then
    randomly check swaps until one improves the score, then
    accept it immediately

    c) annealing (temperature > 0), then check all swaps
    and accept only the best one, if it improves the score
    or if it clears the annealing criteria for the temperature

    returns chart_at_pos, score, uphill_delta
    '''

    original_score = score
    best_score = float('inf')
    best_swap = None
    best_uphill_delta = float('inf')

    if greedy and temperature > 0:
        raise ValueError("greedy and temperature > 0 is not allowed, use either greedy (with temperature=0) or annealing (with temperature>0)")

    # if greedy, make sure to randomize the sequence of checks
    ordered_swap_pairs = random.sample(swap_position_pairs, len(swap_position_pairs)) if greedy else swap_position_pairs

    for i, j in ordered_swap_pairs:
        swapped_char_at_pos = swap_char_at_pos(char_at_pos, i, j)

        if swapped_char_at_pos in cached_distances:
            swapped_distance = cached_distances[swapped_char_at_pos]
        else:
            swapped_distance = hamming_distance(swapped_char_at_pos, center_char_at_pos)
            cached_distances[swapped_char_at_pos] = swapped_distance

        if max_distance is not None and swapped_distance > max_distance:
            continue

        if swapped_char_at_pos in cached_scores:
            swapped_score = cached_scores[swapped_char_at_pos]
        else:
            delta = _calculate_swap_delta(order_1, order_2, order_3, char_at_pos, i, j, swapped_char_at_pos)  # pyright: ignore[reportArgumentType]
            swapped_score = score + delta
            cached_scores[swapped_char_at_pos] = swapped_score

        if swapped_score > score and (swapped_score - score) < best_uphill_delta:
            best_uphill_delta = swapped_score - score

        if swapped_score < best_score:
            best_score = swapped_score
            best_swap = swapped_char_at_pos

            if greedy and swapped_score < score:
                char_at_pos = swapped_char_at_pos
                score = swapped_score
        

    # when greedy, we've already accepted all swaps that are helpful
    if greedy or best_swap is None:
        return (score, char_at_pos, best_uphill_delta)

    # Always accept if it improves the score (delta < 0)
    if best_score < original_score:
        return (best_score, best_swap, best_uphill_delta)
        
    # Accept worse swaps probabilistically based on temperature
    elif temperature > 0:
        delta = best_score - original_score

        # Use Metropolis criterion: accept with probability exp(-delta/temperature)
        # Note: delta is positive here (worse score), so -delta/temperature is negative
        acceptance_probability = math.exp(-delta / temperature)

        if random.random() < acceptance_probability:
            return (best_score, best_swap, best_score - original_score)

        return (score, char_at_pos, best_score - original_score)
    
    return (score, char_at_pos, best_uphill_delta)


def _column_swapping(
    char_at_pos: tuple[int, ...], 
    center_char_at_pos: tuple[int, ...],
    max_distance: int | None,
    score: float, 
    order_1: tuple[tuple[np.ndarray, np.ndarray], ...], 
    order_2: tuple[tuple[np.ndarray, np.ndarray], ...], 
    order_3: tuple[tuple[np.ndarray, np.ndarray], ...],
    pis_at_column: tuple[tuple[int, ...], ...],
    cached_scores: dict[tuple[int, ...], float],
    cached_distances: dict[tuple[int, ...], int],
    temperature: float,
    greedy: bool
) -> tuple[float, tuple[int, ...], float]:
    '''
    check all possible column swaps

    a) steepesthill (temperature == 0 and greedy == False), then
    accept only the best possible swap, and only if it improves
    the score

    b) greedyhill (temperature == 0 and greedy == True), then
    randomly check swaps until one improves the score, then
    accept it immediately

    c) annealing (temperature > 0), then check all swaps
    and accept only the best one, if it improves the score
    or if it clears the annealing criteria for the temperature

    returns chart_at_pos, score, uphill_delta
    '''

    original_score = score
    best_score = float('inf')
    best_swap = None
    best_uphill_delta = float('inf')
    
    if greedy and temperature > 0:
        raise ValueError("greedy and temperature > 0 is not allowed, use either greedy (with temperature=0) or annealing (with temperature>0)")

    for col1, col2 in combinations(range(len(pis_at_column)), 2):
        if len(pis_at_column[col1]) <= 2 or len(pis_at_column[col2]) <= 2:
            continue

        if len(pis_at_column[col1]) > len(pis_at_column[col2]):
            col1, col2 = col2, col1
        
        # col1 is shorter than col2 or they are the same len
        selected_pis_at_col2 = pis_at_column[col2][:len(pis_at_column[col1])]

        # compute the score after swapping all positions in col1 and col2    
        col_swapped_char_at_pos = char_at_pos
        swapped_score = score

        for pi1, pi2 in zip(pis_at_column[col1], selected_pis_at_col2):
            next_swap_char_at_pos = swap_char_at_pos(col_swapped_char_at_pos, pi1, pi2)

            if next_swap_char_at_pos in cached_scores:
                swapped_score = cached_scores[next_swap_char_at_pos]

            else:
                delta = _calculate_swap_delta(order_1, order_2, order_3, col_swapped_char_at_pos, pi1, pi2, next_swap_char_at_pos)  # pyright: ignore[reportArgumentType]
                swapped_score = swapped_score + delta
                cached_scores[next_swap_char_at_pos] = swapped_score

            col_swapped_char_at_pos = next_swap_char_at_pos

        if max_distance is not None:
            swapped_distance = hamming_distance(col_swapped_char_at_pos, center_char_at_pos)
            if swapped_distance > max_distance:
                continue

        if swapped_score > score and (swapped_score - score) < best_uphill_delta:
            best_uphill_delta = swapped_score - score

        if swapped_score < best_score:
            best_score = swapped_score
            best_swap = col_swapped_char_at_pos

            if greedy and swapped_score < score:
                char_at_pos = col_swapped_char_at_pos
                score = swapped_score
    
    if greedy or best_swap is None:
        return (score, char_at_pos, best_uphill_delta)

    if best_score < original_score:
        return (best_score, best_swap, best_score - original_score)

    elif temperature > 0:
        delta = best_score - original_score

        acceptance_probability = math.exp(-delta / temperature)

        if random.random() < acceptance_probability:
            return (best_score, best_swap, best_score - original_score)

        return (score, char_at_pos, best_score - original_score)
    
    return (score, char_at_pos, best_uphill_delta)


def best_swaps(
    char_at_pos: tuple[int, ...], 
    score: float, 
    order_1: tuple[tuple[np.ndarray, np.ndarray], ...], 
    order_2: tuple[tuple[np.ndarray, np.ndarray], ...], 
    order_3: tuple[tuple[np.ndarray, np.ndarray], ...],
    swap_position_pairs: tuple[tuple[int, int], ...],
    cached_scores: dict[tuple[int, ...], float],
    iterations: int,
    max_depth: int
) -> tuple[dict[tuple[int, ...], float], dict[tuple[int, ...], tuple[tuple[int, int], ...]]]:
    '''
    explore swaps recursively to find the most impactful

    for the given layout, we check every possible swap, then iteratively look for additional possible swaps, up to a depth of max_depth

    return a mapping of layouts to scores, and layouts to swaps taken from char_at_pos
    '''

    layout_scores = {char_at_pos: score}
    layout_swaps = {char_at_pos: tuple()}
    layout_heap = [(score, char_at_pos)]
    heapq.heapify(layout_heap)

    for _ in range(iterations):
        if not layout_heap:
            break

        score, char_at_pos = heapq.heappop(layout_heap)

        for i, j in swap_position_pairs:
            swapped_char_at_pos = swap_char_at_pos(char_at_pos, i, j)

            if swapped_char_at_pos in layout_scores:
                continue

            if swapped_char_at_pos in cached_scores:
                swapped_score = cached_scores[swapped_char_at_pos]
            else:
                delta = _calculate_swap_delta(order_1, order_2, order_3, char_at_pos, i, j, swapped_char_at_pos)  # pyright: ignore[reportArgumentType]
                swapped_score = score + delta
                cached_scores[swapped_char_at_pos] = swapped_score

            layout_scores[swapped_char_at_pos] = swapped_score
            layout_swaps[swapped_char_at_pos] = layout_swaps[char_at_pos] + ((i, j),)

            if len(layout_swaps[swapped_char_at_pos]) < max_depth:
                heapq.heappush(layout_heap, (swapped_score, swapped_char_at_pos))

    return layout_scores, layout_swaps


def hamming_distance(char_at_pos_1: tuple[int, ...], char_at_pos_2: tuple[int, ...]) -> int:
    """
    Compute the Hamming distance between two layouts.

    The Hamming distance is the number of positions where the characters differ.
    """
    return sum(1 for i, j in zip(char_at_pos_1, char_at_pos_2) if i != j)


def hamming_distance_cluster_centers(
    sorted_population: list[tuple[int, ...]],
    cluster_threshold: int = 10,
) -> list[int]:
    """
    Simple clustering by best score.

    `sorted_population` must be sorted from best (lowest score) to worst (highest score).
    The cluster center is always the lowest score layout not yet clustered; any subsequent
    layouts within `cluster_threshold` Hamming distance are pulled into that cluster.
    """

    def _hamming_leq(char_at_pos_1: tuple[int, ...], char_at_pos_2: tuple[int, ...], max_distance: int) -> bool:
        """Fast Hamming distance check with early exit."""
        dist = 0
        for i, j in zip(char_at_pos_1, char_at_pos_2):
            if i != j:
                dist += 1
                if dist > max_distance:
                    return False
        return True

    n = len(sorted_population)
    if n == 0:
        return []

    r = cluster_threshold
    k = len(sorted_population[0])
    centers: list[int] = []

    # If r is too large relative to k, partitions won't work (need m=r+1 <= k).
    # Fall back to the simple O(N^2) implementation.
    if r <= 0:
        return list(range(n))
    if r + 1 > k:
        clustered: set[int] = set()
        for i in range(n):
            if i in clustered:
                continue
            centers.append(i)
            for j in range(i + 1, n):
                if j in clustered:
                    continue
                if hamming_distance(sorted_population[i], sorted_population[j]) <= r:
                    clustered.add(j)
        return centers

    # Candidate generation via multiple random partitions of positions into (r+1) blocks.
    # Exact guarantee: if dist(x, y) <= r then for any partition into m=r+1 disjoint blocks,
    # x and y must match at least one whole block. Multiple partitions reduce false candidates.
    m = r + 1
    T = 4  # number of independent partitions (tunable)
    rng = random.Random(0)

    partitions: list[list[list[int]]] = []
    for _ in range(T):
        positions = list(range(k))
        rng.shuffle(positions)
        blocks: list[list[int]] = []
        for j in range(m):
            s = (j * k) // m
            e = ((j + 1) * k) // m
            blocks.append(positions[s:e])
        partitions.append(blocks)

    # tables[t][j][key] -> list of center indices
    tables: list[list[dict[tuple[int, ...], list[int]]]] = [
        [dict() for _ in range(m)] for _ in range(T)
    ]

    for i, x in enumerate(sorted_population):
        # Collect candidate centers that match at least one block under any partition.
        candidates: list[int] = []
        seen: set[int] = set()

        for t in range(T):
            blocks = partitions[t]
            for j, block_positions in enumerate(blocks):
                key = tuple(x[p] for p in block_positions)
                bucket = tables[t][j].get(key)
                if not bucket:
                    continue
                for center_id in bucket:
                    if center_id not in seen:
                        seen.add(center_id)
                        candidates.append(center_id)

        covered = False
        for center_id in candidates:
            if _hamming_leq(x, sorted_population[center_id], r):
                covered = True
                break

        if not covered:
            centers.append(i)
            # Insert this new center into all buckets.
            for t in range(T):
                blocks = partitions[t]
                for j, block_positions in enumerate(blocks):
                    key = tuple(x[p] for p in block_positions)
                    bucket = tables[t][j].get(key)
                    if bucket is None:
                        tables[t][j][key] = [i]
                    else:
                        bucket.append(i)

    return centers