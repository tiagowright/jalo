import math
from itertools import combinations
import numpy as np
import random
import heapq

from model import _calculate_swap_delta
from optim import OptimizerLogger


def swap_char_at_pos(char_at_pos: tuple[int, ...], i: int, j: int) -> tuple[int, ...]:
    return tuple(
        char_at_pos[i] if k == j else
        char_at_pos[j] if k == i else
        char_at_pos[k]
        for k in range(len(char_at_pos))
    )

def improve_layout(
    seed_id: int,
    char_at_pos: tuple[int, ...], 
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

    if greedy and T0 > 0:
        raise ValueError("greedy and T0 > 0 is not allowed, use either greedy (with T0=0) or annealing (with T0>0)")

    if T0 <= 0:
        T0 = 0

    # population is the output of the optimization, the best layouts
    population = {char_at_pos: initial_score}

    # cache stores all layouts found in this execution and caches the score to avoid expensive compute
    cached_scores = {char_at_pos: initial_score}

    # Initialize temperature for simulated annealing
    is_annealing = (T0 > 0)
    initial_temperature = T0
    final_temperature = Tf if Tf > 0 and T0 > 0 else 1e-6 if T0 > 0 else 0


    if not greedy and not is_annealing:
        iterations = 1

    step = 0
    current_char_at_pos = char_at_pos
    current_score = initial_score
    for i in range(iterations):

        # if greedy, then restart each iteration at the seed
        if greedy:
            current_char_at_pos = char_at_pos
            current_score = initial_score

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
                    current_score, current_char_at_pos = _position_swapping(
                        current_char_at_pos, 
                        current_score, 
                        order_1, 
                        order_2, 
                        order_3, 
                        swap_position_pairs, 
                        cached_scores,
                        temperature,
                        greedy
                    )

                    if prev_score/current_score < (1.0 + tolerance):
                        break

                    population[current_char_at_pos] = current_score


            for _ in range(col_swaps_per_step):
                prev_score = current_score
                current_score, current_char_at_pos = _column_swapping(
                    current_char_at_pos, 
                    current_score, 
                    order_1, 
                    order_2, 
                    order_3, 
                    pis_at_column,
                    cached_scores,
                    temperature,
                        greedy
                )

                if prev_score/current_score < (1.0 + tolerance):
                    break
                population[current_char_at_pos] = current_score

            logger.event(seed_id, step, current_score)
            step += 1

            if score_at_start_of_step/current_score < (1.0 + tolerance):
                # population[current_char_at_pos] = current_score
                # this loop made no progress, so we are done on this branch
                break
    
    return population


def _position_swapping(
    char_at_pos: tuple[int, ...], 
    score: float, 
    order_1: tuple[tuple[np.ndarray, np.ndarray], ...], 
    order_2: tuple[tuple[np.ndarray, np.ndarray], ...], 
    order_3: tuple[tuple[np.ndarray, np.ndarray], ...],
    swap_position_pairs: tuple[tuple[int, int], ...],
    cached_scores: dict[tuple[int, ...], float],
    temperature: float,
    greedy: bool
) -> tuple[float, tuple[int, ...]]:
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
    '''

    original_score = score
    best_score = float('inf')
    best_swap = None

    if greedy and temperature > 0:
        raise ValueError("greedy and temperature > 0 is not allowed, use either greedy (with temperature=0) or annealing (with temperature>0)")

    # if greedy, make sure to randomize the sequence of checks
    ordered_swap_pairs = random.sample(swap_position_pairs, len(swap_position_pairs)) if greedy else swap_position_pairs

    for i, j in ordered_swap_pairs:
        swapped_char_at_pos = swap_char_at_pos(char_at_pos, i, j)

        if swapped_char_at_pos in cached_scores:
            swapped_score = cached_scores[swapped_char_at_pos]
        else:
            delta = _calculate_swap_delta(order_1, order_2, order_3, char_at_pos, i, j, swapped_char_at_pos)  # pyright: ignore[reportArgumentType]
            swapped_score = score + delta
            cached_scores[swapped_char_at_pos] = swapped_score

        if swapped_score < best_score:
            best_score = swapped_score
            best_swap = swapped_char_at_pos

            if greedy and swapped_score < score:
                char_at_pos = swapped_char_at_pos
                score = swapped_score


    # when greedy, we've already accepted all swaps that are helpful
    if greedy or best_swap is None:
        return (score, char_at_pos)

    # Always accept if it improves the score (delta < 0)
    if best_score < original_score:
        return (best_score, best_swap)
        
    # Accept worse swaps probabilistically based on temperature
    elif temperature > 0:
        delta = best_score - original_score

        # Use Metropolis criterion: accept with probability exp(-delta/temperature)
        # Note: delta is positive here (worse score), so -delta/temperature is negative
        acceptance_probability = math.exp(-delta / temperature)

        if random.random() < acceptance_probability:
            return (best_score, best_swap)
    
    return (score, char_at_pos)


def _column_swapping(
    char_at_pos: tuple[int, ...], 
    score: float, 
    order_1: tuple[tuple[np.ndarray, np.ndarray], ...], 
    order_2: tuple[tuple[np.ndarray, np.ndarray], ...], 
    order_3: tuple[tuple[np.ndarray, np.ndarray], ...],
    pis_at_column: tuple[tuple[int, ...], ...],
    cached_scores: dict[tuple[int, ...], float],
    temperature: float,
    greedy: bool
) -> tuple[float, tuple[int, ...]]:
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
    '''

    original_score = score
    best_score = float('inf')
    best_swap = None

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

        if swapped_score < best_score:
            best_score = swapped_score
            best_swap = col_swapped_char_at_pos

            if greedy and swapped_score < score:
                char_at_pos = col_swapped_char_at_pos
                score = swapped_score
    
    if greedy or best_swap is None:
        return (score, char_at_pos)

    if best_score < original_score:
        return (best_score, best_swap)

    elif temperature > 0:
        delta = best_score - original_score

        acceptance_probability = math.exp(-delta / temperature)

        if random.random() < acceptance_probability:
            return (best_score, best_swap)
    
    return (score, char_at_pos)


