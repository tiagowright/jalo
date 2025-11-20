import os
import math
from re import T

from itertools import combinations
from typing import List, Tuple, Optional, Any, Iterable
import numpy as np
import random
import logging
import heapq
import queue

import multiprocessing
from numba import jit
from numba import types
from numba.typed import Dict
from tqdm import tqdm

from model import KeyboardModel, NgramType, _calculate_swap_delta
from freqdist import FreqDist
from layout import KeyboardLayout


logger = logging.getLogger(__name__)


class Population:
    def __init__(self, max_size: int, cache_all: bool = False):
        self.max_size = max_size
        self.heap = []
        self.scores = {}
        self.cache_all = cache_all

    def push(self, score: float, char_at_pos: tuple[int, ...]) -> None:
        '''
        adds the new char_at_pos to the population
        '''
        if char_at_pos in self.scores:
            # already in the population, no action
            return

        self.scores[char_at_pos] = score

        # add salt to delta to avoid ties in the heap
        # score += score * 0.001 * random.random()

        if len(self.heap) < self.max_size:
            heapq.heappush(self.heap, (-score, char_at_pos))
        else:
            removed_score, removed_char_at_pos = heapq.heappushpop(self.heap, (-score, char_at_pos))
            if not self.cache_all:
                del self.scores[removed_char_at_pos]

    def top(self) -> tuple[int, ...]:
        '''
        return the char_at_pos that has the best score in the population
        '''
        return self.heap[0][1]


    def sorted(self) -> list[tuple[int, ...]]:
        '''
        return the items in the heap in sorted order
        '''
        return [char_at_pos for score, char_at_pos in sorted(self.heap, reverse=True)]


    def random_item(self) -> np.ndarray:
        '''
        return a random item from the population, but never the top one
        '''
        return random.choice(self.heap[1:])[1]


    def __len__(self) -> int:
        '''
        return the number of items in the heap
        '''
        return len(self.heap)

    def __contains__(self, char_at_pos: tuple[int, ...]) -> bool:
        '''
        return True if the item was ever seen in the population
        '''
        return char_at_pos in self.scores
    
    def __getitem__(self, char_at_pos: tuple[int, ...]) -> float:
        '''
        return the score of the item
        '''
        return self.scores[tuple(char_at_pos)]



# for debugging scores
def assert_scores(char_at_pos, score, model):
    actual_score = model.score_chars_at_positions(char_at_pos)
    if abs(actual_score - score) > abs(0.001 * score):
        raise ValueError(f"score mismatch: {score:.2f} != {actual_score:.2f}")


class Optimizer:
    def __init__(self, model: KeyboardModel, population_size: int = 1000):
        self.model = model

        positions_at_column_map = {}
        for pi, position in enumerate(self.model.hardware.positions):
            if position.col not in positions_at_column_map:
                positions_at_column_map[position.col] = []
            positions_at_column_map[position.col].append(pi)

        self.positions_at_column = tuple(tuple(positions) for positions in positions_at_column_map.values())

        self.swap_position_pairs = tuple(combinations(range(len(self.model.hardware.positions)), 2))

        self.population = Population(max_size=population_size)

    def generate(self, char_seq: list[str], iterations:int = 100, optimizer_iterations:int = 20, score_tolerance:float = 0.01):
        assert len(char_seq) == len(self.model.hardware.positions)

        char_at_pos = np.zeros(len(self.model.hardware.positions), dtype=int)
        for pi, position in enumerate(self.model.hardware.positions):
            char = char_seq[pi]
            try:
                char_at_pos[pi] = self.model.freqdist.char_seq.index(char)
            except ValueError:
                char_at_pos[pi] = self.model.freqdist.char_seq.index(FreqDist.out_of_distribution)


        initial_positions = tuple(
            tuple(np.random.permutation(char_at_pos))
            for _ in range(iterations)
        )

        initial_population = {
            initial_position: self.model.score_chars_at_positions(initial_position)
            for initial_position in initial_positions
        }

        tolerance = score_tolerance * sum(initial_population.values())/len(initial_positions)
        order_1, order_2, order_3 = self._get_FV()

        batch_size = math.ceil(len(initial_positions) / (os.cpu_count() or 1))
        batches = [initial_positions[i:i+batch_size] for i in range(0, len(initial_positions), batch_size)]

        manager = multiprocessing.Manager()
        progress_queue = manager.Queue()
        total_jobs = len(initial_positions)

        tasks = [
            (
                initial_positions_batch,
                tuple(initial_population[initial_position] for initial_position in initial_positions_batch),
                tolerance,
                order_1,
                order_2,
                order_3,
                (),
                self.swap_position_pairs,
                self.positions_at_column,
                optimizer_iterations,
                progress_queue
            )
            for initial_positions_batch in batches
        ]

        with multiprocessing.Pool() as pool:
            results_async = pool.map_async(_optimize_batch_worker, tasks)
            
            with tqdm(total=total_jobs, desc="Generating") as pbar:
                while not results_async.ready():
                    try:
                        # Check for progress updates without blocking
                        if pbar.n < total_jobs:
                            progress_queue.get(timeout=0.1)
                            pbar.update(1)
                    except queue.Empty:
                        # timeout on get, continue loop
                        pass
                
                # update with any remaining items in the queue
                while not progress_queue.empty():
                    try:
                        progress_queue.get_nowait()
                        pbar.update(1)
                    except queue.Empty:
                        break

            results = results_async.get()
            for result_batch in results:
                for new_char_at_pos, score in result_batch.items():
                    self.population.push(score, new_char_at_pos)

                        

    def optimize(self, char_at_pos: np.ndarray, score_tolerance = 0.01, iterations:int = 20, pinned_positions: tuple[int, ...] = ()):
        initial_char_at_pos = tuple(int(x) for x in char_at_pos)
        initial_score = self.model.score_chars_at_positions(initial_char_at_pos)
        tolerance = score_tolerance * initial_score

        order_1, order_2, order_3 = self._get_FV()

        new_population = _optimize(
            initial_char_at_pos,
            initial_score,
            tolerance,
            order_1,
            order_2,
            order_3,
            pinned_positions,
            self.swap_position_pairs,
            self.positions_at_column,
            iterations
        )
        
        for new_char_at_pos, score in new_population.items():
            self.population.push(score, new_char_at_pos)


    def _get_FV(self) -> tuple[tuple[tuple[np.ndarray, np.ndarray], ...], tuple[tuple[np.ndarray, np.ndarray], ...], tuple[tuple[np.ndarray, np.ndarray], ...]]:
        F = self.model.freqdist.to_numpy()
        V = self.model.V

        order_1 = tuple([
            (F[ngramtype], V[ngramtype])
            for ngramtype in NgramType 
            if ngramtype.order == 1 and ngramtype in F and ngramtype in V
        ])
        order_2 = tuple([
            (F[ngramtype], V[ngramtype])
            for ngramtype in NgramType 
            if ngramtype.order == 2 and ngramtype in F and ngramtype in V
        ])
        order_3 = tuple([
            (F[ngramtype], V[ngramtype])
            for ngramtype in NgramType 
            if ngramtype.order == 3 and ngramtype in F and ngramtype in V
        ])

        return order_1, order_2, order_3


def _optimize_worker(args):
    return _optimize(*args)

def _optimize_batch_worker(args):
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
    progress_queue: Any
) -> dict[tuple[int, ...], float]:
    '''
    optimize the layout using hill climbing
    '''

    population_size = 10
    selected_population = {}

    for char_at_pos, initial_score in zip(char_at_pos_list, initial_score_list):
        population = _optimize(char_at_pos, initial_score, tolerance, order_1, order_2, order_3, pinned_positions, swap_position_pairs, positions_at_column, iterations)
        # merge top N
        selected_population.update(heapq.nlargest(population_size, population.items(), key=lambda x: -x[1]))
        
        # keep the top population_size items in selected_population
        selected_population = dict(heapq.nlargest(population_size, selected_population.items(), key=lambda x: -x[1]))

        progress_queue.put(1)

    return selected_population


def _optimize(
    char_at_pos: tuple[int, ...], 
    initial_score: float,
    tolerance: float,
    order_1: tuple[tuple[np.ndarray, np.ndarray], ...], 
    order_2: tuple[tuple[np.ndarray, np.ndarray], ...], 
    order_3: tuple[tuple[np.ndarray, np.ndarray], ...],
    pinned_positions: tuple[int, ...],
    swap_position_pairs: tuple[tuple[int, int], ...],
    positions_at_column: tuple[tuple[int, ...], ...],
    iterations: int
) -> dict[tuple[int, ...], float]:
    '''
    optimize the layout using hill climbing

    this internal function is set up for multiprocessing, so all the inputs are considered immutable,
    and the only outputs are returned to the caller.
    '''

    # print(f"start pid={os.getpid()}\tcache={len(cached_scores)}", flush=True)

    population = {char_at_pos: initial_score}

    # cached_scores = Dict.empty(types.UniTuple(types.int64, len(char_at_pos)), types.float64)
    # cached_scores[char_at_pos] = initial_score
    cached_scores = {char_at_pos: initial_score}


    for i in range(iterations):
        current_char_at_pos = char_at_pos
        current_score = initial_score

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

            # print(f"delta: {delta} vs tolerance: {tolerance}")
            delta = current_score - score_at_start_of_step
            if -tolerance < delta:
                # this loop made no progress, so we are done on this branch
                break
        
    # print(f"end   pid={os.getpid()}\tcache={len(cached_scores)}", flush=True)

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
    simple hill climbing, checks every possible swap, and immediately
    accepts any swap that improves the score

    if N=len(char_at_pos), then there are N*(N-1)/2 swaps to try. We try all,
    sort them by deltas, and apply all that are possible.

    this internal function is meant to be called from _optimize in a multiprocessing context,
    and the only mutable input is population, which is updated with new layouts and shared in a single 
    worker process within _optimize loops.
    '''
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
    positions_at_column: tuple[tuple[int, ...], ...],
    cached_scores: dict[tuple[int, ...], float]
) -> tuple[float, tuple[int, ...]]:

    '''
    simple hill climbing, swapping columns to improve the score the most
    '''
    original_score = score
    best_delta = 0
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
            
        if delta < best_delta:
            best_delta = delta
            best_swap = col_swapped_char_at_pos
    
    if best_swap is not None and best_delta < -tolerance:
        return (original_score + best_delta, best_swap)
    
    return (original_score, char_at_pos)


# def _random_position_swapping(
#     char_at_pos: tuple[int, ...], 
#     score: float, 
#     k: int,
#     order_1: tuple[tuple[np.ndarray, np.ndarray], ...], 
#     order_2: tuple[tuple[np.ndarray, np.ndarray], ...], 
#     order_3: tuple[tuple[np.ndarray, np.ndarray], ...],
#     pinned_positions: tuple[int, ...],
#     swap_position_pairs: tuple[tuple[int, int], ...],
#     population: dict[tuple[int, ...], float],
#     cached_scores: dict[tuple[int, ...], float]
# ) -> tuple[float, tuple[int, ...]]:
#     '''
#     randomly swap k positions
#     '''
#     previous_swapped_char_at_pos = char_at_pos

#     for i, j in random.sample(swap_position_pairs, len(swap_position_pairs)):
#         if i in pinned_positions or j in pinned_positions:
#             continue

#         swapped_char_at_pos = swap_char_at_pos(previous_swapped_char_at_pos, i, j)

#         if swapped_char_at_pos in cached_scores:
#             score = cached_scores[swapped_char_at_pos]
#         else:
#             delta = _calculate_swap_delta(order_1, order_2, order_3, previous_swapped_char_at_pos, i, j, swapped_char_at_pos)  # pyright: ignore[reportArgumentType]
#             score += delta
#             cached_scores[swapped_char_at_pos] = score

#         previous_swapped_char_at_pos = swapped_char_at_pos
    
#     return (score, swapped_char_at_pos)
