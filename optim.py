from model import KeyboardModel, NgramType, _calculate_swap_delta
from freqdist import FreqDist
from layout import KeyboardLayout
from itertools import combinations
from typing import List, Tuple, Optional, Any, Iterable
import numpy as np
import random
import logging
import heapq

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
        # char_at_pos = tuple(char_at_pos)

        if char_at_pos in self.scores:
            # already in the population, no action
            return

        self.scores[char_at_pos] = score

        # add salt to delta to avoid ties in the heap
        score += score * 0.001 * random.random()

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


def swap_char_at_pos(char_at_pos: tuple[int, ...], i: int, j: int) -> tuple[int, ...]:
    return tuple(
        char_at_pos[i] if k == j else
        char_at_pos[j] if k == i else
        char_at_pos[k]
        for k in range(len(char_at_pos))
    )

class Optimizer:
    def __init__(self, model: KeyboardModel, population_size: int = 1000):
        self.model = model

        self.positions_at_column = {}
        for pi, position in enumerate(self.model.hardware.positions):
            if position.col not in self.positions_at_column:
                self.positions_at_column[position.col] = []
            self.positions_at_column[position.col].append(pi)
        
        self.swap_position_pairs = list(combinations(range(len(self.model.hardware.positions)), 2))

        self.population = Population(max_size=population_size)

    def generate(self, char_seq: list[str], iterations:int = 100, optimizer_iterations:int = 20):
        assert len(char_seq) == len(self.model.hardware.positions)

        char_at_pos = np.zeros(len(self.model.hardware.positions), dtype=int)
        for pi, position in enumerate(self.model.hardware.positions):
            char = char_seq[pi]
            try:
                char_at_pos[pi] = self.model.freqdist.char_seq.index(char)
            except ValueError:
                char_at_pos[pi] = self.model.freqdist.char_seq.index(FreqDist.out_of_distribution)

        initial_positions = [
            np.random.permutation(char_at_pos)
            for _ in range(iterations)
        ]

        for current_char_at_position in initial_positions:
            self.optimize(current_char_at_position, iterations=optimizer_iterations)


    def optimize(self, char_at_pos: np.ndarray, score_tolerance = 0.01, iterations:int = 20, pinned_positions: tuple[int, ...] = ()):
        F = self.model.freqdist.to_numpy()
        V = self.model.V

        initial_char_at_pos = tuple(int(x) for x in char_at_pos)
        initial_score = self.model.score_chars_at_positions(char_at_pos)
        tolerance = score_tolerance * initial_score

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


        for i in range(iterations):
            # print(f"iteration {i+1} of {iterations}")
            current_char_at_pos = initial_char_at_pos
            current_score = initial_score

            while True:
                score_at_start_of_step = current_score
                current_score, current_char_at_pos = self.position_swapping(current_char_at_pos, current_score, tolerance, order_1, order_2, order_3, pinned_positions)
                current_score, current_char_at_pos = self.column_swapping(current_char_at_pos, current_score, tolerance, order_1, order_2, order_3, pinned_positions)

                # print(f"delta: {delta} vs tolerance: {tolerance}")
                delta = current_score - score_at_start_of_step
                if -tolerance < delta:
                    break
        

    def position_swapping(
        self, 
        char_at_pos: tuple[int, ...], 
        score: float, 
        tolerance: float,
        order_1: tuple[tuple[np.ndarray, np.ndarray], ...], 
        order_2: tuple[tuple[np.ndarray, np.ndarray], ...], 
        order_3: tuple[tuple[np.ndarray, np.ndarray], ...],
        pinned_positions: tuple[int, ...]
    ) -> tuple[float, tuple[int, ...]]:
        '''
        simple hill climbing, checks every possible swap, and immediately
        accepts any swap that improves the score

        if N=len(char_at_pos), then there are N*(N-1)/2 swaps to try. We try all,
        sort them by deltas, and apply all that are possible.
        '''
        original_score = score
        swap_position_pairs = self.swap_position_pairs.copy()
        random.shuffle(swap_position_pairs)

        for i, j in swap_position_pairs:
            if i in pinned_positions or j in pinned_positions:
                continue

            swapped_char_at_pos = swap_char_at_pos(char_at_pos, i, j)
            if swapped_char_at_pos in self.population:
                swapped_score = self.population[swapped_char_at_pos]
                delta = swapped_score - score
            else:
                delta = _calculate_swap_delta(order_1, order_2, order_3, char_at_pos, i, j, swapped_char_at_pos)  # pyright: ignore[reportArgumentType]

            # accept the swap if it improves the score
            if delta < -tolerance:
                char_at_pos = swapped_char_at_pos
                score += delta
                self.population.push(score, char_at_pos)
        
        return (score, char_at_pos)

    
    def column_swapping(
        self, 
        char_at_pos: tuple[int, ...], 
        score: float, 
        tolerance: float,
        order_1: tuple[tuple[np.ndarray, np.ndarray], ...], 
        order_2: tuple[tuple[np.ndarray, np.ndarray], ...], 
        order_3: tuple[tuple[np.ndarray, np.ndarray], ...],
        pinned_positions: tuple[int, ...]
    ) -> tuple[float, tuple[int, ...]]:

        '''
        simple hill climbing, swapping columns to improve the score the most
        '''
        original_score = score
        best_delta = 0
        best_swap = None

        for col1, col2 in combinations(list(self.positions_at_column.keys()), 2):
            swap_len = min(len(self.positions_at_column[col1]), len(self.positions_at_column[col2]))

            # compute the score after swapping all positions in col1 and col2    
            col_swapped_char_at_pos = char_at_pos
            delta = 0
            for pi1, pi2 in zip(self.positions_at_column[col1], self.positions_at_column[col2]):
                if pi1 in pinned_positions or pi2 in pinned_positions:
                    delta = 0
                    break

                next_swap_char_at_pos = swap_char_at_pos(col_swapped_char_at_pos, pi1, pi2)
                delta += _calculate_swap_delta(order_1, order_2, order_3, col_swapped_char_at_pos, pi1, pi2, next_swap_char_at_pos)  # pyright: ignore[reportArgumentType]
                col_swapped_char_at_pos = next_swap_char_at_pos
                
            if delta < best_delta:
                best_delta = delta
                best_swap = (col1, col2)
        
        if best_swap is not None and best_delta < -tolerance:
            # accept the swap
            for pi1, pi2 in zip(self.positions_at_column[best_swap[0]], self.positions_at_column[best_swap[1]]):
                char_at_pos = swap_char_at_pos(char_at_pos, pi1, pi2)

            self.population.push(original_score + best_delta, char_at_pos)
        
        return (original_score + best_delta, char_at_pos)