from model import KeyboardModel
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

    def push(self, delta: float, char_at_pos: Iterable[float]) -> None:
        '''
        push the item onto the heap, and if the heap is full, pop the worse delta item
        '''
        char_at_pos = tuple(char_at_pos)

        if char_at_pos in self.scores:
            # already in the population, no action
            return

        self.scores[char_at_pos] = delta

        # add salt to delta to avoid ties in the heap
        delta += delta * 0.001 * random.random()

        if len(self.heap) < self.max_size:
            heapq.heappush(self.heap, (-delta, char_at_pos))
        else:
            removed_delta, removed_char_at_pos = heapq.heappushpop(self.heap, (-delta, char_at_pos))
            if not self.cache_all:
                del self.scores[removed_char_at_pos]


    def sorted(self) -> list[np.ndarray]:
        '''
        return the items in the heap in sorted order
        '''
        return [char_at_pos for delta, char_at_pos in sorted(self.heap, reverse=True)]


    def random_item(self) -> np.ndarray:
        '''
        return a random item from the heap
        '''
        return random.choice(self.heap[1:])[1]


    def __len__(self) -> int:
        '''
        return the number of items in the heap
        '''
        return len(self.heap)

    def __contains__(self, char_at_pos: Iterable[float]) -> bool:
        '''
        return True if the item is in the heap
        '''
        return tuple(char_at_pos) in self.scores
    
    def __getitem__(self, char_at_pos: Iterable[float]) -> float:
        '''
        return the score of the item
        '''
        return self.scores[tuple(char_at_pos)]


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

    def optimize(self, char_at_pos: np.ndarray, score_tolerance = 0.1, iterations:int = 1) -> np.ndarray:

        while True:
            delta = self.position_swapping(char_at_pos)
            delta += self.column_swapping(char_at_pos)
            if -score_tolerance < delta:
                break
        
        return char_at_pos

    def position_swapping(self, char_at_pos: np.ndarray) -> float:
        '''
        simple hill climbing, checks every possible swap, and immediately
        accepts any swap that improves the score

        if N=len(char_at_pos), then there are N*(N-1)/2 swaps to try. We try all,
        sort them by deltas, and apply all that are possible.
        '''
        total_delta = 0
        swap_position_pairs = self.swap_position_pairs.copy()
        random.shuffle(swap_position_pairs)

        for i, j in swap_position_pairs:
            swapped_char_at_pos = tuple(
                char_at_pos[i] if k == j else 
                char_at_pos[j] if k == i else 
                char_at_pos[k] 
                for k in range(len(char_at_pos))
            )
            if swapped_char_at_pos in self.population:
                delta = self.population[swapped_char_at_pos]
            else:
                delta = self.model.calculate_swap_delta(char_at_pos, i, j)

            if delta < 0:
                char_at_pos[i], char_at_pos[j] = char_at_pos[j], char_at_pos[i]
                total_delta += delta
                self.population.push(total_delta, char_at_pos)
        
        return total_delta

    
    def column_swapping(self, char_at_pos: np.ndarray) -> float:
        '''
        simple hill climbing, swapping columns to improve the score the most
        '''
        best_delta = 0
        best_swap = None
        for col1, col2 in combinations(list(self.positions_at_column.keys()), 2):
            swap_len = min(len(self.positions_at_column[col1]), len(self.positions_at_column[col2]))
            
            delta = 0
            for pi1, pi2 in zip(self.positions_at_column[col1], self.positions_at_column[col2]):
                delta += self.model.calculate_swap_delta(char_at_pos, pi1, pi2)
                
            if delta < best_delta:
                best_delta = delta
                best_swap = (col1, col2)
        
        if best_swap is not None and best_delta < 0:
            for pi1, pi2 in zip(self.positions_at_column[best_swap[0]], self.positions_at_column[best_swap[1]]):
                char_at_pos[pi1], char_at_pos[pi2] = char_at_pos[pi2], char_at_pos[pi1]
                self.population.push(best_delta, char_at_pos)
        
        return best_delta