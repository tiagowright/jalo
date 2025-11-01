# optimizer.py
#
# Simulated annealing loop tailored to KeyboardModel.  Relies on the model's
# fast delta scoring for swaps so we can explore permutations efficiently.

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple
import math
import numpy as np
from bisect import bisect_left

from keyboard_model import KeyboardModel


@dataclass
class AnnealingStats:
    """Summary metrics produced by the optimizer."""

    best_score: float
    best_layout: np.ndarray
    evaluations: int
    accepted_moves: int
    temperature_schedule: List[float]
    top_layouts: Sequence[Tuple[float, np.ndarray]]


class SimulatedAnnealingOptimizer:
    """
    Simulated annealing on keyboard layouts using swap moves.

    Parameters
    ----------
    model : KeyboardModel
        Pre-configured scoring model.
    rng : Optional[np.random.Generator]
        Custom RNG for reproducibility; defaults to Generator(PCG64()).
    """

    def __init__(self, model: KeyboardModel, rng: Optional[np.random.Generator] = None):
        self.model = model
        self.rng = rng or np.random.default_rng()

    def optimize(
        self,
        initial_layout: np.ndarray,
        *,
        temperature_start: float = 1.0,
        temperature_end: float = 1e-3,
        sweeps: int = 250,
        moves_per_sweep: Optional[int] = None,
        reheat_factor: float = 1.0,
        top_k: int = 10,
    ) -> AnnealingStats:
        """
        Run simulated annealing and return the best layout discovered.

        Parameters
        ----------
        initial_layout : np.ndarray
            Permutation vector representing the starting layout.
        temperature_start : float
            Initial temperature; higher -> more exploratory.
        temperature_end : float
            Final temperature (geometric cooling). Must be > 0.
        sweeps : int
            Number of temperature steps. Each sweep performs `moves_per_sweep` swaps.
        moves_per_sweep : Optional[int]
            How many swap proposals per sweep. Defaults to N * 4 (heuristic).
        reheat_factor : float
            If >1, temperature is briefly reheated whenever no move accepted in a sweep.
        top_k : int
            Number of best layouts to keep track of (sorted by score). Default 10.

        Returns
        -------
        AnnealingStats : best score/layout plus diagnostics.
        """
        layout = np.asarray(initial_layout, dtype=int).copy()
        assert layout.shape == (self.model.N,)
        assert self.model.is_permutation(layout)
        assert temperature_start > 0 and temperature_end > 0
        assert sweeps > 0
        assert top_k > 0

        current_score = self.model.score_layout(layout)
        top_layouts: List[Tuple[float, np.ndarray]] = []
        self._record_layout(top_layouts, current_score, layout, top_k)

        move_budget = moves_per_sweep or (self.model.N * 4)
        evaluations = 0
        accepted_moves = 0
        temperatures: List[float] = []

        # Pre-compute geometric cooling ratio
        cooling_ratio = (temperature_end / temperature_start) ** (1.0 / max(sweeps - 1, 1))
        temperature = temperature_start

        for sweep_idx in range(sweeps):
            temperatures.append(temperature)
            accepted_this_sweep = 0

            for _ in range(move_budget):
                p, q = self._random_swap_indices()
                delta = self.model.delta_swap(layout, p, q)
                evaluations += 1

                if delta <= 0 or self._accept_move(delta, temperature):
                    self.model.apply_swap_inplace(layout, p, q)
                    current_score += delta
                    accepted_moves += 1
                    accepted_this_sweep += 1
                    self._record_layout(top_layouts, current_score, layout, top_k)

            if accepted_this_sweep == 0 and reheat_factor > 1.0:
                temperature = min(temperature_start, temperature * reheat_factor)
            else:
                temperature *= cooling_ratio
                temperature = max(temperature, temperature_end)

        best_score, best_layout = top_layouts[0]

        return AnnealingStats(
            best_score=float(best_score),
            best_layout=best_layout,
            evaluations=evaluations,
            accepted_moves=accepted_moves,
            temperature_schedule=temperatures,
            top_layouts=tuple(top_layouts),
        )

    def _random_swap_indices(self) -> Tuple[int, int]:
        """Pick two distinct positions uniformly at random."""
        i, j = self.rng.integers(0, self.model.N, size=2, endpoint=False)
        while i == j:
            j = int(self.rng.integers(0, self.model.N))
        return int(i), int(j)

    def _accept_move(self, delta: float, temperature: float) -> bool:
        """Metropolis acceptance rule."""
        prob = math.exp(-delta / temperature)
        return bool(self.rng.random() < prob)

    @staticmethod
    def _record_layout(
        top_layouts: List[Tuple[float, np.ndarray]],
        score: float,
        layout: np.ndarray,
        limit: int,
    ):
        """Insert layout into top-k list if it qualifies."""
        new_entry = (float(score), layout.copy())

        # Build sorted insertion by score (ascending)
        scores = [s for s, _ in top_layouts]
        idx = bisect_left(scores, new_entry[0])
        top_layouts.insert(idx, new_entry)

        # Remove duplicates with identical score and layout to avoid ballooning
        deduped: List[Tuple[float, np.ndarray]] = []
        for s, arr in top_layouts:
            if not any(np.array_equal(arr, existing) and abs(existing_score - s) < 1e-12 for existing_score, existing in deduped):
                deduped.append((s, arr))
        top_layouts.clear()
        top_layouts.extend(deduped[:limit])
