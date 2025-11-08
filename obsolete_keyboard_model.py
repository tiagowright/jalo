"""
keyboard_model.py

A NumPy-first model for optimizing keyboard layouts via character n-gram
frequencies and position-based metrics (1-, 2-, and 3-key).

Key features:
- Layout as a permutation vector: char_at_pos[p] = character index at position p.
- Character n-gram distributions (F1/F2/F3) are transformed into position n-grams
  (T1/T2/T3) via pure indexing.
- Dozens of metrics are pre-aggregated into three tensors V1_tot, V2_tot, V3_tot
  (one per order) for very fast scoring.
- Fast delta scoring for swaps: delta_swap(layout, p, q) computes ΔScore in
  O(1)+O(N)+O(N^2) for orders 1,2,3 respectively — great for local search/annealing.

Conventions:
- N = number of characters = number of physical positions (one-to-one assignment).
- F1[i]         = frequency of character i          (sum may be <= 1.0)
- F2[i,j]       = frequency of bigram i->j          (sum may be <= 1.0)
- F3[i,j,k]     = frequency of trigram i->j->k      (sum may be <= 1.0)
- V(order over POSITIONS):
    V1[p]           = cost/score for using position p
    V2[p,q]         = cost/score for consecutive positions p->q
    V3[p,q,r]       = cost/score for triple positions p->q->r
- Score to MINIMIZE:
    Score = <T1, V1_tot> + <T2, V2_tot> + <T3, V3_tot>
  where V*_tot are the pre-aggregated, weighted sums of your many metrics.
"""

from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence, Tuple
import numpy as np

# ---------------------------------------------------------------------------
# Metric model
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Metric:
    """
    A single metric defined over POSITIONS.

    Attributes
    ----------
    order : int
        1 for single-key metric, 2 for bigram metric, 3 for trigram metric.
        note that a skip grap is order 2, since the metric is only based on the first and third character
        
    V : np.ndarray
        Metric tensor over positions:
            order=1: shape (N,)
            order=2: shape (N, N)
            order=3: shape (N, N, N)
    """
    order: int
    V: np.ndarray


# ---------------------------------------------------------------------------
# Objective definition
# ---------------------------------------------------------------------------

class Objective:
    """
    Linear combination of metrics used to score layouts.

    Parameters
    ----------
    terms : Iterable[Tuple[Metric, float]]
        Each entry couples a Metric with its scalar weight in the objective.
    """

    def __init__(self, terms: Iterable[Tuple[Metric, float]]):
        self._terms: List[Tuple[Metric, float]] = [
            (metric, float(weight)) for metric, weight in terms
        ]
        if not self._terms:
            raise ValueError("Objective must contain at least one metric term.")

    @property
    def terms(self) -> Sequence[Tuple[Metric, float]]:
        """Return the metric-weight pairs in insertion order."""
        return tuple(self._terms)

    def preaggregate(self, N: int, dtype: np.dtype) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Reduce the objective terms into weighted tensors over positions.
        """
        V1 = np.zeros((N,), dtype=dtype)
        V2 = np.zeros((N, N), dtype=dtype)
        V3 = np.zeros((N, N, N), dtype=dtype)

        for metric, weight in self._terms:
        	if not weight:
        		continue
            if metric.order == 1:
                assert metric.V.shape == (N,), "Metric(order=1) must be shape (N,)"
                V1 += weight * np.asarray(metric.V, dtype=dtype)
            elif metric.order == 2:
                assert metric.V.shape == (N, N), "Metric(order=2) must be shape (N,N)"
                V2 += weight * np.asarray(metric.V, dtype=dtype)
            elif metric.order == 3:
                assert metric.V.shape == (N, N, N), "Metric(order=3) must be shape (N,N,N)"
                V3 += weight * np.asarray(metric.V, dtype=dtype)
            else:
                raise ValueError("Metric order must be 1, 2, or 3.")

        return V1, V2, V3


# ---------------------------------------------------------------------------
# Keyboard model
# ---------------------------------------------------------------------------

class KeyboardModel:
    """
    Keyboard layout scoring model with pre-aggregated metrics and fast swap deltas.

    Inputs
    ------
    F1, F2, F3 : Optional[np.ndarray]
        Character n-gram frequency distributions.
          - F1[i]           (shape: (N,))
          - F2[i,j]         (shape: (N,N))
          - F3[i,j,k]       (shape: (N,N,N))
        Any of them can be None if not used.

    objective : Objective
        Linear combination of metrics defined over POSITIONS, potentially dozens
        of mixed orders (1, 2, 3). Pre-aggregated once for scoring speed.

    dtype : np.dtype
        Numeric dtype for internal arrays (np.float64 for clarity,
        np.float32 for speed/memory).

    Layout representation
    ---------------------
    A layout is a permutation vector `char_at_pos` with shape (N,)
    where char_at_pos[p] = character index placed at position p.

    Core methods
    ------------
    - transform_ngrams(layout): maps character F* -> position T* via indexing
    - score_layout(layout):     returns scalar Score
    - delta_swap(layout,p,q):   returns ΔScore for swapping positions p and q
    - apply_swap_inplace(layout,p,q): performs the swap in the layout array
    """

    def __init__(
        self,
        F1: Optional[np.ndarray],
        F2: Optional[np.ndarray],
        F3: Optional[np.ndarray],
        objective: Objective,
        dtype: np.dtype = np.float64,
    ):
        # Store frequency tensors (character domain), possibly None
        self.F1 = None if F1 is None else np.asarray(F1, dtype=dtype)
        self.F2 = None if F2 is None else np.asarray(F2, dtype=dtype)
        self.F3 = None if F3 is None else np.asarray(F3, dtype=dtype)
        self.objective = objective
        self.dtype = dtype

        # Infer N from whichever F* is provided
        N = None
        if self.F1 is not None:
            N = self.F1.shape[0]
        if self.F2 is not None:
            N = self.F2.shape[0] if N is None else N
            assert self.F2.shape == (N, N), "F2 must be (N,N)"
        if self.F3 is not None:
            N = self.F3.shape[0] if N is None else N
            assert self.F3.shape == (N, N, N), "F3 must be (N,N,N)"
        assert N is not None, "At least one of F1/F2/F3 must be provided."
        self.N = N

        # Pre-aggregate the objective into three total tensors over POSITIONS
        self.V1_tot, self.V2_tot, self.V3_tot = self._preaggregate_objective(objective)

    # -------------------- Utilities --------------------

    @staticmethod
    def is_permutation(vec: np.ndarray) -> bool:
        """True iff vec contains each integer 0..N-1 exactly once."""
        return np.array_equal(np.sort(vec), np.arange(vec.size))

    # -------------------- Metrics pre-aggregation --------------------

    def _preaggregate_objective(self, objective: Objective):
        """
        Fold all objective terms into three tensors over POSITIONS:
          V1_tot (N,), V2_tot (N,N), V3_tot (N,N,N)
        """
        return objective.preaggregate(self.N, self.dtype)

    def update_objective(self, objective: Objective):
        """
        Replace the objective and recompute V1_tot/V2_tot/V3_tot.
        Call this when metric weights or combinations change.
        """
        self.objective = objective
        self.V1_tot, self.V2_tot, self.V3_tot = self._preaggregate_objective(objective)

    # -------------------- Character n-grams -> Position n-grams --------------------

    def transform_ngrams(
        self,
        char_at_pos: np.ndarray
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Convert character n-gram frequencies (F*) into position n-grams (T*)
        under the given layout (permutation vector).

        Returns
        -------
        (T1, T2, T3) where each is None if the corresponding F* was None.

        Mechanics
        ---------
        T1[p]       = F1[ char_at_pos[p] ]
        T2[p,q]     = F2[ char_at_pos[p], char_at_pos[q] ]
        T3[p,q,r]   = F3[ char_at_pos[p], char_at_pos[q], char_at_pos[r] ]
        """
        C = np.asarray(char_at_pos, dtype=int)
        assert C.shape == (self.N,)
        assert self.is_permutation(C)

        T1 = self.F1[C] if self.F1 is not None else None
        T2 = self.F2[np.ix_(C, C)] if self.F2 is not None else None
        if self.F3 is not None:
            idx = np.ix_(C, C, C)
            T3 = self.F3[idx]
        else:
            T3 = None
        return T1, T2, T3

    # -------------------- Scoring --------------------

    def score_layout(self, char_at_pos: np.ndarray) -> float:
        """
        Compute total Score = <T1, V1_tot> + <T2, V2_tot> + <T3, V3_tot>.
        Only contracts orders whose F* and V*_tot are present/nonzero.
        """
        T1, T2, T3 = self.transform_ngrams(char_at_pos)

        total = 0.0
        if T1 is not None and np.any(self.V1_tot):
            total += float(np.sum(T1 * self.V1_tot))
        if T2 is not None and np.any(self.V2_tot):
            total += float(np.sum(T2 * self.V2_tot))
        if T3 is not None and np.any(self.V3_tot):
            total += float(np.sum(T3 * self.V3_tot))
        return total

    # -------------------- Fast delta for swaps --------------------

    def delta_swap(self, char_at_pos: np.ndarray, p: int, q: int) -> float:
        """
        Compute ΔScore if we swap the characters at positions p and q, without
        recomputing the full score.

        Complexity
        ----------
        - Order-1: O(1)
        - Order-2: O(N)        (only rows/cols p and q change)
        - Order-3: O(N^2)      (only planes touching p or q change; no overlaps)

        Parameters
        ----------
        char_at_pos : np.ndarray
            Current layout permutation vector.
        p, q : int
            Distinct positions to swap.

        Returns
        -------
        float : The change in Score (new - old) if we execute the swap.
        """
        C = np.asarray(char_at_pos, dtype=int)
        N = self.N
        assert C.shape == (N,)
        assert self.is_permutation(C)
        assert 0 <= p < N and 0 <= q < N and p != q

        i = C[p]  # character currently at position p
        j = C[q]  # character currently at position q

        delta = 0.0

        # ---- Order-1 (O(1)) ----
        if self.F1 is not None and np.any(self.V1_tot):
            # T1[p] changes from F1[i] to F1[j], T1[q] vice versa
            # Contribution: (F1[j]-F1[i])*V1[p] + (F1[i]-F1[j])*V1[q]
            delta += (self.F1[j] - self.F1[i]) * (self.V1_tot[p] - self.V1_tot[q])

        # ---- Order-2 (O(N)) ----
        if self.F2 is not None and np.any(self.V2_tot):
            # Row changes: p and q as first index
            cb = C  # char at each position b
            # Row p: F2[j, cb] - F2[i, cb]
            delta += float(np.sum((self.F2[j, cb] - self.F2[i, cb]) * self.V2_tot[p, :]))
            # Row q: F2[i, cb] - F2[j, cb]
            delta += float(np.sum((self.F2[i, cb] - self.F2[j, cb]) * self.V2_tot[q, :]))

            # Column changes: p and q as second index, exclude a in {p,q} to avoid double count
            mask = np.ones(N, dtype=bool)
            mask[[p, q]] = False
            ca = C[mask]  # chars at positions a ≠ p,q
            # Column p: F2[ca, j] - F2[ca, i]
            delta += float(np.sum((self.F2[ca, j] - self.F2[ca, i]) * self.V2_tot[mask, p]))
            # Column q: F2[ca, i] - F2[ca, j]
            delta += float(np.sum((self.F2[ca, i] - self.F2[ca, j]) * self.V2_tot[mask, q]))

        # ---- Order-3 (O(N^2)) ----
        if self.F3 is not None and np.any(self.V3_tot):
            # We touch only planes that include p or q along any axis.
            # Build helper arrays for broadcasting chars at other positions.
            Bb = C[:, None]    # (N,1)
            Cc = C[None, :]    # (1,N)

            # Axis-0 planes: indices (p,*,*) and (q,*,*)
            # Replace char i->j at plane p; j->i at plane q
            plane_new_p = self.F3[j, Bb, Cc]  # (N,N)
            plane_old_p = self.F3[i, Bb, Cc]
            delta += float(np.sum((plane_new_p - plane_old_p) * self.V3_tot[p, :, :]))

            plane_new_q = self.F3[i, Bb, Cc]
            plane_old_q = self.F3[j, Bb, Cc]
            delta += float(np.sum((plane_new_q - plane_old_q) * self.V3_tot[q, :, :]))

            # Axis-1 planes: indices (*,p,*) and (*,q,*), restrict a≠{p,q}
            mask_a = np.ones(N, dtype=bool)
            mask_a[[p, q]] = False
            Ca = C[mask_a]  # chars at positions a ≠ p,q

            mat_new_p = self.F3[Ca[:, None], j, C[None, :]]  # (N-2, N)
            mat_old_p = self.F3[Ca[:, None], i, C[None, :]]
            delta += float(np.sum((mat_new_p - mat_old_p) * self.V3_tot[mask_a, p, :]))

            mat_new_q = self.F3[Ca[:, None], i, C[None, :]]
            mat_old_q = self.F3[Ca[:, None], j, C[None, :]]
            delta += float(np.sum((mat_new_q - mat_old_q) * self.V3_tot[mask_a, q, :]))

            # Axis-2 planes: indices (*,*,p) and (*,*,q), restrict a,b≠{p,q}
            mask_ab = mask_a
            Ca2 = C[mask_ab]
            Cb2 = C[mask_ab]

            grid_new_p = self.F3[Ca2[:, None], Cb2[None, :], j]  # (N-2, N-2)
            grid_old_p = self.F3[Ca2[:, None], Cb2[None, :], i]
            V_edge_p   = self.V3_tot[np.ix_(mask_ab, mask_ab, [p])][:, :, 0]
            delta += float(np.sum((grid_new_p - grid_old_p) * V_edge_p))

            grid_new_q = self.F3[Ca2[:, None], Cb2[None, :], i]
            grid_old_q = self.F3[Ca2[:, None], Cb2[None, :], j]
            V_edge_q   = self.V3_tot[np.ix_(mask_ab, mask_ab, [q])][:, :, 0]
            delta += float(np.sum((grid_new_q - grid_old_q) * V_edge_q))

        return delta

    # -------------------- Convenience helpers --------------------

    @staticmethod
    def apply_swap_inplace(char_at_pos: np.ndarray, p: int, q: int):
        """Swap positions p and q in the layout array (in place)."""
        char_at_pos[p], char_at_pos[q] = char_at_pos[q], char_at_pos[p]

    def score_after_swap(self, char_at_pos: np.ndarray, p: int, q: int) -> float:
        """
        Return score(layout) + delta_swap(layout,p,q) without mutating layout.
        Useful for "peek" evaluation.
        """
        return self.score_layout(char_at_pos) + self.delta_swap(char_at_pos, p, q)


# ---------------------------------------------------------------------------
# Example / smoke test (defines F and V ONCE)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from optimizer import SimulatedAnnealingOptimizer

    rng = np.random.default_rng(42)
    N = 8  # toy example size

    # --- Character n-gram frequencies (define ONCE) ---
    # In your application, load real frequencies and (optionally) normalize so sum<=1.
    F1 = rng.random(N)
    F1 = F1 / F1.sum()

    F2 = rng.random((N, N))
    F2 = F2 / F2.sum()

    F3 = rng.random((N, N, N))
    F3 = F3 / F3.sum()

    # --- Position metrics (define ONCE) ---
    # V1: single-key comfort per position (toy: small gradient)
    V1 = np.linspace(1.0, 0.8, N)

    # V2: bigram "distance" between positions (toy: positions on a 2x4 grid)
    coords = np.array([(r, c) for r in range(2) for c in range(4)])[:N]
    D = np.linalg.norm(coords[:, None, :] - coords[None, :, :], axis=-1)
    V2 = D  # larger distance = larger cost (example)

    # V3: trigram "same-finger adjacency" penalty (toy: penalize repeats on neighbor steps)
    V3 = np.zeros((N, N, N))
    V3 += (np.eye(N)[:, :, None] + np.eye(N)[:, None, :]) * 0.5

    # Bundle many metrics (you could have dozens). Here we just use 3 with weights.
    metrics: List[Metric] = [
        Metric(order=1, V=V1),
        Metric(order=2, V=V2),
        Metric(order=3, V=V3),
    ]
    objective = Objective(
        [
            (metrics[0], 0.1),
            (metrics[1], 1.0),
            (metrics[2], 0.7),
        ]
    )

    # Build the model (pre-aggregates metrics once)
    model = KeyboardModel(F1=F1, F2=F2, F3=F3, objective=objective, dtype=np.float32)

    # Start from identity layout and compute score
    layout = np.arange(N, dtype=int)
    base_score = model.score_layout(layout)
    print(f"[base] score = {base_score:.6f}")

    # Pick a swap and evaluate fast delta
    p, q = 1, 6
    delta = model.delta_swap(layout, p, q)
    # Verify against full recompute after applying the swap
    layout_swapped = layout.copy()
    model.apply_swap_inplace(layout_swapped, p, q)
    full_score = model.score_layout(layout_swapped)

    print(f"[swap {p}<->{q}]  delta = {delta:.6f}")
    print(f"[swap {p}<->{q}]  full  = {full_score:.6f}")
    print(f"[check] base + delta = {base_score + delta:.6f}")
    print(f"[error] abs diff     = {abs((base_score + delta) - full_score):.6e}")

    # Run a short annealing pass starting from the identity layout
    optimizer = SimulatedAnnealingOptimizer(model, rng=rng)
    stats = optimizer.optimize(
        layout,
        temperature_start=0.5,
        temperature_end=1e-3,
        sweeps=50,
        moves_per_sweep=4 * N,
    )
    print(f"[anneal] best score = {stats.best_score:.6f}")
