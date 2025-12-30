# Reverse-Engineering Layout Scoring

This document sketches a practical workflow for inferring an objective
(`objective.py`) that explains a single target layout. The goal is to learn a
linear weight vector over the existing metrics such that the target layout is
optimal (or near-optimal) inside its neighborhood, and layouts generated under
that objective resemble the target.

## Observations We Can Leverage
- `model.KeyboardModel` already exposes every metric contribution before they
  are aggregated by the objective. We can therefore inspect the metric vector
  `m(layout)` by calling each metric in `metrics.METRICS`.
- `solvers.helper` exposes deterministic swap utilities, so we can generate
  controlled perturbations around the target layout and compute both their
  metric vectors and their Hamming distance to the target.
- Optimizers (`optim.py`) expect scores that are linear combinations of the same
  metric set, so once we estimate weights we can reuse all existing tooling.

## Step 1 — Collect Metric Data
1. **Baseline vector**: Evaluate every metric on the target layout. Store this
   as `m*`.
2. **Local samples**: Use `helper.swap_char_at_pos` or `Optimizer.improve` with
   a large `hamming_distance_threshold` to enumerate hundreds/thousands of
   nearby layouts. Record `(layout_id, hamming_dist, metrics_vector)`.
3. **Global anchors (optional)**: Sample layouts drawn from known objectives
   (existing TOML files) so the solver learns what *bad* layouts look like.
4. Persist everything in a dataframe: columns per metric plus metadata (distance,
   original score, etc.). This will feed the optimization problem.

## Step 2 — Frame Weight Recovery as a Ranking Problem
We want a weight vector `w` such that the target scores better than neighbors:

```
wᵀ m* + margin ≤ wᵀ m_i    for every sampled neighbor layout i
```

This can be solved via convex optimization by minimizing a hinge or logistic
loss over these inequalities. One convenient formulation (hinge loss with L1
regularizer) is:

```
min_w  Σ_i max(0, margin + wᵀ(m* - m_i))  +  λ ||w||₁
```

Additional constraints:
- Clamp weights to reasonable ranges (e.g., `w_j ≥ 0` or allow signed weights
  if the user expects metrics that reward/penalize behavior).
- Normalize (`Σ_j |w_j| = 1`) so scaling does not change behavior.

Because the metrics are deterministic, we can use `cvxpy`, scikit-learn’s
`SGDClassifier` (with `hinge` loss and `penalty='l1'`), or even a custom solver
to minimize the objective. The right-hand side uses metric differences rather
than raw scores, so we do not need to know the unknown constant term from the
original objective.

## Step 3 — Regularization and Metric Selection
- Prefer **L1 (lasso) penalty** on the weights to drive many to zero. This
  aligns with the existing objectives where most terms are unused.
- For groups of related metrics (e.g., per-finger stats) you can optionally add
  a group-lasso penalty so entire families drop out together.
- Enforce sparsity explicitly by pruning metrics with |w_j| below a threshold,
  then refitting on the reduced set to remove collinearity.

## Step 4 — Validation Loop
1. Build a temporary `ObjectiveFunction` with the learned weights.
2. Re-run `Optimizer.improve` on the target layout. The layout should remain the
   population best; if not, add the counter-example to the dataset and refit.
3. Run `Optimizer.generate` using the learned objective. Inspect the population
   (and Hamming clusters) to confirm that the generated layouts closely match
   the target or at least remain within the acceptable distance threshold.
4. Compare ranking stability: ensure that metric gradients (`_calculate_swap_delta`)
   push neighbors back toward the target.

## Practical Tips
- **Balance local and distant samples**: include some layouts that are clearly
  worse than the target so the solver learns the global direction of each
  metric.
- **Use margins**: a positive margin (e.g., 0.01 × baseline score) enforces
  separation and prevents degenerate weight solutions.
- **Iterative refinement**: alternate between fitting weights and regenerating
  counter-examples to tighten the approximation (i.e., online convex
  programming).
- **Document assumptions**: when saving the learned weights, record the layout
  used, sampling radius, number of constraints, λ, and whether weights are
  constrained to be non-negative.

Following this pipeline yields an objective whose gradient landscape matches the
observed preference for the target layout. Because we stay inside the existing
linear-metric framework, all current optimizers (`generate`, `improve`, custom
solvers) can use the reconstructed objective without additional changes.
