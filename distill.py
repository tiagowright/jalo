#!/usr/bin/env python
"""
`distill.py` contains functionality to reverse-engineer objective functions from target layouts.
Given a target keyboard layout, it learns a linear weight vector over metrics such that
the target layout is optimal (or near-optimal) within its neighborhood.

`distill.py` can be used from the command line to distill objectives from layout files.
Try `python distill.py --help` to see the available command line options.
"""

import os
import sys
import argparse
import random
import numpy as np
import itertools
import warnings

from itertools import combinations
from dataclasses import dataclass
import numpy as np
import cvxpy as cp
import heapq

from solvers import helper
from layout import KeyboardLayout
from model import KeyboardModel
from objective import ObjectiveFunction
from optim import Optimizer
from metrics import METRICS


@dataclass
class DistillationParams:
    tolerance: float = 1e-4

    nearby_samples: int = 1000
    nearby_samples_max_depth: int = 3

    misclassification_penalty: float = 100.0  # C: ranges 1.0 to 100.0
    stability_penalty: float = 1000.0  # l2_reg (was 0.05) ranges 0.01 to 10.0
    sparsity_penalty: float = 1.0  # l1_reg ranges 0.001 to 1.0


class Distillator:
    """
    A distillator that learns objective functions from target layouts.
    
    Given a target keyboard layout, the distillator samples nearby layouts and learns
    a weight vector over metrics such that the target layout scores better than its neighbors.
    """
    
    def __init__(self, model: KeyboardModel):
        """
        Initialize a Distillator with a KeyboardModel.
        
        Parameters
        ----------
        model : KeyboardModel
            The keyboard model containing hardware, metrics, and frequency distribution.
        """
        self.model = model
        self.swap_position_pairs = tuple(combinations(range(len(self.model.hardware.positions)), 2))


    def _all_swaps(self, char_at_pos: tuple[int, ...]) -> list[tuple[int, ...]]:
        """
        Generate all possible swaps for a given layout.
        """
        return [
            helper.swap_char_at_pos(char_at_pos, i, j)
            for i, j in self.swap_position_pairs
        ]


    def _next_neighborhood(
        self, 
        model: KeyboardModel,
        scores: dict[tuple[int, ...], float], 
        neighborhood: list[tuple[int, ...]], 
        n: int = 500
    ) -> list[tuple[int, ...]]:
        """
        Sample from the next neighborhood of layouts
        """
        optimizer = Optimizer(model)
        order1, order2, order3 = optimizer._get_FV()
        
        # get best n layouts from the current neighborhood
        seeds = heapq.nlargest(n, neighborhood, key=lambda x: -scores[x])

        # for each layout, find the best swaps that improves the score
        next_scores_items = []
        scores_cache = scores.copy()
        for seed in seeds:
            layout_scores, _ = helper.best_swaps(
                seed, scores[seed], order1, order2, order3, self.swap_position_pairs, scores_cache, 1, 1
            )

            # keep the n best (new) layouts
            next_scores_items = heapq.nlargest(
                n, 
                itertools.chain(
                    [
                        # remove any layout that is already known in the scores
                        x for x in layout_scores.items() if x[0] not in scores
                    ], 
                    next_scores_items
                ), 
                key=lambda x: -x[1]
            )
        
        return [x[0] for x in next_scores_items]


    def _estimate_weights(self, M_all: np.ndarray, params: DistillationParams) -> tuple[np.ndarray, np.ndarray]:
        """
        Estimate the weights that make the target layout optimal

        M_all is n x m array of the m metric values for n layouts
        M_all[0, :] is 1 x m array of the m metric values for the target layout

        Returns a 1 x m array of the weights, and 1 x m array of the scores
        """

        m = M_all.shape[1]
        n = M_all.shape[0]

        # normalize the metric values
        mu = np.mean(M_all, axis=0)
        sigma = np.std(M_all, axis=0)
        sigma[sigma < params.tolerance] = 1.0 
        M_norm = (M_all - mu) / sigma

        M0_norm = M_norm[0, :]
        Mi_norm = M_norm[1:, :]
        diff = Mi_norm - M0_norm 

        # Variables
        w_norm = cp.Variable(m)
        slack = cp.Variable(n-1)

        # Hyperparameters
        C = params.misclassification_penalty      # Misclassification/Ranking penalty
        l1_reg = params.sparsity_penalty          # Sparsity
        l2_reg = params.stability_penalty         # Stability (L2)

        # Objective with Elastic Net
        # Note: we use cp.sum_squares(w_norm) for the L2 term ( ||w||_2^2 )
        objective = cp.Minimize(
            l1_reg * cp.norm(w_norm, 1) + 
            l2_reg * cp.sum_squares(w_norm) + 
            # C * cp.sum(slack)
            C * cp.sum(slack) / (n - 1)
        )

        # Constraints
        constraints = [
            diff @ w_norm >= 1 - slack,
            slack >= 0,
            w_norm >= -1,
            w_norm <= 1
        ]

        # Solve
        # OSQP is particularly efficient for Quadratic Programs (which L2 introduces)
        prob = cp.Problem(objective, constraints)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning, module="cvxpy")
            prob.solve(solver=cp.OSQP)

        # Unnormalize to original scale
        w_raw = w_norm.value / sigma

        # compute the scores
        S = M_all @ w_raw

        return w_raw, S


    def distill(
        self, 
        target_layout: KeyboardLayout, 
        depth: int = 8,
        samples_per_layer = 500,
        params: DistillationParams = DistillationParams()

    ) -> tuple[ObjectiveFunction, dict[tuple[int, ...], float]]:
        """
        Learn an objective function that explains why the target layout is optimal.
        
        This method implements the workflow described in WIP/DISTILL.md:
        1. Collect metric data from the target layout and nearby samples
        2. Frame weight recovery as a ranking problem
        3. Solve for weights using optimization
        4. Return an ObjectiveFunction with the learned weights
        
        Parameters
        ----------
        target_layout : KeyboardLayout
            The target layout to reverse-engineer an objective for.
        iterations : int
            Number of iterations or samples to use in the learning process.
            
        Returns
        -------
        ObjectiveFunction
            An objective function with learned weights that makes the target layout optimal.
        Scores: dict[tuple[int, ...], float]
            A dictionary of char_at_pos -> scores for each layout that is better than the target layout 
            for the learned objective function.
        """
        char_at_pos = tuple(self.model.char_at_positions_from_layout(target_layout))
        neighborhood = self._all_swaps(char_at_pos)

        # note that the first layout in this list MUST be the target
        all_layouts = [char_at_pos] + neighborhood
        weights_prev = np.zeros(len(self.model.metrics))

        for di in range(depth):
            M = np.array([
                [
                    metric_values.get(metric, 0)
                    for metric in self.model.metrics
                ]
                for metric_values in [
                    self.model.metric_values_from_char_at_positions(char_at_pos) 
                    for char_at_pos in all_layouts
                ]
            ])

            weights, S = self._estimate_weights(M, params)

            scores = {
                char_at_pos: score
                for char_at_pos, score in zip(all_layouts, S)
            }


            # DEBUG data
            count_better_layouts = np.sum(S[1:] <= scores[char_at_pos])
            best_score = np.min(S[1:])
            diff_square = np.sum((weights - weights_prev) ** 2)
            print(f"Depth {di}, target score: {scores[char_at_pos]:>8.3f}, best score: {best_score:>8.3f}, counting {count_better_layouts:>4d} / {len(S[1:]):>4d}., diff_square: {diff_square:>8.3f}")

            objective = ObjectiveFunction({
                metric: weight
                for metric, weight in zip(self.model.metrics, weights)
                if abs(weight) > params.tolerance
            })

            if di >= depth - 1:
                # save some compute that is not needed searching for the next neighbors
                break

            model = KeyboardModel(
                hardware=self.model.hardware,
                metrics=self.model.metrics,
                objective=objective,
                freqdist=self.model.freqdist
            )

            neighborhood = self._next_neighborhood(model, scores, neighborhood, samples_per_layer)
            all_layouts.extend(neighborhood)
            weights_prev = weights


        return (objective, {
            layout: score 
            for layout, score in scores.items()
            if score < scores[char_at_pos] and layout != char_at_pos
        })




if __name__ == "__main__":
    """
    distill.py can be run as a standalone script from CLI to distill objectives from layouts.
    
    Example usage:
        python distill.py --layout qwerty --hardware ansi --corpus en --iterations 1000
    """

    import tomllib


    @dataclass
    class RunConfig:
        name: str
        name_format: str
        layout: str
        hardware: str
        corpus: str
        depth: int
        samples_per_layer: int
        solver_args: dict

        def __post_init__(self):
            if not self.name and self.name_format:
                all_fields = self.__dict__.copy()
                all_fields.update(self.solver_args)

                try:
                    self.name = self.name_format.format(**all_fields)
                except KeyError as e:
                    # this is usually ok when the RunConfig is a default, but not if it's a user-specified run
                    self.name = f'__name_format_error_key_{e.args[0]}_not_found__'

        @classmethod
        def from_dict(cls, data: dict, defaults: "RunConfig") -> "RunConfig":
            # solver_args are all the keys in the data or defaults that are not one of the named fields in this class
            solver_args = defaults.solver_args.copy()
            solver_args.update({k: v for k, v in data.items() if k not in cls.__dataclass_fields__.keys()})

            return cls(
                name=data.get("name", ''),
                name_format=data.get("name_format", defaults.name_format),
                layout=data.get("layout", defaults.layout),
                hardware=data.get("hardware", defaults.hardware),
                corpus=data.get("corpus", defaults.corpus), 
                depth=data.get("depth", defaults.depth),
                samples_per_layer=data.get("samples_per_layer", defaults.samples_per_layer),
                solver_args=solver_args,
            )
        
        @classmethod
        def runs_from_config(cls, config: dict, defaults: "RunConfig") -> list["RunConfig"]:
            if "default" in config:
                defaults = cls.from_dict(config["default"], defaults)

            runs = []
            for run in config.get("runs", []):
                runs.append(cls.from_dict(run, defaults))

            return runs

        def __str__(self) -> str:
            return "\n".join([
                f"name = {self.name}",
                f"depth = {self.depth}",
                f"samples_per_layer = {self.samples_per_layer}",
                f"solver_args = {self.solver_args}",
                f"hardware = {self.hardware}",
                f"corpus = {self.corpus}",
            ])


    defaults = RunConfig(
        name='',
        layout="sturdy",
        name_format=r'{layout}_{depth}_{samples_per_layer}',
        hardware="ansi",
        corpus="en",
        depth=8,
        samples_per_layer=500,
        solver_args={}
    )


    parser = argparse.ArgumentParser(
        description="Distill objective functions from target keyboard layouts."
    )
    parser.add_argument("config", type=str, help="Path to the configuration file.")
    args = parser.parse_args()

    # Ensure the parent directory is on sys.path
    parent_dir = os.path.dirname(__file__)
    if parent_dir not in sys.path:
        sys.path.insert(0, parent_dir)


    try:
        with open(args.config, "rb") as f:
            try:
                config_dict = tomllib.load(f)
            except tomllib.TOMLDecodeError as e:
                print(f"Error parsing config file {args.config}: {e}")
                sys.exit(1)
    except FileNotFoundError:
        print(f"Error: Config file not found in {args.config}")
        sys.exit(1)
    
    runs = RunConfig.runs_from_config(config_dict, defaults)

    # Import required modules
    from hardware import KeyboardHardware
    from freqdist import FreqDist
    from objective import ObjectiveFunction
    
    try:  
        for run in runs:
            print(f"Starting run: {run.name}")
            hardware = KeyboardHardware.from_name(run.hardware)
            freqdist = FreqDist.from_name(run.corpus)
            initial_objective = ObjectiveFunction.from_name("default")
            target_layout = KeyboardLayout.from_name(run.layout, hardware=hardware)
            model = KeyboardModel(
                hardware=hardware,
                metrics=METRICS,
                objective=initial_objective,
                freqdist=freqdist
            )

            params = DistillationParams()
            for key, value in run.solver_args.items():
                if hasattr(params, key):
                    setattr(params, key, value)
                else:
                    print(f"Warning: {key} is not a valid parameter for DistillationParams")


            distillator = Distillator(model)
            learned_objective, scores = distillator.distill(
                target_layout=target_layout,
                depth=run.depth,
                samples_per_layer=run.samples_per_layer,
                params=DistillationParams(**run.solver_args)
            )

            print(f"objective {learned_objective}")
            print()
            print()
    except KeyboardInterrupt:
        print("Interrupted. Bye bye.")
        sys.exit(1)