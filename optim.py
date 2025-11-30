import os
import math
from re import T

from itertools import combinations
from typing import List, Tuple, Optional, Any, Iterable, Callable
import numpy as np
import random
import heapq
import queue
import time
import csv

import multiprocessing
from functools import partial
from tqdm import tqdm

from model import KeyboardModel, NgramType, _calculate_swap_delta
from freqdist import FreqDist
from layout import KeyboardLayout
from hardware import Finger



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



class OptimizerLogger:
    def __init__(self, optimizer_name: str, batch_name: str, log_runs: bool = True, log_events: bool = True, log_population: bool = True):
        self.optimizer_name = optimizer_name
        self.batch_name = batch_name
        self.log_runs = log_runs
        self.log_events = log_events
        self.log_population = log_population

        self.events_filename = f"{self.optimizer_name}_events.csv"
        self.runs_filename = f"{self.optimizer_name}_runs.csv"
        self.population_filename = f"{self.optimizer_name}_population.csv"

        self.events = []
        self.runs = []
        self.population = {}

    def batch_start(self):
        self.start_time = time.time()

    def batch_end(self, population):
        self.end_time = time.time()
        self.duration = self.end_time - self.start_time

        if self.log_population:
            self.population = population

    def event(self, seed_id: int, step: int, score: float) -> None:
        if not self.log_events:
            return
        self.events.append((seed_id, step, score))

    def run(self, seed_id: int, initial_score: float, final_score: float) -> None:
        if not self.log_runs:
            return
        self.runs.append((seed_id, initial_score, final_score))

    def save(self) -> None:
        if not self.log_events and not self.log_runs and not self.log_population:
            return

        import fcntl
        
        # save events and runs to separate csv files in ./optimizers/logs directory (may need to create it)
        logs_dir = os.path.join(os.path.dirname(__file__), "optimizers", "logs")
        if not os.path.exists(logs_dir):
            os.makedirs(logs_dir)
        
        for file_path, header, rows in (
            (
                os.path.join(logs_dir, self.events_filename),
                ["optimizer_name", "batch_name", "seed_id", "step", "score"],
                [(self.optimizer_name, self.batch_name, seed_id, step, score) for seed_id, step, score in self.events]
            ),
            (
                os.path.join(logs_dir, self.runs_filename),
                ["optimizer_name", "batch_name", "seed_id", "initial_score", "final_score"],
                [(self.optimizer_name, self.batch_name, seed_id, initial_score, final_score) for seed_id, initial_score, final_score in self.runs]
            ),
            (
                os.path.join(logs_dir, self.population_filename),
                ["optimizer_name", "batch_name", "rank", "score"],
                [
                    (self.optimizer_name, self.batch_name, rank, score) 
                    for rank, (char_at_pos, score) in enumerate(sorted(self.population.items(), key=lambda x: x[1]))
                ]
            )
        ):
            if not rows:
                continue

            file_exists = os.path.exists(file_path)
            with open(file_path, "a+") as f:
                # using a lock to make this code multiprocessor safe if writing to the same log file
                fcntl.flock(f, fcntl.LOCK_EX)
                writer = csv.writer(f, delimiter='\t')
                if not file_exists:
                    writer.writerow(header)
                writer.writerows(rows)
                fcntl.flock(f, fcntl.LOCK_UN)



class Optimizer:
    def __init__(
        self, 
        model: KeyboardModel, 
        population_size: int = 1000, 
        solver: str = "genetic", 
        log_runs: bool = False, 
        log_events: bool = False, 
        log_population: bool = False,
        solver_args: dict | None = None
    ):
        self.model = model
        self.solver = solver
        self.solver_args = solver_args or {}
        
        # check solver exists
        if not os.path.exists(os.path.join(os.path.dirname(__file__), "optimizers", solver + ".py")):
            raise ValueError(f"Solver {solver} not found in ./optimizers directory")
        
        # Build wrapper function from solver module name
        full_module_name = f"optimizers.{solver}"
        self.optimizer_function = partial(_optimize_batch_worker_from_module, full_module_name)

        self.log_runs = log_runs
        self.log_events = log_events
        self.log_population = log_population

        self.swap_position_pairs = tuple(combinations(range(len(self.model.hardware.positions)), 2))

        self.population = Population(max_size=population_size)

        # for optimization, we want to be able to try swapping and resequencing whole columns
        # but the naive definition using Position.col is not enough because not all positions
        # in a col use the same finger (e.g., a thumb key on col 4). We also don't want to
        # try all naive permutations of every column, it takes too long to compute, so we will
        # group positions by col and finger into a "column", and we will group together fingers
        # 1,2,7,8 into a larger set, and fingers 3 and 6 into another set, and fingers 0,9.
        pis_at_finger_at_column_map = {}
        for pi, position in enumerate(self.model.hardware.positions):
            if position.finger not in pis_at_finger_at_column_map:
                pis_at_finger_at_column_map[position.finger] = {}
            if position.col not in pis_at_finger_at_column_map[position.finger]:
                pis_at_finger_at_column_map[position.finger][position.col] = []
            pis_at_finger_at_column_map[position.finger][position.col].append(pi)

        finger_groups = [[1,2,7,8],[3,6],[0,9]]
        self.group_of_pis_at_column = tuple(
            tuple(
                tuple(pis_at_finger_at_column_map[Finger(finger_value)][col])
                for finger_value in fg
                for col in pis_at_finger_at_column_map[Finger(finger_value)]
            )
            for fg in finger_groups
        )

        self.pis_at_column = tuple(
            tuple(pis_at_finger_at_column_map[finger][col])
            for finger in pis_at_finger_at_column_map
            for col in pis_at_finger_at_column_map[finger]
        )

        # self.positions_at_column = tuple(tuple(positions) for positions in positions_at_column_map.values())

    def generate(self, char_seq: list[str], iterations:int = 100, score_tolerance:float = 0.001, pinned_positions: tuple[int, ...] = ()):
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

        # update swap_position_pairs, pis_at_column, group_of_pis_at_column to exclude pinned positions
        swap_position_pairs = tuple(
            (i, j) for i, j in self.swap_position_pairs if i not in pinned_positions and j not in pinned_positions
        )
        pis_at_column = tuple(
            pis for pis in self.pis_at_column if not any(pi in pinned_positions for pi in pis)
        )
        group_of_pis_at_column = tuple(
            tuple(pis for pis in group if not any(pi in pinned_positions for pi in pis)) for group in self.group_of_pis_at_column
        )

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
                pinned_positions,
                swap_position_pairs,
                pis_at_column,
                group_of_pis_at_column,
                progress_queue,
                self.population.max_size,
                OptimizerLogger(self.solver, f"batch_{i+1}_of_{len(batches)}_with_{len(initial_positions_batch)}", log_runs=self.log_runs, log_events=self.log_events, log_population=self.log_population),
                self.solver_args
            )
            for i, initial_positions_batch in enumerate(batches)
        ]

        with multiprocessing.Pool() as pool:
            results_async = pool.map_async(self.optimizer_function, tasks)
            
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

        # Import and use the solver module's _optimize function
        import importlib
        full_module_name = f"optimizers.{self.solver}"
        solver_module = importlib.import_module(full_module_name)
        
        new_population = solver_module._optimize(
            initial_char_at_pos,
            initial_score,
            tolerance,
            order_1,
            order_2,
            order_3,
            pinned_positions,
            self.swap_position_pairs,
            self.pis_at_column,
            self.group_of_pis_at_column,
            iterations,
            self.solver_args
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


def _optimize_batch_worker_from_module(module_name: str, args):
    """
    Top-level function that imports an optimizer module and calls its optimize_batch_worker.
    This function is picklable for multiprocessing.
    """
    import importlib
    mod = importlib.import_module(module_name)
    return mod.optimize_batch_worker(args)




if __name__ == "__main__":
    import sys
    import argparse
    import importlib.util
    import time
    import csv
    import tomllib

    from dataclasses import dataclass
    from hardware import KeyboardHardware
    from objective import ObjectiveFunction
    from freqdist import FreqDist
    from model import KeyboardModel
    from metrics import METRICS

    @dataclass
    class OptimizerResult:
        name: str
        optimizer_name: str
        iterations: int
        time_taken: float
        best_score: float
        mean_score: float
        stdev_score: float
        hardware_name: str
        objective_name: str
        corpus_name: str

    @dataclass
    class RunConfig:
        name: str
        name_format: str
        hardware: str
        corpus: str
        iterations: int
        population: int
        objective: str
        solver: str
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
                hardware=data.get("hardware", defaults.hardware),
                corpus=data.get("corpus", defaults.corpus), 
                iterations=data.get("iterations", defaults.iterations),
                population=data.get("population", defaults.population),
                objective=data.get("objective", defaults.objective),
                solver=data.get("solver", defaults.solver),
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
                f"solver = {self.solver}",
                f"iterations = {self.iterations}",
                f"solver_args = {self.solver_args}",
                f"objective = {self.objective}",
                f"population = {self.population}",
                f"hardware = {self.hardware}",
                f"corpus = {self.corpus}",
            ])


    parser = argparse.ArgumentParser(description="Run layout generation experiments with multiple solvers and hyperparameters.")
    parser.add_argument("--config", type=str, default=None, help="path to the config toml file to specify the runs to perform (e.g., ./optimizers/tuning/comparison_config.toml)")
    parser.add_argument("--output", type=str, default="./optimizers/logs/solver_runs_table.csv")
    parser.add_argument("--iterations", type=int, default=1000)
    parser.add_argument("--hardware", type=str, default="ortho")
    parser.add_argument("--objective", type=str, default="default")
    parser.add_argument("--corpus", type=str, default="en")
    parser.add_argument("--solver", type=str, default=None, help="<solver1>[:iterations][,<solver2>[:iterations] ...]")
    parser.add_argument("--log-runs", action="store_true")
    parser.add_argument("--log-events", action="store_true")
    parser.add_argument("--log-population", action="store_true")
    args = parser.parse_args()

    defaults = RunConfig(
        name='',
        name_format=r'{solver}_{iterations}',
        hardware=args.hardware,
        corpus=args.corpus,
        iterations=args.iterations,
        population=1000,
        objective=args.objective,
        solver=args.solver,
        solver_args={}
    )

    # Ensure the parent directory (containing optimizers/) is on sys.path
    parent_dir = os.path.dirname(__file__)
    if parent_dir not in sys.path:
        sys.path.insert(0, parent_dir)
    optimizers_dir = os.path.join(parent_dir, "optimizers")
            

    if args.config is not None:
        with open(args.config, "rb") as f:
            try:
                config_dict = tomllib.load(f)
            except tomllib.TOMLDecodeError as e:
                print(f"Error parsing config file {args.config}: {e}")
                sys.exit(1)
        
        runs = RunConfig.runs_from_config(config_dict, defaults)

        # for every key in config_dict if it is also an args attribute, override it with the value from config_dict
        for key in ['output', 'log_runs', 'log_events', 'log_population']:
            if key in config_dict:
                setattr(args, key, config_dict[key])

    else:
        if args.solver is not None:
            optimizers = {
                tokens[0]: (int(tokens[1]) if len(tokens) > 1 else None) for tokens in (optimizer.split(':') for optimizer in args.solver.split(','))
            }
        else:
            optimizers = {
                optimizer_file[:-3]: None
                for optimizer_file in os.listdir(optimizers_dir) 
                if optimizer_file.endswith(".py") and not optimizer_file.startswith("__")
            }

        runs = []
        for optimizer in optimizers.keys():
            config_dict = defaults.__dict__.copy()
            config_dict['solver'] = optimizer
            config_dict['iterations'] = optimizers[optimizer] or defaults.iterations
            runs.append(RunConfig(**config_dict))


    # print the configuration
    print("Configuration:")
    
    if args.config is not None:
        print(f"Config file: {args.config}")
    else:
        print(f"Solver: {args.solver}")
        print(f"Iterations: {args.iterations}")
        print(f"Hardware: {args.hardware}")
        print(f"Objective: {args.objective}")
        print(f"Corpus: {args.corpus}")

    print(f"Output: {args.output}")

    logging_strs = [key for key in ['log_runs', 'log_events', 'log_population'] if getattr(args, key)]
    if logging_strs:
        print(f"Logging: ", ", ".join(logging_strs))
    else:
        print("Logging nothing")

    print(f"Found {len(runs)} runs")
    print()

    # for each .py in ./optimizers in turn, import optimize_batch_worker
    results = []
    all_top_scores = []

    random.shuffle(runs)
    for run_i, run in enumerate(runs):
        print("-" * 10)
        print(f"Run {run_i+1} of {len(runs)}:")
        print(run)
        print()

        hardware = KeyboardHardware.from_name(run.hardware)
        objective = ObjectiveFunction.from_name(run.objective)
        freqdist = FreqDist.from_name(run.corpus)
        model = KeyboardModel(hardware=hardware, metrics=METRICS, objective=objective, freqdist=freqdist)

        N = len(hardware.positions)
        char_seq = freqdist.char_seq[:N]

        if not run.solver:
            print("Warning: No solver specified, skipping")
            continue

        module_path = os.path.join(optimizers_dir, run.solver + ".py")
        spec = importlib.util.spec_from_file_location(run.solver, module_path)
        if spec is None or spec.loader is None:
            print(f"Warning: Could not load spec for {run.solver}")
            continue
        
        optimizer_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(optimizer_module)
        
        # Register the module in sys.modules so multiprocessing can pickle functions from it
        # Set __name__ to match what pickle will look for
        full_module_name = f"optimizers.{run.solver}"
        optimizer_module.__name__ = full_module_name
        sys.modules[full_module_name] = optimizer_module

        # Also register with the simple name for backwards compatibility
        sys.modules[run.solver] = optimizer_module
        
        optimizer = Optimizer(
            model, 
            solver=run.solver, 
            log_runs=args.log_runs, 
            log_events=args.log_events, 
            log_population=args.log_population,
            solver_args=run.solver_args
        )

        # capture how long it takes to generate
        start_time = time.time()
        optimizer.generate(char_seq=char_seq, iterations=run.iterations)
        end_time = time.time()
        print(f"Time taken: {end_time - start_time:.1f} seconds")
        print()

        top_scores = [model.score_chars_at_positions(char_at_pos) for char_at_pos in optimizer.population.sorted()]
        mean_score = float(np.mean(top_scores))
        stdev_score = float(np.std(top_scores))

        if args.log_population:
            all_top_scores.append(top_scores)

        # assert that top_scores match scores in population.score
        assert all(abs(score - optimizer.population.scores[char_at_pos]) < 0.001*abs(score) for score, char_at_pos in zip(top_scores, optimizer.population.sorted()))

        results.append(OptimizerResult(
            name=run.name,
            hardware_name=hardware.name,
            objective_name=str(objective),
            corpus_name=freqdist.corpus_name,
            optimizer_name=run.solver,
            iterations=run.iterations,
            time_taken=end_time - start_time,
            best_score=min(top_scores),
            mean_score=mean_score,
            stdev_score=stdev_score,
        ))


    # write results to stdout as csv, \t separated
    # append to the file if it already exists
    if os.path.exists(args.output):
        writer = csv.DictWriter(open(args.output, "a"), fieldnames=results[0].__dataclass_fields__.keys(), delimiter='\t')
    else:
        writer = csv.DictWriter(open(args.output, "w"), fieldnames=results[0].__dataclass_fields__.keys(), delimiter='\t')
        writer.writeheader()

    for result in results:
        writer.writerow(result.__dict__)

    if args.log_population and all_top_scores:
        import itertools

        with open(args.output.replace(".csv", "_populations.csv"), "w") as f:
            writer = csv.writer(f, delimiter='\t')
            writer.writerow([result.name for result in results])
            for row in itertools.zip_longest(*all_top_scores, fillvalue=''):
                writer.writerow(row)