import os
import csv
import time
import fcntl


class OptimizerLogger:
    '''
    A logger for an optimizer batch run. Logs events, runs, and population to csv files.
    '''
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

    def event(self, seed_id: int, step: int, score: float, msg: str = '') -> None:
        if not self.log_events:
            return
        self.events.append((seed_id, step, score, msg))

    def run(self, seed_id: int, initial_score: float, final_score: float) -> None:
        if not self.log_runs:
            return
        self.runs.append((seed_id, initial_score, final_score))

    def save(self) -> None:
        if not self.log_events and not self.log_runs and not self.log_population:
            return
        
        # save events and runs to separate csv files in ./solvers/logs directory (may need to create it)
        logs_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "solvers", "logs")
        if not os.path.exists(logs_dir):
            os.makedirs(logs_dir)
        
        for file_path, header, rows in (
            (
                os.path.join(logs_dir, self.events_filename),
                ["optimizer_name", "batch_name", "seed_id", "step", "score", "msg"],
                [(self.optimizer_name, self.batch_name, seed_id, step, score, msg) for seed_id, step, score, msg in self.events]
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
