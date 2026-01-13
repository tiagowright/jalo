import re
import os
import tomllib
from enum import Enum
from typing import Any

from metrics import Metric, METRICS


class ObjectiveFunction:
    '''
    ObjectiveFunction is a linear combination of metrics. The primary purpose is to
    be used as the objective function for the optimizer, summarizing the quality of a keyboard layout.
    
    The optimizer will MINIMIZE the objective function. So think of it as cost or heat function.

    ObjectiveFunction allows other ObjectiveFunctions as input to the linear combination.
    '''

    def __init__(self, metrics: dict['Metric | ObjectiveFunction', float]):
        
        self.metrics = {}

        for metric, weight in metrics.items():
            if isinstance(metric, Metric):
                self.metrics[metric] = weight
            elif isinstance(metric, ObjectiveFunction):
                for sub_metric, sub_weight in metric.metrics.items():
                    if sub_metric in self.metrics:
                        self.metrics[sub_metric] += sub_weight * weight
                    else:
                        self.metrics[sub_metric] = sub_weight * weight
            else:
                raise ValueError(f"Invalid metric: {metric}")

    def __str__(self):
        if not self.metrics:
            return "0"
            
        first_metric = next(iter(self.metrics))

        formatted_weights = {
            metric: f"{weight:.2f}".rstrip("0").rstrip(".").lstrip("+").lstrip("-") for metric, weight in self.metrics.items() if abs(weight) >= 0.005
        }
        formatted_signs = {metric: '- ' if weight < 0 else '+ ' for metric, weight in self.metrics.items()}
        formatted_signs[first_metric] = '-' if self.metrics[first_metric] < 0 else ''

        return ' '.join([
            f"{formatted_signs[metric]}{weight}{metric.name}" for metric, weight in formatted_weights.items()
        ])
          
    def __hash__(self):
        return hash((metric, round(weight, 2)) for metric, weight in self.metrics.items())

    def __repr__(self):
        return f"ObjectiveFunction({self})"

    @classmethod
    def from_name(cls, name: str) -> 'ObjectiveFunction':
        '''
        Create an ObjectiveFunction from a nammed toml file in ./objectives/ directory.

        The file must define a key 'formula', that can be either a string with a formula,
        or a toml table, where each key is a metric name and the value is a weight.

        Example 1:
        formula = "3sfb + 1.5sfs + 100sft"
        
        Example 2:
        [formula]
        sfb = 3.0
        sfs = 1.5
        sft = 100.0
        '''

        # check if the file exists
        file_path = os.path.join('objectives', f'{name}.toml')
        if not os.path.exists(file_path):
            raise ValueError(f"Objective function file not found: {file_path}")

        # load the file
        with open(file_path, 'rb') as file:
            data = tomllib.load(file)

        try:
            formula = data['formula']
        except KeyError:
            raise ValueError(f"Objective function file must define a 'formula' key: {file_path}")


        if isinstance(formula, str):
            return cls.from_formula(formula)
        elif isinstance(formula, dict):
            metrics_by_name = {metric.name: metric for metric in METRICS}
            invalid_metric_names = [metric_name for metric_name in formula.keys() if metric_name not in metrics_by_name]
            if invalid_metric_names:
                raise ValueError(f"Invalid metric names in formula file: {file_path}: {invalid_metric_names}")
            return cls({metrics_by_name[metric_name]: weight for metric_name, weight in formula.items()})
        else:
            raise ValueError(f"Invalid formula: {formula}")

    @classmethod
    def from_formula(cls, formula: str) -> 'ObjectiveFunction':
        '''
        Create an ObjectiveFunction from a formula string.

        The formula is a linear combination of metrics: 
        [+|-][weight_1]<metric_name_1> [+|- [weight_2]<metric_name_2> ...]

        weight_i is a float.
        metric_name_i is the name of a metric in the METRICS list.

        Examples:
        3sfb + 1.5sfs + 100sft
        sfb - sfs


        Note: there is no '*' between the weight and the metric name. The '*' is considered to be a valid
        character in the metric name, so you can have a metric called `sfb*dist`, which can be part of a
        formula, e.g., `28sfb*dist` to compute finger speed
        '''
        metrics = {}
        space_pattern = re.compile(r'\s+')
        sign_pattern = re.compile(r'[+-]')
        float_pattern = re.compile(r'(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?')
        metric_names = sorted([metric.name for metric in METRICS], key = lambda name: -len(name))
        metric_by_hame = {metric.name: metric for metric in METRICS}

        class ParserState(Enum):
            START = 0
            SIGN = 1
            WEIGHT = 2
            METRIC = 3
            END = 4

        def parse_error_message(error: str, i: int, formula: str, state: ParserState) -> str:
            LINE_LEN = 70
            if len(formula) > LINE_LEN:
                start = max(0, i - LINE_LEN//2)
                end = min(len(formula), start + LINE_LEN)
                formula = formula[start:end]
                i -= start

            expected = {
                ParserState.START: "sign, weight, or metric name",
                ParserState.SIGN: "sign +|-",
                ParserState.WEIGHT: "weight (float) or metric name",
                ParserState.METRIC: "metric name",
                ParserState.END: "end of input",
            }
            
            return "\n".join([
                f"Error: {error}, at position {i}: expected {expected[state]}" if error else f"Error: at position {i}: expected {expected[state]}",
                "",
                f"\t{formula}",
                f"\t{'-' * i}^",
                ""
            ])

        state = ParserState.START

        sign = 1
        weight = 1.0
        i = 0
        while i < len(formula):
            match = space_pattern.match(formula, i)
            if match:
                i = match.end()
                continue

            match = sign_pattern.match(formula, i)
            if match:
                if state == ParserState.START or state == ParserState.SIGN:
                    state = ParserState.WEIGHT
                else:
                    raise ValueError(parse_error_message('', i, formula, state))
                sign = 1 if match.group() == '+' else -1
                i = match.end()
                continue

            match = float_pattern.match(formula, i)
            if match:   
                if state in (ParserState.START, ParserState.WEIGHT):
                    state = ParserState.METRIC
                else:
                    raise ValueError(parse_error_message('', i, formula, state))
                weight = float(match.group())
                i = match.end()
                continue

            for metric_name in metric_names:
                if formula.startswith(metric_name, i):
                    i += len(metric_name)
                    break
            else:
                raise ValueError(f"Invalid metric name: {formula[i:]}")

            if state in (ParserState.START, ParserState.WEIGHT, ParserState.METRIC):
                state = ParserState.SIGN
            else:
                raise ValueError(parse_error_message('', i, formula, state))

            if metric_by_hame[metric_name] in metrics:
                raise ValueError(parse_error_message(f"{metric_name} appears multiple times in the formula", i, formula, state))

            metrics[metric_by_hame[metric_name]] = sign * weight
            metric_name = None
            sign = 1
            weight = 1.0

        if state == ParserState.WEIGHT or state == ParserState.METRIC:
            raise ValueError(parse_error_message("Incomplete formula", i, formula, state))

        return cls(metrics)
