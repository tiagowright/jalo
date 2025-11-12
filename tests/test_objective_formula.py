import pytest
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from objective import ObjectiveFunction
from metrics import METRICS

METRIC_BY_NAME = {metric.name: metric for metric in METRICS}

@pytest.mark.parametrize("formula, expected_metrics", [
    ("sfb", {METRIC_BY_NAME["sfb"]: 1.0}),
    ("+sfb", {METRIC_BY_NAME["sfb"]: 1.0}),
    ("-sfb", {METRIC_BY_NAME["sfb"]: -1.0}),
    ("1.5sfb", {METRIC_BY_NAME["sfb"]: 1.5}),
    ("-2.5sfb", {METRIC_BY_NAME["sfb"]: -2.5}),
    ("2sfb - 1.5sfs + sft", {METRIC_BY_NAME["sfb"]: 2.0, METRIC_BY_NAME["sfs"]: -1.5, METRIC_BY_NAME["sft"]: 1.0}),
    ("  2  sfb   -   1.5 sfs   +   sft  ", {METRIC_BY_NAME["sfb"]: 2.0, METRIC_BY_NAME["sfs"]: -1.5, METRIC_BY_NAME["sft"]: 1.0}),
    ("sfb-sfs", {METRIC_BY_NAME["sfb"]: 1.0, METRIC_BY_NAME["sfs"]: -1.0}),
])
def test_valid_formulas(formula, expected_metrics):
    obj = ObjectiveFunction.from_formula(formula)
    assert obj.metrics == expected_metrics

@pytest.mark.parametrize("formula, error_message", [
    ("invalid_metric", "Invalid metric name: invalid_metric"),
    ("sfb + sfb", "sfb appears multiple times in the formula"),
    ("2.5", "Incomplete formula"),
    ("-", "Incomplete formula"),
    ("@sfb", "Invalid metric name: @sfb"),
    ("+2sfb 1sfs 3sft", "expected sign"),
])
def test_invalid_formulas(formula, error_message):
    with pytest.raises(ValueError, match=error_message):
        ObjectiveFunction.from_formula(formula)
