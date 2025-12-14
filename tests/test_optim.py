
import logging
import pytest
import sys
import random
from pathlib import Path
from dataclasses import dataclass


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from freqdist import FreqDist
from hardware import KeyboardHardware
from layout import KeyboardLayout
from metrics import METRICS
from objective import ObjectiveFunction
from model import KeyboardModel
from optim import Optimizer

METRIC_BY_NAME = {metric.name: metric for metric in METRICS}

ALL_CHARS = '''abcdefghijklmnopqrstuvwxyz'"-,./;:<>?'''

GENERATE = '__generate__'

@dataclass(slots=True, frozen=True)
class Scenario:
    """Configuration for building a model with real hardware/layout data."""

    hardware: str
    layout: str
    corpus: str
    objective: ObjectiveFunction
    target_score: float

    oxeylyzer_mode: bool = False

    @property
    def name(self) -> str:
        return f"{self.hardware}_{self.layout}_{self.corpus}_{self.objective}_{self.target_score}"


logger = logging.getLogger(__name__)


SCENARIOS = [
    # 3sfb + 1.5sfs + 1sft
    # Scenario("ansi", "qwerty", "en", ObjectiveFunction({METRIC_BY_NAME["sfb"]: 2.0, METRIC_BY_NAME["sfs"]: 1.5, METRIC_BY_NAME["sft"]: 1.0}), 10.9),
    # Scenario("ansi", "sturdy", "en", ObjectiveFunction({METRIC_BY_NAME["sfb"]: 2.0, METRIC_BY_NAME["sfs"]: 1.5, METRIC_BY_NAME["sft"]: 1.0}), 11.1),
    Scenario("ansi", GENERATE, "en", ObjectiveFunction({METRIC_BY_NAME["sfb"]: 2.0, METRIC_BY_NAME["sfs"]: 1.5, METRIC_BY_NAME["sft"]: 1.0}), 11.0),
    
]


@pytest.mark.parametrize("scenario", SCENARIOS, ids=lambda sc: sc.name)
def test_real_hardware_metric_checks(scenario: Scenario) -> None:
    hardware = KeyboardHardware.from_name(scenario.hardware)
    freqdist = FreqDist.from_name(scenario.corpus)
    model = KeyboardModel(hardware, METRICS, scenario.objective, freqdist)

    optimizer = Optimizer(model, population_size=100)

    if scenario.layout == GENERATE:
        optimizer.generate(
            char_seq=freqdist.char_seq[:len(hardware.positions)],
            seeds=10
        )
    else:
        layout = KeyboardLayout.from_name(scenario.layout, hardware)
        char_at_pos = model.char_at_positions_from_layout(layout)
        base_score = model.score_chars_at_positions(char_at_pos)
        optimizer.improve(tuple(char_at_pos), seeds=10)
    

    char_at_pos = optimizer.population.top()
    optimized_score = model.score_chars_at_positions(char_at_pos)
    assert scenario.layout == GENERATE or optimized_score <= base_score, f"Optimized score {optimized_score} > base {base_score}"
    assert optimized_score <= scenario.target_score, f"Optimized score {optimized_score} > target {scenario.target_score}"

    heap = optimizer.population.heap.copy()
    heap.sort(reverse=True)
    for score, char_at_pos in heap[:10]:
        actual_score = model.score_chars_at_positions(char_at_pos)
        assert abs(actual_score + score) < abs(0.001 * score), f"Score mismatch: {score:.2f} != {actual_score:.2f}"
