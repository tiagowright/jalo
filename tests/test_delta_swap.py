
import logging
import pytest
import sys
import random
from pathlib import Path
from dataclasses import dataclass


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from freqdist import FreqDist, NgramType
from hardware import KeyboardHardware
from layout import KeyboardLayout
from metrics import METRICS, Metric, ObjectiveFunction, use_oxey_mode
from model import KeyboardModel

METRIC_BY_NAME = {metric.name: metric for metric in METRICS}

ALL_CHARS = '''abcdefghijklmnopqrstuvwxyz'"-,./;:<>?'''

@dataclass(slots=True, frozen=True)
class Scenario:
    """Configuration for building a model with real hardware/layout data."""

    hardware: str
    layout: str
    corpus: str
    objective: ObjectiveFunction
    swaps: list[tuple[int, int]]

    oxey_mode: bool = False

    @property
    def name(self) -> str:
        return repr(self.swaps)



logger = logging.getLogger(__name__)


SCENARIOS = [
    Scenario("ortho", "qwerty", "en", ObjectiveFunction({METRIC_BY_NAME["sfb"]: 1.0}), [(0, 1), (2, 3)]),
    Scenario("ortho", "qwerty", "en", ObjectiveFunction({METRIC_BY_NAME["sfb"]: 1.0}), [(2, 13)]),
    Scenario("ortho", "qwerty", "en", ObjectiveFunction({METRIC_BY_NAME["sfb"]: 1.0}), [tuple[int,int](random.sample(range(30), 2)) for _ in range(10)]), # random swaps
    Scenario("ortho", "qwerty", "en", ObjectiveFunction({METRIC_BY_NAME["sfb"]: 1.0}), [tuple[int,int](random.sample(range(30), 2)) for _ in range(100)]), # random swaps
    Scenario("ortho", "qwerty", "en", 
        ObjectiveFunction({METRIC_BY_NAME["sfb"]: 1.0, METRIC_BY_NAME["redirect_bad"]: 1.0, METRIC_BY_NAME["pinky_off"]: 1.0, METRIC_BY_NAME["sfs"]: 1.0}), 
        [(2, 13)]
    ),
    Scenario("ortho", "qwerty", "en", 
        ObjectiveFunction({METRIC_BY_NAME["sfb"]: 1.0, METRIC_BY_NAME["redirect_bad"]: 1.0, METRIC_BY_NAME["pinky_off"]: 1.0, METRIC_BY_NAME["sfs"]: 1.0}), 
        [tuple[int,int](random.sample(range(30), 2)) for _ in range(100)]
    ), # random swaps
]


@pytest.mark.parametrize("scenario", SCENARIOS, ids=lambda sc: sc.name)
def test_real_hardware_metric_checks(scenario: Scenario) -> None:
    hardware = KeyboardHardware.from_name(scenario.hardware)
    layout = KeyboardLayout.from_name(scenario.layout, hardware)
    freqdist = FreqDist.from_name(scenario.corpus)

    model = KeyboardModel(hardware, METRICS, scenario.objective, freqdist)
    char_at_pos = model.char_at_positions_from_layout(layout)
    base_score = model.score_chars_at_positions(char_at_pos)

    for i, j in scenario.swaps:
        delta = model.calculate_swap_delta(char_at_pos, i, j)

        # just compute the full score after the swap and assert that the delta matches
        char_at_pos_swapped = char_at_pos.copy()
        char_at_pos_swapped[i], char_at_pos_swapped[j] = char_at_pos_swapped[j], char_at_pos_swapped[i]
        full_score = model.score_chars_at_positions(char_at_pos_swapped)

        logger.info(f"Swapping {i} <-> {j}:\tdelta={delta:.4f}\texpected={full_score - base_score:.4f}")
        assert pytest.approx(delta, rel=1e-3, abs=1e-4) == full_score - base_score, f"Swap {i} <-> {j} delta: {delta} != {full_score - base_score}"
