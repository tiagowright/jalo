import sys
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from hardware import Finger, KeyboardHardware, Position
from layout import KeyboardLayout, LayoutKey
from metrics import Metric
from objective import ObjectiveFunction
from freqdist import FreqDist, NgramType
from model import KeyboardModel


dummy_freqdist = FreqDist(
    "dummy",
    {
        NgramType.MONOGRAM: {"a": 0.6, "b": 0.3, "c": 0.1},
        NgramType.BIGRAM: {"ab": 0.2, "bc": 0.1, "ca": 0.05},
        NgramType.TRIGRAM: {"abc": 0.05, "bca": 0.02, "cab": 0.01},
    }
)



@pytest.fixture()
def tiny_hardware() -> KeyboardHardware:
    positions = [
        Position(row=0, col=0, x=0.0, y=0.0, finger=Finger.LI, is_home=False),
        Position(row=0, col=1, x=1.0, y=0.0, finger=Finger.LI, is_home=True),
        Position(row=1, col=0, x=0.0, y=1.0, finger=Finger.RI, is_home=True)
    ]
    return KeyboardHardware("tiny", positions)


@pytest.fixture()
def dummy_metrics() -> tuple[Metric, Metric, Metric]:
    mono = Metric(
        name="is_home",
        description="Rewards home-row positions",
        ngramType=NgramType.MONOGRAM,
        function=lambda pos: 1.0 if pos.is_home else 0.0,
    )
    bi = Metric(
        name="same_hand_bigram",
        description="Rewards bigrams pressed with the same hand",
        ngramType=NgramType.BIGRAM,
        function=lambda a, b: 1.0 if a.finger.hand == b.finger.hand else 0.0,
    )
    tri = Metric(
        name="same_hand_trigram",
        description="same hand trigram",
        ngramType=NgramType.TRIGRAM,
        function=lambda a, b, c: 1.0 if (a.finger.hand == b.finger.hand == c.finger.hand) else 0.0,
    )
    return mono, bi, tri


@pytest.fixture()
def dummy_layout(tiny_hardware: KeyboardHardware) -> KeyboardLayout:
    keys = [
        LayoutKey(char=char, row=pos.row, col=pos.col, x=pos.x, y=pos.y, finger=pos.finger, position=pos)
        for pos, char in zip(tiny_hardware.positions, dummy_freqdist.char_seq)
    ]
    return KeyboardLayout(keys=keys, hardware=tiny_hardware, name="abc")


@pytest.fixture()
def model(tiny_hardware: KeyboardHardware, dummy_metrics: tuple[Metric, Metric, Metric]) -> KeyboardModel:
    mono, bi, tri = dummy_metrics
    freqdist = dummy_freqdist
    metrics = [mono, tri]  # omit the bigram metric on purpose
    objective = ObjectiveFunction({mono: 2.0, bi: 1.0})
    return KeyboardModel(
        hardware=tiny_hardware,
        metrics=metrics,
        objective=objective,
        freqdist=freqdist,
    )


def test_metrics_from_objective_are_added(model: KeyboardModel, dummy_metrics: tuple[Metric, Metric, Metric]) -> None:
    _, bi, _ = dummy_metrics
    assert bi in model.metrics, "Metrics referenced by the objective should be added automatically"



def test_position_freqdist_respects_char_order(model: KeyboardModel) -> None:
    char_indices = np.array([2, 0, 1])
    pos_freq = model.position_freqdist(char_indices)

    expected_mono = model.freqdist.to_numpy()[NgramType.MONOGRAM][char_indices]
    expected_bi = model.freqdist.to_numpy()[NgramType.BIGRAM][np.ix_(char_indices, char_indices)]

    assert np.array_equal(pos_freq[NgramType.MONOGRAM], expected_mono)
    assert np.array_equal(pos_freq[NgramType.BIGRAM], expected_bi)


def test_score_contributions_sum_to_total_score(model: KeyboardModel, dummy_layout: KeyboardLayout) -> None:
    contributions = model.score_contributions(dummy_layout)
    total_score = model.score_layout(dummy_layout)

    assert contributions, "Expected at least one metric contribution"
    assert pytest.approx(sum(contributions.values())) == total_score

    tri_metric = next(metric for metric in model.metrics if metric.order == 3)
    assert contributions[tri_metric] == pytest.approx(0.0), "Metrics not in the objective should not contribute"
