
import pytest
import sys
from pathlib import Path
from dataclasses import dataclass
from collections import Counter

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from freqdist import FreqDist, NgramType
from hardware import KeyboardHardware
from layout import KeyboardLayout
from metrics import METRICS, Metric, ObjectiveFunction, use_oxey_mode
from model import KeyboardModel

METRIC_BY_NAME = {metric.name: metric for metric in METRICS}

@dataclass(slots=True, frozen=True)
class Scenario:
    """Configuration for building a model with real hardware/layout data."""

    hardware: str
    layout: str
    text: str
    expectations: dict[str, float]
    oxey_mode: bool = False

    @property
    def name(self) -> str:
        return f"{self.hardware}_{self.layout}_{self.text.replace(' ', '_')}"

def fqt(text: str) -> FreqDist:
    '''
    takes a text string with space separated mono, bi, and trigrams and creates a freqdist (Counter)
    '''
    fq: dict[str | NgramType, dict[str, float]] = {ngramtype: {} for ngramtype in NgramType}

    for char in '''abcdefghijklmnopqrstuvwxyz'"-,./;:<>?''':
        fq[NgramType.MONOGRAM][char] = 0

    for ngram, count in Counter(text.split()).items():
        if len(ngram) == 1:
            fq[NgramType.MONOGRAM][ngram] = count
        elif len(ngram) == 2:
            fq[NgramType.BIGRAM][ngram] = count
            fq[NgramType.SKIPGRAM][ngram] = count
        elif len(ngram) == 3:
            fq[NgramType.TRIGRAM][ngram] = count
        else:
            raise ValueError(f"Invalid ngram: {ngram}")
        
    # normalize the counts
    for ngramtype in fq:
        total = sum(fq[ngramtype].values())
        if total > 0:
            fq[ngramtype] = {ngram: count / total for ngram, count in fq[ngramtype].items()}


    return FreqDist("test-suite", fq)

def one_at_a_time(scenario: Scenario) -> list[Scenario]:
    '''
    helper function breaks one scenario into multiple
    '''
    return [
        Scenario(scenario.hardware, scenario.layout, ngram, scenario.expectations, scenario.oxey_mode)
        for ngram in scenario.text.split()
    ]


SCENARIOS = [
    Scenario('ansi', 'qwerty', 'de fr gt aq sw ju ki lo', {'sfb': 1.0, 'sfs': 1.0}),
    
    # Qwerty RT would be 1U, RG 1.6U, RV  2.02U and RB 2.66U
    Scenario("ansi", "qwerty", "rt", {"sfb": 1.0, "dist_linear": 1}),
    Scenario("ansi", "qwerty", "rg", {"sfb": 1.0, "dist_linear": 1.6}),
    Scenario("ansi", "qwerty", "rv", {"sfb": 1.0, "dist_linear": 2.136}),
    Scenario("ansi", "qwerty", "rb", {"sfb": 1.0, "dist_linear": 2.6575}),
    Scenario("ansi", "qwerty", "rt rg rv rb", {"sfb": 1.0, "sfs": 1.0, "dist_linear": (1+1.6+2.136+2.6575)/4}),

    # scissors
    Scenario("ansi", "qwerty", "qx wc cr ct xr xt qc", {"scissors": 1.0, "rowskip": 1.0}),
    Scenario("ansi", "qwerty", "ni mi i. o/ i/ no mo", {"scissors": 0.0, "rowskip": 1.0}),

    # other potential scissors
    Scenario("ansi", "qwerty", "wz ex o/ i.", {"scissors": 0.0, "rowskip": 1.0}),
    Scenario("ansi", "qwerty", "qs pl", {"scissors": 0.0, "rowskip": 0.0}),
    Scenario("ansi", "qwerty", "aw zs ;o /l", {"scissors": 1.0, "rowskip": 0.0}),

    # SFS
    Scenario("ansi", "qwerty", "br un tg hu rf hn yu", {"sfs": 1.0}),
    Scenario("ansi", "qwerty", "bb nn tt hh rr yy uu", {"sfs": 0.0}),

    # Trigrams
    Scenario("ansi", "qwerty", "sad and our wer you", {"alt": 1/5, "roll": 2/5, "roll3": 1/5, "redirect_total": 2/5}),
    Scenario("ansi", "graphite", "tha the thi", {"roll": 1.0}),

    # y o u q x  k d l w ,  
    # i a e n j  v h t s c
    #  " - r b z  f p m g .
    Scenario("ansi_angle", "inrolly", "you our ion rea", {"roll3": 1.0}),
    Scenario("ansi_angle", "inrolly", "ear ain one are", {"redirect_total": 1.0}),

    # flavors for redirects
    # bad sfs: was
    # bad: eas
    # sfs: far
    # redirect: gas, gag
    # note that oxylyzer considers gag an sfs
    Scenario("ansi", "qwerty", "was eas far gas gag", {"redirect_total": 1.0, "redirect_bad_sfs": 1/5, "redirect_sfs": 1/5, "redirect_bad": 1/5, "redirect": 2/5}),
    Scenario("ansi", "qwerty", "was eas far gas gag", {"redirect_total": 1.0, "redirect_bad_sfs": 1/5, "redirect_sfs": 2/5, "redirect_bad": 1/5, "redirect": 1/5}, oxey_mode=True),

    # LSBs, using the Keyboard layouts doc definition
    Scenario("ansi", "qwerty", "et te ge eg eb be wb sb yi", {"lsb": 1.0}),
    Scenario("ansi", "qwerty", "dt td vd wr wf wv xv ni hi ui ou on ly yo", {"lsb": 0.0}),

    # LSBs, using the Oxeylyzer definition
    Scenario("ansi", "qwerty", "et eg eb dt dg ct cg cb iy ih ky kh kn nk n,", {"lsb": 1.0}, oxey_mode=True),
    Scenario("ansi", "qwerty", "dv db ni wb sb", {"lsb": 0.0}, oxey_mode=True),

]

### skipping some tests today?
# SCENARIOS = SCENARIOS[17:]
# SCENARIOS = one_at_a_time(Scenario("ansi", "qwerty", "ih in ky kh", {"lsb": 1.0}, oxey_mode=True))

@pytest.mark.parametrize("scenario", SCENARIOS, ids=lambda sc: sc.name)
def test_real_hardware_metric_checks(scenario: Scenario) -> None:
    use_oxey_mode(scenario.oxey_mode)
    hardware = KeyboardHardware.from_name(scenario.hardware)
    layout = KeyboardLayout.from_name(scenario.layout, hardware)
    freqdist = fqt(scenario.text)
    model = KeyboardModel(hardware, METRICS, ObjectiveFunction({metric: 1.0 for metric in METRICS}), freqdist)
    metric_scores = model.analyze_layout(layout)

    for metric_name, expected in scenario.expectations.items():
        metric = METRIC_BY_NAME[metric_name]
        actual = metric_scores[metric]
        expected_approx = pytest.approx(expected, rel=1e-3, abs=1e-4)

        assert actual == expected_approx, f"Metric `{metric.name}` expected {expected} but got {actual}"
