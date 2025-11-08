from math import sqrt
from enum import Enum
from dataclasses import dataclass, field
from typing import Callable, Any

from hardware import FingerType, Position, Hand, Finger
from freqdist import NgramType

class Direction(Enum):
    INWARD = 0
    OUTWARD = 1
    REPEAT = 2
    ALTERNATE = 3

    @classmethod
    def which(cls, a: Position, b: Position) -> 'Direction':
        if a.finger.hand != b.finger.hand:
            return Direction.ALTERNATE
        
        if a.finger == b.finger:
            return Direction.REPEAT
        
        if a.finger.value < b.finger.value and a.finger.hand == Hand.LEFT:
            return Direction.INWARD
        elif a.finger.value > b.finger.value and a.finger.hand == Hand.RIGHT:
            return Direction.INWARD

        return Direction.OUTWARD


@dataclass(frozen=True)
class Metric:
    """
    A single metric defined over POSITIONS.
    """
    name: str
    description: str
    ngramType: NgramType

    # for a monogram, single argument is a position
    # for bigram or skipgram, two arguments are positions
    # for trigram, three arguments are positions
    # the function should return a float that is the value of the metric for the ngram given
    function: Callable[[Position], float] | Callable[[Position, Position], float] | Callable[[Position, Position, Position], float] = field(hash=False, compare=False)

    @property
    def order(self) -> int:
        return self.ngramType.order


class ObjectiveFunction:
    '''
    ObjectiveFunction is a linear combination of metrics. The primary purpose is to
    be used as the objective function for the optimizer, summarizing the quality of a keyboard layout.
    
    The optimizer will MINIMIZE the objective function. So think of it as cost or effort function.

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
        formatted_weights = {
            metric: f"{weight:.2f}".strip("0").strip(".") for metric, weight in self.metrics.items()
        }
        
        return " + ".join([f"{weight}{metric.name}" for metric, weight in formatted_weights.items()])
  
    def __hash__(self):
        return hash((metric, round(weight, 2)) for metric, weight in self.metrics.items())

    def __repr__(self):
        return f"ObjectiveFunction({self})"


# Helper functions
def same_finger(a, b):
    return a.finger == b.finger

def same_hand(a, b, c=None):
    if c is None:
        return a.finger.hand == b.finger.hand
    return a.finger.hand == b.finger.hand and a.finger.hand == c.finger.hand

# If True, use oxeylyzer's definition for all metrics that have an oxeylyzer equivalent.
# Where the definitions differ, it is primarily because (i think) the new definition
# is more accurately capture the intent of the metric described in the Keyboard layouts doc.
oxey_mode = False
def use_oxey_mode(mode: bool):
    global oxey_mode
    oxey_mode = mode


# Bigram metrics
# repetition
def rep(a, b):
    return a == b

# single finger bigram
def sfb(a, b):
    return a.finger == b.finger and a != b

# sfb by finger
def sfb_finger(a, b, finger):
    return sfb(a, b) and a.finger == finger

def sfb_finger_0(a, b):
    return sfb_finger(a, b, Finger(0))
def sfb_finger_1(a, b):
    return sfb_finger(a, b, Finger(1))
def sfb_finger_2(a, b):
    return sfb_finger(a, b, Finger(2))
def sfb_finger_3(a, b):
    return sfb_finger(a, b, Finger(3))
def sfb_finger_4(a, b):
    return sfb_finger(a, b, Finger(4))
def sfb_finger_5(a, b):
    return sfb_finger(a, b, Finger(5))
def sfb_finger_6(a, b):
    return sfb_finger(a, b, Finger(6))
def sfb_finger_7(a, b):
    return sfb_finger(a, b, Finger(7))
def sfb_finger_8(a, b):
    return sfb_finger(a, b, Finger(8))
def sfb_finger_9(a, b):
    return sfb_finger(a, b, Finger(9))

# single finger skipgram
def sfs(a, b):
    return sfb(a, b)

# single finger trigram
def sft(a, b, c):
    return sfb(a, b) and sfb(b, c)

# distance squared
def distance_squared(a, b):
    if not same_finger(a, b):
        return 0
    return (a.x - b.x)**2 + (a.y - b.y)**2

# distance
def distance_linear(a, b):
    return sqrt(distance_squared(a, b))

def lsb(a, b):
    '''
    LSB is Lateral Stretch Bigram
    From the "Keyboard layouts" doc:
    Adjacent finger bigrams where the horizontal distance is 2U or greater.
    Semi-adjacent finger bigrams where the horizontal distance is 3.5U or greater.
    '''
    if not same_hand(a,b):
        return False

    if a.finger.type == FingerType.THUMB or b.finger.type == FingerType.THUMB:
        return False
    
    # adjacent fingers
    if abs(a.finger.value - b.finger.value) == 1:
        return abs(a.x - b.x) >= 2
    
    # semi-adjacent fingers
    if abs(a.finger.value - b.finger.value) == 2:
        return abs(a.x - b.x) >= 3.5
    
    return False


# row skip
def rowskip(a, b):
    return abs(a.row - b.row) > 1 and same_hand(a, b)

def scissors(a, b):
    '''
    full scissor bigram (FSB)
    from the Keyboard layouts doc
    The index prefers being lower.
    The middle prefers being higher.
    The ring prefers being higher than index and pinky, but lower than the middle.
    The pinky prefers being lower than middle and ring, but higher than index.

    oxeylyzer also considers FSB with 1 row difference when the fingers are ring and pinky
    and the pinky is higher than the ring
    '''
    if not rowskip(a, b):
        if same_hand(a,b) and (
            (a.finger.type == FingerType.PINKY and b.finger.type == FingerType.RING and a.row > b.row) or
            (a.finger.type == FingerType.RING and b.finger.type == FingerType.PINKY and a.row < b.row)
        ):
            return True
        return False


    if a.row > b.row:
        a, b = b, a
    
    if a.finger.type == FingerType.INDEX:
        return True
    elif a.finger.type == FingerType.MIDDLE:
        return False
    elif a.finger.type == FingerType.RING:
        return b.finger.type == FingerType.MIDDLE
    elif a.finger.type == FingerType.PINKY:
        return (b.finger.type == FingerType.MIDDLE or b.finger.type == FingerType.RING)

    return False

# FSB above are sensible for staggered keyboard planks, but not great for split ortho.
def fsb_split_ortho(a, b):
    # TODO
    pass

def pinky_ring(a,b):
    return same_hand(a,b) and a.row != b.row and (
        (a.finger.type == FingerType.PINKY and b.finger.type == FingerType.RING) or
        (a.finger.type == FingerType.RING and b.finger.type == FingerType.PINKY)
    )

def pinky_off(a):
    return a.finger.type == FingerType.PINKY and not a.is_home

# Trigram metrics

def alternate(a, b, c):
    return a.finger.hand == c.finger.hand and a.finger.hand != b.finger.hand and (
        a.finger != c.finger or a == c
    )

def alternate_sfs(a, b, c):
    return a.finger.hand == c.finger.hand and a.finger.hand != b.finger.hand and (
        a.finger == c.finger and a != c
    )

def roll3(a, b, c):
    return (
        same_hand(a, b, c) and 
        Direction.which(a, b) == Direction.which(b, c) and
        Direction.which(a, b) in (Direction.INWARD, Direction.OUTWARD)
    )

def in_roll(a, b, c):
    return (a.finger.hand != c.finger.hand) and (
        Direction.which(a, b) == Direction.INWARD or 
        Direction.which(b, c) == Direction.INWARD
    )

def out_roll(a, b, c):
    return (a.finger.hand != c.finger.hand) and (
        Direction.which(a, b) == Direction.OUTWARD or 
        Direction.which(b, c) == Direction.OUTWARD
    )

def roll(a, b, c):
    return in_roll(a, b, c) or out_roll(a, b, c) or roll3(a, b, c)


# redirect metrics
def total_redirect(a, b, c):
    return (
        same_hand(a, b, c) and 
        Direction.which(a, b) != Direction.which(b, c) and
        Direction.which(a, b) in (Direction.INWARD, Direction.OUTWARD) and
        Direction.which(b, c) in (Direction.INWARD, Direction.OUTWARD)
    )

def total_redirect_bad(a, b, c):
    return total_redirect(a, b, c) and (
        a.finger.type != FingerType.INDEX and 
        b.finger.type != FingerType.INDEX and 
        c.finger.type != FingerType.INDEX
    )

#
# Note that oxeylyzer definition of redirect with sfs checks only that the fingers are the same,
# so it counts a repetition as a redirect_sfs (e.g. "dad" is a redirect_sfs).
#
def total_redirect_sfs(a, b, c):
    global oxey_mode
    if oxey_mode:
        return total_redirect(a, b, c) and a.finger == c.finger
    return total_redirect(a, b, c) and sfs(a, c)

def redirect_bad_sfs(a, b, c):
    return total_redirect_bad(a, b, c) and total_redirect_sfs(a, b, c)

def redirect_sfs(a, b, c):
    return total_redirect(a, b, c) and total_redirect_sfs(a, b, c) and not redirect_bad_sfs(a, b, c)

def redirect_bad(a, b, c):
    return total_redirect_bad(a, b, c) and not total_redirect_sfs(a, b, c)

def redirect(a, b, c):
    return total_redirect(a, b, c) and not total_redirect_sfs(a, b, c) and not total_redirect_bad(a, b, c)



# finger usage metrics
def finger_usage(finger, a):
    return a.finger == finger

def finger_0(a):
    return finger_usage(Finger(0), a)
def finger_1(a):
    return finger_usage(Finger(1), a)
def finger_2(a):
    return finger_usage(Finger(2), a)
def finger_3(a):
    return finger_usage(Finger(3), a)
def finger_4(a):
    return finger_usage(Finger(4), a)
def finger_5(a):
    return finger_usage(Finger(5), a)
def finger_6(a):
    return finger_usage(Finger(6), a)
def finger_7(a):
    return finger_usage(Finger(7), a)
def finger_8(a):
    return finger_usage(Finger(8), a)
def finger_9(a):
    return finger_usage(Finger(9), a)

def left_hand(a):
    return a.finger.hand == Hand.LEFT

def right_hand(a):
    return a.finger.hand == Hand.RIGHT


METRICS = [
    Metric(name="rep", description="single finger repetition", ngramType=NgramType.BIGRAM, function=rep),
    Metric(name="sfb", description="single finger bigram", ngramType=NgramType.BIGRAM, function=sfb),
    Metric(name="sfs", description="single finger skipgram", ngramType=NgramType.SKIPGRAM, function=sfs),
    Metric(name="sft", description="single finger trigram", ngramType=NgramType.TRIGRAM, function=sft),
    Metric(name="dist", description="ecludean distance squared", ngramType=NgramType.BIGRAM, function=distance_squared),
    Metric(name="dist_linear", description="ecludean distance", ngramType=NgramType.BIGRAM, function=distance_linear),
    Metric(name="lsb", description="lateral stretch bigram", ngramType=NgramType.BIGRAM, function=lsb),
    Metric(name="rowskip", description="row skip", ngramType=NgramType.BIGRAM, function=rowskip),
    Metric(name="scissors", description="full scissor bigram", ngramType=NgramType.BIGRAM, function=scissors),
    Metric(name="pinky_ring", description="pinky ring bigram", ngramType=NgramType.BIGRAM, function=pinky_ring),
    Metric(name="pinky_off", description="pinky off", ngramType=NgramType.MONOGRAM, function=pinky_off),
    Metric(name="same_hand", description="same hand trigram", ngramType=NgramType.TRIGRAM, function=same_hand),
    Metric(name="alt", description="alternates", ngramType=NgramType.TRIGRAM, function=alternate),
    Metric(name="alt_sfs", description="alternates with single finger skipgram", ngramType=NgramType.TRIGRAM, function=alternate_sfs),
    Metric(name="roll", description="total rolls", ngramType=NgramType.TRIGRAM, function=roll),
    Metric(name="in_roll", description="inward rolls", ngramType=NgramType.TRIGRAM, function=in_roll),
    Metric(name="out_roll", description="outward rolls", ngramType=NgramType.TRIGRAM, function=out_roll),
    Metric(name="roll3", description="trigram roll", ngramType=NgramType.TRIGRAM, function=roll3),
    Metric(name="redirect", description="redirect", ngramType=NgramType.TRIGRAM, function=redirect),
    Metric(name="redirect_sfs", description="redirect with single finger skipgram", ngramType=NgramType.TRIGRAM, function=redirect_sfs),
    Metric(name="redirect_bad", description="redirect bad (does not use index finger)", ngramType=NgramType.TRIGRAM, function=redirect_bad),
    Metric(name="redirect_bad_sfs", description="redirect bad with single finger skipgram", ngramType=NgramType.TRIGRAM, function=redirect_bad_sfs),
    Metric(name="total_redirect", description="total redirect", ngramType=NgramType.TRIGRAM, function=total_redirect),
    Metric(name="left_hand", description="left hand usage", ngramType=NgramType.MONOGRAM, function=left_hand),
    Metric(name="right_hand", description="right hand usage", ngramType=NgramType.MONOGRAM, function=right_hand),
] + [
    Metric(name=f"finger_{i}", description=f"finger {i} usage", ngramType=NgramType.MONOGRAM, function=globals()[f"finger_{i}"]) 
    for i in range(10)
] + [
    Metric(name=f"sfb_finger_{i}", description=f"single finger bigram with finger {i}", ngramType=NgramType.BIGRAM, function=globals()[f"sfb_finger_{i}"]) 
    for i in range(10)
]

# assert that the metric names are all unique
assert len(METRICS) == len(set(metric.name for metric in METRICS)), "Metric names must be unique"

if __name__ == "__main__":
    print(METRICS)
