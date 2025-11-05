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
        
        if a == b:
            return Direction.REPEAT
        
        if a.col < b.col and a.finger.hand == Hand.LEFT:
            return Direction.INWARD
        elif a.col > b.col and a.finger.hand == Hand.RIGHT:
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


# Helper functions
def same_finger(a, b):
    return a.finger == b.finger

def same_hand(a, b, c=None):
    if c is None:
        return a.finger.hand == b.finger.hand
    return a.finger.hand == b.finger.hand and a.finger.hand == c.finger.hand



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

# row skip
def rowskip(a, b):
    return abs(a.row - b.row) > 1 and same_hand(a, b)

# full scissor bigram (FSB)
# from the Keyboard layouts doc
# The index prefers being lower.
# The middle prefers being higher.
# The ring prefers being higher than index and pinky, but lower than the middle.
# The pinky prefers being lower than middle and ring, but higher than index.
def fsb(a, b):
    if not rowskip(a, b):
        return False

    if a.row < b.row:
        a, b = b, a
    
    if a.finger.type == FingerType.INDEX:
        return True
    elif a.finger.type == FingerType.MIDDLE:
        return False
    elif a.finger.type == FingerType.RING:
        return b.FingerType == FingerType.MIDDLE
    elif a.finger.type == FingerType.PINKY:
        return (b.FingerType == FingerType.MIDDLE or b.FingerType == FingerType.RING)

    return False

# FSB above are sensible for staggered keyboard planks, but not great for split ortho.
def fsb_split_ortho(a, b):
    # TODO
    pass

# Trigram metrics

def alternate(a, b, c):
    return a.finger.hand == c.finger.hand and a.finger.hand != b.finger.hand

def alternate_sfs(a, b, c):
    return alternate(a, b, c) and sfs(a, c)

def roll3(a, b, c):
    return same_hand(a, b, c) and Direction.which(a, b) == Direction.which(b, c)

def roll(a, b, c):
    return (a.finger.hand == b.finger.hand or b.finger.hand == c.finger.hand) and (a.finger.hand != c.finger.hand)

def in_roll(a, b, c):
    return (a.finger.hand != c.finger.hand) and (
        (a.finger.hand == b.finger.hand and Direction.which(a, b) == Direction.INWARD) or 
        (b.finger.hand == c.finger.hand and Direction.which(b, c) == Direction.INWARD)
    )

def out_roll(a, b, c):
    return (a.finger.hand != c.finger.hand) and (
        (a.finger.hand == b.finger.hand and Direction.which(a, b) == Direction.OUTWARD) or 
        (b.finger.hand == c.finger.hand and Direction.which(b, c) == Direction.OUTWARD)
    )

def redirect(a, b, c):
    return same_hand(a, b, c) and Direction.which(a, b) != Direction.which(b, c)

def redirect_bad(a, b, c):
    return redirect(a, b, c) and (
        a.finger.type() != FingerType.INDEX and 
        b.finger.type() != FingerType.INDEX and 
        c.finger.type() != FingerType.INDEX
    )

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
    Metric(name="rowskip", description="row skip", ngramType=NgramType.BIGRAM, function=rowskip),
    Metric(name="fsb", description="full scissor bigram", ngramType=NgramType.BIGRAM, function=fsb),
    Metric(name="alt", description="alternates", ngramType=NgramType.TRIGRAM, function=alternate),
    Metric(name="alt_sfs", description="alternates with single finger skipgram", ngramType=NgramType.TRIGRAM, function=alternate_sfs),
    Metric(name="roll3", description="trigram roll", ngramType=NgramType.TRIGRAM, function=roll3),
    Metric(name="roll", description="total rolls", ngramType=NgramType.TRIGRAM, function=roll),
    Metric(name="in_roll", description="inward rolls", ngramType=NgramType.TRIGRAM, function=in_roll),
    Metric(name="out_roll", description="outward rolls", ngramType=NgramType.TRIGRAM, function=out_roll),
    Metric(name="redirect", description="redirect", ngramType=NgramType.TRIGRAM, function=redirect),
    Metric(name="redirect_bad", description="redirect bad", ngramType=NgramType.TRIGRAM, function=redirect_bad),
    Metric(name="left_hand", description="left hand usage", ngramType=NgramType.MONOGRAM, function=left_hand),
    Metric(name="right_hand", description="right hand usage", ngramType=NgramType.MONOGRAM, function=right_hand),
] + [
    Metric(name=f"finger_{i}", description=f"finger {i} usage", ngramType=NgramType.MONOGRAM, function=globals()[f"finger_{i}"]) 
    for i in range(10)
] + [
    Metric(name=f"sfb_finger_{i}", description=f"single finger bigram with finger {i}", ngramType=NgramType.BIGRAM, function=globals()[f"sfb_finger_{i}"]) 
    for i in range(10)
]

if __name__ == "__main__":
    print(METRICS)
