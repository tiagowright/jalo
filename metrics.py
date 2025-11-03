from math import sqrt
from enum import Enum
from dataclasses import dataclass
from typing import Callable, Any

from hardware import FingerType, Position, Hand

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
    function: Callable[[Position], float] | Callable[[Position, Position], float] | Callable[[Position, Position, Position], float]

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

# single finger skipgram
def sfs(a, b):
    return sfb(a, b)

# single finger trigram
def sft(a, b, c):
    return sfb(a, b) and sfb(b, c)

# distance squared
def distance_squared(a, b):
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

def roll3(a, b, c):
    return same_hand(a, b, c) and Direction.which(a, b) == Direction.which(b, c)

def roll(a, b, c):
    return (a.finger.hand == b.finger.hand or b.finger.hand == c.finger.hand) and (a.finger.hand != c.finger.hand)

def redirect(a, b, c):
    return same_hand(a, b, c) and Direction.which(a, b) != Direction.which(b, c)

METRICS = [
    Metric(name="rep", description="single finger repetition", ngramType=NgramType.BIGRAM, function=rep),
    Metric(name="sfb", description="single finger bigram", ngramType=NgramType.BIGRAM, function=sfb),
    Metric(name="sfs", description="single finger skipgram", ngramType=NgramType.SKIPGRAM, function=sfs),
    Metric(name="sft", description="single finger trigram", ngramType=NgramType.TRIGRAM, function=sft),
    Metric(name="dist", description="ecludean distance squared", ngramType=NgramType.BIGRAM, function=distance_squared),
    Metric(name="dist_linear", description="ecludean distance", ngramType=NgramType.BIGRAM, function=distance_linear),
    Metric(name="rowskip", description="row skip", ngramType=NgramType.BIGRAM, function=rowskip),
    Metric(name="fsb", description="full scissor bigram", ngramType=NgramType.BIGRAM, function=fsb),
]

if __name__ == "__main__":
    print(METRICS)