from math import sqrt
from enum import Enum
from dataclasses import dataclass, field
from typing import Callable, Any

from hardware import FingerType, Position, Hand, Finger
from freqdist import NgramType

import re
from functools import cache

class Direction(Enum):
    INWARD = 0
    OUTWARD = 1
    REPEAT = 2
    ALTERNATE = 3

    @classmethod
    @cache
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
oxeylyzer_mode = False
def use_oxeylyzer_mode(mode: bool):
    global oxeylyzer_mode
    oxeylyzer_mode = mode

#
# Bigram metrics
#

# repetition
def rep(a, b):
    return a == b

# single finger bigram
def sfb(a, b):
    return a.finger == b.finger and a != b

# "good" sfbs, where the finger moves down a row
def sfb_rake(a, b):
    return sfb(a, b) and a.row + 1 == b.row

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

# sfb distances
def sfb_distance(a, b):
    return distance_squared(a, b) if sfb(a, b) else 0

def sfs_distance(a, b):
    return distance_squared(a, b) if sfs(a, b) else 0

def sft_distance(a, b, c):
    return distance_squared(a, b) + distance_squared(b, c) if sft(a, b, c) else 0


def lsb(a, b):
    '''
    LSB is Lateral Stretch Bigram

    From the Keyboard layouts doc:
    Adjacent finger bigrams where the horizontal distance is 2U or greater.
    Semi-adjacent finger bigrams where the horizontal distance is 3.5U or greater.

    Oxeylyzer definition:
    Middle and Index bigram, where Index is in the center column, with two exceptions:
    - on the right hand when index is on the bottom and middle is on the top
    - on the left hand when index is on the bottom and middle is on the home row
    (e.g., on ansi keyboard, qwerty "ni" and "db" are not LSB)
    '''
    if not same_hand(a,b):
        return False

    if a.finger.type == FingerType.THUMB or b.finger.type == FingerType.THUMB:
        return False
    
    if oxeylyzer_mode:
        if a.finger.type == FingerType.INDEX and b.finger.type == FingerType.MIDDLE:
            pass
        elif a.finger.type == FingerType.MIDDLE and b.finger.type == FingerType.INDEX:
            a, b = b, a
        else:
            return False
        
        if a.finger.hand == Hand.RIGHT and a.row - b.row >= 2:
            return False
        if a.finger.hand == Hand.LEFT and b.is_home and a.row == b.row + 1:
            return False

        return a.col == 4 or a.col == 5

    # Keyboard layouts doc definition
    # adjacent fingers
    if abs(a.finger.value - b.finger.value) == 1:
        return abs(a.x - b.x) >= 2
    
    # semi-adjacent fingers
    if abs(a.finger.value - b.finger.value) == 2:
        return abs(a.x - b.x) >= 3.5
    
    return False


# row skip
def rowskip(a, b):
    return abs(a.row - b.row) > 1 and same_hand(a, b) and a.finger.type != FingerType.THUMB and b.finger.type != FingerType.THUMB

def scissors_oxeylyzer_mode(a, b):
    '''
    oxeylyzer definition of scissors is encoded as a list of finger positions,
    and includes some single row stretches when pinky is higher than ring.

    instead of listing position locations, in this implementation we are 
    inferring the rules from the positions and rewriting as statements so that
    it can be more flexible to different hardware setups.
    '''
    if not rowskip(a, b):
        if same_hand(a,b) and (
            (a.finger.type == FingerType.PINKY and b.finger.type == FingerType.RING and a.row < b.row) or
            (a.finger.type == FingerType.RING and b.finger.type == FingerType.PINKY and a.row > b.row)
        ):
            return True
        return False

    if not same_hand(a, b):
        return False

    if a.finger.type.value > b.finger.type.value:
        a, b = b, a
    if a.finger.type == FingerType.PINKY and b.finger.type == FingerType.RING:
        return True
    if a.finger.type == FingerType.RING:
        return b.finger.type == FingerType.MIDDLE
    if a.finger.type == FingerType.MIDDLE:
        return b.finger.type == FingerType.INDEX and (
            (a.finger.hand == Hand.LEFT and abs(b.x - a.x) >= 2.0) or
            a.row > b.row
        )
    return False

def scissors_ortho(a,b):
    '''
    any time the pinky is in a different row than the ring, except when off by one with pinky lower
    any time the ring and middle are in a skip row, or the middle is lower than the ring
    
    any skip row where the middle is lower than the index
    any skip row where the index is in the center column, with the middle
    the index in the center column and lower than the middle (even a single row)

    scissors:
    qs qx ax zw pl p. ;. /o
    wc ex ,o .i ok l, sc wd
    cr ct ,u ,y in ev db eg ih kn
    be dt cg h, yk
    '''
    
    if not same_hand(a, b):
        return False

    if a.finger.type.value > b.finger.type.value:
        a, b = b, a

    if a.finger.type == FingerType.PINKY:
        return (
            b.finger.type == FingerType.RING and 
            a.row != b.row and 
            not (abs(a.row - b.row) == 1 and a.row > b.row)
        )

    elif a.finger.type == FingerType.RING:
        return b.finger.type == FingerType.MIDDLE and (
            rowskip(a, b) or 
            a.row < b.row
        )

    elif a.finger.type == FingerType.MIDDLE:
        return b.finger.type == FingerType.INDEX and (
            (a.row != b.row and abs(a.x - b.x) >= 2.0) or
            (rowskip(a, b) and a.row > b.row)
        )


    return False


def scissors(a, b):
    '''
    full scissor bigram (FSB)
    from the Keyboard layouts doc
    The index prefers being lower.
    The middle prefers being higher.
    The ring prefers being higher than index and pinky, but lower than the middle.
    The pinky prefers being lower than middle and ring, but higher than index.
    (implied: no scissors on thumbs)
    '''

    if a.finger.type == FingerType.THUMB or b.finger.type == FingerType.THUMB:
        return False

    if not same_hand(a, b):
        return False

    if oxeylyzer_mode:
        return scissors_oxeylyzer_mode(a, b)

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

#
# Trigram metrics
#
# note that trigram metrics are the most expensive to compute, so a little bit of care
# here can save you a second or more when creating a model instance.
#
def alternate_total(a, b, c):
    return a.finger.hand == c.finger.hand and a.finger.hand != b.finger.hand

def alternate(a, b, c):
    return alternate_total(a, b, c) and not alternate_sfs(a, b, c)

def alternate_sfs(a, b, c):
    global oxeylyzer_mode
    if oxeylyzer_mode:
        return a.finger.hand == c.finger.hand and a.finger.hand != b.finger.hand and (
            a.finger == c.finger
        )

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
def redirect_total(a, b, c):
    direction_a_b = Direction.which(a, b)
    direction_b_c = Direction.which(b, c)
    return (
        same_hand(a, b, c) and (
            (direction_a_b == Direction.INWARD and direction_b_c == Direction.OUTWARD) or
            (direction_a_b == Direction.OUTWARD and direction_b_c == Direction.INWARD)
        )    
    )

def redirect_bad_total(a, b, c):
    return redirect_total(a, b, c) and (
        a.finger.type != FingerType.INDEX and 
        b.finger.type != FingerType.INDEX and 
        c.finger.type != FingerType.INDEX
    )

#
# Note that oxeylyzer definition of redirect with sfs checks only that the fingers are the same,
# so it counts a repetition as a redirect_sfs (e.g. "dad" is a bad redirect sfs).
#
def redirect_sfs_total(a, b, c):
    global oxeylyzer_mode
    if oxeylyzer_mode:
        return redirect_total(a, b, c) and a.finger == c.finger
    return redirect_total(a, b, c) and sfs(a, c)

def redirect_bad_sfs(a, b, c):
    return redirect_bad_total(a, b, c) and redirect_sfs_total(a, b, c)

def redirect_sfs(a, b, c):
    return redirect_total(a, b, c) and redirect_sfs_total(a, b, c) and not redirect_bad_sfs(a, b, c)

def redirect_bad(a, b, c):
    return redirect_bad_total(a, b, c) and not redirect_sfs_total(a, b, c)

def redirect(a, b, c):
    return redirect_total(a, b, c) and not redirect_sfs_total(a, b, c) and not redirect_bad_total(a, b, c)



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

def heat(a):
    return a.heat

def home(a):
    return a.is_home


METRICS = [
    Metric(name="rep", description="single finger repetition", ngramType=NgramType.BIGRAM, function=rep),
    Metric(name="sfb", description="single finger bigram", ngramType=NgramType.BIGRAM, function=sfb),
    Metric(name="sfb_rake", description="single finger bigram where the finger moves down a row", ngramType=NgramType.BIGRAM, function=sfb_rake),
    Metric(name="sfs", description="single finger skipgram", ngramType=NgramType.SKIPGRAM, function=sfs),
    Metric(name="sft", description="single finger trigram", ngramType=NgramType.TRIGRAM, function=sft),
    Metric(name="home", description="home position", ngramType=NgramType.MONOGRAM, function=home),
    Metric(name="dist", description="ecludean distance squared", ngramType=NgramType.BIGRAM, function=distance_squared),
    Metric(name="dist_linear", description="ecludean distance", ngramType=NgramType.BIGRAM, function=distance_linear),
    Metric(name="sfb_dist", description="single finger bigram distance squared", ngramType=NgramType.BIGRAM, function=sfb_distance),
    Metric(name="sfs_dist", description="single finger skipgram distance squared", ngramType=NgramType.SKIPGRAM, function=sfs_distance),
    Metric(name="sft_dist", description="single finger trigram distance squared", ngramType=NgramType.TRIGRAM, function=sft_distance),
    Metric(name="heat", description="heat of a key press", ngramType=NgramType.MONOGRAM, function=heat),
    Metric(name="lsb", description="lateral stretch bigram", ngramType=NgramType.BIGRAM, function=lsb),
    Metric(name="rowskip", description="row skip where a bigram on the same hand is separated by more than 1 row", ngramType=NgramType.BIGRAM, function=rowskip),
    Metric(name="scissors", description="full scissor bigram", ngramType=NgramType.BIGRAM, function=scissors),
    Metric(name="scissors_ortho", description="ortho scissor bigram", ngramType=NgramType.BIGRAM, function=scissors_ortho),
    Metric(name="pinky_ring", description="pinky ring bigram", ngramType=NgramType.BIGRAM, function=pinky_ring),
    Metric(name="pinky_off", description="pinky is not home", ngramType=NgramType.MONOGRAM, function=pinky_off),
    Metric(name="same_hand", description="same hand trigram", ngramType=NgramType.TRIGRAM, function=same_hand),
    Metric(name="alt", description="alternates", ngramType=NgramType.TRIGRAM, function=alternate),
    Metric(name="alt_sfs", description="alternates with single finger skipgram", ngramType=NgramType.TRIGRAM, function=alternate_sfs),
    Metric(name="alt_total", description="alternates total", ngramType=NgramType.TRIGRAM, function=alternate_total),
    Metric(name="roll", description="total rolls", ngramType=NgramType.TRIGRAM, function=roll),
    Metric(name="in_roll", description="inward rolls", ngramType=NgramType.TRIGRAM, function=in_roll),
    Metric(name="out_roll", description="outward rolls", ngramType=NgramType.TRIGRAM, function=out_roll),
    Metric(name="roll3", description="trigram roll", ngramType=NgramType.TRIGRAM, function=roll3),
    Metric(name="redirect", description="redirects that are not sfs or bad", ngramType=NgramType.TRIGRAM, function=redirect),
    Metric(name="redirect_sfs", description="redirect with single finger skipgram", ngramType=NgramType.TRIGRAM, function=redirect_sfs),
    Metric(name="redirect_bad", description="redirect bad (does not use index finger)", ngramType=NgramType.TRIGRAM, function=redirect_bad),
    Metric(name="redirect_bad_sfs", description="redirect bad with single finger skipgram", ngramType=NgramType.TRIGRAM, function=redirect_bad_sfs),
    Metric(name="redirect_total", description="total redirect", ngramType=NgramType.TRIGRAM, function=redirect_total),
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
