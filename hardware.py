'''
These classes define the physical attributes of keyboard hardware
'''

from dataclasses import dataclass
from enum import Enum, unique
from typing import List
from collections import defaultdict
import os

@unique
class Hand(Enum):
    """
    Represent the hand of a keyboard user.
    """
    LEFT = 0
    RIGHT = 1

@unique
class FingerType(Enum):
    """
    Represent the type of a finger.
    """
    PINKY = 0
    RING = 1
    MIDDLE = 2
    INDEX = 3
    THUMB = 4

@unique
class Finger(Enum):
    """
    Represent the finger of a keyboard key.
    """
    LP = 0
    LR = 1
    LM = 2
    LI = 3
    LT = 4

    RT = 5
    RI = 6
    RM = 7
    RR = 8
    RP = 9

    @property
    def hand(self) -> Hand:
        """Return the hand that owns this finger."""
        return Hand(self.value // 5)

    def type(self) -> FingerType:
        """
        Return the type of the finger.
        """
        if self.value < 5:
            return FingerType(self.value)

        return FingerType(9 - self.value)
    

@dataclass(frozen=True)
class Position:
    """
    Represent the physical and logical position of a keyboard key.

    Attributes
    ----------
    row : int
        The logical row index of the key.
    col : int
        The logical column index of the key.
    x : float
        The X-coordinate of the key in physical units (e.g., millimeters).
    y : float
        The Y-coordinate of the key in physical units (e.g., millimeters).
    finger : Finger
        The finger assigned to press the key (domain-specific encoding).
    """
    row: int
    col: int
    x: float
    y: float
    finger: Finger
    is_home: bool = False
    effort: float = 0.0

    def __eq__(self, other: 'Position') -> bool:
        return self.row == other.row and self.col == other.col and self.finger == other.finger

    def __hash__(self) -> int:
        return hash((self.row, self.col, self.finger))


class KeyboardHardware:
    """
    Represent the physical layout of a keyboard.

    Attributes
    ----------
    positions : List[Position]
        The positions of the keys on the keyboard.
    rows : List[int]
        The rows of the keyboard.
    cols : List[int]
        The columns of the keyboard.
    finger_to_positions : Dict[Finger, List[Position]]
        The positions of the keys assigned to each finger.
    """
    def __init__(self, name: str, positions: List[Position]):
        self.name = name
        self.positions = sorted(positions, key=lambda x: (x.row, x.col))

        self.rows = list(set(position.row for position in self.positions))
        self.cols = list(set(position.col for position in self.positions))

        self.finger_to_positions = defaultdict(list)
        self.grid = defaultdict(lambda: defaultdict(list))
        for position in self.positions:
            self.finger_to_positions[position.finger].append(position)
            self.grid[position.row][position.col].append(position)

    def __len__(self) -> int:
        return len(self.positions)

    @classmethod
    def from_name(cls, name: str) -> 'KeyboardHardware':
        """
        Create a KeyboardHardware instance from a name.
        
        Parameters
        ----------
        name : str
            The name of the keyboard hardware module in the keebs directory.
        
        Returns
        -------
        KeyboardHardware
            The keyboard hardware instance from the specified module.
        """
        # programtically find the file/module with the specified name then return the KEYBOARD element from it
        # the file/module is in the keebs directory
        keebs_dir = os.path.join('keebs', name)
        import importlib
        module = importlib.import_module('keebs.' + name)
        return module.KEYBOARD


    def str(self, show_finger_numbers: bool = False, show_finger_names: bool = False, show_stagger: bool = False) -> str:
        '''
        Show the keyboard layout in a human-readable format.

        Parameters
        ----------
        show_finger_numbers : bool
            Show the finger numbers.
        show_finger_names : bool
            Show the finger names.
        show_stagger : bool
            Show the stagger. [TODO: implement]
        '''

        def _str_position(position: Position) -> str:
            s = ''
            if show_finger_numbers:
                s += f'{position.finger.value:1d}'
            if show_finger_names:
                s += f'{position.finger.name}'
            if not s:
                s = '.'
            return s

        sorted_positions = sorted(self.positions, key=lambda x: (x.row, x.col))

        if not sorted_positions:
            return ''
        
        prev_position = sorted_positions[0]
        prev_row_x = prev_position.x
        row_stagger = ''
        s = _str_position(prev_position)

        for position in sorted_positions[1:]:
            if position.row != prev_position.row:
                s += '\n'
                if show_stagger:
                    if position.x > prev_row_x:
                        row_stagger += ' '
                    elif position.x < prev_row_x and len(row_stagger) > 0:
                        row_stagger = row_stagger[:-1]
                    s += row_stagger
                    prev_row_x = position.x


            else:
                if position.col != prev_position.col:
                    s += ' '
                if show_stagger and position.finger.hand != prev_position.finger.hand:
                    s += '  '
            s += _str_position(position)
            prev_position = position
        
        return s

if __name__ == "__main__":
    for name in ['ansi', 'ortho']:
        keyboard = KeyboardHardware.from_name(name)
        print(f'{name}:')
        print(keyboard.str(show_finger_numbers=True, show_stagger=True))
        print()
        print(keyboard.str(show_finger_names=True, show_stagger=True))
        print()
        print()
