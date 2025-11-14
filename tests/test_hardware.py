import unittest
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from layout import KeyboardLayout
from hardware import KeyboardHardware

class TestHardware(unittest.TestCase):
    def test_ansi_32(self):
        hw = KeyboardHardware.from_name('ansi_32')
        layout = KeyboardLayout.from_name('hdneu', hw)
        self.assertIsNotNone(layout)

    def test_ansi_angle(self):
        hw = KeyboardHardware.from_name('ansi_angle')
        layout = KeyboardLayout.from_name('inrolly', hw)
        self.assertIsNotNone(layout)

    def test_ansi(self):
        hw = KeyboardHardware.from_name('ansi')
        layout = KeyboardLayout.from_name('qwerty', hw)
        self.assertIsNotNone(layout)

    # def test_ortho_22(self):
    #     hw = KeyboardHardware.from_name('ortho_22')
    #     layout = KeyboardLayout.from_name('qwerty', hw)
    #     self.assertIsNotNone(layout)

    def test_ortho_thumb(self):
        hw = KeyboardHardware.from_name('ortho_thumb')
        layout = KeyboardLayout.from_name('hdgold', hw)
        self.assertIsNotNone(layout)

    def test_ortho(self):
        hw = KeyboardHardware.from_name('ortho')
        layout = KeyboardLayout.from_name('qwerty', hw)
        self.assertIsNotNone(layout)

if __name__ == '__main__':
    unittest.main()
