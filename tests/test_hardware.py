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

    def test_ortho_thumb(self):
        hw = KeyboardHardware.from_name('ortho_thumb')
        layout = KeyboardLayout.from_name('hdgold', hw)
        self.assertIsNotNone(layout)

    def test_ortho(self):
        hw = KeyboardHardware.from_name('ortho')
        layout = KeyboardLayout.from_name('qwerty', hw)
        self.assertIsNotNone(layout)

    # use the hint
    def test_ansi_32_hint(self):
        layout = KeyboardLayout.from_name('hdneu')
        self.assertIsNotNone(layout)
        self.assertEqual(layout.hardware.name, 'ansi_32')

    def test_ansi_angle_hint(self):
        layout = KeyboardLayout.from_name('inrolly')
        self.assertIsNotNone(layout)
        self.assertEqual(layout.hardware.name, 'ansi_angle')


if __name__ == '__main__':
    unittest.main()
