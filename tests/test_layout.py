import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from hardware import KeyboardHardware, Finger, Hand
from layout import KeyboardLayout, LayoutKey


def test_load_layout_by_name():
    """Test loading a layout by name from the layouts directory."""
    layout = KeyboardLayout.from_name('qwerty', KeyboardHardware.from_name('ansi'))
    assert layout.name == 'qwerty'
    assert layout.hardware.name == 'ansi'
    assert len(layout.keys) > 0


def test_load_layout_with_hardware_hint():
    """Test loading a layout that specifies its hardware in a comment."""
    layout = KeyboardLayout.from_name('hdneu')
    assert layout.name == 'hdneu'
    assert layout.hardware.name is not None


def test_layout_from_text():
    """Test creating a layout from a text grid."""
    text_grid = """q w e r t  y u i o p
a s d f g  h j k l ;
z x c v b  n m , . /"""
    hardware = KeyboardHardware.from_name('ortho')
    layout = KeyboardLayout.from_text(text_grid, 'test_layout', hardware)
    assert layout.name == 'test_layout'
    assert len(layout.keys) > 0


def test_layout_str_representation():
    """Test that __str__ produces a readable grid format."""
    layout = KeyboardLayout.from_name('qwerty', KeyboardHardware.from_name('ansi'))
    layout_str = str(layout)
    assert isinstance(layout_str, str)
    assert len(layout_str) > 0
    # Should have multiple lines (rows)
    assert '\n' in layout_str


def test_layout_char_to_key_mapping():
    """Test that char_to_key mapping is correctly populated."""
    layout = KeyboardLayout.from_name('qwerty', KeyboardHardware.from_name('ansi'))
    assert 'q' in layout.char_to_key
    assert 'w' in layout.char_to_key
    assert layout.char_to_key['q'].char == 'q'
    assert layout.char_to_key['w'].char == 'w'


def test_layout_key_at_position_mapping():
    """Test that key_at_position mapping is correctly populated."""
    layout = KeyboardLayout.from_name('qwerty', KeyboardHardware.from_name('ansi'))
    # Check that all hardware positions have corresponding keys
    for position in layout.hardware.positions:
        assert position in layout.key_at_position
        assert layout.key_at_position[position].position == position


def test_layout_validation_all_positions_covered():
    """Test that layout validation ensures all hardware positions are covered."""
    hardware = KeyboardHardware.from_name('ansi')
    # Create a layout missing some keys
    incomplete_keys = [
        LayoutKey.from_position(pos, 'a')
        for pos in hardware.positions[:-1]  # Missing one position
    ]
    with pytest.raises(ValueError, match="Keys must point to all positions"):
        KeyboardLayout(incomplete_keys, hardware, 'incomplete')


def test_layout_validation_unique_positions():
    """Test that layout validation ensures keys point to unique positions."""
    hardware = KeyboardHardware.from_name('ansi')
    position = hardware.positions[0]
    # Create keys pointing to the same position
    duplicate_keys = [
        LayoutKey.from_position(position, 'a'),
        LayoutKey.from_position(position, 'b'),
    ]
    with pytest.raises(ValueError, match="Keys must point to unique positions"):
        KeyboardLayout(duplicate_keys, hardware, 'duplicate')


def test_mirror():
    """Test that mirror() correctly mirrors a layout by swapping left and right hands."""
    layout = KeyboardLayout.from_name('hdpm')
    mirrored = layout.mirror()
    
    # Verify mirrored layout has correct name
    assert mirrored.name == 'hdpm mirrored'
    
    # Verify the string representation matches expected output
    # The user-provided format includes blank lines between rows for readability,
    # but str() produces no blank lines, so we extract and compare content lines
    expected_str = """b y o u ;   x l d p f z

c i e a ,   k h t n s q

/ = ' . -   j m g w v  

        r"""
    
    actual_str = str(mirrored)
    
    # Extract non-empty lines from both (ignoring blank lines in expected)
    expected_lines = [line.rstrip() for line in expected_str.split('\n') if line.strip()]
    actual_lines = [line.rstrip() for line in actual_str.split('\n') if line.strip()]
    
    assert actual_lines == expected_lines, \
        f"Mirrored layout mismatch:\n  actual:\n{chr(10).join('    ' + line for line in actual_lines)}\n  expected:\n{chr(10).join('    ' + line for line in expected_lines)}"


def test_invert():
    """Test that invert() correctly inverts a layout for the specified hand."""
    layout = KeyboardLayout.from_name('hdpm')
    inverted = layout.invert(hand=Hand.RIGHT)
    
    # Verify inverted layout has correct name
    assert inverted.name == 'hdpm inverted right hand'
    
    # Verify the string representation matches expected output
    # The user-provided format includes blank lines between rows for readability,
    # but str() produces no blank lines, so we extract and compare content lines
    expected_str = """f p d l x   - . ' = / z

s n t h k   , a e i c q

v w g m j   ; u o y b  

        r"""
    
    actual_str = str(inverted)
    
    # Extract non-empty lines from both (ignoring blank lines in expected)
    expected_lines = [line.rstrip() for line in expected_str.split('\n') if line.strip()]
    actual_lines = [line.rstrip() for line in actual_str.split('\n') if line.strip()]
    
    assert actual_lines == expected_lines, \
        f"Inverted layout mismatch:\n  actual:\n{chr(10).join('    ' + line for line in actual_lines)}\n  expected:\n{chr(10).join('    ' + line for line in expected_lines)}"


def test_mirror_with_custom_name():
    """Test that mirror() accepts a custom name."""
    layout = KeyboardLayout.from_name('qwerty', KeyboardHardware.from_name('ansi'))
    mirrored = layout.mirror('custom_mirrored')
    assert mirrored.name == 'custom_mirrored'


def test_mirror_preserves_hardware():
    """Test that mirror() preserves the hardware."""
    layout = KeyboardLayout.from_name('hdpm')
    mirrored = layout.mirror()
    assert mirrored.hardware == layout.hardware
    assert mirrored.hardware.name == layout.hardware.name


def test_mirror_creates_new_layout():
    """Test that mirror() creates a new layout instance."""
    layout = KeyboardLayout.from_name('qwerty', KeyboardHardware.from_name('ansi'))
    mirrored = layout.mirror()
    assert mirrored is not layout
    assert mirrored.keys is not layout.keys

