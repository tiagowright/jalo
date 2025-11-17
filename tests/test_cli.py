
import unittest
from pathlib import Path
from unittest.mock import patch, MagicMock
from io import StringIO
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from jalo import main

class TestJaloCLI(unittest.TestCase):

    @patch('jalo.JaloShell')
    def test_analyze_command(self, mock_shell):
        instance = mock_shell.return_value
        main(['analyze', 'qwerty'])
        instance.do_analyze.assert_called_once_with('qwerty')

    @patch('jalo.JaloShell')
    def test_contributions_command(self, mock_shell):
        instance = mock_shell.return_value
        main(['contributions', 'qwerty', 'dvorak'])
        instance.do_contributions.assert_called_once_with('qwerty dvorak')

    @patch('jalo.JaloShell')
    def test_generate_command(self, mock_shell):
        instance = mock_shell.return_value
        main(['generate', '50', '10'])
        instance.do_generate.assert_called_once_with('50 10')

    @patch('jalo.JaloShell')
    def test_generate_command_defaults(self, mock_shell):
        instance = mock_shell.return_value
        main(['generate'])
        instance.do_generate.assert_called_once_with('100 20')

    @patch('jalo.JaloShell')
    def test_improve_command(self, mock_shell):
        instance = mock_shell.return_value
        main(['improve', 'qwerty', '5'])
        instance.do_improve.assert_called_once_with('qwerty 5')

    @patch('jalo.JaloShell')
    def test_improve_command_defaults(self, mock_shell):
        instance = mock_shell.return_value
        main(['improve', 'qwerty'])
        instance.do_improve.assert_called_once_with('qwerty 10')

    @patch('jalo.JaloShell')
    def test_metrics_command(self, mock_shell):
        instance = mock_shell.return_value
        main(['metrics'])
        instance.do_metrics.assert_called_once_with('')

    @patch('jalo.JaloShell')
    def test_compare_command(self, mock_shell):
        instance = mock_shell.return_value
        main(['compare', 'qwerty', 'dvorak'])
        instance.do_compare.assert_called_once_with('qwerty dvorak')

if __name__ == '__main__':
    unittest.main()
