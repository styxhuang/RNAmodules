import os
from pathlib import Path
import unittest
from unittest.mock import patch
from io import StringIO

TEST_DIR = Path(os.path.dirname(__file__))
class TestArgparse(unittest.TestCase):

    def test_rna_short_main(self):
        from RNA import main
        config_file     = TEST_DIR / 'assets_small_train_data' / 'config.yaml'
        output_file     = TEST_DIR / 'Results' / 'test_1'

        test_args = [
            "-i", str(config_file),
            "-o", str(output_file),
            "-t",
            "-p",
            "-v"
        ]
        with patch("sys.stdout", new=StringIO()) as fake_out:
            parsed_args = main(test_args)

    def test_rna_large_main(self):
        from RNA import main
        config_file     = TEST_DIR / 'assets_large_train_data' / 'config.yaml'
        output_file     = TEST_DIR / 'Results' / 'test_2'

        test_args = [
            "-i", str(config_file),
            "-o", str(output_file),
            "-t",
            "-p",
            "-v"
        ]
        with patch("sys.stdout", new=StringIO()) as fake_out:
            parsed_args = main(test_args)

if __name__ == "__main__":
    unittest.main()