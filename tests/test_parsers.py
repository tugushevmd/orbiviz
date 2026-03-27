"""Tests for _parsers.py — auto_parse and format detection."""
import pytest
from pathlib import Path

from pyqchem.viz._parsers import auto_parse

DATA_DIR = Path(__file__).parent / "data"


class TestAutoParse:
    def test_xyz(self):
        result = auto_parse(DATA_DIR / "water.xyz")
        assert result["format"] == "xyz"
        assert len(result["atoms"]) == 3

    def test_unsupported_format(self):
        # Create a temp file with unknown extension
        tmp = DATA_DIR / "dummy.zzz"
        try:
            tmp.write_text("dummy")
            with pytest.raises(ValueError, match="Unsupported"):
                auto_parse(tmp)
        finally:
            tmp.unlink(missing_ok=True)
