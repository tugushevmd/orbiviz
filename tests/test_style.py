"""Tests for _style.py — visual theming."""
from pyqchem.viz._style import (
    text_color_from_rgba,
    get_display_radius,
    get_fill_alpha,
    get_atom_color,
    get_atom_edge_color,
)


class TestTextColor:
    def test_dark_background_gives_white(self):
        assert text_color_from_rgba((0.0, 0.0, 0.0, 1.0)) == "white"

    def test_light_background_gives_dark(self):
        assert text_color_from_rgba((1.0, 1.0, 1.0, 1.0)) == "#222222"


class TestDisplayRadius:
    def test_hydrogen_smallest(self):
        assert get_display_radius("H") < get_display_radius("C")

    def test_unknown_element_fallback(self):
        r = get_display_radius("Xx")
        assert r > 0


class TestAtomColor:
    def test_known_elements(self):
        assert get_atom_color("C") == "#909090"
        assert get_atom_color("N") == "#3050F8"

    def test_unknown_fallback(self):
        assert get_atom_color("Xx") == "#777777"

    def test_edge_color(self):
        assert get_atom_edge_color("C") == "#555555"
