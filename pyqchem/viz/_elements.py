"""Full periodic table data for visualization: radii, CPK colors, masses."""
from __future__ import annotations

from typing import NamedTuple


class ElementData(NamedTuple):
    Z: int
    symbol: str
    name: str
    mass: float
    covalent_radius: float   # Å, Cordero et al. 2008
    vdw_radius: float        # Å, Bondi / Mantina
    cpk_color: str           # hex, Jmol/CPK scheme
    edge_color: str          # hex, darker variant for outlines


# fmt: off
_DATA = [
    # Z   sym   name            mass       cov_r  vdw_r  cpk_color  edge_color
    (  1, "H",  "Hydrogen",      1.008,    0.31,  1.20,  "#FFFFFF", "#AAAAAA"),
    (  2, "He", "Helium",        4.003,    0.28,  1.40,  "#D9FFFF", "#88CCCC"),
    (  3, "Li", "Lithium",       6.941,    1.28,  1.82,  "#CC80FF", "#8844BB"),
    (  4, "Be", "Beryllium",     9.012,    0.96,  1.53,  "#C2FF00", "#88BB00"),
    (  5, "B",  "Boron",        10.811,    0.84,  1.92,  "#FFB5B5", "#CC8888"),
    (  6, "C",  "Carbon",       12.011,    0.76,  1.70,  "#909090", "#555555"),
    (  7, "N",  "Nitrogen",     14.007,    0.71,  1.55,  "#3050F8", "#1830AA"),
    (  8, "O",  "Oxygen",       15.999,    0.66,  1.52,  "#FF0D0D", "#CC0000"),
    (  9, "F",  "Fluorine",     18.998,    0.57,  1.47,  "#90E050", "#60AA30"),
    ( 10, "Ne", "Neon",         20.180,    0.58,  1.54,  "#B3E3F5", "#77AACC"),
    ( 11, "Na", "Sodium",       22.990,    1.66,  2.27,  "#AB5CF2", "#7733BB"),
    ( 12, "Mg", "Magnesium",    24.305,    1.41,  1.73,  "#8AFF00", "#66BB00"),
    ( 13, "Al", "Aluminium",    26.982,    1.21,  1.84,  "#BFA6A6", "#887777"),
    ( 14, "Si", "Silicon",      28.086,    1.11,  2.10,  "#F0C8A0", "#AA8866"),
    ( 15, "P",  "Phosphorus",   30.974,    1.07,  1.80,  "#FF8000", "#CC6600"),
    ( 16, "S",  "Sulfur",       32.065,    1.05,  1.80,  "#FFFF30", "#BBBB00"),
    ( 17, "Cl", "Chlorine",     35.453,    1.02,  1.75,  "#1FF01F", "#15AA15"),
    ( 18, "Ar", "Argon",        39.948,    1.06,  1.88,  "#80D1E3", "#559EAA"),
    ( 19, "K",  "Potassium",    39.098,    2.03,  2.75,  "#8F40D4", "#662299"),
    ( 20, "Ca", "Calcium",      40.078,    1.76,  2.31,  "#3DFF00", "#2EBB00"),
    ( 21, "Sc", "Scandium",     44.956,    1.70,  2.11,  "#E6E6E6", "#AAAAAA"),
    ( 22, "Ti", "Titanium",     47.867,    1.60,  1.87,  "#BFC2C7", "#888C90"),
    ( 23, "V",  "Vanadium",     50.942,    1.53,  1.79,  "#A6A6AB", "#777780"),
    ( 24, "Cr", "Chromium",     51.996,    1.39,  1.89,  "#8A99C7", "#5E6E8E"),
    ( 25, "Mn", "Manganese",    54.938,    1.39,  1.97,  "#9C7AC7", "#6E558E"),
    ( 26, "Fe", "Iron",         55.845,    1.32,  1.94,  "#E06633", "#AA4422"),
    ( 27, "Co", "Cobalt",       58.933,    1.26,  1.92,  "#F090A0", "#AA6070"),
    ( 28, "Ni", "Nickel",       58.693,    1.24,  1.63,  "#50D050", "#38AA38"),
    ( 29, "Cu", "Copper",       63.546,    1.32,  1.40,  "#C88033", "#8E5A22"),
    ( 30, "Zn", "Zinc",         65.380,    1.22,  1.39,  "#7D80B0", "#585A80"),
    ( 31, "Ga", "Gallium",      69.723,    1.22,  1.87,  "#C28F8F", "#886666"),
    ( 32, "Ge", "Germanium",    72.630,    1.20,  2.11,  "#668F8F", "#476666"),
    ( 33, "As", "Arsenic",      74.922,    1.19,  1.85,  "#BD80E3", "#8555AA"),
    ( 34, "Se", "Selenium",     78.971,    1.20,  1.90,  "#FFA100", "#CC8000"),
    ( 35, "Br", "Bromine",      79.904,    1.20,  1.85,  "#A62929", "#772020"),
    ( 36, "Kr", "Krypton",      83.798,    1.16,  2.02,  "#5CB8D1", "#408090"),
    ( 37, "Rb", "Rubidium",     85.468,    2.20,  3.03,  "#702EB0", "#501E80"),
    ( 38, "Sr", "Strontium",    87.620,    1.95,  2.49,  "#00FF00", "#00BB00"),
    ( 39, "Y",  "Yttrium",      88.906,    1.90,  2.19,  "#94FFFF", "#66BBBB"),
    ( 40, "Zr", "Zirconium",    91.224,    1.75,  1.86,  "#94E0E0", "#66AAAA"),
    ( 41, "Nb", "Niobium",      92.906,    1.64,  2.07,  "#73C2C9", "#508A90"),
    ( 42, "Mo", "Molybdenum",   95.950,    1.54,  2.09,  "#54B5B5", "#3A8080"),
    ( 43, "Tc", "Technetium",   98.000,    1.47,  2.09,  "#3B9E9E", "#2A7070"),
    ( 44, "Ru", "Ruthenium",   101.070,    1.46,  2.07,  "#248F8F", "#1A6666"),
    ( 45, "Rh", "Rhodium",     102.906,    1.42,  1.95,  "#0A7D8C", "#075A66"),
    ( 46, "Pd", "Palladium",   106.420,    1.39,  2.02,  "#006985", "#004D62"),
    ( 47, "Ag", "Silver",      107.868,    1.45,  1.72,  "#C0C0C0", "#888888"),
    ( 48, "Cd", "Cadmium",     112.414,    1.44,  1.58,  "#FFD98F", "#CCAA66"),
    ( 49, "In", "Indium",      114.818,    1.42,  1.93,  "#A67573", "#775350"),
    ( 50, "Sn", "Tin",         118.710,    1.39,  2.17,  "#668080", "#475858"),
    ( 51, "Sb", "Antimony",    121.760,    1.39,  2.06,  "#9E63B5", "#6E4580"),
    ( 52, "Te", "Tellurium",   127.600,    1.38,  2.06,  "#D47A00", "#AA6200"),
    ( 53, "I",  "Iodine",      126.904,    1.39,  1.98,  "#940094", "#660066"),
    ( 54, "Xe", "Xenon",       131.293,    1.40,  2.16,  "#429EB0", "#2E7080"),
    ( 55, "Cs", "Caesium",     132.905,    2.44,  3.43,  "#57178F", "#3D1066"),
    ( 56, "Ba", "Barium",      137.327,    2.15,  2.68,  "#00C900", "#008E00"),
    ( 57, "La", "Lanthanum",   138.905,    2.07,  2.43,  "#70D4FF", "#4E96BB"),
    ( 58, "Ce", "Cerium",      140.116,    2.04,  2.42,  "#FFFFC7", "#BBBB88"),
    ( 59, "Pr", "Praseodymium",140.908,    2.03,  2.40,  "#D9FFC7", "#99BB88"),
    ( 60, "Nd", "Neodymium",   144.242,    2.01,  2.39,  "#C7FFC7", "#88BB88"),
    ( 61, "Pm", "Promethium",  145.000,    1.99,  2.38,  "#A3FFC7", "#72BB88"),
    ( 62, "Sm", "Samarium",    150.360,    1.98,  2.36,  "#8FFFC7", "#63BB88"),
    ( 63, "Eu", "Europium",    151.964,    1.98,  2.35,  "#61FFC7", "#44BB88"),
    ( 64, "Gd", "Gadolinium",  157.250,    1.96,  2.34,  "#45FFC7", "#30BB88"),
    ( 65, "Tb", "Terbium",     158.925,    1.94,  2.33,  "#30FFC7", "#22BB88"),
    ( 66, "Dy", "Dysprosium",  162.500,    1.92,  2.31,  "#1FFFC7", "#15BB88"),
    ( 67, "Ho", "Holmium",     164.930,    1.92,  2.30,  "#00FF9C", "#00BB6E"),
    ( 68, "Er", "Erbium",      167.259,    1.89,  2.29,  "#00E675", "#00AA55"),
    ( 69, "Tm", "Thulium",     168.934,    1.90,  2.27,  "#00D452", "#009E3A"),
    ( 70, "Yb", "Ytterbium",   173.045,    1.87,  2.26,  "#00BF38", "#008E2A"),
    ( 71, "Lu", "Lutetium",    174.967,    1.87,  2.24,  "#00AB24", "#007E1A"),
    ( 72, "Hf", "Hafnium",     178.490,    1.75,  2.23,  "#4DC2FF", "#368EBB"),
    ( 73, "Ta", "Tantalum",    180.948,    1.70,  2.22,  "#4DA6FF", "#3678BB"),
    ( 74, "W",  "Tungsten",    183.840,    1.62,  2.18,  "#2194D6", "#166E9E"),
    ( 75, "Re", "Rhenium",     186.207,    1.51,  2.16,  "#267DAB", "#1B587E"),
    ( 76, "Os", "Osmium",      190.230,    1.44,  2.16,  "#266696", "#1B486E"),
    ( 77, "Ir", "Iridium",     192.217,    1.41,  2.13,  "#175487", "#103C62"),
    ( 78, "Pt", "Platinum",    195.084,    1.36,  1.75,  "#D0D0E0", "#9090A0"),
    ( 79, "Au", "Gold",        196.967,    1.36,  1.66,  "#FFD123", "#CCAA1A"),
    ( 80, "Hg", "Mercury",     200.592,    1.32,  1.55,  "#B8B8D0", "#808098"),
    ( 81, "Tl", "Thallium",    204.383,    1.45,  1.96,  "#A6544D", "#773C36"),
    ( 82, "Pb", "Lead",        207.200,    1.46,  2.02,  "#575961", "#3D3E44"),
    ( 83, "Bi", "Bismuth",     208.980,    1.48,  2.07,  "#9E4FB5", "#6E3880"),
    ( 84, "Po", "Polonium",    209.000,    1.40,  1.97,  "#AB5C00", "#7E4400"),
    ( 85, "At", "Astatine",    210.000,    1.50,  2.02,  "#754F45", "#523830"),
    ( 86, "Rn", "Radon",       222.000,    1.50,  2.20,  "#428296", "#2E5C6E"),
]
# fmt: on


ELEMENTS: dict[str, ElementData] = {}
Z_TO_SYMBOL: dict[int, str] = {}

for _row in _DATA:
    _z, _sym, _name, _mass, _cov, _vdw, _cpk, _edge = _row
    ELEMENTS[_sym] = ElementData(_z, _sym, _name, _mass, _cov, _vdw, _cpk, _edge)
    Z_TO_SYMBOL[_z] = _sym

# Cleanup module namespace
del _row, _z, _sym, _name, _mass, _cov, _vdw, _cpk, _edge


def get_element(identifier: int | str) -> ElementData:
    """Look up element by symbol (str) or atomic number (int)."""
    if isinstance(identifier, int):
        sym = Z_TO_SYMBOL.get(identifier)
        if sym is None:
            raise ValueError(f"Unknown atomic number: {identifier}")
        return ELEMENTS[sym]
    sym = identifier.strip().capitalize()
    if len(sym) > 1:
        sym = sym[0] + sym[1:].lower()
    if sym not in ELEMENTS:
        raise ValueError(f"Unknown element symbol: {identifier!r}")
    return ELEMENTS[sym]
