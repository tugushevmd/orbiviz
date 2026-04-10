"""Top-level entry point for Orbiviz.

Forwards to the unified visualization CLI:

    python main.py charge --input-file mol.out --charge-scheme chelpg --output q.png
    python main.py auto mol.out

Equivalent to ``python -m pyqchem.viz`` once the package is installed.
"""
from pyqchem.viz.cli import main


if __name__ == "__main__":
    main()
