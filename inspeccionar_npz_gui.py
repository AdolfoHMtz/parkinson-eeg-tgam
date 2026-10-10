"""
Lanzador de GUI para inspeccionar archivos .npz del proyecto.

Uso:
    python inspeccionar_npz_gui.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.preprocesamiento.gui_npz import main


if __name__ == "__main__":
    main()
