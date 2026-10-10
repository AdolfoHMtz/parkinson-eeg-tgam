"""
Lanzador de GUI para el pipeline de preprocesamiento.

Uso:
    python preprocesar_datos_gui.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.preprocesamiento.gui_pipeline import main


if __name__ == "__main__":
    main()
