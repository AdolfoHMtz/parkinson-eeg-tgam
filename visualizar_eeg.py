"""
Lanzador de la visualizacion EEG en tiempo real.

Uso:
    python visualizar_eeg.py              (dispositivo real, autodetecta COM)
    python visualizar_eeg.py --demo       (datos simulados para prueba)
    python visualizar_eeg.py COM5         (puerto COM especifico)
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.visualizacion.app import main

if __name__ == "__main__":
    main()
