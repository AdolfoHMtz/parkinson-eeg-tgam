"""
Lectura en tiempo real del dispositivo NeuroSky TGAM via puerto serial.

Conecta al dispositivo, parsea el protocolo binario y muestra los datos
en consola: senal cruda (Raw EEG), 8 bandas de frecuencia, atencion,
meditacion y calidad de senal.

Uso:
    python leer_tgam.py           (autodetecta puerto COM)
    python leer_tgam.py COM5      (usa puerto especifico)
    python leer_tgam.py --raw     (muestra solo datos raw a 512 Hz)
"""

import sys
import time
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.conexion.serial_manager import conectar, desconectar, BAUD_RATE
from src.parser.tgam_parser import TGAMParser, DatosTGAM


def barra(valor, maximo=100, ancho=20):
    """Genera una barra de progreso simple."""
    lleno = int((valor / maximo) * ancho) if maximo > 0 else 0
    return "[" + "#" * lleno + "-" * (ancho - lleno) + "]"


class LectorTGAM:
    """Encapsula la logica de lectura y visualizacion del TGAM."""

    def __init__(self, solo_raw=False):
        self.solo_raw = solo_raw
        self.paquete_count = 0
        # Ultimo paquete con bandas validas (llegan ~1/seg, no a 512 Hz)
        self._ultimo_con_bandas = None

    def on_datos(self, datos: DatosTGAM):
        """Callback invocado por el parser con cada paquete decodificado."""
        if self.solo_raw:
            print(f"Raw: {datos.raw_wave:>6d}  |  Signal: {datos.poor_signal}")
            return

        # Las bandas EEG solo llegan ~1/seg. Los raw llegan a 512 Hz.
        # Evitar crear dict en cada paquete raw: verificar un campo directamente.
        if datos.actualizo_eeg_power:
            self._ultimo_con_bandas = datos
            self.paquete_count += 1
            self._mostrar(datos)

    def _mostrar(self, datos: DatosTGAM):
        """Muestra el resumen de datos en consola."""
        print("\033[2J\033[H")
        print("=" * 65)
        print("  NEUROSKY TGAM - Lectura en tiempo real")
        print("=" * 65)

        # Calidad de senal
        if datos.poor_signal == 0:
            senal_txt = "EXCELENTE"
        elif datos.poor_signal < 50:
            senal_txt = "BUENA"
        elif datos.poor_signal < 100:
            senal_txt = "REGULAR"
        elif datos.poor_signal < 200:
            senal_txt = "MALA"
        else:
            senal_txt = "SIN CONTACTO"
        print(f"\n  Calidad de senal:  {datos.poor_signal:>3d}/200  ({senal_txt})")

        print(f"\n  Attention:     {barra(datos.attention)} {datos.attention:>3d}/100")
        print(f"  Meditation:    {barra(datos.meditation)} {datos.meditation:>3d}/100")

        if datos.blink_strength > 0:
            print(f"  Blink:         {datos.blink_strength}")

        print(f"\n  Raw EEG:       {datos.raw_wave:>6d}")

        bandas = datos.bandas_dict()
        print(f"\n  {'Banda':<14s}  {'Valor':>10s}  {'Grafica'}")
        print("  " + "-" * 50)
        max_banda = max(bandas.values(), default=1) or 1
        for nombre, valor in bandas.items():
            barra_val = int((valor / max_banda) * 25)
            print(f"  {nombre:<14s}  {valor:>10,d}  {'#' * barra_val}")

        print(f"\n  Paquete #{self.paquete_count}")
        print("  Presiona Ctrl+C para salir")


def main():
    puerto = None
    solo_raw = False
    for arg in sys.argv[1:]:
        if arg == "--raw":
            solo_raw = True
        elif arg.upper().startswith("COM") or arg.startswith("/dev/"):
            puerto = arg

    print("=" * 50)
    print("  NEUROSKY TGAM - Lector Python")
    print(f"  Protocolo: {BAUD_RATE} baud, 8N1")
    print("=" * 50)

    ser = conectar(puerto)
    if ser is None:
        print("\nNo se pudo conectar. Verifica:")
        print("  1. Que el adaptador BT-USB este conectado")
        print("  2. Que el dispositivo TGAM este encendido")
        print("  3. Que ningun otro programa use el puerto (cierra Realterm)")
        return

    lector = LectorTGAM(solo_raw=solo_raw)
    parser = TGAMParser(on_datos=lector.on_datos)

    print("\nLeyendo datos del dispositivo...")
    print("Presiona Ctrl+C para detener\n")

    try:
        while True:
            if ser.in_waiting > 0:
                datos_raw = ser.read(ser.in_waiting)
                for byte in datos_raw:
                    parser.parse_byte(byte)
            else:
                time.sleep(0.001)

    except KeyboardInterrupt:
        print("\n\nDetenido por el usuario.")
        print(f"  Paquetes exitosos:  {parser.paquetes_ok}")
        print(f"  Errores checksum:   {parser.paquetes_error}")

    finally:
        desconectar(ser)


if __name__ == "__main__":
    main()
