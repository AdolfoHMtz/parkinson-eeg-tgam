"""
Modulo de conexion serial para el dispositivo NeuroSky TGAM.
Autodetecta el puerto COM del adaptador BT-USB y establece la conexion
a 57600 baud, 8N1.
"""

import serial
import serial.tools.list_ports
import time


BAUD_RATE = 57600
DATA_BITS = serial.EIGHTBITS
STOP_BITS = serial.STOPBITS_ONE
PARITY = serial.PARITY_NONE
TIMEOUT = 1  # segundos


def listar_puertos():
    """Devuelve una lista de puertos COM disponibles con su descripcion."""
    puertos = serial.tools.list_ports.comports()
    return [(p.device, p.description, p.hwid) for p in puertos]


def autodetectar_puerto():
    """
    Intenta detectar automaticamente el puerto COM del dispositivo TGAM.
    Busca puertos que contengan palabras clave de Bluetooth/serial en su descripcion.
    Si encuentra uno solo, lo devuelve. Si hay varios, le pide al usuario que elija.
    """
    puertos = listar_puertos()

    if not puertos:
        return None

    # Si solo hay un puerto, usarlo directamente
    if len(puertos) == 1:
        return puertos[0][0]

    # Buscar puertos con palabras clave de Bluetooth
    palabras_clave = ["bluetooth", "bt", "serial", "com"]
    candidatos = []
    for device, desc, hwid in puertos:
        desc_lower = desc.lower()
        if any(palabra in desc_lower for palabra in palabras_clave):
            candidatos.append((device, desc))

    if len(candidatos) == 1:
        return candidatos[0][0]

    # Si no se puede decidir automaticamente, mostrar opciones
    return None


def seleccionar_puerto():
    """
    Muestra los puertos disponibles y permite al usuario seleccionar uno.
    Si la autodeteccion encuentra un unico candidato, lo usa directamente.
    """
    puerto_auto = autodetectar_puerto()
    if puerto_auto:
        print(f"Puerto detectado automaticamente: {puerto_auto}")
        return puerto_auto

    puertos = listar_puertos()
    if not puertos:
        print("ERROR: No se encontraron puertos COM disponibles.")
        print("Verifica que el adaptador BT-USB este conectado.")
        return None

    print("\nPuertos COM disponibles:")
    print("-" * 60)
    for i, (device, desc, hwid) in enumerate(puertos):
        print(f"  [{i + 1}] {device} - {desc}")
    print("-" * 60)

    while True:
        try:
            opcion = input(f"Selecciona un puerto (1-{len(puertos)}): ").strip()
            indice = int(opcion) - 1
            if 0 <= indice < len(puertos):
                return puertos[indice][0]
            print("Opcion no valida.")
        except ValueError:
            print("Ingresa un numero.")
        except KeyboardInterrupt:
            print("\nCancelado.")
            return None


def conectar(puerto=None):
    """
    Abre la conexion serial al puerto especificado (o autodetectado).
    Retorna el objeto serial abierto, o None si falla.
    """
    if puerto is None:
        puerto = seleccionar_puerto()

    if puerto is None:
        return None

    try:
        ser = serial.Serial(
            port=puerto,
            baudrate=BAUD_RATE,
            bytesize=DATA_BITS,
            stopbits=STOP_BITS,
            parity=PARITY,
            timeout=TIMEOUT,
        )
        time.sleep(0.5)
        ser.reset_input_buffer()
        ser.reset_output_buffer()
        print(f"Conectado a {puerto} a {BAUD_RATE} baud")
        return ser
    except serial.SerialException as e:
        print(f"ERROR al conectar a {puerto}: {e}")
        return None


def desconectar(ser):
    """Cierra la conexion serial de forma segura."""
    if ser and ser.is_open:
        ser.close()
        print("Conexion serial cerrada.")
