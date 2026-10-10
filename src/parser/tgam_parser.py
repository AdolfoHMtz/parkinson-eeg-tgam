"""
Parser del protocolo binario TGAM (ThinkGear ASIC Module).

Implementacion basada en:
- 03_Codigo_Fuente_Referencia/ReadParseTGAM_C#/Parser.cs
- 01_Documentacion/TGAM Communications/ThinkGear/ThinkGearStreamParser.c
- 03_Codigo_Fuente_Referencia/thinkgear_testapp_C++Sample/thinkgear.h

Protocolo de paquetes:
  [SYNC 0xAA][SYNC 0xAA][PLENGTH][PAYLOAD...][CHECKSUM]

Codigos de datos en el payload:
  0x02 - Poor Signal Level (1 byte, 0=buena, 200=sin contacto)
  0x04 - Attention eSense (1 byte, 0-100)
  0x05 - Meditation eSense (1 byte, 0-100)
  0x16 - Blink Strength (1 byte, 0-255)
  0x80 - Raw Wave Value (2 bytes, signed 16-bit, 512 Hz)
  0x83 - EEG Power (24 bytes: 8 bandas x 3 bytes cada una)
"""

from copy import copy
from dataclasses import dataclass
from typing import Optional, Callable


# Constantes del protocolo
SYNC_BYTE = 0xAA
EXCODE_BYTE = 0x55
MAX_PAYLOAD_LENGTH = 170

# Codigos de datos
CODE_POOR_SIGNAL = 0x02
CODE_ATTENTION = 0x04
CODE_MEDITATION = 0x05
CODE_BLINK_STRENGTH = 0x16
CODE_RAW_WAVE = 0x80
CODE_EEG_POWER = 0x83

# Nombres de las 8 bandas de frecuencia (en orden del protocolo)
BANDAS_EEG = [
    "delta",       # 0.5 - 2.75 Hz
    "theta",       # 3.5 - 6.75 Hz
    "low_alpha",   # 7.5 - 9.25 Hz
    "high_alpha",  # 10 - 12 Hz
    "low_beta",    # 12.75 - 20 Hz
    "high_beta",   # 20.25 - 28 Hz
    "low_gamma",   # 30 - 40 Hz
    "high_gamma",  # 40 - 50 Hz
]

# Estados de la maquina de estados
STATE_SYNC = 1
STATE_SYNC_CHECK = 2
STATE_PAYLOAD_LENGTH = 3
STATE_PAYLOAD = 4
STATE_CHECKSUM = 5


@dataclass
class DatosTGAM:
    """Estructura que contiene los datos parseados de un paquete TGAM."""
    poor_signal: int = 200       # 0=buena senal, 200=sin contacto
    attention: int = 0           # 0-100
    meditation: int = 0          # 0-100
    blink_strength: int = 0      # 0-255
    raw_wave: int = 0            # signed 16-bit (-32768 a 32767)
    # Bandas de frecuencia EEG (valores unsigned 24-bit)
    delta: int = 0
    theta: int = 0
    low_alpha: int = 0
    high_alpha: int = 0
    low_beta: int = 0
    high_beta: int = 0
    low_gamma: int = 0
    high_gamma: int = 0
    # Flags de actualizacion del paquete actual
    actualizo_poor_signal: bool = False
    actualizo_attention: bool = False
    actualizo_meditation: bool = False
    actualizo_blink_strength: bool = False
    actualizo_raw_wave: bool = False
    actualizo_eeg_power: bool = False

    def bandas_dict(self):
        """Devuelve las 8 bandas como diccionario."""
        return {nombre: getattr(self, nombre) for nombre in BANDAS_EEG}

    def senal_buena(self):
        """Retorna True si la calidad de senal es aceptable (< 50)."""
        return self.poor_signal < 50

    @property
    def actualizo_metricas_lentas(self):
        """True si este paquete trae bandas/eSense/poor signal."""
        return (
            self.actualizo_eeg_power
            or self.actualizo_attention
            or self.actualizo_meditation
            or self.actualizo_poor_signal
        )


class TGAMParser:
    """
    Maquina de estados que parsea el stream de bytes del protocolo TGAM.

    Uso:
        parser = TGAMParser(on_datos=mi_callback)
        for byte in stream:
            parser.parse_byte(byte)

    El callback on_datos se invoca cada vez que se parsea un paquete completo
    y valido (checksum correcto). Recibe un objeto DatosTGAM con los valores.
    """

    def __init__(self, on_datos: Optional[Callable[[DatosTGAM], None]] = None):
        self.on_datos = on_datos
        self._payload = bytearray(256)
        self.paquetes_ok = 0
        self.paquetes_error = 0
        self.reset()

    def reset(self):
        """Reinicia el parser a su estado inicial."""
        self._state = STATE_SYNC
        self._payload_length = 0
        self._payload_bytes_received = 0
        self._payload_sum = 0
        self.datos = DatosTGAM()

    def parse_byte(self, byte_val: int) -> int:
        """
        Procesa un byte del stream serial.

        Retorna:
            0  - Paquete aun no completo
            1  - Paquete parseado exitosamente
           -2  - Error de checksum
           -3  - Longitud de payload invalida
        """
        if self._state == STATE_SYNC:
            if byte_val == SYNC_BYTE:
                self._state = STATE_SYNC_CHECK

        elif self._state == STATE_SYNC_CHECK:
            if byte_val == SYNC_BYTE:
                self._state = STATE_PAYLOAD_LENGTH
            else:
                self._state = STATE_SYNC

        elif self._state == STATE_PAYLOAD_LENGTH:
            if byte_val > MAX_PAYLOAD_LENGTH:
                self._state = STATE_SYNC
                return -3
            self._payload_length = byte_val
            self._payload_bytes_received = 0
            self._payload_sum = 0
            self._state = STATE_PAYLOAD

        elif self._state == STATE_PAYLOAD:
            self._payload[self._payload_bytes_received] = byte_val
            self._payload_bytes_received += 1
            self._payload_sum += byte_val
            if self._payload_bytes_received >= self._payload_length:
                self._state = STATE_CHECKSUM

        elif self._state == STATE_CHECKSUM:
            self._state = STATE_SYNC
            checksum_esperado = (~self._payload_sum) & 0xFF

            if byte_val != checksum_esperado:
                self.paquetes_error += 1
                return -2

            self.paquetes_ok += 1
            self._parse_payload()
            return 1

        return 0

    def _parse_payload(self):
        """Decodifica el contenido del payload en datos individuales."""
        self._reset_flags_paquete()
        i = 0
        while i < self._payload_length:
            while i < self._payload_length and self._payload[i] == EXCODE_BYTE:
                i += 1

            if i >= self._payload_length:
                break

            code = self._payload[i]
            i += 1

            if code >= 0x80:
                if i >= self._payload_length:
                    break
                value_length = self._payload[i]
                i += 1
            else:
                value_length = 1

            if i + value_length > self._payload_length:
                break

            self._process_code(code, i, value_length)
            i += value_length

        if self.on_datos:
            self.on_datos(copy(self.datos))

    def _reset_flags_paquete(self):
        """Limpia flags para marcar solo los campos presentes en este paquete."""
        self.datos.actualizo_poor_signal = False
        self.datos.actualizo_attention = False
        self.datos.actualizo_meditation = False
        self.datos.actualizo_blink_strength = False
        self.datos.actualizo_raw_wave = False
        self.datos.actualizo_eeg_power = False

    def _process_code(self, code: int, offset: int, length: int):
        """Procesa un codigo de datos especifico del payload."""

        if code == CODE_POOR_SIGNAL:
            self.datos.poor_signal = self._payload[offset]
            self.datos.actualizo_poor_signal = True

        elif code in (CODE_ATTENTION, CODE_MEDITATION):
            valor = self._payload[offset]
            if valor <= 100:
                if code == CODE_ATTENTION:
                    self.datos.attention = valor
                    self.datos.actualizo_attention = True
                else:
                    self.datos.meditation = valor
                    self.datos.actualizo_meditation = True

        elif code == CODE_BLINK_STRENGTH:
            self.datos.blink_strength = self._payload[offset]
            self.datos.actualizo_blink_strength = True

        elif code == CODE_RAW_WAVE:
            if length == 2:
                self.datos.raw_wave = int.from_bytes(
                    self._payload[offset:offset + 2], byteorder="big", signed=True
                )
                self.datos.actualizo_raw_wave = True

        elif code == CODE_EEG_POWER:
            if length == 24:
                self._parse_eeg_power(offset)
                self.datos.actualizo_eeg_power = True

    def _parse_eeg_power(self, offset: int):
        """Parsea las 8 bandas de potencia EEG (3 bytes big-endian cada una)."""
        for idx, nombre in enumerate(BANDAS_EEG):
            pos = offset + (idx * 3)
            valor = int.from_bytes(self._payload[pos:pos + 3], byteorder="big")
            setattr(self.datos, nombre, valor)
