"""
briscas/agents/base.py

Definición de la interfaz base para todos los agentes de Briscas.
"""

from typing import List
from briscas.core import EstadoBriscas, Carta


class Jugador:
    """
    Interfaz para cualquier jugador de Briscas.

    Atributos:
        nombre (str): nombre identificador del agente.
        mano (List[Carta]): cartas que el jugador tiene en mano.
    """
    def __init__(self, nombre: str):
        self.nombre = nombre
        self.mano: List[Carta] = []

    def seleccionar_carta(self, estado: EstadoBriscas) -> Carta:
        """
        Debe ser sobrecargado por cada subclase.
        Parámetros:
            estado (EstadoBriscas): estado actual del juego.
        Retorna:
            Carta: carta elegida para jugar.
        """
        raise NotImplementedError("Este método debe implementarse en cada agente")

    def __repr__(self) -> str:
        return self.nombre
