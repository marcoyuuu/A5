"""
briscas/agents/random_agent.py

Agente que selecciona cartas completamente al azar.
"""

import random
from briscas.agents.base import Jugador
from briscas.core import EstadoBriscas, Carta


class AgenteAleatorio(Jugador):
    """
    Elige siempre una carta al azar de su mano.
    """
    def __init__(self, nombre: str = "Rand", seed: int = None):
        super().__init__(nombre)
        self.rng = random.Random(seed)

    def seleccionar_carta(self, estado: EstadoBriscas) -> Carta:
        carta = self.rng.choice(self.mano)
        self.mano.remove(carta)
        return carta
