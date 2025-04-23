"""
briscas/game.py

Motor de juego principal. Orquesta:
  - creación de EstadoBriscas
  - bucle principal de juego
  - retorno de puntuaciones finales
"""

from typing import Sequence, Dict
from briscas.core import EstadoBriscas, Baraja
from briscas.agents import Jugador


class BriscasGame:
    """
    Controla una partida de Briscas:
      - jugadores    : lista de objetos Jugador
      - seguir_palo  : si se fuerza seguir palo
      - seed         : semilla para Baraja
    """
    def __init__(self,
                 jugadores: Sequence[Jugador],
                 seguir_palo: bool = False,
                 seed: int = None):
        self.jugadores = tuple(jugadores)
        self.seguir_palo = seguir_palo
        self.seed = seed
        self.estado: EstadoBriscas = None  # se inicializa en iniciar()

    def iniciar(self):
        """Crea un EstadoBriscas nuevo y reparte manos a los jugadores."""
        self.estado = EstadoBriscas(self.jugadores, self.seguir_palo)
        for j in self.jugadores:
            j.mano = list(self.estado.manos[j])

    def jugar(self) -> Dict[Jugador, int]:
        """
        Ejecuta el bucle de juego hasta terminar:
        - turno por turno, llama a jugador.seleccionar_carta()
        - aplica resultado en estado
        Devuelve el Counter de puntuaciones finales.
        """
        if self.estado is None:
            self.iniciar()

        while True:
            if self.estado.es_terminal():
                break
            legales = self.estado.acciones_legales()
            if not legales:
                break
            jugador = self.estado.jugadores[self.estado.turno]
            jugador.mano = list(self.estado.manos[jugador])
            carta = jugador.seleccionar_carta(self.estado)
            self.estado = self.estado.resultado(carta)

        return self.estado.puntos
