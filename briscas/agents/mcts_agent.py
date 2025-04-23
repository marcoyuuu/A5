"""
briscas/agents/mcts_agent.py

Agente MCTS (Monte Carlo Tree Search) para Briscas.
"""

import math
import time
import random
from typing import List, Optional
from briscas.agents.base import Jugador
from briscas.core import EstadoBriscas, Carta


class NodoMCTS:
    """
    Nodo para MCTS:
      - estado  : EstadoBriscas en este nodo.
      - padre   : NodoMCTS del que proviene (None para la raíz).
      - carta   : Carta jugada para llegar aquí (None en la raíz).
      - hijos   : lista de sucesores.
      - N       : número de visitas.
      - Q       : recompensa acumulada.
    """
    def __init__(self, estado: EstadoBriscas, padre: Optional["NodoMCTS"] = None, carta: Optional[Carta] = None):
        self.estado = estado
        self.padre = padre
        self.carta = carta
        self.hijos: List["NodoMCTS"] = []
        self.N: int = 0
        self.Q: float = 0.0

    def fully_expanded(self) -> bool:
        return len(self.hijos) == len(self.estado.acciones_legales())


class AgenteMCTS(Jugador):
    """
    Agente basado en Monte Carlo Tree Search (MCTS) para Briscas.

    Parámetros:
      - iteraciones: número máximo de simulaciones por jugada.
      - tiempo_max : tiempo máximo por jugada (segundos), opcional.
      - c          : constante de exploración utilizada en la fórmula UCT.
      - seed       : semilla para reproducibilidad.

    Justificación del parámetro 'c':
      La constante de exploración 'c' regula el equilibrio entre exploración y explotación
      en el criterio UCT (Upper Confidence Bound for Trees):

          UCT = (Q_i / N_i) + c * sqrt(ln(N_p) / N_i)

      Donde:
        - Q_i : recompensa acumulada del nodo hijo.
        - N_i : número de visitas del nodo hijo.
        - N_p : número de visitas del nodo padre.

      Un valor bajo de 'c' favorece la explotación (acciones conocidas con buen desempeño),
      mientras que un valor alto promueve la exploración (nodos menos visitados).

      Basado en la literatura estándar y en evaluaciones empíricas realizadas en este proyecto,
      se estableció un valor por defecto de c = 1.4 con 500 iteraciones, 
      ya que ofreció un balance óptimo entre rendimiento y eficiencia en el contexto del juego de Briscas.
    """
    def __init__(self,
                 nombre: str = "MCTS",
                 iteraciones: int = 500,
                 tiempo_max: Optional[float] = None,
                 c: float = 1.4,
                 seed: Optional[int] = None):
        super().__init__(nombre)
        self.iteraciones = iteraciones
        self.tiempo_max = tiempo_max
        self.c = c
        self.rng = random.Random(seed)

    def seleccionar_carta(self, estado: EstadoBriscas) -> Carta:
        raiz = NodoMCTS(estado)
        inicio = time.perf_counter()

        for _ in range(self.iteraciones):
            if self.tiempo_max and (time.perf_counter() - inicio) >= self.tiempo_max:
                break
            nodo = self._seleccionar(raiz)
            recompensa = self._simular(nodo.estado)
            self._retropropagar(nodo, recompensa)

        mejor = max(raiz.hijos, key=lambda h: h.N)
        self.mano.remove(mejor.carta)
        return mejor.carta

    def _seleccionar(self, nodo: NodoMCTS) -> NodoMCTS:
        while True:
            if nodo.estado.es_terminal() or not nodo.estado.acciones_legales():
                return nodo
            if not nodo.fully_expanded():
                return self._expandir(nodo)
            nodo = self._mejor_uct(nodo)

    def _expandir(self, nodo: NodoMCTS) -> NodoMCTS:
        legales = nodo.estado.acciones_legales()
        jugadas_usadas = {h.carta for h in nodo.hijos}
        for carta in legales:
            if carta not in jugadas_usadas:
                nuevo_estado = nodo.estado.resultado(carta)
                hijo = NodoMCTS(nuevo_estado, padre=nodo, carta=carta)
                nodo.hijos.append(hijo)
                return hijo
        raise RuntimeError("No hay acciones por expandir")

    def _mejor_uct(self, nodo: NodoMCTS) -> NodoMCTS:
        logN = math.log(nodo.N + 1)
        def uct_score(h):
            return (h.Q / (h.N + 1e-9) +
                    self.c * math.sqrt(logN / (h.N + 1e-9)))
        return max(nodo.hijos, key=uct_score)

    def _simular(self, estado: EstadoBriscas) -> float:
        s = estado._copia()
        while not s.es_terminal():
            legales = s.acciones_legales()
            if not legales:
                break
            c = self.rng.choice(legales)
            s = s.resultado(c)
        return s.utilidad(self)

    def _retropropagar(self, nodo: NodoMCTS, recompensa: float):
        while nodo:
            nodo.N += 1
            nodo.Q += recompensa
            nodo = nodo.padre
