# briscas/agents/heuristic_agent.py
"""
briscas/agents/heuristic_agent.py

Agente heurístico para Briscas:
  - Prioriza cartas capaces de ganar la baza y con más puntos.
  - Si no puede ganar, descarta cartas sin puntos.
  - Reserva triunfos para situaciones donde no quede otra opción.
  - Modo agresivo: juega siempre las de más puntos, aun cuando no gane la baza.
"""

import random
from typing import List
from briscas.core import Carta, EstadoBriscas, ORDEN, VALORES_PUNTOS
from briscas.agents import Jugador

class AgenteHeuristico(Jugador):
    def __init__(self,
                 nombre: str = "Heuristico",
                 modo_agresivo: bool = False,
                 seed: int = None):
        """
        :param modo_agresivo: si True, prefiere cartas altas aún cuando no gane la baza.
        """
        super().__init__(nombre)
        self.modo_agresivo = modo_agresivo
        self.rng = random.Random(seed)

    def seleccionar_carta(self, estado: EstadoBriscas) -> Carta:
        mano: List[Carta] = list(self.mano)  # copia local
        legales = estado.acciones_legales()
        # 1) Si solo hay una opción:
        if len(legales) == 1:
            c = legales[0]
            self.mano.remove(c)
            return c

        # Determinar la mejor carta de la baza actual
        mejor_baza = None
        if estado.baza:
            palo_seguidor = estado.baza[0][1].palo
            mejor_baza = estado.baza[0][1]
            for _, carta in estado.baza[1:]:
                if carta.es_mayor_que(mejor_baza, palo_seguidor, estado.vida.palo):
                    mejor_baza = carta

        # 2) ¿Hay cartas que ganen la baza? (y si no, modo_agresivo)
        def gana_baza(c: Carta) -> bool:
            if mejor_baza is None:
                return True
            return c.es_mayor_que(
                mejor_baza,
                estado.baza[0][1].palo if estado.baza else c.palo,
                estado.vida.palo
            )

        candidatos = [c for c in mano if gana_baza(c)]
        if candidatos:
            # jugar la de máximo valor en puntos, y en empate la de más alto rango
            def key_ganar(c: Carta):
                return (c.puntos, -ORDEN.index(c.valor))
            eleccion = max(candidatos, key=key_ganar)

            self.mano.remove(eleccion)
            return eleccion

        # 3) No puede ganar la baza
        #    si no es agresivo, descartar cartas sin puntos y no triunfo
        sin_puntos = [c for c in mano if c.puntos == 0 and c.palo != estado.vida.palo]
        if sin_puntos and not self.modo_agresivo:
            # tirar la de menor rango
            eleccion = min(sin_puntos, key=lambda c: ORDEN.index(c.valor))
            self.mano.remove(eleccion)
            return eleccion

        # 4) modo agresivo: jugar la de más puntos posible
        if self.modo_agresivo:
            eleccion = max(mano, key=lambda c: (c.puntos, -ORDEN.index(c.valor)))
            self.mano.remove(eleccion)
            return eleccion

        # 5) por defecto, tirar la carta de menor valor (de puntos o rango)
        eleccion = min(mano, key=lambda c: (c.puntos, ORDEN.index(c.valor)))
        self.mano.remove(eleccion)
        return eleccion
