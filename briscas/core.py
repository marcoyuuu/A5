"""
briscas/core.py

Modelo de dominio para el juego de Briscas:
- Carta         : representación de una carta de la baraja española.
- Baraja        : baraja mezclable con reparto de cartas.
- EstadoBriscas : estado completo de una partida (turno, manos, baza, puntuación, etc.).
"""

import random
from collections import Counter
from typing import List, Tuple, Optional

# ------------------------------------------------------------------
# Constantes globales
# ------------------------------------------------------------------

VALORES_PUNTOS = {1: 11, 3: 10, 12: 4, 11: 3, 10: 2, 7: 0, 6: 0, 5: 0, 4: 0, 2: 0}
PALOS = ("oros", "copas", "espadas", "bastos")
ORDEN = (1, 3, 12, 11, 10, 7, 6, 5, 4, 2)  # de mayor a menor importancia

# ------------------------------------------------------------------
# Clases de dominio
# ------------------------------------------------------------------

class Carta:
    """Representa una carta de la baraja española."""
    def __init__(self, valor: int, palo: str):
        self.valor = valor
        self.palo = palo
        self.puntos = VALORES_PUNTOS[valor]

    def es_mayor_que(self, otra: "Carta", palo_seguidor: str, palo_triunfo: str) -> bool:
        """
        Compara dos cartas según reglas de Briscas.
        - Primero, si comparten palo, decide por orden de valor.
        - Luego, triunfo > no triunfo.
        - Finalmente, seguidor > otro palo.
        """
        if self.palo == otra.palo:
            return ORDEN.index(self.valor) < ORDEN.index(otra.valor)
        if self.palo == palo_triunfo and otra.palo != palo_triunfo:
            return True
        if self.palo != palo_triunfo and otra.palo == palo_triunfo:
            return False
        # ni triunfo ni mismo palo
        return self.palo == palo_seguidor

    def __repr__(self) -> str:
        nombres = {1: "As", 3: "Tres", 12: "Rey", 11: "Caballo", 10: "Sota"}
        n = nombres.get(self.valor, str(self.valor))
        return f"{n} de {self.palo}"


class Baraja:
    """Baraja de 40 cartas que se puede barajar y repartir."""
    def __init__(self, seed: Optional[int] = None):
        self._rng = random.Random(seed)
        self.cartas = [Carta(v, p) for p in PALOS for v in VALORES_PUNTOS]
        self._rng.shuffle(self.cartas)

    def robar(self) -> Optional[Carta]:
        """Extrae la carta superior o devuelve None si está vacía."""
        return self.cartas.pop() if self.cartas else None


class EstadoBriscas:
    """
    Estado completo de la partida:
      - jugadores: tupla de agentes
      - seguir_palo: si se fuerza seguir palo al jugar
      - vida: carta de triunfo
      - manos: cartas en mano de cada jugador
      - mazo: resto de cartas
      - baza: lista de (jugador, carta) de la baza actual
      - puntos: contador de puntos acumulados
      - turno: índice del jugador que mueve
    """
    def __init__(self, jugadores: Tuple, seguir_palo: bool):
        self.jugadores = jugadores
        self.seguir_palo = seguir_palo
        mazo = Baraja()
        self.vida = mazo.robar()
        # repartir 3 cartas a cada jugador
        self.manos = {j: [mazo.robar() for _ in range(3)] for j in jugadores}
        self.mazo = mazo
        self.baza: List[Tuple] = []
        self.puntos = Counter()
        self.turno = 0

    def acciones_legales(self) -> List[Carta]:
        """Cartas que el jugador actual puede jugar."""
        jugador = self.jugadores[self.turno]
        mano = self.manos[jugador]
        if not self.seguir_palo or not self.baza:
            return list(mano)
        palo_seguidor = self.baza[0][1].palo
        seguidor = [c for c in mano if c.palo == palo_seguidor]
        return seguidor or list(mano)

    def resultado(self, carta: Carta) -> "EstadoBriscas":
        """
        Aplica la acción de jugar 'carta' y devuelve un nuevo EstadoBriscas.
        - Añade la carta a la baza.
        - Si la baza se completa, calcula ganador, suma puntos y reparte nuevas cartas.
        """
        nuevo = self._copia()
        jugador = nuevo.jugadores[nuevo.turno]
        nuevo.manos[jugador].remove(carta)
        nuevo.baza.append((jugador, carta))

        # si se completa la baza:
        if len(nuevo.baza) == len(nuevo.jugadores):
            ganador = EstadoBriscas._ganador(nuevo.baza, nuevo.vida.palo)
            puntos_baza = sum(c.puntos for _, c in nuevo.baza)
            nuevo.puntos[ganador] += puntos_baza
            nuevo.baza.clear()
            # reparto de cartas en orden ganador → los demás
            orden_robo = [ganador] + [j for j in nuevo.jugadores if j != ganador]
            for j in orden_robo:
                rc = nuevo.mazo.robar()
                if rc:
                    nuevo.manos[j].append(rc)
            nuevo.turno = nuevo.jugadores.index(ganador)
        else:
            # siguiente jugador
            nuevo.turno = (nuevo.turno + 1) % len(nuevo.jugadores)

        return nuevo

    def es_terminal(self) -> bool:
        """La partida termina cuando ninguna mano tiene cartas."""
        return all(len(m) == 0 for m in self.manos.values())

    def utilidad(self, jugador) -> int:
        """
        Para 2 jugadores, utilidad = diferencia de puntos.
        Para más jugadores, simplemente devuelve sus puntos.
        """
        if len(self.jugadores) == 2:
            otro = [j for j in self.jugadores if j != jugador][0]
            return self.puntos[jugador] - self.puntos[otro]
        return self.puntos[jugador]

    @staticmethod
    def _ganador(baza: List[Tuple], palo_triunfo: str):
        palo_seguidor = baza[0][1].palo
        ganador, carta_max = baza[0]
        for j, c in baza[1:]:
            if c.es_mayor_que(carta_max, palo_seguidor, palo_triunfo):
                ganador, carta_max = j, c
        return ganador

    def _copia(self) -> "EstadoBriscas":
        """Copia profunda del estado para simulaciones."""
        nb = EstadoBriscas.__new__(EstadoBriscas)
        nb.jugadores = self.jugadores
        nb.seguir_palo = self.seguir_palo
        nb.vida = self.vida
        # duplicar mazo
        nb.mazo = Baraja()
        nb.mazo.cartas = list(self.mazo.cartas)
        # duplicar manos y demás atributos
        nb.manos = {j: list(m) for j, m in self.manos.items()}
        nb.baza = list(self.baza)
        nb.puntos = Counter(self.puntos)
        nb.turno = self.turno
        return nb
