"""
briscas/agents/dnn_agent.py
---------------------------
Agente que selecciona la carta a jugar mediante una red neuronal (Keras/TensorFlow)
entrenada por imitación de MCTS.
"""

import os
from typing import List, Tuple, Optional
import numpy as np
import tensorflow as tf

from briscas.agents.base import Jugador
from briscas.core import Carta, EstadoBriscas, PALOS, VALORES_PUNTOS

# Número total de cartas (4 palos × 10 valores)
N_CARTAS = len(PALOS) * len(VALORES_PUNTOS)

MODEL_PATH = os.path.join(
    os.path.dirname(__file__),
    "models",
    "briscas_dnn.keras"  # usa el formato Keras nativo
)

def _one_hot_carta(carta: Optional[Carta]) -> np.ndarray:
    """Codifica una Carta (o None) como one-hot sobre 40 posiciones."""
    vec = np.zeros(N_CARTAS, dtype=np.float32)
    if carta is None:
        return vec
    valores = list(VALORES_PUNTOS.keys())
    palo_idx = PALOS.index(carta.palo)
    val_idx = valores.index(carta.valor)
    vec[palo_idx * len(valores) + val_idx] = 1.0
    return vec


def _legal_mask(mano: List[Carta]) -> np.ndarray:
    """Mask binario de tamaño 40 con 1s en las cartas que están en la mano."""
    mask = np.zeros(N_CARTAS, dtype=np.float32)
    for c in mano:
        idx = _one_hot_carta(c).argmax()
        mask[idx] = 1.0
    return mask


def encode_state(
    estado: EstadoBriscas,
    jugador: "AgenteDNN"
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Codifica el estado de Briscas para la DNN:
      - mano_vec: 3 cartas x 40 = 120
      - palo de triunfo: 4
      - última carta jugada: 40
      - puntos propios y ajenos normalizados: 2
    Total: 120 + 4 + 40 + 2 = 166
    """
    # Mano (3 cartas; si hay menos, relleno con None)
    mano = estado.manos[jugador]
    mano_vecs = [_one_hot_carta(c) for c in mano]
    while len(mano_vecs) < 3:
        mano_vecs.append(_one_hot_carta(None))
    mano_vec = np.concatenate(mano_vecs)

    # Palo de triunfo
    palo_vec = np.zeros(len(PALOS), dtype=np.float32)
    palo_vec[PALOS.index(estado.vida.palo)] = 1.0

    # Última baza
    ultima = estado.baza[-1][1] if estado.baza else None
    rival_vec = _one_hot_carta(ultima)

    # Puntos normalizados
    propios = estado.puntos[jugador] / 120.0
    ajenos = (
        max(estado.puntos[j] for j in estado.jugadores if j != jugador) / 120.0
        if len(estado.jugadores) > 1 else 0.0
    )
    puntos_vec = np.array([propios, ajenos], dtype=np.float32)

    state_vec = np.concatenate([mano_vec, palo_vec, rival_vec, puntos_vec])
    legal_mask = _legal_mask(mano)
    return state_vec, legal_mask


class AgenteDNN(Jugador):
    _modelo: Optional[tf.keras.Model] = None

    def __init__(self, nombre: str = "DNN"):
        super().__init__(nombre)
        if AgenteDNN._modelo is None:
            if not os.path.exists(MODEL_PATH):
                raise FileNotFoundError(
                    f"Modelo no encontrado: {MODEL_PATH}. Entrénalo antes."
                )
            # Carga el modelo en CPU
            AgenteDNN._modelo = tf.keras.models.load_model(MODEL_PATH)

    def seleccionar_carta(self, estado: EstadoBriscas) -> Carta:
        # Codificamos el estado
        state_vec, legal_mask = encode_state(estado, self)
        # Inferencia (será en CPU, porque hemos deshabilitado GPUs)
        logits = AgenteDNN._modelo.predict(state_vec[None, :], verbose=0)[0]
        masked = logits * legal_mask
        if masked.sum() == 0:
            # Ninguna carta legal? (debería costar, pero por si acaso)
            masked = legal_mask
        probs = masked / masked.sum()
        idx = int(np.argmax(probs))

        # Traducimos el índice a palo+valor
        valores = list(VALORES_PUNTOS.keys())
        palo_idx, val_idx = divmod(idx, len(valores))
        palo = PALOS[palo_idx]
        valor = valores[val_idx]

        # Sacamos la carta de la mano
        for carta in self.mano:
            if carta.palo == palo and carta.valor == valor:
                self.mano.remove(carta)
                return carta

        # Backup: devolvemos la primera legal
        fallback = estado.acciones_legales()[0]
        self.mano.remove(fallback)
        return fallback
