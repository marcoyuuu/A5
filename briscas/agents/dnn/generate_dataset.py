#!/usr/bin/env python3
"""
generate_dataset.py

Simula partidas MCTS vs Aleatorio en paralelo para generar
un dataset de estados y acciones (Imitation Learning) para Briscas.

Salida:
    data/X.npy  -- array de shape (N,206)
    data/y.npy  -- array de shape (N,), con etiquetas 0–39
"""

import os
# — Suprimir logs de TensorFlow (sólo errores graves) —
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import logging
logging.getLogger("tensorflow").setLevel(logging.ERROR)

import numpy as np
import tensorflow as tf
from tqdm import tqdm
from multiprocessing import Pool, cpu_count

from briscas.game import BriscasGame
from briscas.agents.mcts_agent import AgenteMCTS
from briscas.agents.random_agent import AgenteAleatorio
from briscas.agents.dnn_agent import encode_state
from briscas.core import PALOS, VALORES_PUNTOS

# ——————————————————————————————
#  Parámetros de la generación
# ——————————————————————————————
N_PARTIDAS = 1000      # total de partidas a simular
MCTS_ITER  = 300       # iteraciones MCTS por jugada
MCTS_C     = 1.4       # parámetro UCT

DATA_DIR = "data"      # directorio donde se guardan X.npy e y.npy
os.makedirs(DATA_DIR, exist_ok=True)

# ——————————————————————————————
#  Construcción del mapeo carta → índice [0..39]
# ——————————————————————————————
CARD_INDEX = {}
idx = 0
for palo in PALOS:
    for valor in VALORES_PUNTOS:
        CARD_INDEX[(valor, palo)] = idx
        idx += 1

def simulate_one_game(_: int):
    """
    Simula UNA partida MCTS vs Aleatorio y
    devuelve dos listas de longitud variable (un ejemplo por jugada del MCTS):
        X_loc = [state_vec_1, state_vec_2, ...]
        y_loc = [label_1,     label_2,     ...]
    """
    X_loc = []
    y_loc = []

    # Creamos agentes y juego desde cero
    mcts  = AgenteMCTS("MCTS", iteraciones=MCTS_ITER, c=MCTS_C)
    rival = AgenteAleatorio("Rival")
    juego = BriscasGame([mcts, rival], seguir_palo=True)
    juego.iniciar()

    # Repetimos mientras haya acciones legales
    while True:
        legales = juego.estado.acciones_legales()
        if not legales:
            break

        turno   = juego.estado.turno
        jugador = juego.jugadores[turno]

        # Sincronizamos la mano real del agente con el estado interno
        jugador.mano = list(juego.estado.manos[jugador])

        if isinstance(jugador, AgenteMCTS):
            # Codificamos el estado y pedimos la carta al MCTS
            state_vec, _ = encode_state(juego.estado, jugador)
            carta_sel    = jugador.seleccionar_carta(juego.estado)
            key          = (carta_sel.valor, carta_sel.palo)
            label        = CARD_INDEX[key]

            X_loc.append(state_vec)
            y_loc.append(label)
        else:
            # Acción aleatoria del rival
            carta_sel = jugador.seleccionar_carta(juego.estado)

        # Avanzamos al siguiente estado
        juego.estado = juego.estado.resultado(carta_sel)

    return X_loc, y_loc

def main():
    # Cantidad de workers = núcleos-1 para no saturar
    n_workers = max(1, cpu_count() - 1)

    X_list = []
    y_list = []

    # Pool de procesos; repartimos UN juego por tarea
    with Pool(n_workers) as pool:
        for X_loc, y_loc in tqdm(
            pool.imap_unordered(simulate_one_game, range(N_PARTIDAS)),
            total=N_PARTIDAS,
            desc="Generando partidas"
        ):
            X_list.extend(X_loc)
            y_list.extend(y_loc)

    # Apilamos y guardamos
    X = np.stack(X_list, axis=0)
    y = np.array(y_list, dtype=np.int64)
    np.save(os.path.join(DATA_DIR, "X.npy"), X)
    np.save(os.path.join(DATA_DIR, "y.npy"), y)

    print(f"\n✔ Dataset generado: X.shape={X.shape}, y.shape={y.shape}")

if __name__ == "__main__":
    main()
