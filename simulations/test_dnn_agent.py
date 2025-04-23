# test_all_agents.py

import os
# ocultar logs de TensorFlow / Abseil
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import logging
logging.getLogger('tensorflow').setLevel(logging.ERROR)
try:
    import absl.logging as _abllog
    _abllog.set_verbosity(_abllog.ERROR)
    _abllog.set_stderrthreshold('error')
except ImportError:
    pass
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=DeprecationWarning)

from multiprocessing import Pool, cpu_count
from tqdm import tqdm
import random

from briscas.agents.dnn_agent import AgenteDNN
from briscas.agents.random_agent import AgenteAleatorio
from briscas.agents.heuristic_agent import AgenteHeuristico
from briscas.agents.mcts_agent import AgenteMCTS
from briscas.game import BriscasGame

# función que simula UNA partida y devuelve el nombre del ganador
def jugar_1_partida(args):
    AgentClass1, AgentClass2, seed = args
    random.seed(seed)
    j1 = AgentClass1("A1")
    j2 = AgentClass2("A2")
    # alternar orden para mayor equidad
    jugadores = [j1, j2] if seed % 2 == 0 else [j2, j1]
    juego = BriscasGame(jugadores)
    juego.iniciar()
    while not juego.estado.es_terminal():
        legales = juego.estado.acciones_legales()
        if not legales:
            break
        jugador = juego.estado.jugadores[juego.estado.turno]
        jugador.mano = list(juego.estado.manos[jugador])
        carta = jugador.seleccionar_carta(juego.estado)
        juego.estado = juego.estado.resultado(carta)
    puntos = juego.estado.puntos
    ganador = max(puntos, key=lambda j: puntos[j])
    return ganador.nombre

def evaluar_enfrentamiento(AgentClass1, AgentClass2, n_partidas=500):
    args = [(AgentClass1, AgentClass2, seed) for seed in range(n_partidas)]
    wins = { 'A1': 0, 'A2': 0 }
    with Pool(processes=cpu_count()) as pool:
        for ganador in tqdm(pool.imap_unordered(jugar_1_partida, args),
                             total=len(args),
                             desc=f"{AgentClass1.__name__} vs {AgentClass2.__name__}",
                             unit="partida"):
            wins[ganador] += 1
    return wins

if __name__ == "__main__":
    confrontaciones = [
        (AgenteDNN, AgenteAleatorio),
        (AgenteDNN, AgenteHeuristico),
        (AgenteDNN, AgenteMCTS),
    ]
    resultados = {}
    for A, B in confrontaciones:
        wins = evaluar_enfrentamiento(A, B, n_partidas=500)
        resultados[(A.__name__, B.__name__)] = wins

    print("\n=== Resultados finales ===")
    for (Aname, Bname), wins in resultados.items():
        print(f"{Aname} vs {Bname}: {wins['A1']} – {wins['A2']} (de 500)")
