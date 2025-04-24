# test_dnn_agent.py

import os
import matplotlib
matplotlib.use('Agg')
# 1) Suprimir logs de TensorFlow / Abseil
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
warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib")
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
import random
import pandas as pd
import matplotlib.pyplot as plt

from briscas.agents.dnn_agent import AgenteDNN
from briscas.agents.random_agent import AgenteAleatorio
from briscas.agents.heuristic_agent import AgenteHeuristico
from briscas.agents.mcts_agent import AgenteMCTS
from briscas.game import BriscasGame

OUTPUT_DIR = "dnn_evaluation"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def jugar_partida_datos(args):
    """
    Simula UNA partida y devuelve un dict con:
    - matchup: "<A>_vs_<B>"
    - seed
    - winner: nombre del agente ganadora
    - points_<A>, points_<B>: puntos finales
    - turns: nº de turnos jugados
    """
    AgentClass1, AgentClass2, seed = args
    label1 = AgentClass1.__name__.replace('Agente','')
    label2 = AgentClass2.__name__.replace('Agente','')
    random.seed(seed)
    # crear agentes con sus etiquetas
    j1 = AgentClass1(label1)
    j2 = AgentClass2(label2)
    # alternar orden
    jugadores = [j1, j2] if seed % 2 == 0 else [j2, j1]
    juego = BriscasGame(jugadores)
    juego.iniciar()
    turns = 0
    while not juego.estado.es_terminal():
        legales = juego.estado.acciones_legales()
        if not legales:
            break
        jugador = juego.estado.jugadores[juego.estado.turno]
        jugador.mano = list(juego.estado.manos[jugador])
        carta = jugador.seleccionar_carta(juego.estado)
        juego.estado = juego.estado.resultado(carta)
        turns += 1

    puntos = juego.estado.puntos
    ganador = max(puntos, key=lambda j: puntos[j])
    return {
        'matchup': f"{label1}_vs_{label2}",
        'seed': seed,
        'winner': ganador.nombre,
        f'points_{label1}': puntos[j1],
        f'points_{label2}': puntos[j2],
        'turns': turns
    }

def eval_matchup(AgentClass1, AgentClass2, n=500):
    label1 = AgentClass1.__name__.replace('Agente','')
    label2 = AgentClass2.__name__.replace('Agente','')
    args = [(AgentClass1, AgentClass2, seed) for seed in range(n)]
    results = []
    with Pool(cpu_count()) as pool:
        for res in tqdm(pool.imap_unordered(jugar_partida_datos, args),
                        total=n,
                        desc=f"{label1} vs {label2}",
                        unit="partida"):
            results.append(res)
    df = pd.DataFrame(results)
    # guardar CSV
    csv_name = os.path.join(OUTPUT_DIR, f"results_{label1}_vs_{label2}.csv")
    df.to_csv(csv_name, index=False)
    print(f"> Datos guardados en {csv_name}")
    return df

def analizar_y_plot(df):
    matchup = df['matchup'].iloc[0]
    A, _, B = matchup.partition('_vs_')
    # métricas
    winsA = (df['winner']==A).mean()*100
    winsB = (df['winner']==B).mean()*100
    avgA = df[f'points_{A}'].mean()
    avgB = df[f'points_{B}'].mean()
    stdA = df[f'points_{A}'].std()
    stdB = df[f'points_{B}'].std()
    avg_turns = df['turns'].mean()

    print(f"\n🔹 Evaluación: {A} vs {B} ({len(df)} partidas)")
    print("─"*50)
    print(f"- % Victorias {A:8}: {winsA:5.1f}%")
    print(f"- % Victorias {B:8}: {winsB:5.1f}%\n")
    print(f"- Puntos promedio {A:3}: {avgA:5.1f} (σ={stdA:.1f})")
    print(f"- Puntos promedio {B:3}: {avgB:5.1f} (σ={stdB:.1f})\n")
    print(f"- Duración promedio: {avg_turns:.1f} turnos")

    # 1) Histograma de puntos de A
    plt.figure()
    plt.hist(df[f'points_{A}'], bins=20)
    plt.title(f"Histograma puntos {A}")
    plt.xlabel("Puntos")
    plt.ylabel("Frecuencia")
    fname = os.path.join(OUTPUT_DIR, f"hist_{A}_vs_{B}.png")
    plt.savefig(fname)
    plt.close()

    # 2) Boxplot comparativo
    plt.figure()
    plt.boxplot([df[f'points_{A}'], df[f'points_{B}']], labels=[A, B])
    plt.title(f"Boxplot puntos {A} vs {B}")
    plt.ylabel("Puntos")
    fname = os.path.join(OUTPUT_DIR, f"boxplot_{A}_vs_{B}.png")
    plt.savefig(fname)
    plt.close()

    # 3) Barras de % victorias
    plt.figure()
    plt.bar([A, B], [winsA, winsB])
    plt.title(f"% Victorias {A} vs {B}")
    plt.ylabel("Porcentaje")
    fname = os.path.join(OUTPUT_DIR, f"barras_{A}_vs_{B}.png")
    plt.savefig(fname)
    plt.close()

    # 4) Scatter duración vs diferencia de puntos
    df['Diff'] = df[f'points_{A}'] - df[f'points_{B}']
    plt.figure()
    plt.scatter(df['turns'], df['Diff'], alpha=0.6)
    plt.title(f"Duración vs (Puntos {A} - Puntos {B})")
    plt.xlabel("Turnos"); plt.ylabel("Diferencia de puntos")
    fname = os.path.join(OUTPUT_DIR, f"scatter_turns_diff_{A}_vs_{B}.png")
    plt.savefig(fname)
    plt.close()

if __name__ == "__main__":
    confronts = [
        (AgenteDNN,        AgenteAleatorio),
        (AgenteDNN,        AgenteHeuristico),
        (AgenteDNN,        AgenteMCTS),
    ]
    for A, B in confronts:
        df = eval_matchup(A, B, n=500)
        analizar_y_plot(df)
