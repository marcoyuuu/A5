# simulations/run_simulation.py

import os
import time
from multiprocessing import Pool, cpu_count
from typing import List, Tuple

import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

from briscas.game import BriscasGame
from briscas.agents import AgenteMCTS, AgenteAleatorio, AgenteHeuristico
from briscas.utils import Logger

# Parámetros MCTS óptimos (sección 4.4)
ITER_MCTS   = 500    # iteraciones por decisión
C_MCTS      = 1.4    # constante de exploración óptima
N_PARTIDAS  = 500
BASE_OUTPUT = "output"


def crear_agentes(modo_agresivo: bool, seed: int) -> List:
    """
    Devolvemos los 3 agentes, variando sólo el flag 'modo_agresivo'
    del heurístico (el seguimiento de palo se maneja en BriscasGame).
    """
    return [
        # Ahora pasamos iteraciones, c y seed para reproducibilidad
        AgenteMCTS("MCTS",
                   iteraciones=ITER_MCTS,
                   tiempo_max=None,
                   c=C_MCTS,
                   seed=seed),
        AgenteHeuristico("Heuristico", modo_agresivo=modo_agresivo),
        AgenteAleatorio("Rand")
    ]


def ejecutar_partida(args: Tuple[int, bool, bool, int]) -> dict:
    """
    args = (i, modo_agresivo, seguir_palo, seed)
    """
    i, modo_agresivo, seguir_palo, seed = args

    # 1) Creamos los agentes (incluyendo seed para MCTS)
    jugadores = crear_agentes(modo_agresivo, seed)
    # 2) Creamos el juego con el flag seguir_palo
    # 2) Creamos el juego con seed fija para reproducibilidad y seguir_palo=True
    juego      = BriscasGame(jugadores,
                             seguir_palo=seguir_palo,
                             seed=seed + i)

    # 3) Ejecutamos la partida
    t0      = time.perf_counter()
    puntos  = juego.jugar()
    dur     = time.perf_counter() - t0
    ganador = max(puntos, key=puntos.get).nombre

    # 4) Armamos la fila de resultados
    fila = {
        "Partida": i,
        "Ganador": ganador,
        "Dur":     dur
    }
    for j in jugadores:
        fila[f"P_{j.nombre}"] = puntos[j]

    return fila


def simular_partidas(n: int, modo_agresivo: bool, seguir_palo: bool, seed: int, logger: Logger) -> pd.DataFrame:
    """
    Simula n partidas en paralelo, mostrando barra de progreso con tqdm.
    """
    logger.log(f" Simulando {n} partidas (agresivo={modo_agresivo}, seguir_palo={seguir_palo})…")
    args = [(i, modo_agresivo, seguir_palo, seed) for i in range(1, n + 1)]
    resultados = []

    print(f"▶ Simulando {len(args)} partidas usando {cpu_count()} CPUs…")
    with Pool(processes=cpu_count()) as pool:
        for resultado in tqdm(pool.imap_unordered(ejecutar_partida, args),
                              total=n,
                              desc="Partidas",
                              unit="ptda"):
            resultados.append(resultado)

    return pd.DataFrame(resultados)


def save_descriptive_stats(df: pd.DataFrame, outdir: str):
    cols = [c for c in df.columns if c.startswith("P_")] + ["Dur"]
    df[cols].describe().round(2).to_csv(os.path.join(outdir, 'estadisticas_descriptivas.csv'))


def save_win_stats(df: pd.DataFrame, outdir: str):
    w = df['Ganador'].value_counts()
    p = (w / len(df) * 100).round(2)
    with open(os.path.join(outdir, 'victorias.txt'), 'w') as f:
        f.write("Conteo victorias:\n"); f.write(w.to_string())
        f.write("\n\n% victorias:\n"); f.write(p.to_string())


def plot_all(df: pd.DataFrame, outdir: str):
    os.makedirs(outdir, exist_ok=True)

    # 1) Conteo de victorias
    plt.figure()
    df['Ganador'].value_counts().plot(
        kind='bar',
        title='Victorias por agente'
    )
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, 'victorias_bar.png'))
    plt.close()

    # 2) Boxplot de puntos
    pts = [c for c in df.columns if c.startswith("P_")]
    plt.figure()
    df[pts].boxplot()
    plt.title('Distribución de puntos por agente')
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, 'boxplot_puntos.png'))
    plt.close()

    # 3) Histograma de duración
    plt.figure()
    df['Dur'].hist(bins=30)
    plt.title('Histograma de duración de partidas')
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, 'hist_duracion.png'))
    plt.close()

    # 4) Scatter MCTS vs Rand (si existen ambas columnas)
    if 'P_MCTS' in df.columns and 'P_Rand' in df.columns:
        plt.figure()
        plt.scatter(df['P_MCTS'], df['P_Rand'], alpha=0.6)
        plt.title('Puntos MCTS vs Rand')
        plt.xlabel('P_MCTS'); plt.ylabel('P_Rand')
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, 'scatter_puntos.png'))
        plt.close()

        # 5) Scatter duración vs diferencia de puntos
        df['Diff'] = df['P_MCTS'] - df['P_Rand']
        plt.figure()
        plt.scatter(df['Dur'], df['Diff'], alpha=0.6)
        plt.title('Duración vs (MCTS − Rand)')
        plt.xlabel('Dur (s)'); plt.ylabel('Diff Puntos')
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, 'scatter_dur_diff.png'))
        plt.close()


def main():
    logger = Logger(level=1)

    # Probamos ambos modos: normal y agresivo
    for modo in (False, True):
        etiqueta = "agresivo" if modo else "normal"
        outdir   = os.path.join(BASE_OUTPUT, etiqueta)
        os.makedirs(outdir, exist_ok=True)

        # 1) Simulación
        df = simular_partidas(
            n=N_PARTIDAS,
            modo_agresivo=modo,
            seguir_palo=True,    # siempre forzamos seguir palo
            seed=42,             # semilla base reproducible
            logger=logger
        )
        csv_path = os.path.join(outdir, "res.csv")
        df.to_csv(csv_path, index=False)
        logger.log(f" Resultados '{etiqueta}' → {csv_path}")

        # 2) Estadísticas y gráficas
        save_descriptive_stats(df, outdir)
        save_win_stats(df, outdir)
        plot_all(df, outdir)
        logger.log(f" Gráficas '{etiqueta}' en {outdir}\n")


if __name__ == "__main__":
    main()
