import os
import time
import itertools
from multiprocessing import Pool, cpu_count
from typing import Tuple, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm  # Barra de progreso

from briscas.game import BriscasGame
from briscas.agents import AgenteMCTS, AgenteAleatorio

# Parámetros del diseño factorial
C_VALUES    = [0.5, 1.0, 1.4, 2.0]
ITER_VALUES = [100, 500, 1000]
N_PARTIDAS  = 50
SEED_BASE   = 42

# Directorios y archivos de salida
OUT_DIR   = "mcts_evaluation"
CSV_RAW   = os.path.join(OUT_DIR, "mcts_raw_results.csv")
CSV_AGG   = os.path.join(OUT_DIR, "mcts_agg_results.csv")
PLOTS_DIR = os.path.join(OUT_DIR, "plots")


def run_game(args: Tuple[float, int, int]) -> dict:
    """
    Ejecuta una partida MCTS vs Aleatorio.
    args = (c, iteraciones, seed)
    """
    c, n_iters, seed = args

    mcts = AgenteMCTS("MCTS", iteraciones=n_iters, c=c)
    rand = AgenteAleatorio("Rand")
    juego = BriscasGame([mcts, rand], seed=seed)

    t0 = time.perf_counter()
    puntos = juego.jugar()
    duration = time.perf_counter() - t0

    return {
        "c": c,
        "iteraciones": n_iters,
        "seed": seed,
        "duration": duration,
        "P_MCTS": puntos[mcts],
        "P_Rand": puntos[rand],
        "victory": int(puntos[mcts] > puntos[rand])
    }


def simulate_all() -> pd.DataFrame:
    """
    Ejecuta todas las simulaciones con barra de progreso.
    """
    os.makedirs(OUT_DIR, exist_ok=True)

    tasks = [
        (c, iters, SEED_BASE + int(c * 100) + iters + i)
        for c, iters in itertools.product(C_VALUES, ITER_VALUES)
        for i in range(N_PARTIDAS)
    ]

    total = len(tasks)
    print(f"▶ Simulando {total} partidas usando {cpu_count()} CPUs…")

    results = []
    with Pool(processes=cpu_count()) as pool:
        for r in tqdm(pool.imap_unordered(run_game, tasks), total=total, desc="Progreso"):
            results.append(r)

    df_raw = pd.DataFrame(results)
    df_raw.to_csv(CSV_RAW, index=False)
    print(f"✔ Resultados crudos guardados en {CSV_RAW}")
    return df_raw


def aggregate(df: pd.DataFrame) -> pd.DataFrame:
    """
    Agrega métricas por configuración (c, iteraciones).
    """
    agg = df.groupby(["c", "iteraciones"]).agg(
        victory_rate = ("victory", lambda s: 100 * s.mean()),
        duration_mean= ("duration", "mean"),
        duration_std = ("duration", "std"),
        Pm_mean      = ("P_MCTS", "mean"),
        Pm_std       = ("P_MCTS", "std"),
        Pr_mean      = ("P_Rand", "mean"),
        Pr_std       = ("P_Rand", "std"),
    ).reset_index()

    agg.to_csv(CSV_AGG, index=False)
    print(f"✔ Resultados agregados guardados en {CSV_AGG}")
    return agg


def plot_heatmap(agg: pd.DataFrame):
    """Genera heatmap del porcentaje de victorias."""
    ptab = agg.pivot(index="c", columns="iteraciones", values="victory_rate")
    plt.figure(figsize=(6, 5))
    sns.heatmap(ptab, annot=True, fmt=".1f", cmap="YlGnBu")
    plt.title("Victory Rate (%) – MCTS vs Random")
    plt.ylabel("Constante de exploración c")
    plt.xlabel("Iteraciones")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "heatmap_victory_rate.png"))
    plt.close()


def plot_duration(agg: pd.DataFrame):
    """Gráfico de líneas: duración promedio vs iteraciones."""
    plt.figure(figsize=(6, 4))
    sns.lineplot(
        data=agg,
        x="iteraciones", y="duration_mean", hue="c", marker="o",
        err_style="band", err_kws={"alpha": 0.3}
    )
    plt.title("Duración promedio de partidas")
    plt.ylabel("Duración (s)")
    plt.xlabel("Iteraciones")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "lineplot_duration.png"))
    plt.close()


def plot_boxplot(df: pd.DataFrame):
    """Boxplot de puntos obtenidos por MCTS según c."""
    plt.figure(figsize=(6, 4))
    sns.boxplot(data=df, x="c", y="P_MCTS")
    plt.title("Distribución de puntos MCTS por c")
    plt.xlabel("Constante de exploración c")
    plt.ylabel("Puntos MCTS")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "boxplot_MCTS_points.png"))
    plt.close()


def main():
    # 1️⃣ Simulación
    df_raw = simulate_all()

    # 2️⃣ Agregación de resultados
    df_agg = aggregate(df_raw)

    # 3️⃣ Generación de gráficos
    os.makedirs(PLOTS_DIR, exist_ok=True)
    plot_heatmap(df_agg)
    plot_duration(df_agg)
    plot_boxplot(df_raw)

    print(f"✔ Gráficos guardados en {PLOTS_DIR}")


if __name__ == "__main__":
    main()
