import os
import sys
import time
import subprocess
from src.config import BENCHMARK_PROFILES
from src.visualization import PlotGenerator

PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
VEHICLES = ["rover", "drone"]
MODES = ["complete"]


def run_benchmark_suite():
    total_runs = len(BENCHMARK_PROFILES) * len(VEHICLES) * len(MODES)
    run_idx = 1

    print(f"🚀 INICIANDO BENCHMARK MATRICIAL DE GREEN FL ({total_runs} ejecuciones)")

    for mode in MODES:
        for (model, dataset), cfg in BENCHMARK_PROFILES.items():
            for vehicle in VEHICLES:
                print(f"\n==================================================================")
                print(f" RUN [{run_idx}/{total_runs}] | Modelo: {model} | Dataset: {dataset} | Vehículo: {vehicle}")
                print(f" Info: {cfg['description']}")
                print(f" Estrategia: {cfg['unfreeze_strategy'].upper()} | Res: {cfg['resolution']}")
                print(f"==================================================================")
                
                cmd = [
                    sys.executable, "main.py",
                    "--model", model,
                    "--dataset", dataset,
                    "--profile", vehicle,
                    "--mode", mode
                ]
                
                try:
                    subprocess.run(cmd, check=True)
                except subprocess.CalledProcessError as e:
                    print(f"❌ Error en la ejecución de {model}-{dataset}: {e}")
                
                run_idx += 1


    print("\n📊 Plot comparativa final (Rover vs. Drone)...")
    for mode in MODES:
        logs_dir = os.path.join("results", "logs", mode)
        output_fig = os.path.join("results", "figures", mode, "pareto_frontier.png")
        PlotGenerator.plot_pareto_from_logs(logs_dir=logs_dir, output_path=output_fig)


if __name__ == "__main__":
    run_benchmark_suite()
