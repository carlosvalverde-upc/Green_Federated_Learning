import argparse
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.models import ModelFactory
from src.datasets import DatasetFactory
from src.fl_core import MissionAwareClient, FederatedServer
from src.telemetry import TelemetryLogger
from src.visualization import PlotGenerator
from src.config import BENCHMARK_PROFILES


def parse_args():
    parser = argparse.ArgumentParser(description="Green Federated Learning Framework - Single Run")
    parser.add_argument(
        "--model",
        type=str,
        default="squeezenet",
        help="Modelo de visión (squeezenet, mobilenet_v3_small, mobilenet_v2, efficientnet, resnet18, resnet34)"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="cifar10",
        help="Dataset (mnist, svhn, cifar10, eurosat, gtsrb, plantvillage, pets, cifar100, flowers102)"
    )
    parser.add_argument(
        "--profile", "--battery_profile",
        dest="profile",
        type=str,
        default="rover",
        choices=["rover", "drone"],
        help="Perfil del vehículo (rover, drone)"
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="complete",
        choices=["debug", "complete"],
        help="Modo de ejecución: 'debug' (Early Stopping) o 'complete' (50 rondas)"
    )
    parser.add_argument(
        "--freeze",
        type=str,
        default=None,
        choices=["full", "partial", "head_only"],
        help="Sobrescribir estrategia de congelado (full, partial, head_only). Si es None, usa config.py"
    )
    parser.add_argument(
        "--num_rounds",
        type=int,
        default=50,
        help="Número máximo de rondas FL"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=None,
        help="Tamaño de lote (si es None, usa config.py)"
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=None,
        help="Learning rate (si es None, usa config.py)"
    )
    parser.add_argument(
        "--soc_threshold",
        type=float,
        default=20.0,
        help="Umbral de batería crítica (SoC %)"
    )
    parser.add_argument(
        "--no_plots",
        action="store_true",
        help="Desactivar generación de gráficos"
    )
    return parser.parse_args()
    

def run_experiment(args):
    enable_early_stop = (args.mode == "debug")
    
    # 1. Recuperar perfil ad-hoc
    pair_key = (args.model.lower(), args.dataset.lower())
    cfg = BENCHMARK_PROFILES.get(pair_key, {})

    unfreeze_strategy = args.freeze or cfg.get("unfreeze_strategy", "head_only")
    batch_size = args.batch_size or cfg.get("batch_size", 32)
    lr = args.lr or cfg.get("lr", 1e-3)
    epochs = cfg.get("epochs", 1)
    resolution = cfg.get("resolution", (64, 64))
    max_train_samples = cfg.get("max_train_samples", 1200)
    max_test_samples = cfg.get("max_test_samples", 500)

    exp_name = f"{args.model}_{args.dataset}_{args.profile}"
    logs_dir = os.path.join("results", "logs", args.mode)
    figures_dir = os.path.join("results", "figures", args.mode)
    os.makedirs(logs_dir, exist_ok=True)
    os.makedirs(figures_dir, exist_ok=True)

    csv_path = os.path.join(logs_dir, f"{exp_name}.csv")
    figure_path = os.path.join(figures_dir, f"{exp_name}_summary.png")

    print("=" * 60)
    print(f"   GREEN FEDERATED LEARNING -  [{args.mode.upper()}]   ")
    print("=" * 60)
    print(f"Modelo:            {args.model}")
    print(f"Dataset:           {args.dataset}")
    print(f"Descongelado:      {unfreeze_strategy.upper()}")
    print(f"Resolución:        {resolution[0]}x{resolution[1]}")
    print(f"Batch Size:        {batch_size}")
    print(f"Épocas Locales:    {epochs}")
    print(f"Muestras Train:    {max_train_samples}")
    print(f"Muestras Test:     {max_test_samples}")
    print(f"Learning Rate:     {lr}")
    print(f"Perfil Vehículo:   {args.profile} (Umbral: {args.soc_threshold}%)")
    print(f"Modo:              {args.mode} (Early Stopping: {enable_early_stop})")
    print("-" * 60)

    logger = TelemetryLogger(filepath=csv_path, model=args.model, dataset=args.dataset, profile=args.profile)

    train_loader, test_loader, num_classes = DatasetFactory.get_dataset(
        dataset_name=args.dataset,
        batch_size=batch_size,
        max_train_samples=max_train_samples,
        max_test_samples=max_test_samples,
        resolution=resolution,
        client_id=0,
        num_clients=1
    )

    model = ModelFactory.create_model(model_name=args.model, num_classes=num_classes)
    client = MissionAwareClient(
        client_id=0,
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        battery_profile=args.profile,
        soc_threshold=args.soc_threshold,
        epochs=epochs,
        unfreeze_strategy=unfreeze_strategy,
        lr=lr,
        weight_decay=cfg.get("weight_decay", 1e-4)
    )

    server = FederatedServer(clients=[client], logger=logger)
    server.run_rounds(num_rounds=args.num_rounds, patience_rounds=3, enable_early_stop=enable_early_stop)

    if not args.no_plots:
        PlotGenerator.plot_experiment_summary(csv_filepath=csv_path, output_path=figure_path)


if __name__ == "__main__":
    cli_args = parse_args()
    run_experiment(cli_args)
