import gc
import time
import random
import numpy as np
from typing import List, Optional, Dict, Any, Tuple
from .client import MissionAwareClient
from .strategies import GreenFedAvgStrategy
from src.telemetry import TelemetryLogger
from src.datasets.loader import calculate_entropy


class FederatedServer:
    def __init__(
        self,
        clients: List[MissionAwareClient],
        logger: Optional[TelemetryLogger] = None,
        clients_per_round: int = 1,
        w_soc: float = 0.35,
        w_entropy: float = 0.30,
        w_priority: float = 0.15,
        w_comm_cost: float = 0.20,
        patience_rounds: int = 6,
        min_acc_delta: float = 0.005,
        offline_rate: float = 0.15
    ):
        self.clients = clients
        self.logger = logger
        self.global_parameters = self.clients[0].get_parameters()
        self.start_time = time.time()

        self.clients_per_round = min(clients_per_round, len(clients))
        self.w_soc = w_soc
        self.w_entropy = w_entropy
        self.w_priority = w_priority
        self.w_comm_cost = w_comm_cost

        self.patience_rounds = patience_rounds
        self.min_acc_delta = min_acc_delta
        self.offline_rate = offline_rate

    def _score_and_select_clients(self, available_client_ids: List[int]) -> List[int]:
        scores = []
        for client in self.clients:
            if client.client_id not in available_client_ids:
                continue

            soc_norm = client.battery.get_soc_percentage() / 100.0
            h_d = calculate_entropy(client.train_loader.dataset) if hasattr(client.train_loader, "dataset") else 1.0
            entropy_norm = h_d / 3.32

            priority_norm = getattr(client, "mission_priority", 1.0) / 3.0
            comm_penalty_norm = 0.15

            s_i = (
                self.w_soc * soc_norm +
                self.w_entropy * entropy_norm +
                self.w_priority * priority_norm -
                self.w_comm_cost * comm_penalty_norm
            )
            scores.append((client.client_id, s_i))

        scores.sort(key=lambda x: x[1], reverse=True)
        return [cid for cid, _ in scores[:self.clients_per_round]]

    def _check_accuracy_stagnation(
        self,
        current_round: int,
        acc_history: List[float]
    ) -> Tuple[bool, str]:
        if current_round <= self.patience_rounds:
            return False,

        recent_accs = acc_history[-self.patience_rounds:]
        acc_gain = recent_accs[-1] - recent_accs[0]

        if acc_gain < self.min_acc_delta:
            return True, (
                f"ESTANCAMIENTO DE PRECISIÓN (Ronda {current_round}). "
                f"Ganancia en las últimas {self.patience_rounds} rondas: {acc_gain*100:.2f}% "
                f"(inferior al umbral mínimo de {self.min_acc_delta*100:.2f}%)."
            )

        return False, 

    def _compute_green_metrics(
        self,
        total_energy_j: float,
        acc_initial: float,
        acc_final: float,
        null_updates_count: int
    ) -> Dict[str, float]:
        acc_delta_pct = max(0.01, (acc_final - acc_initial) * 100.0)
        ear = total_energy_j / acc_delta_pct
        eoa = null_updates_count * 75.0

        return {"ear_j_pct": ear, "eoa_j": eoa}

    def run_rounds(
        self, 
        num_rounds: int = 50, 
        max_null_rounds: int = 5,
        enable_early_stop: bool = True
    ):
        mode_str = f"EARLY STOPPING POR ESTANCAMIENTO (Patience: {self.patience_rounds})" if enable_early_stop else "COMPLETO (Sin Parada)"
        print(f"\n--- Iniciando Simulación FL [{mode_str}] sobre {len(self.clients)} Nodos ---")
        
        consecutive_null_rounds = 0
        total_accumulated_energy_j = 0.0
        null_updates_count = 0
        loss_history, acc_history = [], []

        for r in range(1, num_rounds + 1):
            round_start_time = time.time()

            available_ids = [
                client.client_id for client in self.clients 
                if random.random() > self.offline_rate
            ]

            selected_ids = self._score_and_select_clients(available_ids)

            fit_results = []
            active_clients_in_round = 0
            round_energy_j = 0.0
            client_metrics: Dict[int, Dict[str, Any]] = {}

            for client in self.clients:
                cid = client.client_id

                if cid in selected_ids:
                    params, num_samples, metrics = client.fit(self.global_parameters, current_round=r)
                    if metrics.get("status") == "TRAINED" and num_samples > 0:
                        fit_results.append((params, num_samples))
                        active_clients_in_round += 1
                    else:
                        null_updates_count += 1
                else:
                    metrics = client.step_idle(elapsed_seconds=2.5)
                    if cid not in available_ids:
                        metrics["status"] = "OFFLINE"

                client_metrics[cid] = metrics
                round_energy_j += metrics.get("energy_total_j", 0.0)

            total_accumulated_energy_j += round_energy_j

            if active_clients_in_round > 0:
                self.global_parameters = GreenFedAvgStrategy.aggregate(fit_results)
                consecutive_null_rounds = 0
            else:
                consecutive_null_rounds += 1

            round_accs, round_losses = [], []
            print(f"\n[Ronda {r}/{num_rounds}] - Seleccionados: {selected_ids} | Disponibles: {available_ids}")

            for client in self.clients:
                acc, test_loss, _ = client.evaluate(self.global_parameters)
                round_accs.append(acc)
                round_losses.append(test_loss)
                soc = client.battery.get_soc_percentage()

                c_meta = client_metrics.get(client.client_id, {})
                c_status = c_meta.get("status", "SKIP")
                c_train_loss = c_meta.get("loss", 0.0)
                c_energy = c_meta.get("energy_total_j", 0.0)
                c_time = c_meta.get("elapsed_s", 0.0)

                print(
                    f"  └─ [Cliente {client.client_id}] Status: {c_status:<11} | "
                    f"Train Loss: {c_train_loss:.4f} | Test Acc Local: {acc*100:.2f}% | "
                    f"SoC Batería: {soc:.1f}% | Energía: {c_energy:.1f} J | Tiempo: {c_time:.2f}s"
                )

                if self.logger:
                    self.logger.log_round(
                        fl_round=r,
                        client_id=client.client_id,
                        soc_pct=soc,
                        status=c_status,
                        train_loss=c_train_loss,
                        test_acc=acc,
                        energy_j=c_energy
                    )

            round_elapsed_s = time.time() - round_start_time

            avg_round_acc = sum(round_accs) / max(1, len(round_accs))
            avg_round_loss = sum(round_losses) / max(1, len(round_losses))
            
            acc_history.append(avg_round_acc)
            loss_history.append(avg_round_loss)

            print(
                f"Resumen Ronda {r}: Tiempo: {round_elapsed_s:.2f}s | "
                f"Acc Global Media: {avg_round_acc*100:.2f}% | "
                f"Loss Media: {avg_round_loss:.4f} | Energía Ronda: {round_energy_j:.1f} J\n"
            )

            should_stop, reason = self._check_accuracy_stagnation(r, acc_history)

            if enable_early_stop and should_stop:
                green_stats = self._compute_green_metrics(
                    total_accumulated_energy_j, acc_history[0], avg_round_acc, null_updates_count
                )
                print("\n" + "="*70)
                print(f"EARLY STOPPING ACTIVADO: {reason}")
                print("="*70)
                print(f"  • EAR (Energy-to-Accuracy Ratio): {green_stats['ear_j_pct']:.2f} J/%")
                print(f"  • Energía Total Consumida:        {total_accumulated_energy_j:.1f} J")
                print(f"  • Tiempo Total de Ejecución:      {time.time() - self.start_time:.2f} s")
                print("="*70 + "\n")
                break

            if consecutive_null_rounds >= max_null_rounds:
                print("\nParada por Baterías Agotadas en la Flota.")
                break

            gc.collect()


def run_server(server_address: str = "0.0.0.0:8080", num_rounds: int = 10, min_clients: int = 2):
    print(f"Servidor FL escuchando en {server_address} (Configurado para {num_rounds} rondas)...")
