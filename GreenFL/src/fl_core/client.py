import time
from typing import Dict, Tuple, List, Any, Optional
import numpy as np
import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
from torch.utils.data import DataLoader

from src.battery import VirtualBattery
from src.telemetry.cellular_comm import DynamicCellularChannel

torch.set_num_threads(4)
torch.set_num_interop_threads(1)


class MissionAwareClient:
    def __init__(
        self,
        client_id: int,
        model: nn.Module,
        train_loader: DataLoader,
        test_loader: DataLoader,
        battery: Optional[VirtualBattery] = None,
        battery_profile: str = "drone",
        soc_threshold: float = 15.0,
        epochs: int = 2,
        unfreeze_strategy: str = "auto",
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        pruning_amount: float = 0.20,
        total_mission_time_s: float = 1800.0,
        fps: float = 30.0,
        e_infer_j: float = 0.10,
        e_safety_buffer_j: float = 5000.0,
        seconds_per_sample_epoch: float = 0.03
    ):
        self.client_id = client_id
        self.model = model
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.epochs = epochs
        self.unfreeze_strategy = unfreeze_strategy
        self.pruning_amount = pruning_amount
        self.seconds_per_sample_epoch = seconds_per_sample_epoch

        # Parámetros físicos de misión y supervivencia
        self.total_mission_time_s = total_mission_time_s
        self.fps = fps
        self.e_infer_j = e_infer_j
        self.e_safety_buffer_j = e_safety_buffer_j
        self.mission_elapsed_time_s = 0.0

        # Histórico de consumo energético de rondas pasadas
        self.history_energy_costs: List[float] = []

        self.cellular_channel = DynamicCellularChannel()
        self.battery = battery if isinstance(battery, VirtualBattery) else VirtualBattery(profile_name=battery_profile)
        self.soc_threshold = soc_threshold

        self.device = torch.device("cpu")
        self.model.to(self.device)
        self.lr = lr
        self.weight_decay = weight_decay
        self.criterion = nn.CrossEntropyLoss()

        self.optimizer: Optional[torch.optim.Optimizer] = None
        self._current_strategy: Optional[str] = None
        self._pruning_applied: bool = False

    def compute_survival_needs(self, current_elapsed_time_s: float) -> float:
        """Cálculo estricto de energía mínima requerida para completar la misión y retornar."""
        p_idle = getattr(self.battery, "p_idle", 8.0)
        p_kinematic = getattr(self.battery, "p_kinematics", getattr(self.battery, "p_kinematic", 35.0))
        p_npu_infer = self.e_infer_j * self.fps

        total_survival_power_w = p_idle + p_kinematic + p_npu_infer
        remaining_mission_time_s = max(0.0, self.total_mission_time_s - current_elapsed_time_s)

        return total_survival_power_w * remaining_mission_time_s

    def predict_training_cost(self) -> float:
        """Estimación base teórica usada únicamente cuando aún no existen datos históricos."""
        num_samples = len(self.train_loader.dataset) if hasattr(self.train_loader, "dataset") else 2400
        simulated_comp_time_s = num_samples * self.epochs * self.seconds_per_sample_epoch

        p_kinematic = getattr(self.battery, "p_kinematics", getattr(self.battery, "p_kinematic", 35.0))
        p_inference = getattr(self.battery, "p_inference", 10.0)
        p_active = p_kinematic + p_inference

        e_comp_est = p_active * simulated_comp_time_s
        e_comm_est = 1500.0  # Estimación inicial de transmisión celular
        return e_comp_est + e_comm_est

    def evaluate_null_update_condition(self, current_elapsed_time_s: float) -> Tuple[bool, float, float, float]:
        """
        Proyección de Supervivencia Prospectiva:
        Usa el promedio móvil del consumo real de las últimas rondas (o estimación teórica)
        para proyectar si la energía tras entrenar garantiza la supervivencia + buffer.
        """
        e_current = self.battery.current_energy_j

        if self.history_energy_costs:
            e_train_est = float(np.mean(self.history_energy_costs[-3:]))
        else:
            e_train_est = self.predict_training_cost()

        e_projected = e_current - e_train_est
        e_survival = self.compute_survival_needs(current_elapsed_time_s)

        should_trigger_null = e_projected < (e_survival + self.e_safety_buffer_j)
        return should_trigger_null, e_projected, e_survival, e_train_est

    def _configure_optimizer(self):
        soc = self.battery.get_soc_percentage()
        if self.unfreeze_strategy == "auto":
            curr_strategy = "full" if soc > 50.0 else "head_only"
        else:
            curr_strategy = self.unfreeze_strategy

        if self.optimizer is None or self._current_strategy != curr_strategy:
            self._current_strategy = curr_strategy

            if curr_strategy == "full":
                for param in self.model.parameters():
                    param.requires_grad = True
                effective_lr = 3e-4
            else:
                for param in self.model.parameters():
                    param.requires_grad = False
                head_names = ["fc", "classifier", "head", "heads"]
                for name, module in self.model.named_children():
                    if name in head_names:
                        for param in module.parameters():
                            param.requires_grad = True
                effective_lr = self.lr

            trainable_params = [p for p in self.model.parameters() if p.requires_grad]
            self.optimizer = torch.optim.AdamW(trainable_params, lr=effective_lr, weight_decay=self.weight_decay)

    def _apply_pruning(self, current_round: int = 1):
        if current_round <= 5 or self.pruning_amount <= 0.0 or self._pruning_applied:
            return

        for name, module in self.model.named_modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                prune.l1_unstructured(module, name="weight", amount=self.pruning_amount)
                prune.remove(module, "weight")
        
        self._pruning_applied = True

    def get_parameters(self) -> List[torch.Tensor]:
        return [val.detach().cpu().clone() for val in self.model.state_dict().values()]

    def set_parameters(self, parameters: List[torch.Tensor]):
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = {k: torch.tensor(v) if not isinstance(v, torch.Tensor) else v for k, v in params_dict}
        self.model.load_state_dict(state_dict, strict=True)

    def step_idle(self, elapsed_seconds: float = 20.0) -> Dict[str, Any]:
        e_idle = self.battery.consume_training_step(
            elapsed_seconds=elapsed_seconds,
            active_kinematics=True,
            active_inference=True,
            epochs=0
        )
        self.mission_elapsed_time_s += elapsed_seconds
        return {
            "status": "IDLE / SKIPPED",
            "loss": 0.0,
            "soc_pct": self.battery.get_soc_percentage(),
            "energy_total_j": e_idle,
            "energy_comp_j": e_idle,
            "energy_comm_j": 0.0,
            "elapsed_s": elapsed_seconds
        }

    def fit(self, parameters: List[torch.Tensor], current_round: int = 1, current_elapsed_time_s: float = 0.0) -> Tuple[List[torch.Tensor], int, Dict[str, Any]]:
        self.set_parameters(parameters)

        should_null, e_proj, e_surv, e_est = self.evaluate_null_update_condition(current_elapsed_time_s)

        if should_null:
            idle_metrics = self.step_idle(elapsed_seconds=20.0)
            idle_metrics["status"] = "NULL UPDATE"
            return self.get_parameters(), 0, idle_metrics

        self._configure_optimizer()

        self.model.train()
        if self._current_strategy == "head_only":
            for m in self.model.modules():
                if isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                    m.eval()

        total_loss, samples_processed = 0.0, 0
        start_time = time.time()

        for _ in range(self.epochs):
            for x, y in self.train_loader:
                x, y = x.to(self.device), y.to(self.device)

                self.optimizer.zero_grad(set_to_none=True)
                output = self.model(x)
                loss = self.criterion(output, y)
                loss.backward()

                nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()

                total_loss += loss.item() * len(y)
                samples_processed += len(y)

        self._apply_pruning(current_round=current_round)
        elapsed_time = max(0.001, time.time() - start_time)

        channel_cond = self.cellular_channel.sample_channel(env_degradation_mean=0.3)
        effective_pruning = self.pruning_amount if self._pruning_applied else 0.0
        payload_bits = sum(p.numel() * 32 for p in self.model.parameters()) * (1.0 - effective_pruning)
        e_comm_joules, comm_time_s = self.cellular_channel.calculate_comm_energy(payload_bits, channel_cond)

        e_comp = self.battery.consume_training_step(
            elapsed_seconds=elapsed_time,
            active_kinematics=True,
            active_inference=True,
            epochs=self.epochs
        )

        self.battery.current_energy_j = max(0.0, self.battery.current_energy_j - e_comm_joules)
        total_drained_joules = e_comp + e_comm_joules
        total_time_s = elapsed_time + comm_time_s
        self.mission_elapsed_time_s += total_time_s

        self.history_energy_costs.append(total_drained_joules)

        avg_loss = total_loss / max(1, samples_processed)

        return self.get_parameters(), samples_processed, {
            "status": "TRAINED",
            "loss": avg_loss,
            "soc_pct": self.battery.get_soc_percentage(),
            "energy_total_j": total_drained_joules,
            "energy_comp_j": e_comp,
            "energy_comm_j": e_comm_joules,
            "elapsed_s": total_time_s
        }

    def evaluate(self, parameters: List[torch.Tensor]) -> Tuple[float, float, int]:
        self.set_parameters(parameters)
        self.model.eval()
        correct, total, total_loss = 0, 0, 0.0

        with torch.inference_mode():
            for x, y in self.test_loader:
                x, y = x.to(self.device), y.to(self.device)
                output = self.model(x)
                loss = self.criterion(output, y)
                total_loss += loss.item() * len(y)
                _, predicted = torch.max(output, 1)
                total += y.size(0)
                correct += (predicted == y).sum().item()

        acc = correct / max(1, total)
        avg_loss = total_loss / max(1, total)
        return acc, avg_loss, total


GreenFlowerClient = MissionAwareClient
