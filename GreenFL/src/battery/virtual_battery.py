from typing import Union
from .profiles import BATTERY_PROFILES, ROVER_PROFILE, VehicleProfile


class VirtualBattery:
    def __init__(
        self,
        profile_name: str = "rover",
        capacity_wh: float = 100,
        **kwargs
    ):
        p_raw = kwargs.get("profile_type", kwargs.get("profile", profile_name))

        if isinstance(p_raw, VehicleProfile):
            self.profile = p_raw
        else:
            self.profile = BATTERY_PROFILES.get(str(p_raw).lower(), ROVER_PROFILE)

        self.capacity_wh = capacity_wh
        self.capacity_j = capacity_wh * 3600.0
        self.current_energy_j = self.capacity_j

    def get_soc_percentage(self) -> float:
        return max(0.0, (self.current_energy_j / self.capacity_j) * 100.0)

    def drain_time(self, time_delta: float, active_kinematics: bool = True, active_inference: bool = True) -> float:
        """
        Resta consumo continuo en función del tiempo:
        E_time = (P_idle + P_kinematic + (E_infer * FPS)) * delta_t
        """
        p_idle = self.profile.p_idle
        p_kinematic = self.profile.p_kinematic if active_kinematics else 0.0
        
        e_infer = getattr(self.profile, 'e_infer', 0.0)
        fps = getattr(self.profile, 'fps', 0.0)
        p_infer = (e_infer * fps) if active_inference else getattr(self.profile, 'p_infer', 0.0)

        p_continuous = p_idle + p_kinematic + p_infer
        drained_joules = p_continuous * max(0.0, time_delta)

        self.current_energy_j = max(0.0, self.current_energy_j - drained_joules)
        return drained_joules

    def drain_workload(self, epochs: int, cpu_train_joules_override: float = None) -> float:
        """
        Resta consumo por cómputo de entrenamiento local:
        E_workload = (E_train * epochs)
        """
        if cpu_train_joules_override is not None and cpu_train_joules_override > 0:
            drained_joules = cpu_train_joules_override
        else:
            e_train = getattr(self.profile, 'e_train', 120.0)
            drained_joules = e_train * max(0, epochs)

        self.current_energy_j = max(0.0, self.current_energy_j - drained_joules)
        return drained_joules

    def consume_training_step(
        self, 
        elapsed_seconds: float, 
        cpu_train_joules: float = 0.0, 
        active_kinematics: bool = True, 
        active_inference: bool = True,
        epochs: int = 1
    ) -> float:
        """
        Calcula la descarga combinada utilizando drain_time y drain_workload.
        """
        time_joules = self.drain_time(
            time_delta=elapsed_seconds, 
            active_kinematics=active_kinematics, 
            active_inference=active_inference
        )
        workload_joules = self.drain_workload(
            epochs=epochs, 
            cpu_train_joules_override=cpu_train_joules
        )
        return time_joules + workload_joules
