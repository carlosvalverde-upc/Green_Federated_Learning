from dataclasses import dataclass

@dataclass
class VehicleProfile:
    name: str
    p_kinematic: float  
    p_idle: float       
    e_train: float      
    e_infer: float      
    fps: float          

    # Parámetros RF
    p_tx_min: float = 0.5
    p_tx_max: float = 4.5
    p_tail: float = 2.0
    tau_tail: float = 10.0

    def update_measured_metrics(self, e_train: float = None, e_infer: float = None, fps: float = None):
        """
        Sobrescribe dinámicamente los valores teóricos con las mediciones
        instantáneas reales capturadas por el INA219 / PMIC.
        """
        if e_train is not None and e_train > 0:
            self.e_train = e_train
        if e_infer is not None and e_infer > 0:
            self.e_infer = e_infer
        if fps is not None and fps > 0:
            self.fps = fps


ROVER_PROFILE = VehicleProfile("rover", p_kinematic=20.0, p_idle=2.7789, e_train=6.1546, e_infer=0.018, fps=15.0)
DRONE_PROFILE = VehicleProfile("drone", p_kinematic=20.0, p_idle=2.7789, e_train=6.1546, e_infer=0.018, fps=15.0)

BATTERY_PROFILES = {
    "rover": ROVER_PROFILE,
    "drone": DRONE_PROFILE,
}
