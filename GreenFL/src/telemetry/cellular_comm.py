import numpy as np
from dataclasses import dataclass
from typing import Tuple


@dataclass
class ChannelConditions:
    gamma: float           # Factor de degradación del entorno [0.0 = Óptimo, 1.0 = Límite de celda]
    p_tx_w: float          # Potencia de emisión RF en Watts
    per: float             # Packet Error Rate
    r_up_mbps: float       # Tasa de subida (Mbps)


class DynamicCellularChannel:
    """
    Modelado dinámico 4G/5G (3GPP TS 38.331).
    Correlaciona estocásticamente la potencia RF, la tasa de transmisión y el error de paquete.
    """
    def __init__(
        self,
        p_tx_min: float = 0.5, p_tx_max: float = 4.5,
        per_min: float = 0.001, per_max: float = 0.20,
        r_min_mbps: float = 1.0, r_max_mbps: float = 50.0,
        p_tail_w: float = 2.0, tau_tail_s: float = 10.0
    ):
        self.p_tx_min, self.p_tx_max = p_tx_min, p_tx_max
        self.per_min, self.per_max = per_min, per_max
        self.r_min_mbps, self.r_max_mbps = r_min_mbps, r_max_mbps
        self.p_tail_w, self.tau_tail_s = p_tail_w, tau_tail_s

    def sample_channel(self, env_degradation_mean: float = 0.3) -> ChannelConditions:
        gamma = np.clip(np.random.normal(env_degradation_mean, 0.15), 0.0, 1.0)
        p_tx = np.clip(self.p_tx_min + gamma * (self.p_tx_max - self.p_tx_min) + np.random.normal(0, 0.05), self.p_tx_min, self.p_tx_max)
        per_sigm = 1.0 / (1.0 + np.exp(-8.0 * (gamma - 0.6)))
        per = np.clip(self.per_min + (self.per_max - self.per_min) * per_sigm + np.random.normal(0, 0.005), self.per_min, self.per_max)
        r_up = np.clip(self.r_max_mbps * ((1.0 - gamma) ** 1.5) + self.r_min_mbps, self.r_min_mbps, self.r_max_mbps)
        return ChannelConditions(gamma=gamma, p_tx_w=p_tx, per=per, r_up_mbps=r_up)

    def calculate_comm_energy(self, payload_bits: float, cond: ChannelConditions) -> Tuple[float, float]:
        effective_bits = payload_bits / max(0.001, (1.0 - cond.per))
        r_up_bps = cond.r_up_mbps * 1e6
        t_tx = effective_bits / max(1e3, r_up_bps)
        e_active = cond.p_tx_w * t_tx
        e_tail = self.p_tail_w * self.tau_tail_s
        return e_active + e_tail, t_tx + self.tau_tail_s
