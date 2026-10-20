"""
green_metrics.py - Métricas de Eficiencia Energética para Green Edge AI.
Ref: Schwartz et al. (2020) "Green AI", CACM.
"""

from typing import List, Dict, Any


class GreenMetricsCalculator:
    
    @staticmethod
    def compute_ear(total_energy_joules: float, acc_initial: float, acc_final: float) -> float:
        """EAR = E_total / (Acc_final - Acc_inicial) [J/%]"""
        acc_delta = max(0.0001, acc_final - acc_initial)
        return total_energy_joules / acc_delta

    @staticmethod
    def compute_gesi(e_green: float, e_baseline_50: float) -> float:
        """GESI = (1 - (E_green / E_baseline_50)) * 100 [%]"""
        if e_baseline_50 <= 0: return 0.0
        return max(0.0, (1.0 - (e_green / e_baseline_50)) * 100.0)

    @staticmethod
    def compute_eoa(null_update_count: int, e_avg_train_round: float) -> float:
        """EOA = Null_Updates_Prevenidos * E_media_ronda [J]"""
        return null_update_count * e_avg_train_round

    @staticmethod
    def compute_npe(acc_history: List[float], energy_history: List[float]) -> List[float]:
        """NPE(r) = (Acc(r) - Acc(r-1)) / E_ronda(r) [%/J]"""
        npe = [0.0]
        for r in range(1, len(acc_history)):
            d_acc = acc_history[r] - acc_history[r - 1]
            e_r = max(0.001, energy_history[r])
            npe.append(d_acc / e_r)
        return npe