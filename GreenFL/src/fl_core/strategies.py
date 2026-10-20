from typing import List, Tuple, Dict, Any, Optional, Union
import torch

try:
    import flwr as fl
    from flwr.common import Parameters, FitRes, Scalar, ndarrays_to_parameters, parameters_to_ndarrays
    from flwr.server.client_proxy import ClientProxy
    from flwr.server.strategy.aggregate import aggregate as flwr_aggregate
    FLWR_AVAILABLE = True
    BaseStrategy = fl.server.strategy.FedAvg
except ImportError:
    FLWR_AVAILABLE = False
    BaseStrategy = object


class GreenFedAvg(BaseStrategy):
    def __init__(self, energy_awareness: bool = True, **kwargs):
        if FLWR_AVAILABLE and issubclass(BaseStrategy, object) and BaseStrategy != object:
            super().__init__(**kwargs)
        self.energy_awareness = energy_awareness

    @staticmethod
    def aggregate(fit_results: List[Tuple[List[torch.Tensor], int]]) -> List[torch.Tensor]:
        valid_results = [(params, n) for params, n in fit_results if n > 0]
        if not valid_results:
            return fit_results[0][0] if fit_results else []

        total_samples = sum(n for _, n in valid_results)
        aggregated_params = [torch.zeros_like(p, dtype=torch.float32) for p in valid_results[0][0]]

        for params, n_samples in valid_results:
            weight = n_samples / total_samples
            for i, p in enumerate(params):
                aggregated_params[i] += p.to(torch.float32) * weight

        return aggregated_params

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[Any, Any]],
        failures: List[Union[Tuple[Any, Any], BaseException]],
    ) -> Tuple[Optional[Any], Dict[str, Any]]:
        if not FLWR_AVAILABLE:
            raise RuntimeError("Flower (flwr) no está instalado en el entorno actual.")

        if not results:
            return None, {}

        active_results = [
            (client, fit_res)
            for client, fit_res in results
            if fit_res.metrics.get("status") != "SKIPPED_LOW_BATT" and fit_res.num_examples > 0
        ]

        if not active_results:
            return None, {}

        weights_results = [
            (parameters_to_ndarrays(fit_res.parameters), fit_res.num_examples)
            for _, fit_res in active_results
        ]

        aggregated_ndarrays = flwr_aggregate(weights_results)
        parameters_aggregated = ndarrays_to_parameters(aggregated_ndarrays)

        #Muestreo de telemetría agregada
        total_energy = sum(fit_res.metrics.get("energy_j", 0.0) for _, fit_res in active_results)
        metrics = {
            "active_clients": len(active_results),
            "total_energy_j": float(total_energy),
        }

        return parameters_aggregated, metrics


#alias
GreenFedAvgStrategy = GreenFedAvg
