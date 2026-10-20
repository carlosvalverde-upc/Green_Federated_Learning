import csv
import os
import time


def get_cpu_temperature() -> float:
    try:
        with open("/sys/class/thermal/thermal_zone0/temp", "r") as f:
            return round(float(f.read().strip()) / 1000.0, 1)
    except Exception:
        return 0.0


class BenchmarkLogger:
    def __init__(
        self,
        filepath: str = "",
        model: str = "",
        dataset: str = "",
        profile: str = "",
        experiment_name: str = "",
        **kwargs
    ):
        if not filepath and experiment_name:
            self.filepath = os.path.join("logs", f"{experiment_name}.csv")
        elif filepath:
            self.filepath = filepath
        else:
            self.filepath = os.path.join("logs", "benchmark_telemetry.csv")

        self.experiment_name = experiment_name
        self.model = model
        self.dataset = dataset
        self.profile = profile
        self.start_time = time.time()
        self.last_round_time = self.start_time

        dir_path = os.path.dirname(self.filepath)
        if dir_path:
            os.makedirs(dir_path, exist_ok=True)

        self.headers = [
            "fl_round", "model", "dataset", "battery_profile", "client_id",
            "soc_pct", "status", "train_loss", "test_acc",
            "energy_consumed_j", "avg_power_w", "round_time_s",
            "elapsed_time_s", "cpu_temp_c", "comm_bytes"
        ]

        if not os.path.exists(self.filepath) or os.path.getsize(self.filepath) == 0:
            with open(self.filepath, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(self.headers)
                f.flush()
                os.fsync(f.fileno())

    def log_round(
        self,
        fl_round: int,
        client_id: int = 0,
        soc_pct: float = 100.0,
        status: str = "TRAINED",
        train_loss: float = 0.0,
        test_acc: float = 0.0,
        energy_j: float = 0.0,
        model: str = "",
        dataset: str = "",
        profile: str = "",
        comm_bytes: int = 0,
        **kwargs
    ):
        now = time.time()
        elapsed_s = round(now - self.start_time, 2)
        round_s = round(now - self.last_round_time, 2)
        self.last_round_time = now

        if energy_j == 0.0:
            energy_j = kwargs.get(
                "energy_consumed_j",
                kwargs.get("energy_total_j", kwargs.get("total_joules", 0.0))
            )

        m = model if model else self.model
        d = dataset if dataset else self.dataset
        p = profile if profile else self.profile

        avg_power_w = round(energy_j / round_s, 3) if round_s > 0 else 0.0
        cpu_temp = get_cpu_temperature()

        row = [
            fl_round, m, d, p, client_id,
            round(soc_pct, 2), str(status).upper(), round(train_loss, 4), round(test_acc, 4),
            round(energy_j, 2), avg_power_w, round_s,
            elapsed_s, cpu_temp, comm_bytes
        ]

        with open(self.filepath, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(row)
            f.flush()
            os.fsync(f.fileno())

    def log_metric(self, *args, **kwargs):
        if "energy_consumed_j" in kwargs and "energy_j" not in kwargs:
            kwargs["energy_j"] = kwargs.pop("energy_consumed_j")
        elif "energy_total_j" in kwargs and "energy_j" not in kwargs:
            kwargs["energy_j"] = kwargs.pop("energy_total_j")
        elif "total_joules" in kwargs and "energy_j" not in kwargs:
            kwargs["energy_j"] = kwargs.pop("total_joules")

        kwargs.pop("round_time_s", None)
        kwargs.pop("elapsed_time_s", None)
        self.log_round(*args, **kwargs)


TelemetryLogger = BenchmarkLogger
