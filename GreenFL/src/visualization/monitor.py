import csv
import os
import re
import subprocess
import sys
import time
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

LOG_FILE = "native_pmic_telemetry.csv"
PLOT_FILE = "native_pmic_plot.png"
SAMPLING_INTERVAL = 0.05  # Muestreo ~20 Hz


class RPi5PMICReader:

  """Lector del PMIC nativo (Renesas/DA9091) de la Raspberry Pi 5."""

  def __init__(self):
    self.vcgencmd_bin = self._find_vcgencmd()

  def _find_vcgencmd(self) -> str:
    for path in ["/usr/bin/vcgencmd", "/bin/vcgencmd", "vcgencmd"]:
      try:
        subprocess.check_output([path, "version"], stderr=subprocess.DEVNULL)
        return path
      except Exception:
        continue
    return "vcgencmd"

  def get_raw_output(self) -> str:
    try:
      return subprocess.check_output(
          [self.vcgencmd_bin, "pmic_read_adc"], text=True
      )
    except Exception as e:
      return f"ERROR: {e}"

  def read_pmic_power_w(self) -> float:
    raw_str = self.get_raw_output()
    if "ERROR" in raw_str or not raw_str.strip():
      return self.read_sysfs_hwmon()

    voltages = {}
    currents = {}

    pattern = re.compile(
        r"([A-Za-z0-9_]+?_([VA]))\b.*=\s*([0-9.]+)", re.IGNORECASE
    )

    for line in raw_str.splitlines():
      match = pattern.search(line)
      if match:
        full_key, unit_type, val_str = match.groups()
        val = float(val_str)
        clean_key = re.sub(r"_[VA]$", "", full_key, flags=re.IGNORECASE).upper()

        if unit_type.upper() == "V":
          voltages[clean_key] = val
        elif unit_type.upper() == "A":
          currents[clean_key] = val

    main_rails = ["EXT5V", "5V_HEAD", "5V_SYS", "SYS_5V"]
    for rail in main_rails:
      if rail in voltages and rail in currents:
        return round(voltages[rail] * currents[rail], 4)

    total_power = 0.0
    for rail, current in currents.items():
      if rail in voltages:
        total_power += voltages[rail] * current

    if total_power > 0:
      return round(total_power, 4)

    return self.read_sysfs_hwmon()

  def read_sysfs_hwmon(self) -> float:
    try:
      hwmon_base = "/sys/class/hwmon"
      if os.path.exists(hwmon_base):
        for d in os.listdir(hwmon_base):
          hw_dir = os.path.join(hwmon_base, d)
          name_path = os.path.join(hw_dir, "name")
          if os.path.exists(name_path):
            with open(name_path, "r") as f:
              if "pmic" in f.read().lower() or "rpi" in f.read().lower():
                p_file = os.path.join(hw_dir, "power1_input")
                if os.path.exists(p_file):
                  with open(p_file, "r") as pf:
                    return round(float(pf.read().strip()) / 1e6, 4)
    except Exception:
      pass
    return 0.0


def generate_plot_and_stats(
    csv_path: str,
    plot_path: str,
    median_window: int = 21,
    smoothing_span: int = 40,
):
  """Calcula las métricas estadísticas (media, min, max, joules) y genera la gráfica."""
  if not os.path.exists(csv_path) or os.path.getsize(csv_path) == 0:
    print("CSV de telemetría no encontrado o vacío.")
    return

  df = pd.read_csv(csv_path)
  if df.empty or len(df) < 2:
    print("Datos insuficientes para calcular la media y generar el gráfico.")
    return

  df.columns = [str(c).strip().lower() for c in df.columns]

  t = df["elapsed_time_s"].values
  p_raw = df["pmic_power_w"].values

  mean_power = float(np.mean(p_raw))
  min_power = float(np.min(p_raw))
  max_power = float(np.max(p_raw))
  duration_s = float(t[-1] - t[0])
  total_samples = len(p_raw)

  # Integración trapezoidal compatible con NumPy 1.x (np.trapz) y NumPy 2.x (np.trapezoid)
  trapz_fn = getattr(np, "trapezoid", getattr(np, "trapz", None))
  total_joules = (
      float(trapz_fn(p_raw, x=t)) if len(p_raw) > 1 else mean_power * duration_s
  )

  print("\n" + "=" * 65)
  print("RESUMEN ESTADÍSTICO DE CONSUMO ELÉCTRICO (PMIC)")
  print("=" * 65)
  print(f" Muestras Totales:       {total_samples} lecturas")
  print(f" Duración del Registro:  {duration_s:.2f} segundos")
  print(f" Potencia Mínima:        {min_power:.4f} W")
  print(f" Potencia Máxima:        {max_power:.4f} W")
  print(f" POTENCIA MEDIA:       {mean_power:.4f} W")
  print(f" ENERGÍA TOTAL:        {total_joules:.4f} Joules (J)")
  print("=" * 65)

  df_plot = pd.DataFrame({"t": t, "p_raw": p_raw})
  df_plot["p_median"] = (
      df_plot["p_raw"]
      .rolling(window=median_window, center=True, min_periods=1)
      .median()
  )
  df_plot["p_smooth"] = df_plot["p_median"].ewm(span=smoothing_span).mean()

  fig, ax = plt.subplots(figsize=(11, 5), dpi=300)

  # Curva en bruto atenuada al fondo
  ax.plot(
      df_plot["t"],
      df_plot["p_raw"],
      color="#1f77b4",
      alpha=0.18,
      linewidth=0.8,
      label="Raw Native PMIC ADC",
  )

  # Curva filtrada
  ax.plot(
      df_plot["t"],
      df_plot["p_smooth"],
      color="#d62728",
      linewidth=2.0,
      label="Filtered System Power (W)",
  )

  # Línea de Potencia Media
  ax.axhline(
      mean_power,
      color="#2ca02c",
      linestyle="--",
      linewidth=1.8,
      label=f"Mean Power: {mean_power:.3f} W",
  )

  ax.set_title(
      "Raspberry Pi 5 Native PMIC Hardware Power Profile",
      fontsize=12,
      fontweight="bold",
      pad=12,
  )
  ax.set_xlabel("Elapsed Time (seconds)", fontweight="bold")
  ax.set_ylabel("Instantaneous Power Draw (W)", fontweight="bold")
  ax.grid(True, linestyle=":", alpha=0.6)

  ax.legend(
      bbox_to_anchor=(1.02, 1),
      loc="upper left",
      borderaxespad=0,
      frameon=True,
      facecolor="white",
      framealpha=0.9,
      edgecolor="lightgray",
  )

  plt.savefig(plot_path, bbox_inches="tight")
  plt.close()
  print(f"Gráfico guardado en: {plot_path}")


def start_native_pmic_monitoring():
  pmic_reader = RPi5PMICReader()
  initial_power = pmic_reader.read_pmic_power_w()

  print("=" * 65)
  print("MONITOR DE POTENCIA NATIVO PMIC (Raspberry Pi 5)")
  print(f"Lectura inicial PMIC: {initial_power:.3f} W")
  print(f"Registrando telemetría en: {LOG_FILE}")
  print("Presiona [CTRL + C] para detener el registro y calcular la media.")
  print("=" * 65)

  with open(LOG_FILE, mode="w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["timestamp_unix", "elapsed_time_s", "pmic_power_w"])

  start_time = time.time()

  try:
    while True:
      current_time = time.time()
      elapsed_s = round(current_time - start_time, 3)
      power_w = pmic_reader.read_pmic_power_w()

      with open(LOG_FILE, mode="a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([current_time, elapsed_s, power_w])

      sys.stdout.write(
          f"\rTiempo: {elapsed_s:.1f}s | Potencia PMIC:"
          f" {power_w:.3f} W   "
      )
      sys.stdout.flush()

      time.sleep(SAMPLING_INTERVAL)

  except KeyboardInterrupt:
    print("\n\nMonitoreo detenido por el usuario.")
    print("Calculando media y procesando gráfica...")
    generate_plot_and_stats(LOG_FILE, PLOT_FILE)


if __name__ == "__main__":
  start_native_pmic_monitoring()
