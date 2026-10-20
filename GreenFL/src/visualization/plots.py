import glob
import os
import matplotlib.pyplot as plt
import pandas as pd


class PlotGenerator:

  @staticmethod
  def cure_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Freezes test accuracy upon training stagnation or battery depletion."""
    df = df.copy()
    df.columns = [str(c).strip().lower() for c in df.columns]

    if "fl_round" not in df.columns:
      return df

    rounds = sorted(df["fl_round"].unique())
    stagnant_r = None

    soc_col = next(
        (c for c in ["soc_pct", "soc", "battery_soc"] if c in df.columns), None
    )

    for r in rounds:
      sub = df[df["fl_round"] == r]
      statuses = sub["status"].astype(str).str.upper().tolist()
      no_training = any("NULL" in s for s in statuses) or not any(
          "TRAINED" in s for s in statuses
      )
      battery_depleted = (
          (sub[soc_col].max() <= 0.0) if soc_col is not None else False
      )

      if no_training or battery_depleted:
        stagnant_r = r
        break

    if stagnant_r is not None:
      valid_r = [r for r in rounds if r < stagnant_r]
      last_valid_acc = float(
          df[
              df["fl_round"] == (max(valid_r) if valid_r else stagnant_r)
          ]["test_acc"].mean()
      )
      df.loc[df["fl_round"] >= stagnant_r, "test_acc"] = last_valid_acc

    return df

  @classmethod
  def plot_experiment_summary(
      cls,
      csv_filepath: str,
      output_path: str = "",
  ):
    """Generates temporal evolution plot (Green SoC % circles vs Blue Accuracy % squares)."""
    if not os.path.exists(csv_filepath) or os.path.getsize(csv_filepath) == 0:
      print(f"[Warning] CSV file empty or missing: {csv_filepath}")
      return

    try:
      raw_df = pd.read_csv(csv_filepath)
      if raw_df.empty:
        return

      df = cls.cure_dataframe(raw_df)

      if "client_id" not in df.columns:
        df["client_id"] = 0

      if "fl_round" not in df.columns:
        df["fl_round"] = range(1, len(df) + 1)

      if "soc_pct" not in df.columns:
        df["soc_pct"] = 100.0

      acc_col = next(
          (
              c
              for c in ["test_acc", "accuracy", "acc", "global_acc"]
              if c in df.columns
          ),
          "test_acc",
      )

      if df[acc_col].max() <= 1.0 and df[acc_col].max() > 0:
        df[acc_col] = df[acc_col] * 100.0

      os.makedirs(os.path.dirname(output_path), exist_ok=True)
      fig, ax1 = plt.subplots(figsize=(10, 4.5), dpi=300)

      # 1. Axis Y1: Battery SoC (%) -> Green with circle markers
      color_soc = "tab:green"
      for client_id, group in df.groupby("client_id"):
        ax1.plot(
            group["fl_round"],
            group["soc_pct"],
            label="Battery SoC (%)",
            color=color_soc,
            marker="o",
            markersize=5,
            linewidth=2,
            linestyle="-",
            alpha=0.9,
        )

      ax1.set_xlabel("FL Training Round", fontweight="bold", fontsize=11)
      ax1.set_ylabel(
          "State of Charge - SoC (%)",
          color=color_soc,
          fontweight="bold",
          fontsize=11,
      )
      ax1.tick_params(axis="y", labelcolor=color_soc)
      ax1.set_ylim(-5, 105)
      ax1.grid(True, linestyle=":", alpha=0.6)

      # 2. Axis Y2: Global Accuracy (%) -> Blue with square markers
      color_acc = "tab:blue"
      ax2 = ax1.twinx()
      avg_acc = df.groupby("fl_round")[acc_col].mean()
      ax2.plot(
          avg_acc.index,
          avg_acc.values,
          label="Mean Accuracy (%)",
          color=color_acc,
          marker="s",
          markersize=5,
          linewidth=2,
          linestyle="--",
          alpha=0.9,
      )
      ax2.set_ylabel(
          "Accuracy (%)", color=color_acc, fontweight="bold", fontsize=11
      )
      ax2.tick_params(axis="y", labelcolor=color_acc)
      ax2.set_ylim(-5, 105)

      # Lower left legend
      lines1, labels1 = ax1.get_legend_handles_labels()
      lines2, labels2 = ax2.get_legend_handles_labels()
      ax1.legend(
          lines1 + lines2,
          labels1 + labels2,
          loc="lower left",
          frameon=True,
          facecolor="white",
          framealpha=0.9,
          edgecolor="lightgray",
      )

      exp_title = (
          os.path.basename(csv_filepath)
          .replace("cured_", "")
          .replace(".csv", "")
      )
      plt.title(
          f"Battery Evolution vs. Global Accuracy\n[{exp_title}]",
          fontsize=12,
          fontweight="bold",
          pad=12,
      )

      fig.tight_layout()
      plt.savefig(output_path, dpi=300)
      plt.close()
      print(f"[PlotGenerator] Individual plot saved: {output_path}")

    except Exception as e:
      print(f"[Error] Failed to process {csv_filepath}: {e}")

  @classmethod
  def plot_pareto_from_logs(
      cls, logs_dir: str, output_path: str = ""
  ):
    """Builds Pareto Frontier plot with numbered markers and external 2-column legend."""
    if not os.path.exists(logs_dir):
      print(f"[Warning] Directory not found: {logs_dir}")
      return

    csv_files = sorted(glob.glob(os.path.join(logs_dir, "*.csv")))
    if not csv_files:
      print(f"[Warning] No CSV logs found in: {logs_dir}")
      return

    summary_data = []

    for idx, csv_path in enumerate(csv_files, start=1):
      if os.path.getsize(csv_path) == 0:
        continue

      try:
        raw_df = pd.read_csv(csv_path)
        if raw_df.empty:
          continue

        df = cls.cure_dataframe(raw_df)
        final_r = (
            int(df["fl_round"].max()) if "fl_round" in df.columns else len(df)
        )
        last_df = (
            df[df["fl_round"] == final_r]
            if "fl_round" in df.columns
            else df.iloc[-1:]
        )

        acc_col = next(
            (c for c in ["test_acc", "accuracy", "acc"] if c in df.columns),
            "test_acc",
        )
        final_acc = float(last_df[acc_col].mean())
        if final_acc <= 1.0 and final_acc > 0:
          final_acc *= 100.0

        soc_col = next((c for c in ["soc_pct", "soc"] if c in df.columns), None)
        final_soc = float(last_df[soc_col].mean()) if soc_col else 0.0

        clean_fname = (
            os.path.basename(csv_path)
            .replace("cured_", "")
            .replace(".csv", "")
        )
        vehicle = (
            df["battery_profile"].iloc[0]
            if "battery_profile" in df.columns
            else ("drone" if "drone" in clean_fname.lower() else "rover")
        )

        summary_data.append({
            "id": idx,
            "name": clean_fname,
            "acc": final_acc,
            "soc": final_soc,
            "vehicle": str(vehicle).lower(),
        })
      except Exception as e:
        print(f"[Warning] Error parsing {csv_path}: {e}")

    if not summary_data:
      return

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig, ax = plt.subplots(figsize=(11, 6), dpi=300)

    for r in summary_data:
      is_drone = r["vehicle"] == "drone"
      marker = "o" if is_drone else "^"
      color = "#1f77b4" if is_drone else "#d62728"

      ax.scatter(
          r["soc"],
          r["acc"],
          s=120,
          color=color,
          marker=marker,
          edgecolors="black",
          linewidth=0.8,
          label=f"[{r['id']}] {r['name']}",
      )
      ax.annotate(
          str(r["id"]),
          (r["soc"], r["acc"]),
          textcoords="offset points",
          xytext=(5, 5),
          fontsize=9,
          fontweight="bold",
      )

    ax.set_title(
        "Pareto: Energy Efficiency vs. Accuracy",
        fontsize=13,
        fontweight="bold",
        pad=12,
    )
    ax.set_xlabel(
        "Remaining Energy (%) [Final SoC]", fontsize=11, fontweight="bold"
    )
    ax.set_ylabel("Final Accuracy (%)", fontsize=11, fontweight="bold")
    ax.set_xlim(-5, 105)
    ax.set_ylim(-5, 105)
    ax.grid(True, linestyle="--", alpha=0.5)

    ax.legend(
        title="Experiments",
        bbox_to_anchor=(1.02, 1),
        loc="upper left",
        ncols=2,
        fontsize=7,
        title_fontsize=9,
        frameon=True,
        edgecolor="lightgray",
    )

    plt.savefig(output_path, bbox_inches="tight")
    plt.close()
    print(f"[PlotGenerator] Pareto plot saved: {output_path}")


if __name__ == "__main__":
  base_dir = ""
  folders = ["curated_plots", "curated_plots_mission_aware"]
  out_base = os.path.join(base_dir, "plots traducidos")

  for folder in folders:
    in_dir = os.path.join(base_dir, folder)
    out_dir = os.path.join(out_base, folder)

    if os.path.exists(in_dir):
      print(f"Processing folder: {folder}...")
      os.makedirs(out_dir, exist_ok=True)

      for f in sorted(glob.glob(os.path.join(in_dir, "*.csv"))):
        out_png = os.path.join(out_dir, os.path.basename(f).replace(".csv", ".png"))
        PlotGenerator.plot_experiment_summary(f, out_png)

      pareto_out = os.path.join(out_dir, "pareto_frontier.png")
      PlotGenerator.plot_pareto_from_logs(in_dir, pareto_out)
