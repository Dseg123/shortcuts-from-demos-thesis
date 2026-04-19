"""Plot training-step sweep curves from the wide-format online_stats.csv.

Plots multiple metrics for a single (env, residual) combo on the same axes.
Each metric becomes a line whose x-axis is the training step.

Edit the CONFIG block to choose metrics and the combo, then run:
    python experiments/plot_sweeps.py
"""

import re
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


# ---- CONFIG ----
HERE       = Path(__file__).resolve().parent
csv_path   = HERE / "online_stats.csv"
save_path  = HERE / "sweep_plot.png"

# Pick metrics to plot (one line per metric). Available per-step suffixes:
#   n2n_success_rate, n2n_env_success_rate
#   s2s_success_rate, s2s_graph_success_rate, s2s_env_success_rate
#   n2n_avg_steps_success, n2n_se_steps_success
#   n2n_avg_steps_env_success, n2n_se_steps_env_success
#   s2s_avg_steps_success, s2s_se_steps_success
#   s2s_avg_steps_graph_success, s2s_se_steps_graph_success
#   s2s_avg_steps_env_success, s2s_se_steps_env_success
#   edges_added, train_time_s
#
# Each entry is (metric_name, optional_se_col_name).
metrics = [
    ("n2n_avg_steps_success",       "n2n_se_steps_success"),
    ("s2s_avg_steps_graph_success", "s2s_se_steps_graph_success"),
]

# The single (env, residual) combo to plot for.
combo = ("transport", True)


# ---- LOAD ----
df = pd.read_csv(csv_path)
df["residual"] = df["residual"].astype(str).str.lower() == "true"

env, residual = combo
rows = df[(df["env"] == env) & (df["residual"] == residual)]
if rows.empty:
    raise SystemExit(f"No row for ({env}, residual={residual}).")
row = rows.iloc[0]


def resolve_step_cols(metric_name: str) -> tuple[list[int], list[str]]:
    """Return sorted (steps, col_names) for every step{N}_{metric_name} column."""
    pattern = re.compile(rf"^step(\d+)_{re.escape(metric_name)}$")
    found = []
    for col in df.columns:
        m = pattern.match(col)
        if m:
            found.append((int(m.group(1)), col))
    found.sort()
    return [s for s, _ in found], [c for _, c in found]


# ---- PLOT ----
plt.figure(figsize=(8, 5))

for metric, metric_se in metrics:
    steps, col_names = resolve_step_cols(metric)
    if not col_names:
        print(f"Warning: no columns for metric '{metric}' — skipping.")
        continue

    values = (pd.to_numeric(row[col_names], errors="coerce"))  # cap y-axis for readability
    # every value should be at most 1000
    values = values.clip(upper=1000)

    errs = None
    if metric_se:
        _, se_cols = resolve_step_cols(metric_se)
        if len(se_cols) == len(col_names):
            errs = pd.to_numeric(row[se_cols], errors="coerce")
        else:
            print(f"Warning: SE columns for '{metric_se}' don't align — no error bars.")

    if errs is not None:
        plt.errorbar(steps, values, yerr=errs,
                     marker="o", linewidth=2, capsize=4, label=("multi-task" if metric.startswith("n2n") else "single-task"))
    else:
        plt.plot(steps, values, marker="o", linewidth=2, label=metric)

combo_label = f"{env}"
plt.title(f"{combo_label}: metrics vs training steps", fontsize=14)
plt.xlabel("Training step", fontsize=12)
plt.ylabel("Value", fontsize=12)

plt.grid(True, linestyle="--", alpha=0.5)
plt.legend()
plt.tight_layout()
plt.savefig(save_path, dpi=150)
print(f"Saved {save_path}")
plt.show()
