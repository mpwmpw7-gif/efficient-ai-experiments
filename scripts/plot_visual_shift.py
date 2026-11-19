"""Build the visual comparison figure from the recorded measurements."""
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
results = json.loads((ROOT / "results" / "visual_evidence.json").read_text(encoding="utf-8"))
families = [results["families"][key] for key in ("sensor_shift", "banded_confirmation")]
methods = [("A0", "Original", "#41586e"), ("A1", "Aligned", "#a8523a"), ("A2", "Unaligned", "#bd9461")]
xticks = np.arange(len(families))
width = 0.23

fig, (ax_evidence, ax_loss) = plt.subplots(1, 2, figsize=(9.0, 3.5), layout="constrained")
for offset, (method, label, colour) in enumerate(methods, start=-1):
    fitted = [family["methods"][method]["calibrated"] for family in families]
    evidence = [100 * f["events"]["neutral_false_3to1"] / f["events"]["neutral_events"] for f in fitted]
    loss = [f["outcome"]["bits"] for f in fitted]
    ax_evidence.bar(xticks + width * offset, evidence, width, label=label, color=colour)
    ax_loss.bar(xticks + width * offset, loss, width, color=colour)

for axis, title, label in [(ax_evidence, "False decisive evidence", "Neutral events with ≥3:1 odds (%)"),
                           (ax_loss, "Belief prediction loss", "Bits per outcome (lower is better)")]:
    axis.set_title(title, loc="left", fontweight="semibold")
    axis.set_ylabel(label)
    axis.set_xticks(xticks, ["Prior sensor shift", "New obstacle + sensor"])
    axis.spines[["top", "right"]].set_visible(False)
    axis.grid(axis="y", color="#e5e5e5", linewidth=0.7)
    axis.set_axisbelow(True)
ax_evidence.set_ylim(0, 100)
ax_evidence.legend(frameon=False, fontsize=8, loc="upper center")
(ROOT / "assets").mkdir(exist_ok=True)
fig.savefig(ROOT / "assets" / "visual-shift.png", dpi=150, facecolor="white")
