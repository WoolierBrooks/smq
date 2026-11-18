"""Draw the README hero figure from real draws of the synthetic generator.

    python docs/make_assets.py
"""

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from smq import simulate_scores

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets")
COLORS = ["#B03A48", "#3C6E9F", "#D4A017"]
INK = "#1F2937"

# vertices of the 2-simplex in the plane
V = np.array([[0.0, 0.0], [1.0, 0.0], [0.5, np.sqrt(3) / 2]])


def to_plane(p):
    return p @ V


def main():
    os.makedirs(OUT, exist_ok=True)
    prevalence = [0.5, 0.3, 0.2]
    merging = [0.0, 0.35, 0.7, 1.0]

    fig, axes = plt.subplots(1, len(merging), figsize=(12, 3.5))
    for ax, m in zip(axes, merging):
        scores, labels = simulate_scores(3000, prevalence, m, random_state=7)
        xy = to_plane(scores)
        for k in range(3):
            pts = xy[labels == k]
            ax.scatter(pts[:, 0], pts[:, 1], s=2.2, alpha=0.45, color=COLORS[k], linewidths=0)
        tri = np.vstack([V, V[:1]])
        ax.plot(tri[:, 0], tri[:, 1], color=INK, lw=1)
        ax.set_title(f"m = {m:g}", color=INK, fontsize=12)
        ax.set_aspect("equal")
        ax.axis("off")
    fig.suptitle(
        "Synthetic classifier scores on the simplex (3 classes, prevalence 0.5 / 0.3 / 0.2)",
        color=INK, fontsize=12, y=1.02,
    )
    fig.text(0.5, -0.04, "merging factor m: from separable classes (left) to indistinguishable ones (right)",
             ha="center", color="#6B7280", fontsize=10)
    for ext in ("png", "svg"):
        fig.savefig(os.path.join(OUT, f"synthetic_scores.{ext}"), dpi=200, bbox_inches="tight",
                    transparent=False, facecolor="white")
    print("saved", OUT)


if __name__ == "__main__":
    main()
