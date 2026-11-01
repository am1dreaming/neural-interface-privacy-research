"""Draw the privacy/utility tradeoff from results/defence_tradeoff.json.

    python experiments/plot_defence_tradeoff.py            # needs matplotlib

One panel per defence family and attacker. In each panel: the undefended encoder (gray) and the
settings of that family (blue, joined in order of strength), with 95% intervals over seeds.
Left = the attacker names the person less often, up = the decoder still works. The best place to
be is the top left. Rows are the attacker: the nonlinear MLP probe (the strong one) and the
linear probe. One accent colour on purpose: the families are panels, not colours.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

# palette.md, light mode
SURFACE, INK, INK2, MUTED = "#fcfcfb", "#0b0b0b", "#52514e", "#898781"
GRID, AXIS, ACCENT = "#e1e0d9", "#c3c2b7", "#2a78d6"

# (name prefix, panel title, does a LARGER setting mean a STRONGER defence?)
FAMILIES = [
    ("noise", "Noise added to the embedding", True),
    ("pca", "Keep only k main directions", False),   # fewer directions kept = stronger
    ("id-null", "Remove m identity directions", True),
    ("adversarial", "Train against a person-guesser", True),
]
PROBE_ROWS = [("mlp", "MLP probe (strong attacker)"), ("linear", "Linear probe")]


def setting_value(name: str) -> float:
    return float(re.search(r"=([0-9.]+)", name).group(1))


def plot_tradeoff(summary: dict, path: str | Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    d = summary["defences"]
    chance_id, chance_top1 = summary["chance_identity"], summary["chance_top1"]
    probes = [(p, t) for p, t in PROBE_ROWS if p in d["none"]]

    import logging

    from matplotlib import font_manager

    logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)
    have = {f.name for f in font_manager.fontManager.ttflist}
    families = [f for f in ("Inter", "Helvetica Neue", "Arial") if f in have][:1] or ["DejaVu Sans"]
    plt.rcParams.update({"font.family": families, "text.color": INK})
    fig, axes = plt.subplots(len(probes), len(FAMILIES), figsize=(14, 3.6 * len(probes) + 1.0),
                             sharex=False, sharey=True, squeeze=False)
    fig.patch.set_facecolor(SURFACE)

    for r, (probe, probe_title) in enumerate(probes):
        xs_all = [d[n][probe]["ci_high"] for n in d] + [d[n][probe]["ci_low"] for n in d] + [chance_id]
        lo, hi = max(0.0, min(xs_all) - 0.05), min(1.0, max(xs_all) + 0.05)
        for c, (key, title, larger_is_stronger) in enumerate(FAMILIES):
            ax = axes[r][c]
            ax.set_facecolor(SURFACE)
            names = sorted((n for n in d if n.startswith(key)), key=setting_value)
            base = d["none"]
            ax.axvline(chance_id, color=AXIS, lw=1, zorder=1)
            ax.text(chance_id, 1.0, " chance", transform=ax.get_xaxis_transform(), color=MUTED, fontsize=8,
                    va="top", ha="left")
            ax.axhline(chance_top1, color=AXIS, lw=1, zorder=1)
            if names:
                pts = [(d[n][probe]["mean"], d[n]["top1"]["mean"]) for n in names]
                ax.plot([p[0] for p in pts], [p[1] for p in pts], color=ACCENT, lw=2, solid_capstyle="round",
                        solid_joinstyle="round", zorder=3)
                for n, (x, y) in zip(names, pts):
                    ax.errorbar(x, y, xerr=[[x - d[n][probe]["ci_low"]], [d[n][probe]["ci_high"] - x]],
                                yerr=[[y - d[n]["top1"]["ci_low"]], [d[n]["top1"]["ci_high"] - y]],
                                fmt="none", ecolor=ACCENT, elinewidth=1, alpha=0.45, zorder=2)
                    ax.plot(x, y, "o", ms=9, color=ACCENT, mec=SURFACE, mew=2, zorder=4)
                k_strong = -1 if larger_is_stronger else 0  # only the strongest setting is labelled: the rest is in the table
                n, (x, y) = names[k_strong], pts[k_strong]
                ax.annotate("strongest: " + n.split(" ", 1)[1], (x, y), xytext=(0, -15), textcoords="offset points",
                            color=INK2, fontsize=8, ha="center")
            bx, by = base[probe]["mean"], base["top1"]["mean"]
            ax.errorbar(bx, by, xerr=[[bx - base[probe]["ci_low"]], [base[probe]["ci_high"] - bx]],
                        yerr=[[by - base["top1"]["ci_low"]], [base["top1"]["ci_high"] - by]],
                        fmt="none", ecolor=MUTED, elinewidth=1, alpha=0.6, zorder=2)
            ax.plot(bx, by, "o", ms=9, color=MUTED, mec=SURFACE, mew=2, zorder=5)
            if c == 0 and r == 0:
                ax.annotate("no defence", (bx, by), xytext=(8, -14), textcoords="offset points", color=INK2, fontsize=8)
            ax.set_xlim(lo, hi)
            ax.set_ylim(0, max(0.55, base["top1"]["ci_high"] + 0.08))
            ax.grid(True, color=GRID, lw=1, ls="-", zorder=0)
            ax.set_axisbelow(True)
            for side in ("top", "right"):
                ax.spines[side].set_visible(False)
            for side in ("left", "bottom"):
                ax.spines[side].set_color(AXIS)
            ax.tick_params(colors=MUTED, labelsize=8, length=0)
            if r == 0:
                ax.set_title(title, fontsize=10, color=INK, loc="left", pad=8)
            if c == 0:
                ax.set_ylabel("decoding top-1\n(new users, unseen sentences)", fontsize=8.5, color=INK2)
            ax.set_xlabel(f"attacker accuracy, {probe_title.split(' (')[0].lower()}", fontsize=8.5, color=INK2)
            if c == 0:
                ax.text(0, 1.2, probe_title, transform=ax.transAxes, fontsize=9.5, color=INK2, weight="bold")

    from matplotlib.lines import Line2D

    handles = [Line2D([], [], marker="o", ls="", ms=9, color=MUTED, mec=SURFACE, mew=2, label="no defence"),
               Line2D([0], [0], marker="o", ls="-", lw=2, ms=9, color=ACCENT, mec=SURFACE, mew=2,
                      label="defended; settings joined in order of strength")]
    fig.legend(handles=handles, loc="upper right", bbox_to_anchor=(0.995, 0.985), frameon=False, fontsize=9,
               labelcolor=INK2)
    fig.suptitle("Price of privacy: what each defence takes from the decoder and from the attacker",
                 x=0.012, y=0.995, ha="left", fontsize=13, color=INK, weight="bold")
    fig.text(0.012, 0.947 if len(probes) > 1 else 0.915,
             f"Synthetic data, {summary['n_seeds']} seeds, 95% intervals. Best is the top left: more private and still accurate. "
             "Thin lines: chance.",
             fontsize=9, color=INK2, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.925 if len(probes) > 1 else 0.88), h_pad=2.2, w_pad=1.6)
    fig.savefig(path, dpi=160, facecolor=SURFACE)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", default="results/defence_tradeoff.json")
    parser.add_argument("--out", default="results/defence_tradeoff.png")
    args = parser.parse_args()
    data = json.loads(Path(args.results).read_text())
    plot_tradeoff(data["summary"], args.out)
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
