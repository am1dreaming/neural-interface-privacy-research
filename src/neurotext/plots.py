"""All figures of the audit, from the JSON files the experiments write. One command:

    python experiments/make_plots.py            # or: python -m neurotext.plots

Rules for every figure: 95% intervals over seeds; a line for chance wherever a number has one (1/n for
accuracy, 0.5 for EER / AUC); bars start at zero; one style; labels in English and Russian.
A figure whose input file is missing is skipped with a message, so the command works at every stage of the project.

  fig1_leakage_vs_chance    identity_leakage.json    accuracy of the attacker against chance
  fig2_scaling_curve        scaling_curve.json       accuracy and EER against the number of people
  fig3_defence_tradeoff     defence_tradeoff.json    leakage against kept utility (and EER if --verify was used)
  fig4_feature_ablation     feature_ablation.json    what each feature group gives alone / what is lost without it
  fig5_encoders             encoder_robustness.json  the same leakage for four architectures (+ encoders_table.md)
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

PALETTE = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B3", "#937860", "#DA8BC3", "#8C8C8C"]
CHANCE = "#222222"

REP_RU = {"trained encoder": "обученный энкодер", "untrained encoder": "необученный энкодер",
          "band power (all features)": "спектр (все признаки)",
          "band power, PCA to embedding size": "спектр, PCA до размера эмбеддинга"}
GROUP_RU = {"rms": "амплитуда", "delta": "дельта", "theta": "тета", "alpha": "альфа", "beta": "бета",
            "gamma": "гамма", "alpha_peak": "альфа-пик"}
PROBE_RU = {"linear": "линейный", "mlp": "MLP"}


def bi(en: str, ru: str) -> str:
    return f"{en} / {ru}"


def _style() -> None:
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                         "axes.grid": True, "grid.alpha": 0.25, "axes.axisbelow": True,
                         "figure.dpi": 130, "savefig.dpi": 160, "legend.frameon": False})


def _load(path: Path):
    if not path.exists():
        print(f"skip: {path} not found")
        return None
    return json.loads(path.read_text())


def _err(ci: dict | None):
    if not ci or any(k not in ci for k in ("mean", "ci_low", "ci_high")) or np.isnan(ci["ci_low"]):
        return None
    return ci["mean"] - ci["ci_low"], ci["ci_high"] - ci["mean"]


def _mean(ci):
    return float(ci["mean"]) if ci else np.nan


def _bars(ax, labels, cis, color=PALETTE[0], width=0.8, offset=0.0, label=None, rotation=25):
    x = np.arange(len(labels)) + offset
    ax.bar(x, [_mean(c) for c in cis], width, color=color, label=label)
    for xi, c in zip(x, cis):
        e = _err(c)
        if e:
            ax.errorbar(xi, c["mean"], yerr=[[e[0]], [e[1]]], color="black", capsize=3, lw=1)
    ax.set_xticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=rotation, ha="right", fontsize=8)
    ax.set_ylim(bottom=0)


def _hline(ax, y, text):
    ax.axhline(y, color=CHANCE, ls="--", lw=1)
    ax.text(ax.get_xlim()[1], y, f" {text}", va="bottom", ha="right", fontsize=8, color=CHANCE)


def _save(fig, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {path}")
    return path


# ---------------------------------------------------------------- figure 1
def fig1_leakage(results: Path, out: Path):
    d = _load(results / "identity_leakage.json")
    if not d:
        return None
    pops = d["summary"]["populations"]
    fig, axes = plt.subplots(1, len(pops), figsize=(6.2 * len(pops), 4.4), squeeze=False)
    titles = {"new_users": bi("new people (never seen by the encoder)", "новые люди"),
              "everyone": bi("all people", "все люди")}
    for ax, (pop, body) in zip(axes[0], pops.items()):
        reps = list(body["representations"])
        probes = [p for p in ("linear", "mlp") if p in next(iter(body["representations"].values()))]
        w = 0.8 / len(probes)
        for k, p in enumerate(probes):
            _bars(ax, [f"{r}\n{REP_RU.get(r, '')}" for r in reps],
                  [body["representations"][r][p]["accuracy"] for r in reps], PALETTE[k], w, (k - (len(probes) - 1) / 2) * w,
                  f"{p} / {PROBE_RU[p]}")
        ax.set_ylim(0, 1.05)
        _hline(ax, body["chance"], bi("chance", "шанс") + f" {body['chance']:.2f}")
        ax.set_title(titles.get(pop, pop))
        ax.set_ylabel(bi("attacker accuracy", "точность атакующего"))
        ax.legend(loc="lower left")
    fig.suptitle(bi("Identity leaks from the representations (95% CI over seeds)",
                    "Личность утекает из представлений (95% ДИ по сидам)"))
    return _save(fig, out / "fig1_leakage_vs_chance.png")


# ---------------------------------------------------------------- figure 2
def fig2_scaling(results: Path, out: Path):
    d = _load(results / "scaling_curve.json")
    if not d:
        return None
    s = d["summary"]
    ns = s["n_list"]
    fin = _load(results / "final_scaling.json")
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11.5, 5.6))
    for k, rep in enumerate(s["reps"]):
        col = PALETTE[k % len(PALETTE)]
        for probe, ls in (("linear", "-"), ("mlp", "--")):
            if probe not in s["reps"][rep]:
                continue
            cis = [s["reps"][rep][probe][str(n)]["accuracy"] for n in ns]
            m = np.array([_mean(c) for c in cis])
            lo = np.array([c["ci_low"] if c else np.nan for c in cis])
            hi = np.array([c["ci_high"] if c else np.nan for c in cis])
            a1.plot(ns, m, ls, color=col, marker="o", ms=3, label=f"{rep} ({probe})" if probe == "linear" else None)
            a1.fill_between(ns, lo, hi, color=col, alpha=0.12)
        for scorer, ls in (("lda", "-"), ("cosine", ":")):
            cis = [s["verify"][rep][scorer][str(n)]["eer"] for n in ns] if s.get("verify", {}).get(rep) else []
            if cis and any(cis):
                m = np.array([_mean(c) for c in cis])
                a2.plot(ns, m, ls, color=col, marker="o", ms=3, label=f"{rep} ({scorer})")
                a2.fill_between(ns, [c["ci_low"] if c else np.nan for c in cis], [c["ci_high"] if c else np.nan for c in cis],
                                color=col, alpha=0.12)
    if fin:
        fs = fin["summary"]
        fns = fs["n_list"]
        rep = "band power (all features)"
        a1.plot(fns, [_mean(fs["reps"][rep]["linear"][str(n)]["accuracy"]) for n in fns], "D", mfc="none", color="black",
                label=bi("final run, features, all people", "финальный прогон, признаки"))
        if fs.get("verify", {}).get(rep):
            a2.plot(fns, [_mean(fs["verify"][rep]["cosine"][str(n)]["eer"]) for n in fns], "D", mfc="none", color="black")
    grid = np.array(sorted(set(ns + (fin["summary"]["n_list"] if fin else []))))
    a1.plot(grid, 1 / grid, color=CHANCE, ls=":", lw=1.4, label=bi("chance 1/n", "шанс 1/n"))
    a1.set_xscale("log")
    a1.set_ylim(0, 1.05)
    a1.set_xlabel(bi("number of people", "число людей"))
    a1.set_ylabel(bi("closed-set accuracy", "точность идентификации"))
    a1.set_title(bi("Identification", "Идентификация"))
    a1.legend(fontsize=7, loc="upper center", bbox_to_anchor=(0.5, -0.17), ncol=2)
    a2.axhline(0.5, color=CHANCE, ls=":", lw=1.4, label=bi("chance 0.5", "шанс 0.5"))
    a2.set_xscale("log")
    a2.set_ylim(0, 0.6)
    a2.set_xlabel(bi("number of people", "число людей"))
    a2.set_ylabel(bi("EER (lower = more leakage)", "EER (ниже = больше утечка)"))
    a2.set_title(bi("Open-set verification", "Открытая верификация"))
    a2.legend(fontsize=7, loc="upper center", bbox_to_anchor=(0.5, -0.17), ncol=2)
    for a in (a1, a2):
        a.set_xticks(grid)
        a.set_xticklabels([str(n) for n in grid])
    fig.suptitle(bi("Leakage against the number of people (95% CI over seeds)",
                    "Утечка и число людей (95% ДИ по сидам)"))
    return _save(fig, out / "fig2_scaling_curve.png")


# ---------------------------------------------------------------- figure 3
def fig3_tradeoff(results: Path, out: Path):
    d = _load(results / "defence_tradeoff.json")
    if not d:
        return None
    s = d["summary"]
    probes = [p for p in ("linear", "mlp") if p in next(iter(s["defences"].values()))]
    has_eer = any(v.get("eer_lda") for v in s["defences"].values())
    n = len(probes) + int(has_eer)
    fig, axes = plt.subplots(1, n, figsize=(5.3 * n, 4.6), squeeze=False)
    for ax, key in zip(axes[0], probes + (["eer_lda"] if has_eer else [])):
        for k, (name, row) in enumerate(s["defences"].items()):
            x, y = row["top1"], row.get(key)
            if not x or not y:
                continue
            ex, ey = _err(x), _err(y)
            ax.errorbar(x["mean"], y["mean"], xerr=[[ex[0]], [ex[1]]] if ex else None, yerr=[[ey[0]], [ey[1]]] if ey else None,
                        fmt="o", color=PALETTE[k % len(PALETTE)], capsize=2, lw=1)
            ax.annotate(name, (x["mean"], y["mean"]), fontsize=7, xytext=(4, 4), textcoords="offset points")
        ax.axvline(s["chance_top1"], color=CHANCE, ls=":", lw=1)
        if key == "eer_lda":
            ax.axhline(0.5, color=CHANCE, ls="--", lw=1)
            ax.set_ylabel(bi("EER of LDA verification (0.5 = chance)", "EER верификации (0.5 = шанс)"))
            ax.set_title(bi("open-set attacker", "открытая верификация"))
            ax.set_ylim(0, 0.6)
        else:
            ax.axhline(s["chance_identity"], color=CHANCE, ls="--", lw=1)
            ax.set_ylabel(bi("attacker accuracy", "точность атакующего") + f" ({key})")
            ax.set_title(f"{key} / {PROBE_RU[key]}")
            ax.set_ylim(0, 1.0)
        ax.set_xlim(0, None)
        ax.set_xlabel(bi("decoder top-1 (utility)", "точность декодера top-1 (полезность)"))
    fig.suptitle(bi("Price of privacy: best defence is low-right of the cloud in leakage, right in utility",
                    "Цена защиты: чем левее-ниже по утечке и правее по полезности, тем лучше"), fontsize=10)
    return _save(fig, out / "fig3_defence_tradeoff.png")


# ---------------------------------------------------------------- figure 4
def fig4_ablation(results: Path, out: Path):
    d = _load(results / "feature_ablation.json")
    if not d:
        return None
    s = d["summary"]
    st = s["settings"]
    groups = [k[len("only "):] for k in st if k.startswith("only ")]
    probes = [p for p in ("linear", "mlp") if p in st["all features"]]
    full = {p: _mean(st["all features"][p]["accuracy"]) for p in probes}
    fig, axes = plt.subplots(1, 3, figsize=(17, 5.4), sharey=True)
    panels = [(bi("one group alone", "только одна группа"), [f"only {g}" for g in groups], groups),
              (bi("everything except one group", "всё, кроме одной группы"), [f"all minus {g}" for g in groups], groups),
              (bi("recording-artefact controls", "контроли артефактов"),
               ["all features"] + [k for k in st if k.startswith("control:")],
               ["all", "amplitude normalised", "edges trimmed", "normalised + trimmed"])]
    for ax, (title, keys, names) in zip(axes, panels):
        w = 0.8 / len(probes)
        labels = [f"{n} / {GROUP_RU[n]}" if n in GROUP_RU else n for n in names]
        for k, p in enumerate(probes):
            _bars(ax, labels, [st[key][p]["accuracy"] for key in keys], PALETTE[k], w, (k - (len(probes) - 1) / 2) * w,
                  f"{p} / {PROBE_RU[p]}", rotation=40)
        for k, p in enumerate(probes):
            ax.axhline(full[p], color=PALETTE[k], ls=":", lw=1)
        ax.set_ylim(0, 1.2)
        ax.set_yticks(np.arange(0, 1.01, 0.2))
        _hline(ax, s["chance"], bi("chance", "шанс") + f" {s['chance']:.2f}")
        ax.set_title(title)
    axes[0].set_ylabel(bi("attacker accuracy", "точность атакующего"))
    axes[0].legend(loc="upper left", fontsize=8)
    fig.suptitle(bi("Where the leakage comes from (exploratory; dotted = all features; 95% CI over seeds)",
                    "Откуда утечка (разведочно; пунктир = все признаки; 95% ДИ по сидам)"))
    return _save(fig, out / "fig4_feature_ablation.png")


# ---------------------------------------------------------------- figure 5
def fig5_encoders(results: Path, out: Path):
    d = _load(results / "encoder_robustness.json")
    if not d:
        return None
    enc = d["summary"]["encoders"]
    probes = [p for p in ("linear", "mlp") if p in d["summary"]["probes"]]
    dims = sorted({v["out_dim"] for v in enc.values()})
    ref_dim = 32 if 32 in dims else dims[-1]
    names = sorted({v["encoder"] for v in enc.values()})
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4.6))
    w = 0.8 / len(probes)
    labels = [n for n in names if f"{n}-{ref_dim}" in enc]
    for k, p in enumerate(probes):
        _bars(a1, labels, [enc[f"{n}-{ref_dim}"]["trained"][p]["accuracy"] for n in labels], PALETTE[k], w,
              (k - (len(probes) - 1) / 2) * w, f"{p} / {PROBE_RU[p]}")
    chance = enc[f"{labels[0]}-{ref_dim}"]["trained"][probes[0]]["chance"]
    a1.set_ylim(0, 1.05)
    _hline(a1, chance, bi("chance", "шанс") + f" {chance:.2f}")
    a1.set_title(bi(f"four encoders, {ref_dim}-d, new people", f"четыре энкодера, {ref_dim}-мерные, новые люди"))
    a1.set_ylabel(bi("attacker accuracy", "точность атакующего"))
    a1.legend(loc="lower left")
    if len(dims) > 1:
        for k, n in enumerate(names):
            ds = [x for x in dims if f"{n}-{x}" in enc]
            cis = [enc[f"{n}-{x}"]["trained"][probes[-1]]["accuracy"] for x in ds]
            a2.plot(ds, [_mean(c) for c in cis], "-o", color=PALETTE[k], label=n)
            a2.fill_between(ds, [c["ci_low"] if c else np.nan for c in cis], [c["ci_high"] if c else np.nan for c in cis],
                            color=PALETTE[k], alpha=0.12)
        a2.axhline(chance, color=CHANCE, ls="--", lw=1)
        a2.set_xscale("log", base=2)
        a2.set_xticks(dims)
        a2.set_xticklabels([str(x) for x in dims])
        a2.set_ylim(0, 1.05)
        a2.set_xlabel(bi("embedding size", "размер эмбеддинга"))
        a2.set_ylabel(bi(f"attacker accuracy ({probes[-1]})", f"точность атакующего ({probes[-1]})"))
        a2.set_title(bi("leakage against embedding size", "утечка и размер эмбеддинга"))
        a2.legend(fontsize=8)
    else:
        a2.axis("off")
        a2.text(0.5, 0.5, bi("run with --out-dims 4 8 16 32 64\nfor the embedding-size panel",
                             "запустите с --out-dims 4 8 16 32 64\nдля панели про размер эмбеддинга"), ha="center", va="center")
    fig.suptitle(bi("Is the leakage specific to one architecture? (95% CI over seeds)",
                    "Утечка зависит от архитектуры? (95% ДИ по сидам)"))
    _table(enc, probes, out / "encoders_table.md")
    return _save(fig, out / "fig5_encoders.png")


def _table(enc: dict, probes, path: Path) -> None:
    def cell(ci, signed=False):
        if not ci:
            return "n/a"
        f = "+.3f" if signed else ".3f"
        return f"{format(ci['mean'], f)} [{format(ci['ci_low'], f)}, {format(ci['ci_high'], f)}]"

    head = "| encoder | size | " + " | ".join(f"{p} acc" for p in probes) + " | LDA EER | trained - untrained (" + probes[0] + ") |"
    lines = [head, "|" + "---|" * (4 + len(probes))]
    for label, v in enc.items():
        tmu = v["trained_minus_untrained"].get(probes[0])
        lines.append(f"| {v['encoder']} | {v['out_dim']} | " + " | ".join(cell(v["trained"][p]["accuracy"]) for p in probes)
                     + f" | {cell(v['verify_trained']['lda']['eer'])} | {cell(tmu, True) if tmu else 'n/a'} |")
    path.write_text("\n".join(lines) + "\n")
    print(f"saved {path}")


def make_all(results: str | Path = "results") -> list[Path]:
    _style()
    results = Path(results)
    made = []
    for fn in (fig1_leakage, fig2_scaling, fig3_tradeoff, fig4_ablation, fig5_encoders):
        p = fn(results, results)
        if p:
            made.append(p)
    return made


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    make_all(ap.parse_args().results)
