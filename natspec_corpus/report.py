"""Tables and figures, generated rather than typed.

Built before any real number exists, on purpose. A table whose shape is fixed
in advance cannot be reshaped afterwards to flatter a result — which column to
show, which split to report, which ablation to include are all decisions that
should be made without knowing how they come out.

Two conventions the write-up depends on:

**The primary metrics lead.** Claim support rate and defect-free rate come
first in every table; BLEU, ROUGE-L and coverage follow as secondary. n-gram
overlap with one human reference is a weak proxy for whether a comment is
true, and the ordering says so without an argument.

**Nothing is hidden by averaging.** Every table reports the three evidence
conditions separately — Σ(f) with a body, interface declaration without one,
and no Σ(f) at all — because 51% of the validation split falls in the middle
group and one pooled mean would obscure the ablation that matters most.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence

PRIMARY = [
    ("claim_support_rate", "Claim support"),
    ("defect_free_rate", "Defect-free"),
    ("verifier_pass_rate", "Verifier pass"),
]

SECONDARY = [
    ("notice_bleu", "notice BLEU"),
    ("notice_rouge_l", "notice ROUGE-L"),
    ("param_bleu", "param BLEU"),
    ("param_coverage", "param coverage"),
    ("return_bleu", "return BLEU"),
]

CONDITIONS = [("with_sigma", "Σ(f), with body"),
              ("without_sigma", "no Σ(f)"),
              ("all", "all functions")]


def flatten(block: dict) -> Dict[str, Optional[float]]:
    """One `evaluate.aggregate` block as a flat metric row."""
    out: Dict[str, Optional[float]] = {"n": block.get("n", 0)}
    for key, _ in PRIMARY:
        out[key] = block.get(key)
    for kind in ("notice", "dev", "param", "return"):
        sub = block.get(kind) or {}
        for metric in ("bleu", "rouge_l", "coverage"):
            out[f"{kind}_{metric}"] = sub.get(metric)
    return out


def _fmt(v, places: int = 3) -> str:
    if v is None:
        return "—"
    if isinstance(v, bool):
        return "yes" if v else "no"
    if isinstance(v, str):
        return v                # already formatted (an interval, a label)
    if isinstance(v, int):
        return str(v)
    try:
        return f"{v:.{places}f}"
    except (TypeError, ValueError):
        return str(v)


def markdown_table(headers: Sequence[str], rows: Sequence[Sequence]) -> str:
    head = "| " + " | ".join(headers) + " |"
    rule = "| " + " | ".join("---" for _ in headers) + " |"
    body = ["| " + " | ".join(_fmt(c) for c in r) + " |" for r in rows]
    return "\n".join([head, rule, *body])


def latex_table(headers: Sequence[str], rows: Sequence[Sequence], *,
                caption: str = "", label: str = "") -> str:
    spec = "l" + "r" * (len(headers) - 1)
    lines = [r"\begin{table}[t]", r"\centering",
             rf"\begin{{tabular}}{{{spec}}}", r"\toprule",
             " & ".join(_esc(h) for h in headers) + r" \\", r"\midrule"]
    lines += [" & ".join(_esc(_fmt(c)) for c in r) + r" \\" for r in rows]
    lines += [r"\bottomrule", r"\end{tabular}"]
    if caption:
        lines.append(rf"\caption{{{_esc(caption)}}}")
    if label:
        lines.append(rf"\label{{{label}}}")
    lines.append(r"\end{table}")
    return "\n".join(lines)


def _esc(s: str) -> str:
    out = str(s)
    for a, b in (("\\", r"\textbackslash{}"), ("&", r"\&"), ("%", r"\%"),
                 ("_", r"\_"), ("#", r"\#"), ("$", r"\$")):
        out = out.replace(a, b)
    return out.replace("Σ", r"$\Sigma$").replace("—", "--")


# --------------------------------------------------------------------------
# the tables
# --------------------------------------------------------------------------

def main_table(results: Dict[str, dict], labels: Dict[str, str], *,
               condition: str = "all") -> tuple:
    """One row per configuration: primary metrics, then secondary."""
    headers = ["Configuration", "n"] + [h for _, h in PRIMARY] \
        + [h for _, h in SECONDARY]
    rows = []
    for name in sorted(results):
        flat = flatten((results[name] or {}).get(condition) or {})
        rows.append([labels.get(name, name), flat.get("n", 0)]
                    + [flat.get(k) for k, _ in PRIMARY]
                    + [flat.get(k) for k, _ in SECONDARY])
    return headers, rows


def condition_table(result: dict) -> tuple:
    """One configuration, broken out by evidence condition."""
    headers = ["Condition", "n"] + [h for _, h in PRIMARY] + ["notice BLEU"]
    rows = []
    for key, label in CONDITIONS:
        flat = flatten((result or {}).get(key) or {})
        rows.append([label, flat.get("n", 0)]
                    + [flat.get(k) for k, _ in PRIMARY]
                    + [flat.get("notice_bleu")])
    return headers, rows


def ablation_table_rows(table: dict, labels: Dict[str, str]) -> tuple:
    """`stats.ablation_table` as a table, with the honest columns.

    `within noise` is beside `p (Holm)` deliberately: a corrected p-value below
    0.05 on a difference smaller than the seed-to-seed spread is not a result,
    and putting the two columns side by side makes that impossible to miss.
    """
    headers = ["Ablation", "n", "Δ vs full", "95% CI", "p (Holm)",
               "Cliff's δ", "magnitude", "within noise"]
    rows = []
    for name, c in sorted(table.get("comparisons", {}).items()):
        rows.append([
            labels.get(name, name), c["n"], _fmt(c["diff"]),
            f"[{_fmt(c['ci_low'])}, {_fmt(c['ci_high'])}]",
            _fmt(c["p_adjusted"], 4), _fmt(c["cliffs_delta"]),
            c["magnitude"], "yes" if c["within_noise"] else "no",
        ])
    return headers, rows


def seed_table(table: dict) -> tuple:
    headers = ["Configuration", "seeds", "mean", "sd"]
    rows = [[name, s["seeds"], s["mean"], s["sd"]]
            for name, s in sorted(table.get("seed_spread", {}).items())]
    return headers, rows


# --------------------------------------------------------------------------
# the figure
# --------------------------------------------------------------------------

def ablation_figure(table: dict, labels: Dict[str, str], path: Path) -> Path:
    """Each ablation's difference from the full system, with its interval.

    A point with a whisker, not a bar: the quantity is a difference with an
    uncertainty, and a bar chart of differences invites reading the bar's
    length as the result while hiding the interval that decides whether it is
    one.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    comps = table.get("comparisons", {})
    names = sorted(comps, key=lambda n: comps[n]["diff"])
    if not names:
        raise ValueError("no comparisons to plot")
    y = list(range(len(names)))
    diffs = [comps[n]["diff"] for n in names]
    lo = [comps[n]["diff"] - comps[n]["ci_low"] for n in names]
    hi = [comps[n]["ci_high"] - comps[n]["diff"] for n in names]

    fig, ax = plt.subplots(figsize=(7.2, 0.5 * len(names) + 1.6))
    ax.axvline(0.0, color="#888888", linewidth=1, zorder=1)
    ax.errorbar(diffs, y, xerr=[lo, hi], fmt="o", capsize=3, linewidth=1.4,
                markersize=5, color="#1f4e79", ecolor="#7f9db9", zorder=2)
    ax.set_yticks(y)
    ax.set_yticklabels([labels.get(n, n) for n in names])
    ax.set_xlabel("difference from the full system (negative = worse)")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="x", color="#eeeeee", zorder=0)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


# --------------------------------------------------------------------------
# writing it all out
# --------------------------------------------------------------------------

def write_report(out_dir: Path, *, results: Dict[str, dict],
                 labels: Dict[str, str], ablations: Optional[dict] = None,
                 manifest: Optional[dict] = None,
                 figure: bool = True) -> Dict[str, Path]:
    """Every table in Markdown and LaTeX, the figure, and the manifest."""
    out_dir = Path(out_dir)
    (out_dir / "tables").mkdir(parents=True, exist_ok=True)
    written: Dict[str, Path] = {}

    def emit(name: str, headers, rows, caption: str) -> None:
        md = out_dir / "tables" / f"{name}.md"
        tex = out_dir / "tables" / f"{name}.tex"
        md.write_text(f"**{caption}**\n\n" + markdown_table(headers, rows) + "\n",
                      encoding="utf-8")
        tex.write_text(latex_table(headers, rows, caption=caption,
                                   label=f"tab:{name}") + "\n", encoding="utf-8")
        written[f"{name}.md"] = md
        written[f"{name}.tex"] = tex

    h, r = main_table(results, labels)
    emit("main", h, r, "Results by configuration, all functions.")

    for cond, label in (("with_sigma", "functions with a fact table"),
                        ("without_sigma", "functions without one")):
        h, r = main_table(results, labels, condition=cond)
        emit(f"main_{cond}", h, r, f"Results by configuration, {label}.")

    if "C1" in results:
        h, r = condition_table(results["C1"])
        emit("conditions", h, r,
             "The full system, broken out by evidence condition.")

    if ablations:
        h, r = ablation_table_rows(ablations, labels)
        emit("ablations", h, r,
             "Each ablation against the full system, Holm-corrected.")
        h, r = seed_table(ablations)
        emit("seeds", h, r, "Seed-to-seed spread per configuration.")
        if figure:
            try:
                written["ablations.png"] = ablation_figure(
                    ablations, labels, out_dir / "figures" / "ablations.png")
            except Exception as e:                       # noqa: BLE001
                written["ablations.png.error"] = out_dir / "figures" / "error.txt"
                written["ablations.png.error"].parent.mkdir(parents=True,
                                                            exist_ok=True)
                written["ablations.png.error"].write_text(str(e))

    if manifest is not None:
        p = out_dir / "manifest.json"
        p.write_text(json.dumps(manifest, indent=1, sort_keys=True),
                     encoding="utf-8")
        written["manifest.json"] = p
    return written
