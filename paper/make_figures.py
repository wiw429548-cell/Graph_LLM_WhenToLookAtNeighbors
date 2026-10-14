"""Builds every figure, table and in-text number of paper/main.tex from the experiment outputs.

    python paper/make_figures.py          (from the project root; ~1 min, CPU only)

Inputs : results/analysis/<llm>/<ds>.json, results/llm/<llm>/<ds>/raw.pkl, results/gnn/<ds>.json,
         results/distill/<llm>/<ds>.json, data/processed/<ds>.pkl
Outputs: paper/figures/*.pdf, paper/tables/*.tex, paper/numbers.tex  (no number in the paper is typed by hand)
"""
import json
import pickle
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker  # noqa: E402
from matplotlib import font_manager  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402
from scipy.stats import wilcoxon, spearmanr  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RES, PROC = ROOT / "results", ROOT / "data" / "processed"
OUT_F, OUT_T = ROOT / "paper" / "figures", ROOT / "paper" / "tables"
OUT_F.mkdir(parents=True, exist_ok=True)
OUT_T.mkdir(parents=True, exist_ok=True)

M7, MQ3, M3, M32 = ("Qwen_Qwen2_5_7B_Instruct", "Qwen_Qwen3_8B", "Qwen_Qwen2_5_3B_Instruct",
                    "Qwen_Qwen2_5_32B_Instruct_AWQ")
MODEL_NAME = {M3: "Qwen2.5-3B", M7: "Qwen2.5-7B", MQ3: "Qwen3-8B", M32: "Qwen2.5-32B"}
DS = ["cora", "citeseer", "pubmed", "arxiv", "wikics", "instagram", "reddit", "history", "photo", "computer",
      "cornell", "texas", "washington", "wisconsin", "actor", "amazon_ratings"]
LOW_H = ["cornell", "texas", "washington", "wisconsin", "actor", "amazon_ratings"]
SCALE_DS = ["cora", "pubmed", "wikics", "history", "cornell", "actor"]
NAME = {"cora": "Cora", "citeseer": "CiteSeer", "pubmed": "PubMed", "arxiv": "arXiv", "wikics": "WikiCS",
        "instagram": "Instagram", "reddit": "Reddit", "history": "History", "photo": "Photo",
        "computer": "Computers", "cornell": "Cornell", "texas": "Texas", "washington": "Washington",
        "wisconsin": "Wisconsin", "actor": "Actor", "amazon_ratings": "Amazon-R."}
TF, LR = "NB (Pi=est.) [NeedNbr-TF]", "learned g_phi [NeedNbr-L]"

# ------------------------------------------------------------------ style: gray/white canvas + a few accent hues
BLUE, ORANGE, AQUA, VIOLET = "#2a78d6", "#e8613a", "#17a673", "#6a4fc9"
BLUE_L, ORANGE_L, AQUA_L, VIOLET_L = "#d6e6fa", "#fbdccf", "#d2f0e3", "#e6e0f7"
INK, INK2, GRAY, GRAY2, GRAY3, GRID = "#1f1f1e", "#52514e", "#8c8b87", "#b4b3ae", "#d9d8d3", "#ffffff"
PANEL, PANEL2 = "#f4f3ef", "#e9e7e1"
FONT_DIR = Path.home() / "AppData/Local/Programs/MiKTeX/fonts/opentype/public/libertine"
for f in ("LinBiolinum_R.otf", "LinBiolinum_RB.otf", "LinBiolinum_RI.otf"):
    if (FONT_DIR / f).exists():
        font_manager.fontManager.addfont(str(FONT_DIR / f))
plt.rcParams.update({
    "font.family": "Linux Biolinum O" if (FONT_DIR / "LinBiolinum_R.otf").exists() else "DejaVu Sans",
    "mathtext.fontset": "custom", "mathtext.rm": "Linux Biolinum O", "mathtext.it": "Linux Biolinum O:italic",
    "mathtext.bf": "Linux Biolinum O:bold",
    "font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 7.8, "xtick.labelsize": 7, "ytick.labelsize": 7,
    "legend.fontsize": 6.8, "axes.facecolor": PANEL, "axes.edgecolor": PANEL, "axes.linewidth": 0.0,
    "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2, "xtick.major.size": 0, "ytick.major.size": 0,
    "xtick.major.pad": 3, "ytick.major.pad": 3, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.9,
    "axes.axisbelow": True, "axes.spines.top": False, "axes.spines.right": False, "axes.spines.left": False,
    "axes.spines.bottom": False, "legend.frameon": False, "figure.dpi": 200, "savefig.dpi": 600,
    "savefig.bbox": "tight", "savefig.pad_inches": 0.03, "pdf.fonttype": 42, "axes.titleweight": "bold",
    "axes.titlecolor": INK, "axes.titlelocation": "left", "axes.titlepad": 6, "lines.solid_capstyle": "round",
    "lines.dash_capstyle": "round"})
COL_W, TEXT_W = 3.33, 7.0  # ACM sigconf column / text width (in)


def save(fig, name):
    fig.savefig(OUT_F / f"{name}.pdf")
    fig.savefig(OUT_F / "png" / f"{name}.png")
    plt.close(fig)


def zero_line(ax, axis="h"):
    (ax.axhline if axis == "h" else ax.axvline)(0, color=INK2, lw=0.8, zorder=2)


def js(m, ds):
    p = RES / "analysis" / m / f"{ds}.json"
    return json.load(open(p, encoding="utf-8")) if p.exists() else None


def raw(m, ds):
    p = RES / "llm" / m / ds / "raw.pkl"
    return pickle.load(open(p, "rb")) if p.exists() else None


def mean(xs):
    xs = [x for x in xs if x is not None]
    return float(np.mean(xs)) if xs else float("nan")


NUM = {}  # \newcommand macros for the text


def num(key, val, fmt="{:.1f}"):
    NUM[key] = fmt.format(val)


def pct(x):  # accuracy fraction -> percentage points
    return 100.0 * x


A7 = {d: js(M7, d) for d in DS}
AQ = {d: js(MQ3, d) for d in DS}
A3 = {d: js(M3, d) for d in SCALE_DS}
A32 = {d: js(M32, d) for d in SCALE_DS}
GNN = {d: json.load(open(RES / "gnn" / f"{d}.json", encoding="utf-8")) for d in DS}


def curve(a, router, strat="cns"):
    return a["curves"][f"{strat}|{router}"]


def acc_at(a, router, b=0.2, strat="cns"):
    return next(c["acc"] for c in curve(a, router, strat)["curve"] if abs(c["budget"] - b) < 1e-9)


# =================================================================== Table 1: datasets
def dataset_table():
    rows = []
    for d in DS:
        g = pickle.load(open(PROC / f"{d}.pkl", "rb"))
        ei = g["edge_index"]
        und = {(min(a, b), max(a, b)) for a, b in ei.T.tolist() if a != b}
        a = A7[d]
        rows.append((NAME[d], len(g["y"]), len(und), len(g["label_names"]), a["homophily"], a["adj_homophily"],
                     a["tokens_f0"], a["tokens_f1_only"]["random"], d in LOW_H))
    L = [r"\begin{table}[t]", r"\centering\small", r"\caption{The 16 text-attributed graphs. $h$: edge homophily; "
         r"$h_{\mathrm{adj}}$: adjusted homophily~\cite{platonov2023characterizing}; tokens: mean prompt length of "
         r"the text-only prompt $f_0$ and of the graph-augmented prompt $f_1$ with five random neighbours "
         r"(Qwen2.5 tokenizer, 1{,}000 test nodes).}\label{tab:data}",
         r"\setlength{\tabcolsep}{3.2pt}", r"\begin{tabular}{@{}lrrrrrrr@{}}", r"\toprule",
         r"Dataset & \#Nodes & \#Edges & $C$ & $h$ & $h_{\mathrm{adj}}$ & tok $f_0$ & tok $f_1$\\", r"\midrule"]
    for i, r in enumerate(rows):
        if i == 10:
            L.append(r"\midrule")
        L.append(f"{r[0]} & {r[1]:,} & {r[2]:,} & {r[3]} & {r[4]:.2f} & {r[5]:.2f} & {r[6]:.0f} & {r[7]:.0f}\\\\"
                 .replace(",", "{,}"))
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    (OUT_T / "datasets.tex").write_text("\n".join(L), encoding="utf-8")


# =================================================================== Table 2: main results (7B, full width)
def main_table():
    cols = [("f0", "Text only"), ("random", "Random"), ("similar", "Similar"), ("cns", r"\textsc{CNS}"),
            ("keywords", "Keywords"), ("summary", "Summary"), ("graphicl", "GraphICL$^\\dagger$"),
            ("tf20", r"\textsc{NeedNbr}"), ("orc20", "Oracle"), ("gcn", "GCN$^\\dagger$")]

    def vals(a, d):
        v = {"f0": a["acc_f0"], **{k: a["acc_f1_full"].get(k) for k in
                                    ("random", "similar", "cns", "keywords", "summary", "graphicl")},
             "tf20": acc_at(a, TF), "orc20": acc_at(a, "oracle"), "gcn": GNN[d]["gcn"]["test_eval_acc_mean"]}
        return v

    L = [r"\begin{table*}[t]", r"\centering\small",
         r"\caption{Accuracy (\%) of Qwen2.5-7B-Instruct on 1{,}000 test nodes per dataset. "
         r"Columns 2--7 add the context of five neighbours to \emph{every} node; \textsc{NeedNbr} and Oracle add "
         r"it (with \textsc{CNS} neighbours) only to the 20\% of nodes ranked highest by the training-free "
         r"\textsc{NeedNbr} score or by the true per-node gain (an upper bound that uses test labels). "
         r"Cells are shaded by the change over text-only inference (\colorbox{gainbg}{gain}, "
         r"\colorbox{lossbg}{loss} of more than 2 points). $^\dagger$Uses the 20 labels per class "
         r"(GraphICL as in-context demonstrations, GCN for training). Best LLM configuration per row in bold "
         r"(Oracle excluded).}\label{tab:main}",
         r"\setlength{\tabcolsep}{4.2pt}",
         r"\begin{tabular}{@{}l c cccccc c >{\color{oraclegray}}c c@{}}", r"\toprule",
         r" & & \multicolumn{6}{c}{Neighbour context for all nodes} & \multicolumn{2}{c}{Routed 20\%} & \\",
         r"\cmidrule(lr){3-8}\cmidrule(lr){9-10}",
         "Dataset & " + " & ".join(c[1] for c in cols[:8]) + r" & Oracle & GCN$^\dagger$\\", r"\midrule"]
    avg = {c[0]: [] for c in cols}
    for i, d in enumerate(DS):
        if i == 10:
            L.append(r"\midrule")
        v = vals(A7[d], d)
        llm_keys = [c[0] for c in cols[:8]]
        best = max(v[k] for k in llm_keys if v[k] is not None)
        cells = []
        for k, _ in cols:
            x = v[k]
            avg[k].append(x)
            s = f"{pct(x):.1f}"
            if k in llm_keys and abs(x - best) < 1e-9:
                s = r"\textbf{" + s + "}"
            if k not in ("f0", "gcn", "orc20"):
                dlt = x - v["f0"]
                if dlt > 0.02:
                    s = r"\cellcolor{gainbg}" + s
                elif dlt < -0.02:
                    s = r"\cellcolor{lossbg}" + s
            cells.append(s)
        L.append(f"{NAME[d]} & " + " & ".join(cells) + r"\\")
    L.append(r"\midrule")
    av = {k: mean(x) for k, x in avg.items()}
    llm_keys = [c[0] for c in cols[:8]]
    best = max(av[k] for k in llm_keys)
    L.append(r"\textit{Average} & " + " & ".join(
        (r"\textbf{%.1f}" if (k in llm_keys and abs(av[k] - best) < 1e-9) else "%.1f") % pct(av[k])
        for k, _ in cols) + r"\\")
    # Qwen3-8B average row (no GraphICL run for the second LLM)
    q = {k: [] for k, _ in cols}
    for d in DS:
        a = AQ[d]
        v = vals(a, d)
        for k in q:
            q[k].append(v[k])
    L.append(r"\textit{Avg.\ Qwen3-8B} & " + " & ".join(
        ("--" if any(x is None for x in q[k]) else "%.1f" % pct(mean(q[k]))) for k, _ in cols) + r"\\")
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table*}"]
    (OUT_T / "main.tex").write_text("\n".join(L), encoding="utf-8")
    for k in av:
        num(f"avg{k.replace('2', 'two').replace('0', 'zero').replace('1', 'one')}", pct(av[k]))
    num("orcgain", pct(av["orc20"] - av["f0"]), "{:+.1f}")
    num("qorcgain", pct(mean(q["orc20"]) - mean(q["f0"])), "{:+.1f}")
    num("qavgfzero", pct(mean(q["f0"])))
    num("qavgrandom", pct(mean(q["random"])))
    num("qavgcns", pct(mean(q["cns"])))
    num("qavgsummary", pct(mean(q["summary"])))
    num("qavgtf", pct(mean(q["tf20"])))
    num("qavgorc", pct(mean(q["orc20"])))
    # how many datasets: f1-random below f0
    n_loss = sum(A7[d]["acc_f1_full"]["random"] < A7[d]["acc_f0"] for d in DS)
    num("nlossrandom", n_loss, "{:d}")
    n_loss_q = sum(AQ[d]["acc_f1_full"]["random"] < AQ[d]["acc_f0"] for d in DS)
    num("nlossrandomq", n_loss_q, "{:d}")
    n_gicl = sum(A7[d]["acc_f1_full"]["graphicl"] > A7[d]["acc_f0"] for d in DS)
    num("ngiclwins", n_gicl, "{:d}")
    n_gcn = sum(GNN[d]["gcn"]["test_eval_acc_mean"] > max(
        [A7[d]["acc_f0"]] + [A7[d]["acc_f1_full"][s] for s in A7[d]["acc_f1_full"]]) for d in DS)
    num("ngcnwins", n_gcn, "{:d}")


# =================================================================== Figure 1: per-node helped / hurt
def fig_helped_hurt():
    rows = list(DS)
    H = np.array([A7[d]["gain_breakdown"]["random"]["helped"] for d in rows])
    U = np.array([A7[d]["gain_breakdown"]["random"]["hurt"] for d in rows])
    # mean row at the bottom, separated from the graphs
    y = np.r_[np.arange(len(rows))[::-1] + 1.4, 0]
    H, U = np.r_[H, H.mean()], np.r_[U, U.mean()]
    labels = [NAME[d] for d in rows] + ["Mean (16)"]
    fig, ax = plt.subplots(figsize=(COL_W, 2.95))
    ax.axhspan(1.4 - 0.5, 1.4 + 5.5, color=PANEL2, lw=0, zorder=0)  # low-homophily block
    ax.axhspan(-0.55, 0.55, color="white", lw=0, zorder=0)
    ax.barh(y, 100 * H, height=0.66, color=BLUE, label="helped (wrong $\\to$ right)", zorder=3)
    ax.barh(y, -100 * U, height=0.66, color=ORANGE, label="hurt (right $\\to$ wrong)", zorder=3)
    ax.scatter(100 * (H - U), y, s=17, marker="D", color=INK, edgecolor="white", linewidth=0.7, zorder=4,
               label="net change (what benchmarks report)")
    for yi, h, u in zip(y, H, U):
        ax.text(100 * h + 0.8, yi, f"{100 * h:.0f}", va="center", ha="left", fontsize=5.6, color=BLUE)
        ax.text(-100 * u - 0.8, yi, f"{100 * u:.0f}", va="center", ha="right", fontsize=5.6, color=ORANGE)
    zero_line(ax, "v")
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.get_yticklabels()[-1].set_fontweight("bold")
    ax.set_xlabel("share of the test nodes (%)")
    ax.set_xlim(-56, 30)
    ax.set_ylim(-0.7, y.max() + 0.7)
    ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{abs(v):.0f}"))
    ax.grid(axis="y", visible=False)
    ax.text(29, 1.4 + 5.2, "low homophily", fontsize=6.2, color=INK2, va="top", ha="right", style="italic")
    ax.text(29, y[0] + 0.2, "homophilic", fontsize=6.2, color=INK2, va="top", ha="right", style="italic")
    ax.text(-55, y[0] + 0.9, "$\\leftarrow$ hurt", fontsize=6.6, color=ORANGE, ha="left", weight="bold")
    ax.text(28, y[0] + 0.9, "helped $\\rightarrow$", fontsize=6.6, color=BLUE, ha="right", weight="bold")
    h, lb = ax.get_legend_handles_labels()
    ax.legend([h[1], h[2], h[0]], [lb[1], lb[2], lb[0]], loc="lower center", bbox_to_anchor=(0.45, 1.03), ncol=2,
              handlelength=1.0, columnspacing=0.9, handletextpad=0.4, fontsize=6.4)
    save(fig, "helped_hurt")
    g = [A7[d]["gain_breakdown"] for d in DS]
    num("helpedrandom", pct(mean([x["random"]["helped"] for x in g])))
    num("hurtrandom", pct(mean([x["random"]["hurt"] for x in g])))
    num("anyfix", pct(mean([x["_any_strategy_fixes"] for x in g])))
    num("helpedgicl", pct(mean([x["graphicl"]["helped"] for x in g])))
    num("hurtgicl", pct(mean([x["graphicl"]["hurt"] for x in g])))
    num("helpedq", pct(mean([AQ[d]["gain_breakdown"]["random"]["helped"] for d in DS])))
    num("hurtq", pct(mean([AQ[d]["gain_breakdown"]["random"]["hurt"] for d in DS])))


# =================================================================== Figure 3: why routing fails (pooled flips)
def pooled_flips(m, datasets, strat="random", nq=5):
    """per ambiguity quintile (within each dataset, rank-based), pooled: acc f0/f1, helped/hurt share"""
    acc0, acc1, helped, hurt, fix_rate, break_rate = [np.zeros(nq) for _ in range(6)]
    cnt = np.zeros(nq)
    w0 = np.zeros(nq)
    r0 = np.zeros(nq)
    for d in datasets:
        r = raw(m, d)
        if r is None or strat not in r["P1"]:
            continue
        n = r["n_test"]
        y = r["y"][:n]
        P0 = r["P0"][:n]
        c0 = P0.argmax(1) == y
        c1 = np.asarray(r["P1"][strat])[:n].argmax(1) == y
        A = np.array([f["A"] for f in r["feats"]["est"]])[:n]
        for q, idx in enumerate(np.array_split(np.argsort(A, kind="stable"), nq)):
            cnt[q] += len(idx)
            acc0[q] += c0[idx].sum()
            acc1[q] += c1[idx].sum()
            helped[q] += (~c0[idx] & c1[idx]).sum()
            hurt[q] += (c0[idx] & ~c1[idx]).sum()
            w0[q] += (~c0[idx]).sum()
            r0[q] += c0[idx].sum()
    return dict(acc0=acc0 / cnt, acc1=acc1 / cnt, helped=helped / cnt, hurt=hurt / cnt,
                fix=helped / np.maximum(w0, 1), brk=hurt / np.maximum(r0, 1))


def fig_ambiguity():
    s7 = pooled_flips(M7, DS, "random")
    s7c = pooled_flips(M7, DS, "cns")
    s32 = pooled_flips(M32, SCALE_DS, "random")
    fig, axs = plt.subplots(1, 3, figsize=(TEXT_W, 1.85), gridspec_kw={"wspace": 0.34})
    x = np.arange(1, 6)
    xt = ["Q1\nlow", "Q2", "Q3", "Q4", "Q5\nhigh"]
    # (a) accuracy, shaded by the sign of the change
    ax = axs[0]
    a0, a1 = 100 * s7["acc0"], 100 * s7["acc1"]
    ax.fill_between(x, a0, a1, where=a1 <= a0, color=ORANGE, alpha=0.18, lw=0, interpolate=True)
    ax.fill_between(x, a0, a1, where=a1 > a0, color=BLUE, alpha=0.25, lw=0, interpolate=True)
    ax.plot(x, a0, color=INK2, lw=1.8, marker="o", ms=4.2, mec="white", mew=0.7, label="text only $f_0$", zorder=4)
    ax.plot(x, a1, color=BLUE, lw=1.8, marker="o", ms=4.2, mec="white", mew=0.7, label="+ 5 random neighbours",
            zorder=4)
    ax.plot(x, 100 * s7c["acc1"], color=BLUE, lw=1.0, ls=(0, (2.5, 1.5)), marker="s", ms=2.8, alpha=0.8,
            label="+ 5 CNS neighbours", zorder=3)
    ax.set_title("(a) Accuracy by ambiguity")
    ax.set_ylabel("accuracy (%)")
    ax.legend(loc="lower left", handlelength=1.8)
    # (b) mirrored columns: helped up, hurt down
    ax = axs[1]
    hp, hu = 100 * s7["helped"], 100 * s7["hurt"]
    ax.bar(x, hp, 0.6, color=BLUE, label="helped", zorder=3)
    ax.bar(x, -hu, 0.6, color=ORANGE, label="hurt", zorder=3)
    ax.plot(x, hp - hu, color=INK, lw=1.1, marker="D", ms=3.6, mec="white", mew=0.6, label="net", zorder=4)
    for xi, a, b in zip(x, hp, hu):
        ax.text(xi, a + 0.8, f"{a:.0f}", ha="center", va="bottom", fontsize=5.8, color=BLUE)
        ax.text(xi, -b - 0.8, f"{b:.0f}", ha="center", va="top", fontsize=5.8, color=ORANGE)
    zero_line(ax)
    ax.set_title("(b) Who is helped, who is hurt")
    ax.set_ylabel("share of nodes in quintile (%)")
    ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{abs(v):.0f}"))
    ax.set_ylim(-25, 31)
    ax.legend(loc="upper left", ncol=3, handlelength=1.0, columnspacing=0.8)
    # (c) conditional flip rates, 7B solid / 32B dashed
    ax = axs[2]
    ax.fill_between(x, 100 * s7["fix"], 100 * s7["brk"], color=ORANGE, alpha=0.12, lw=0)
    ax.plot(x, 100 * s7["brk"], color=ORANGE, lw=1.8, marker="o", ms=4.2, mec="white", mew=0.7,
            label="P(broken | $f_0$ right)")
    ax.plot(x, 100 * s7["fix"], color=BLUE, lw=1.8, marker="o", ms=4.2, mec="white", mew=0.7,
            label="P(fixed | $f_0$ wrong)")
    ax.plot(x, 100 * s32["brk"], color=ORANGE, lw=1.0, ls=(0, (2.5, 1.5)), marker="^", ms=3.4, alpha=0.9)
    ax.plot(x, 100 * s32["fix"], color=BLUE, lw=1.0, ls=(0, (2.5, 1.5)), marker="^", ms=3.4, alpha=0.9)
    ax.text(5.15, 100 * s7["brk"][-1], "7B", fontsize=6.2, color=INK2, va="center")
    ax.text(5.15, 100 * s32["brk"][-1] + 1.5, "32B", fontsize=6.2, color=INK2, va="center")
    ax.set_title("(c) Conditional flip rates")
    ax.set_ylabel("rate (%)")
    ax.legend(loc="upper left", handlelength=1.6)
    ax.set_ylim(0, 72)
    ax.set_xlim(0.7, 5.6)
    for ax in axs:
        ax.set_xticks(x)
        ax.set_xticklabels(xt, fontsize=6.4)
        ax.set_xlabel("text-only ambiguity quintile", fontsize=7)
        ax.grid(axis="x", visible=False)
    save(fig, "ambiguity")
    num("qonezero", pct(s7["acc0"][0]))
    num("qoneone", pct(s7["acc1"][0]))
    num("qfivezero", pct(s7["acc0"][4]))
    num("qfiveone", pct(s7["acc1"][4]))
    num("qonehelped", pct(s7["helped"][0]))
    num("qonehurt", pct(s7["hurt"][0]))
    num("qfivehelped", pct(s7["helped"][4]))
    num("qfivehurt", pct(s7["hurt"][4]))
    num("qfivefix", pct(s7["fix"][4]))
    num("qfivebrk", pct(s7["brk"][4]))
    num("qonebrk", pct(s7["brk"][0]))
    num("qfivefixthirtytwo", pct(s32["fix"][4]))
    num("qfivebrkthirtytwo", pct(s32["brk"][4]))


# =================================================================== Figure 4: budget curves + router table
ROUTERS = [("random", "Random", GRAY, (0, (3, 2)), 1.3),
           ("degree", "Degree", GRAY2, "-", 0.9),
           ("A_only (confidence)", "Ambiguity $A$", GRAY2, "-", 0.9),
           ("LLM self-assessment", "LLM self-assess.", INK2, (0, (1, 1.2)), 1.3),
           (TF, "NeedNbr-TF", BLUE, "-", 2.0),
           (LR, "NeedNbr-L", AQUA, "-", 1.6),
           ("oracle", "Oracle (bound)", VIOLET, "-", 1.8)]


def avg_curve(As, router, strat="cns"):
    cs = [curve(a, router, strat)["curve"] for a in As if f"{strat}|{router}" in a["curves"]]
    b = [c["budget"] for c in cs[0]]
    d = np.mean([[c["acc"] - a["acc_f0"] for c in cc] for cc, a in zip(cs, As)], axis=0)
    t = np.mean([[c["tokens_per_node"] for c in cc] for cc in cs], axis=0)
    return np.array(b), d, t


def fig_budget():
    fig, axs = plt.subplots(1, 2, figsize=(COL_W, 2.3), gridspec_kw={"wspace": 0.12}, sharey=True)
    for ax, (A, title) in zip(axs, [([A7[d] for d in DS], "(a) Qwen2.5-7B"),
                                    ([AQ[d] for d in DS], "(b) Qwen3-8B")]):
        b, rnd, _ = avg_curve(A, "random")
        _, orc, _ = avg_curve(A, "oracle")
        ax.fill_between(100 * b, 100 * rnd, 100 * orc, color=VIOLET_L, lw=0, zorder=1)
        k = int(np.argmax(orc - rnd))
        ax.text(100 * b[k] + 4, 100 * (rnd[k] + orc[k]) / 2, "headroom", color=VIOLET, fontsize=6.4,
                style="italic", va="center")
        for key, lab, col, ls, lw in ROUTERS:
            bb, dlt, _ = avg_curve(A, key)
            ax.plot(100 * bb, 100 * dlt, color=col, ls=ls, lw=lw, label=lab,
                    zorder=5 if col in (BLUE, AQUA, VIOLET) else 4)
        zero_line(ax)
        ax.set_title(title, fontsize=7.6)
        ax.set_xlabel("routing budget (%)")
        ax.set_xticks([0, 20, 50, 100])
        ax.set_xlim(0, 100)
    axs[0].set_ylabel("$\\Delta$ accuracy vs. text only (pts)")
    h, lab = axs[0].get_legend_handles_labels()
    fig.legend(h, lab, loc="upper center", bbox_to_anchor=(0.52, 0.0), ncol=4, handlelength=1.8, fontsize=6.2,
               columnspacing=0.9)
    save(fig, "budget")


def router_table():
    rows = [("random", "Random"), ("degree", "Degree"),
            ("A_only (confidence)", r"Ambiguity $A$"), ("A_only (calibrated)", r"$A$, calibrated"),
            ("D_only", r"Discrim.\ $D$"), ("AxD", r"$A\times D$"),
            ("LLM self-assessment", "Self-assess."),
            ("NB (Pi=identity)", r"NB, $\Pi=I$"), ("NB (Pi=zero-shot est.)", r"NB, label-free $\Pi$"),
            (TF, r"\textsc{NeedNbr}-TF"),
            ("NB with MLP pseudo-labels (no neighbour LLM calls)", r"NB, classifier labels"),
            (LR, r"\textsc{NeedNbr}-L"), ("oracle", r"\textcolor{oraclegray}{Oracle}")]
    models = [(M7, A7, DS), (MQ3, AQ, DS), (M32, A32, SCALE_DS)]
    L = [r"\begin{table}[t]", r"\centering\small",
         r"\caption{Routing quality with \textsc{CNS} neighbours. AUC: area under the accuracy--budget curve "
         r"(budgets 0--100\%), averaged over datasets; $\rho$: Spearman correlation between router score and the "
         r"true per-node gain on the test nodes, averaged over datasets; W/L: datasets on which the router's "
         r"accuracy at a 20\% budget is higher/lower than that of the random router (averaged over 200 "
         r"permutations). \textsc{NeedNbr}-TF: NB with $\Pi$ from the training labels; \textsc{NeedNbr}-L: learned "
         r"on validation nodes; Oracle: ranks by the true gain (test labels).}\label{tab:routing}",
         r"\setlength{\tabcolsep}{3.0pt}", r"\begin{tabular}{@{}lccccccc@{}}", r"\toprule",
         r" & \multicolumn{3}{c}{Qwen2.5-7B (16)} & \multicolumn{3}{c}{Qwen3-8B (16)} & 32B (6)\\",
         r"\cmidrule(lr){2-4}\cmidrule(lr){5-7}\cmidrule(lr){8-8}",
         r"Router & AUC & $\rho$ & W/L & AUC & $\rho$ & W/L & AUC\\", r"\midrule"]
    for key, lab in rows:
        cells = []
        for mi, (m, A, dsl) in enumerate(models):
            aucs = [curve(A[d], key)["area"] for d in dsl]
            cells.append(f"{pct(mean(aucs)):.1f}")
            if mi == 2:
                break
            if key in ("random", "oracle"):
                cells += ["--", "--"]
                continue
            rho = mean([A[d]["spearman_vs_true_gain"].get(f"cns|{key}") for d in dsl])
            w = sum(acc_at(A[d], key) > acc_at(A[d], "random") + 1e-9 for d in dsl)
            lo = sum(acc_at(A[d], key) < acc_at(A[d], "random") - 1e-9 for d in dsl)
            cells += [f"{rho:+.2f}", f"{w}/{lo}"]
        if key == "oracle":
            L.append(r"\midrule")
        L.append(lab + " & " + " & ".join(cells) + r"\\")
        if key == "random":
            L.append(r"\midrule")
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    (OUT_T / "routing.tex").write_text("\n".join(L), encoding="utf-8")
    a7 = {k: mean([curve(A7[d], k)["area"] for d in DS]) for k, _ in rows}
    num("aucrandom", pct(a7["random"]))
    num("auctf", pct(a7[TF]))
    num("auclr", pct(a7[LR]))
    num("aucorc", pct(a7["oracle"]))
    num("aucself", pct(a7["LLM self-assessment"]))
    num("aucA", pct(a7["A_only (confidence)"]))
    best_nonoracle = max(v for k, v in a7.items() if k != "oracle")
    num("aucbest", pct(best_nonoracle))
    num("rhotf", mean([A7[d]["spearman_vs_true_gain"][f"cns|{TF}"] for d in DS]), "{:.2f}")
    num("rholr", mean([A7[d]["spearman_vs_true_gain"][f"cns|{LR}"] for d in DS]), "{:.2f}")
    num("rhoself", mean([A7[d]["spearman_vs_true_gain"]["cns|LLM self-assessment"] for d in DS]), "{:.2f}")
    rhos = [r for d in DS for k, v in A7[d]["spearman_vs_true_gain"].items() if k.startswith("cns|") for r in [v]
            if r is not None]
    num("rhomax", max(rhos), "{:.2f}")
    # Wilcoxon NeedNbr-TF@20 vs random router @20 (random = permutation-average), across 16 datasets
    tf = [acc_at(A7[d], TF) for d in DS]
    rr = [acc_at(A7[d], "random") for d in DS]
    num("wilcoxtfrandom", wilcoxon(tf, rr).pvalue, "{:.2f}")
    num("tfminusrandom", pct(mean(np.array(tf) - np.array(rr))), "{:+.1f}")
    num("tftwenty", pct(mean(tf)))
    num("randomtwenty", pct(mean(rr)))
    num("orctwenty", pct(mean([acc_at(A7[d], "oracle") for d in DS])))
    # auto thresholds
    auto = [A7[d]["auto_threshold"]["NeedNbr-auto (NB, val-tuned threshold)"] for d in DS]
    num("autoacc", pct(mean([x["acc"] for x in auto])))
    num("autonever", sum(x["routed_frac"] < 0.02 for x in auto), "{:d}")
    num("autoalways", sum(x["routed_frac"] > 0.9 for x in auto), "{:d}")


# =================================================================== Figure 5: scale + local homophily
def fig_scale_homophily():
    fig, axs = plt.subplots(1, 2, figsize=(TEXT_W, 1.9), gridspec_kw={"width_ratios": [1.0, 1.25], "wspace": 0.22})
    ax = axs[0]
    models = [(M3, A3, "3B"), (M7, A7, "7B"), (M32, A32, "32B")]
    xs = np.arange(3)
    ends = []
    for d in SCALE_DS:
        ys = [100 * (A[d]["acc_f1_full"]["random"] - A[d]["acc_f0"]) for _, A, _ in models]
        col = ORANGE if d in LOW_H else BLUE
        ax.plot(xs, ys, color=col, lw=1.0, marker="o", ms=3.4, mec="white", mew=0.5, alpha=0.75, zorder=3)
        ends.append((ys[-1], NAME[d], col))
    avg = [mean([100 * (A[d]["acc_f1_full"]["random"] - A[d]["acc_f0"]) for d in SCALE_DS]) for _, A, _ in models]
    ax.plot(xs, avg, color=INK, lw=2.4, marker="s", ms=5, mec="white", mew=0.8, zorder=5)
    for xi, v in zip(xs, avg):
        ax.text(xi, v - 3.2, f"{v:+.1f}", ha="center", va="top", fontsize=6.4, color=INK, weight="bold", zorder=6,
                bbox=dict(boxstyle="round,pad=0.15", fc=PANEL, ec="none"))
    ends.sort()
    placed = []
    for yv, lab, col in ends:  # spread the right-hand labels so they do not collide
        yl = max(yv, placed[-1] + 4.2) if placed else yv
        placed.append(yl)
        ax.annotate(lab, (xs[-1], yv), xytext=(xs[-1] + 0.14, yl), textcoords="data", fontsize=6.3, color=col,
                    va="center", arrowprops=dict(arrowstyle="-", color=GRAY2, lw=0.5, shrinkA=0, shrinkB=2))
    zero_line(ax)
    ax.set_xticks(xs)
    ax.set_xticklabels(["Qwen2.5-3B", "Qwen2.5-7B", "Qwen2.5-32B"])
    ax.grid(axis="x", visible=False)
    ax.set_xlim(-0.25, 2.8)
    ax.set_ylabel("$\\Delta$ acc. of $f_1$ vs. $f_0$ (pts)")
    ax.set_title("(a) Effect of neighbours vs. model size")
    ax.legend(handles=[Line2D([], [], color=BLUE, marker="o", ms=3.5, lw=1.0, label="homophilic"),
                       Line2D([], [], color=ORANGE, marker="o", ms=3.5, lw=1.0, label="low homophily"),
                       Line2D([], [], color=INK, marker="s", ms=4, lw=2.2, label="mean")],
              loc="upper left", ncol=3, handlelength=1.2, columnspacing=0.7, fontsize=6.2)
    ax.set_ylim(-35, 33)
    # (b) pooled local homophily of the target: delta f1-random vs f0 per bin, for 3B / 7B / 32B (6 shared graphs)
    ax = axs[1]
    bins = ["[0.00,0.25)", "[0.25,0.50)", "[0.50,0.75)", "[0.75,1.00]"]
    lab = ["0–.25", ".25–.5", ".5–.75", ".75–1"]
    w = 0.26
    for i, (m, A, nm) in enumerate(models):
        dl = []
        for bn in bins:
            n_, s0, s1 = 0, 0.0, 0.0
            for d in SCALE_DS:
                for row in A[d]["by_local_homophily"]:
                    if row["bin"] == bn:
                        n_ += row["n"]
                        s0 += row["acc_f0"] * row["n"]
                        s1 += row["acc_f1_random"] * row["n"]
            dl.append(100 * (s1 - s0) / max(n_, 1))
        col = [GRAY2, GRAY, BLUE][i]
        xb = np.arange(4) + (i - 1) * w
        ax.bar(xb, dl, w * 0.9, color=col, label=MODEL_NAME[m], zorder=3)
        for xv, v in zip(xb, dl):
            ax.text(xv, v + (0.4 if v >= 0 else -0.4), f"{v:+.1f}", ha="center", va="bottom" if v >= 0 else "top",
                    fontsize=5.4, color=INK2 if i < 2 else BLUE)
    zero_line(ax)
    ax.axvspan(2.55, 3.45, color=BLUE_L, alpha=0.6, lw=0, zorder=0)
    ax.text(3.0, -13.0,"neighbours carry\nthe right label", ha="center", fontsize=6.0, color=BLUE, style="italic")
    ax.set_xticks(np.arange(4))
    ax.set_xticklabels([f"local $h$ {x}" for x in lab])
    ax.grid(axis="x", visible=False)
    ax.set_ylim(-21.5, 4)
    ax.set_ylabel("$\\Delta$ acc. vs. $f_0$ (pts)")
    ax.set_title("(b) By local homophily of the target node (6 graphs, pooled)")
    ax.legend(loc="lower left", ncol=1, handlelength=1.0)
    save(fig, "scale_homophily")
    for (m, A, nm), v in zip(models, avg):
        num(f"scaledelta{nm.lower().replace('3b', 'three').replace('7b', 'seven').replace('32b', 'thirtytwo')}", v,
            "{:+.1f}")


def scale_table():
    L = [r"\begin{table}[t]", r"\centering\small",
         r"\caption{Scaling within the Qwen2.5 family on the six graphs run with all three sizes: accuracy (\%) "
         r"of text-only inference, of graph-augmented inference for all nodes, and of \textsc{NeedNbr}-TF at a "
         r"20\% budget; Oracle@20\% is the upper bound for routing. $f_1{<}f_0$: graphs on which random neighbours "
         r"lower the accuracy.}\label{tab:scale}",
         r"\setlength{\tabcolsep}{3.2pt}", r"\begin{tabular}{@{}llcccccc@{}}", r"\toprule",
         r"Model & $f_1{<}f_0$ & $f_0$ & Random & Similar & \textsc{CNS} & \textsc{NeedNbr} & \textcolor{oraclegray}{Oracle}\\",
         r"\midrule"]
    for m, A in [(M3, A3), (M7, A7), (M32, A32)]:
        v = [mean([A[d]["acc_f0"] for d in SCALE_DS])] + \
            [mean([A[d]["acc_f1_full"][s] for d in SCALE_DS]) for s in ("random", "similar", "cns")] + \
            [mean([acc_at(A[d], TF) for d in SCALE_DS]), mean([acc_at(A[d], "oracle") for d in SCALE_DS])]
        nl = sum(A[d]["acc_f1_full"]["random"] < A[d]["acc_f0"] for d in SCALE_DS)
        L.append(f"{MODEL_NAME[m]} & {nl}/6 & " + " & ".join(
            (r"\textcolor{oraclegray}{%.1f}" if i == 5 else "%.1f") % pct(x) for i, x in enumerate(v)) + r"\\")
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    (OUT_T / "scale.tex").write_text("\n".join(L), encoding="utf-8")


# =================================================================== Figure 6: option-prior shift (mechanism)
def fig_prior_shift():
    pts = []
    for m, dsl in [(M7, DS), (M32, SCALE_DS), (M3, SCALE_DS)]:
        for d in dsl:
            r = raw(m, d)
            n, nv = r["n_test"], r["n_val"]
            y = r["y"][:n]
            P0, P1 = r["P0"][:n], np.asarray(r["P1"]["random"])[:n]
            C = P0.shape[1]

            def top(P):
                return np.bincount(P.argmax(1), minlength=C).max() / n

            def kl_pred(P):  # KL(pred-label distribution || true label distribution)
                p = np.bincount(P.argmax(1), minlength=C) / n + 1e-3
                q = np.bincount(y, minlength=C) / n + 1e-3
                p, q = p / p.sum(), q / q.sum()
                return float((p * np.log(p / q)).sum())
            pts.append(dict(m=m, d=d, kl0=kl_pred(P0), kl1=kl_pred(P1), top0=top(P0), top1=top(P1),
                            dacc=float((P1.argmax(1) == y).mean() - (P0.argmax(1) == y).mean())))
    p7 = [p for p in pts if p["m"] == M7]
    x7 = np.array([p["kl1"] - p["kl0"] for p in p7])
    y7 = np.array([100 * p["dacc"] for p in p7])
    rho = spearmanr(x7, y7).correlation
    fig, ax = plt.subplots(figsize=(COL_W, 2.1))
    ax.axvspan(0, 1.1, ymin=0, ymax=0.62, color=ORANGE_L, alpha=0.55, lw=0, zorder=0)
    ax.text(1.02, -30.5, "distorted and worse", ha="right", fontsize=6.2, color=ORANGE, style="italic")
    keep = x7 < 1.1  # History is off scale; fit on the visible 7B points
    k, b0 = np.polyfit(x7[keep], y7[keep], 1)
    xx = np.linspace(-0.6, 1.0, 50)
    ax.plot(xx, k * xx + b0, color=BLUE, lw=1.0, ls=(0, (3, 1.5)), alpha=0.8, zorder=2)
    style = {M7: (BLUE, "o", "Qwen2.5-7B (16 graphs)"), M32: (VIOLET, "^", "Qwen2.5-32B (6)"),
             M3: (GRAY, "s", "Qwen2.5-3B (6)")}
    for m, (c, mk, lab) in style.items():
        P = [p for p in pts if p["m"] == m]
        ax.scatter([p["kl1"] - p["kl0"] for p in P], [100 * p["dacc"] for p in P], s=24, color=c, marker=mk,
                   edgecolor="white", linewidth=0.6, label=lab, zorder=3)
    hist = next(p for p in pts if p["m"] == M7 and p["d"] == "history")
    ax.set_xlim(-0.75, 1.1)
    ax.annotate(f"History ($\\Delta$KL = {hist['kl1'] - hist['kl0']:.1f}, off scale)", (-0.75, 100 * hist["dacc"]),
                xytext=(-0.6, 100 * hist["dacc"] - 1.5), fontsize=6.2, color=INK2, va="center",
                arrowprops=dict(arrowstyle="->", color=BLUE, lw=0.8))
    for p in p7:
        if p["d"] in ("pubmed", "wikics", "actor"):
            ax.annotate(NAME[p["d"]], (p["kl1"] - p["kl0"], 100 * p["dacc"]), xytext=(4, 2),
                        textcoords="offset points", fontsize=6.2, color=INK2)
    ax.text(0.97, 0.97, f"Spearman $\\rho$ = {rho:.2f} (7B)", transform=ax.transAxes, ha="right", va="top",
            fontsize=6.6, color=BLUE, weight="bold",
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=BLUE_L, lw=0.8))
    zero_line(ax)
    zero_line(ax, "v")
    ax.set_xlabel("change in label-distribution mismatch, KL(pred $\\|$ true)")
    ax.set_ylabel("$\\Delta$ accuracy (pts)")
    ax.legend(loc="lower left", handletextpad=0.3, fontsize=6.3)
    save(fig, "prior_shift")
    num("klrho", rho, "{:.2f}")
    pm = {p["d"]: p for p in p7}
    num("pubmedtopzero", pct(pm["pubmed"]["top0"]), "{:.0f}")
    num("pubmedtopone", pct(pm["pubmed"]["top1"]), "{:.0f}")
    num("actortopone", pct(pm["actor"]["top1"]), "{:.0f}")
    # option-prior correction estimated on the 300 validation nodes (CNS prompt: the only f1 computed on val)
    rec = []
    for d in DS:
        r = raw(M7, d)
        n, nv = r["n_test"], r["n_val"]
        te, va = np.arange(n), np.arange(n, n + nv)
        y = r["y"]
        P0, P1 = r["P0"], np.asarray(r["P1"]["cns"])
        a0 = (P0[te].argmax(1) == y[te]).mean()
        a1 = (P1[te].argmax(1) == y[te]).mean()
        a1c = ((P1[te] / P1[va].mean(0)).argmax(1) == y[te]).mean()
        a0c = ((P0[te] / P0[va].mean(0)).argmax(1) == y[te]).mean()
        rec.append((a0, a1, a0c, a1c))
    rec = np.array(rec)
    num("pcfzero", pct(rec[:, 0].mean()))
    num("pcfone", pct(rec[:, 1].mean()))
    num("pcfzeroc", pct(rec[:, 2].mean()))
    num("pcfonec", pct(rec[:, 3].mean()))


# =================================================================== Figure 7: accuracy-token frontier (7B)
def fig_tokens():
    fig, ax = plt.subplots(figsize=(COL_W, 2.25))
    A = [A7[d] for d in DS]
    f0 = mean([a["acc_f0"] for a in A])
    t0 = mean([a["tokens_f0"] for a in A])
    ax.axhspan(100 * f0, 62, xmin=0, xmax=1, color=BLUE_L, alpha=0.35, lw=0, zorder=0)
    ax.text(265, 61.4, "shaded: better than text only", ha="left", va="top", fontsize=6.0, color=BLUE,
            style="italic")
    ax.axhline(100 * f0, color=INK2, lw=0.7, ls=(0, (2, 2)), zorder=2)
    for key, lab, col, ls, lw in [("random", "random", GRAY, (0, (3, 2)), 1.2),
                                  (TF, "NeedNbr-TF", BLUE, "-", 1.8),
                                  ("oracle", "oracle", VIOLET, "-", 1.6)]:
        b, dlt, t = avg_curve(A, key)
        ax.plot(t, 100 * (f0 + dlt), color=col, ls=ls, lw=lw, marker="o", ms=2.6, mec="white", mew=0.4, label=lab,
                zorder=4)
    pts = [("Text only", t0, f0, INK, "o")]
    for s, lab in [("random", "Random"), ("similar", "Similar"), ("cns", "CNS"), ("keywords", "Keywords"),
                   ("summary", "Summary"), ("graphicl", "GraphICL$^\\dagger$")]:
        tt = mean([a["tokens_f1_only"][s] for a in A])  # graph-augmented prompt replaces the text-only one
        pts.append((lab, tt, mean([a["acc_f1_full"][s] for a in A]), GRAY, "s"))
    ens = mean([a["llm_gnn"]["acc_f0_gcn_ensemble"] for a in A])
    casc = mean([a["llm_gnn"]["acc_needNbr20_to_gcn"] for a in A])
    pts.append(("$f_0$+GCN$^\\dagger$", t0, ens, AQUA, "D"))
    pts.append(("NeedNbr$\\to$GCN$^\\dagger$", t0, casc, AQUA, "D"))
    offs = {"Text only": (-4, -10), "Random": (5, 3), "Similar": (-30, -7), "CNS": (4, -6), "Keywords": (4, -1),
            "Summary": (4, -1), "GraphICL$^\\dagger$": (-8, 5), "$f_0$+GCN$^\\dagger$": (6, -2),
            "NeedNbr$\\to$GCN$^\\dagger$": (6, -2)}
    for lab, t, acc, c, mk in pts:
        ax.scatter([t], [100 * acc], s=34 if c == AQUA else (22 if c != INK else 34), color=c, marker=mk,
                   edgecolor="white", linewidth=0.6, zorder=5)
        ax.annotate(lab, (t, 100 * acc), xytext=offs.get(lab, (4, 0)), textcoords="offset points", fontsize=6.2,
                    color=AQUA if c == AQUA else (INK2 if c != INK else INK), zorder=6,
                    weight="bold" if c == AQUA else "normal")
    ax.annotate("", (t0 * 1.0, 100 * casc - 0.3), (t0 * 1.0, 100 * f0 + 0.3),
                arrowprops=dict(arrowstyle="-|>", color=AQUA, lw=1.0, mutation_scale=7))
    ax.text(t0 * 0.93, 100 * (casc + f0) / 2, f"+{round(100 * casc, 1) - round(100 * f0, 1):.1f}", color=AQUA, fontsize=6.4, ha="right",
            va="center", weight="bold")
    ax.set_xscale("log")
    ax.set_xticks([300, 500, 1000, 2000])
    ax.set_xticklabels(["300", "500", "1k", "2k"])
    ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.grid(which="minor", visible=False)
    ax.set_xlim(255, 2100)
    ax.set_xlabel("prompt tokens per node (log scale)")
    ax.set_ylabel("mean accuracy, 16 graphs (%)")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=3, fontsize=6.3, handlelength=1.6)
    ax.set_ylim(46.5, 61.8)
    save(fig, "tokens")
    num("tokzero", t0, "{:.0f}")
    num("tokcns", mean([a["tokens_f1_only"]["cns"] for a in A]), "{:.0f}")
    num("tokgicl", mean([a["tokens_f1_only"]["graphicl"] for a in A]), "{:.0f}")
    num("toksummary", mean([a["tokens_f1_only"]["summary"] for a in A]), "{:.0f}")
    num("tokkeywords", mean([a["tokens_f1_only"]["keywords"] for a in A]), "{:.0f}")
    num("toktf", mean([next(c["tokens_per_node"] for c in curve(a, TF)["curve"] if abs(c["budget"] - .2) < 1e-9)
                       for a in A]), "{:.0f}")
    num("ensacc", pct(ens))
    num("cascacc", pct(casc))
    num("gcnacc", pct(mean([GNN[d]["gcn"]["test_eval_acc_mean"] for d in DS])))
    num("gcnaccq", pct(mean([AQ[d]["llm_gnn"]["acc_needNbr20_to_gcn"] for d in DS])))
    num("sigms", mean([A7[d]["overhead"]["sig_sec_per_node"] * 1000 for d in DS]), "{:.2f}")
    num("cnsms", mean([A7[d]["overhead"]["cns_sec_per_node"] * 1000 for d in DS]), "{:.1f}")


# =================================================================== Table: neighbour selection + k
def selection_table():
    strat = [("cns", r"\textsc{CNS}"), ("cns_nored", r"\textsc{CNS} w/o redundancy"),
             ("cns_id", r"\textsc{CNS}, $\Pi=I$"), ("random", "Random 1-hop"), ("similar", "Similar (SNS-style)"),
             ("cns_hints", r"\textsc{CNS} + label hints"), ("keywords", "Keywords (GraphiT-style)"),
             ("summary", "LLM summary of neighbours"), ("graphicl", r"GraphICL-style$^\dagger$")]
    L = [r"\begin{table}[t]", r"\centering\small",
         r"\caption{Neighbour context applied to all nodes: mean accuracy (\%) and prompt tokens per node over "
         r"the 16 graphs (Qwen2.5-7B; Qwen3-8B has no GraphICL run), and the number of graphs on which the "
         r"context beats text-only inference ($>f_0$).}\label{tab:selection}",
         r"\setlength{\tabcolsep}{3.0pt}", r"\begin{tabular}{@{}lccccc@{}}", r"\toprule",
         r" & \multicolumn{3}{c}{Qwen2.5-7B} & \multicolumn{2}{c}{Qwen3-8B}\\",
         r"\cmidrule(lr){2-4}\cmidrule(lr){5-6}",
         r"Context & Acc. & $>f_0$ & Tokens & Acc. & $>f_0$\\", r"\midrule",
         r"Text only ($f_0$) & %.1f & -- & %.0f & %.1f & --\\" % (
             pct(mean([A7[d]["acc_f0"] for d in DS])), mean([A7[d]["tokens_f0"] for d in DS]),
             pct(mean([AQ[d]["acc_f0"] for d in DS]))), r"\midrule"]
    for s, lab in strat:
        a7 = mean([A7[d]["acc_f1_full"][s] for d in DS])
        w7 = sum(A7[d]["acc_f1_full"][s] > A7[d]["acc_f0"] for d in DS)
        t7 = mean([A7[d]["tokens_f1_only"][s] for d in DS])
        if all(s in AQ[d]["acc_f1_full"] for d in DS):
            aq = f"{pct(mean([AQ[d]['acc_f1_full'][s] for d in DS])):.1f}"
            wq = str(sum(AQ[d]["acc_f1_full"][s] > AQ[d]["acc_f0"] for d in DS))
        else:
            aq, wq = "--", "--"
        L.append(f"{lab} & {pct(a7):.1f} & {w7} & {t7:.0f} & {aq} & {wq}\\\\")
        if s == "similar":
            L.append(r"\midrule")
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    (OUT_T / "selection.tex").write_text("\n".join(L), encoding="utf-8")
    # k sensitivity (7B, 5 graphs)
    ks = ["cora", "pubmed", "history", "cornell", "actor"]
    K = {d: json.load(open(RES / "analysis" / M7 / f"ksens_{d}.json", encoding="utf-8"))["k_sensitivity"] for d in ks}
    for k in (3, 5, 10):
        for s in ("random", "cns"):
            num(f"k{['', '', '', 'three', '', 'five', '', '', '', '', 'ten'][k]}{s}",
                pct(mean([K[d][f"k={k}"][s]["all"] for d in ks])))
    num("kfzero", pct(mean([K[d]["k=3"]["acc_f0"] for d in ks])))
    w7 = {s: sum(A7[d]["acc_f1_full"][s] > A7[d]["acc_f0"] for d in DS) for s in ("cns", "random", "similar")}
    num("winscns", w7["cns"], "{:d}")
    num("winsrandom", w7["random"], "{:d}")
    num("winssimilar", w7["similar"], "{:d}")


# =================================================================== Table: GNN / distillation
def gnn_table():
    L = [r"\begin{table}[t]", r"\centering\small",
         r"\caption{Using the 20 labels per class without any LLM call: mean accuracy (\%, 5 seeds) of models "
         r"trained on frozen \texttt{bge-base} embeddings, next to the LLM configurations (Qwen2.5-7B). "
         r"$\to$GCN: the 20\% of nodes routed by \textsc{NeedNbr}-TF are answered by the GCN instead of a "
         r"graph-augmented prompt. $^*$Averaged over the \Nhtwogcnn{} graphs on which the 2-hop adjacency fits "
         r"full-batch training.}\label{tab:gnn}",
         r"\setlength{\tabcolsep}{3.0pt}", r"\begin{tabular}{@{}lcccc@{}}", r"\toprule",
         r"Method & Hom.\ (10) & Low-$h$ (6) & All (16) & LLM tok.\\", r"\midrule"]
    hom = [d for d in DS if d not in LOW_H]

    def row(lab, f, tok):
        v = [pct(mean([f(d) for d in grp])) for grp in (hom, LOW_H, DS)]
        L.append(f"{lab} & {v[0]:.1f} & {v[1]:.1f} & {v[2]:.1f} & {tok}\\\\")
    for key, lab in [("mlp", "MLP"), ("gcn", "GCN"), ("sage", "GraphSAGE"), ("gat", "GAT"), ("gprgnn", "GPR-GNN"),
                     ("h2gcn", "H2GCN-lite"), ("label_propagation", "Label propagation")]:
        ok = [d for d in DS if "test_eval_acc_mean" in GNN[d].get(key, {})]
        if len(ok) < len(DS):  # H2GCN is skipped on graphs whose 2-hop adjacency does not fit full-batch
            v = [mean([GNN[d][key]["test_eval_acc_mean"] for d in grp if d in ok]) for grp in (hom, LOW_H, DS)]
            L.append(f"{lab}$^*$ & {pct(v[0]):.1f} & {pct(v[1]):.1f} & {pct(v[2]):.1f} & 0\\\\")
            num("htwogcnn", len(ok), "{:d}")
            num("htwogcnhom", len([d for d in ok if d in hom]), "{:d}")
        else:
            row(lab, lambda d, key=key: GNN[d][key]["test_eval_acc_mean"], "0")
    L.append(r"\midrule")
    row(r"LLM $f_0$ (text only)", lambda d: A7[d]["acc_f0"], f"{mean([A7[d]['tokens_f0'] for d in DS]):.0f}")
    row(r"LLM GraphICL-style", lambda d: A7[d]["acc_f1_full"]["graphicl"],
        f"{mean([A7[d]['tokens_f1_only']['graphicl'] for d in DS]):.0f}")
    row(r"$f_0$ + GCN (prob.\ average)", lambda d: A7[d]["llm_gnn"]["acc_f0_gcn_ensemble"],
        f"{mean([A7[d]['tokens_f0'] for d in DS]):.0f}")
    row(r"\textsc{NeedNbr}-TF@20\% $\to$GCN", lambda d: A7[d]["llm_gnn"]["acc_needNbr20_to_gcn"],
        f"{mean([A7[d]['tokens_f0'] for d in DS]):.0f}")
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    (OUT_T / "gnn.tex").write_text("\n".join(L), encoding="utf-8")
    best_low = max(GNN[d][k]["test_eval_acc_mean"] for d in ["cornell"] for k in GNN[d])
    num("mlplow", pct(mean([GNN[d]["mlp"]["test_eval_acc_mean"] for d in LOW_H])))
    num("gcnlow", pct(mean([GNN[d]["gcn"]["test_eval_acc_mean"] for d in LOW_H])))
    num("gcnhom", pct(mean([GNN[d]["gcn"]["test_eval_acc_mean"] for d in hom])))
    num("fzerohom", pct(mean([A7[d]["acc_f0"] for d in hom])))
    num("fzerolow", pct(mean([A7[d]["acc_f0"] for d in LOW_H])))
    _ = best_low
    # distillation
    D = {d: json.load(open(RES / "distill" / M7 / f"{d}.json", encoding="utf-8")) for d in ["arxiv", "history", "computer",
                                                                          "amazon_ratings"]}
    L = [r"\begin{table}[t]", r"\centering\small",
         r"\caption{Distillation: a GCN trained only on Qwen2.5-7B pseudo-labels of 2{,}000 nodes (no ground-truth "
         r"labels for training or model selection). PL: pseudo-label accuracy; GCN: test accuracy (\%, mean of "
         r"3 seeds); Tok: LLM prompt tokens (millions).}\label{tab:distill}",
         r"\setlength{\tabcolsep}{2.4pt}", r"\begin{tabular}{@{}lccccccccc@{}}", r"\toprule",
         r" & \multicolumn{3}{c}{f0 only} & \multicolumn{3}{c}{\textsc{NeedNbr} 20\%} & "
         r"\multicolumn{3}{c}{$f_1$ for all}\\",
         r"\cmidrule(lr){2-4}\cmidrule(lr){5-7}\cmidrule(lr){8-10}",
         r"Graph & PL & GCN & Tok & PL & GCN & Tok & PL & GCN & Tok\\", r"\midrule"]
    for d, x in D.items():
        cells = []
        for k in ("f0 only", "NeedNbr routing (20%)", "graph-augmented for all"):
            cells += [f"{pct(x[k]['pseudo_label_acc']):.1f}", f"{pct(x[k]['gcn_test_acc']):.1f}",
                      f"{x[k]['llm_prompt_tokens'] / 1e6:.2f}"]
        L.append(f"{NAME[d]} & " + " & ".join(cells) + r"\\")
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    (OUT_T / "distill.tex").write_text("\n".join(L), encoding="utf-8")


# =================================================================== misc numbers quoted in the text
def misc_numbers():
    num("histtopclass", 0, "{:d}")
    r = raw(M7, "history")
    n = r["n_test"]
    pred = r["P0"][:n].argmax(1)
    num("histcatchall", pct((pred == 9).mean()), "{:.0f}")
    num("histcatchallthree", pct((raw(M3, "history")["P0"][:r["n_test"]].argmax(1) == 9).mean()), "{:.0f}")
    num("histtrue", pct((r["y"][:n] == 9).mean()), "{:.1f}")
    num("actorfzero", pct(A7["actor"]["acc_f0"]))
    num("actorfone", pct(A7["actor"]["acc_f1_full"]["random"]))
    num("actorfonethirtytwo", pct(A32["actor"]["acc_f1_full"]["random"]))
    num("actorfzerothirtytwo", pct(A32["actor"]["acc_f0"]))
    num("histfzero", pct(A7["history"]["acc_f0"]))
    num("histfone", pct(A7["history"]["acc_f1_full"]["random"]))
    # dataset-level homophily vs effect (7B, 16 graphs)
    hadj = [A7[d]["adj_homophily"] for d in DS]
    dl = [A7[d]["acc_f1_full"]["random"] - A7[d]["acc_f0"] for d in DS]
    num("homrho", spearmanr(hadj, dl).correlation, "{:.2f}")
    num("homp", spearmanr(hadj, dl).pvalue, "{:.2f}")
    # pooled local-homophily [0.75,1] bin, 7B and 32B, all graphs they ran
    for m, A, tag in [(M7, A7, "seven"), (M32, A32, "thirtytwo")]:
        n_, s0, s1 = 0, 0.0, 0.0
        for d, a in A.items():
            for row in a["by_local_homophily"]:
                if row["bin"] == "[0.75,1.00]":
                    n_ += row["n"]
                    s0 += row["acc_f0"] * row["n"]
                    s1 += row["acc_f1_random"] * row["n"]
        num(f"hihomzero{tag}", pct(s0 / n_))
        num(f"hihomone{tag}", pct(s1 / n_))
        num(f"hihomn{tag}", n_, "{:,d}")
    r = raw(M7, "actor")
    n = r["n_test"]
    p1 = np.asarray(r["P1"]["random"])[:n].argmax(1)
    top = np.bincount(p1).argmax()
    num("actortopclasstrue", pct((r["y"][:n] == top).mean()), "{:.0f}")
    num("actortopclassletter", "ABCDE"[top], "{}")
    num("histcatchallall", 100 * (raw(M7, "history")["y"] == 9).mean() if False else 100 * float(
        (pickle.load(open(PROC / "history.pkl", "rb"))["y"] == 9).mean()), "{:.1f}")
    # dataset-level decision: val-tuned threshold (NeedNbr-auto) vs text only, and the per-dataset best of f0 / f1
    for A, tag in [(A7, ""), (AQ, "q")]:
        au = [A[d]["auto_threshold"]["NeedNbr-auto (NB, val-tuned threshold)"]["acc"] for d in DS]
        f0 = [A[d]["acc_f0"] for d in DS]
        num(f"auto{tag}acc", pct(mean(au)))
        num(f"auto{tag}p", wilcoxon(au, f0).pvalue, "{:.2f}")
        num(f"auto{tag}wins", sum(a > b + 1e-9 for a, b in zip(au, f0)), "{:d}")
        num(f"auto{tag}losses", sum(a < b - 1e-9 for a, b in zip(au, f0)), "{:d}")
        num(f"dsbest{tag}", pct(mean([max(A[d]["acc_f0"], A[d]["acc_f1_full"]["cns"]) for d in DS])))
    # calibration
    num("calibfzero", pct(mean([A7[d].get("acc_f0_calibrated") for d in DS])))
    # self-assessment cost: one extra short call per node -> report tokens? (not stored) skip
    # number of LLM calls overall
    calls = sum(A[d]["llm_stats"]["option_calls"] for A, dl_ in [(A7, DS), (AQ, DS), (A3, SCALE_DS),
                                                                 (A32, SCALE_DS)] for d in dl_)
    num("ncalls", calls / 1e3, "{:.0f}")
    # validation sizes (the learned router and the auto threshold are fitted on val_gain)
    nv_small = [A7[d]["n_val"] for d in DS if A7[d]["n_val"] < 300]
    num("nvalmin", min(nv_small), "{:d}")
    num("nvalmax", max(nv_small), "{:d}")
    # cost view (ii): NeedNbr-TF @20% incl. the neighbours' text-only calls
    c2 = {d: next(c["tokens_per_node"] for c in A7[d]["curves"][f"cns_costview2|{TF}"]["curve"]
                  if abs(c["budget"] - 0.2) < 1e-9) for d in DS}
    lo, hi = min(c2, key=c2.get), max(c2, key=c2.get)
    num("costtwomin", c2[lo] / 1e3, "{:.1f}k")
    num("costtwomax", c2[hi] / 1e3, "{:.1f}k")
    num("costtwominds", NAME[lo], "{}")
    num("costtwomaxds", NAME[hi], "{}")
    # paired significance per graph (exact McNemar, p < 0.05), NeedNbr-TF @20% (CNS) vs alternatives
    def nsig(A, key, sign):
        return sum(1 for d in DS if A[d]["significance_needNbr20"][key]["p_mcnemar"] < 0.05
                   and np.sign(A[d]["significance_needNbr20"][key]["delta_acc"]) == sign)
    for A, tag in [(A7, ""), (AQ, "q")]:
        k = "random router @20% (seeded draw)"
        num(f"sigrnd{tag}", nsig(A, k, 1) + nsig(A, k, -1), "{:d}")
        num(f"sigfzerobetter{tag}", nsig(A, "f0 (text only)", 1), "{:d}")
        num(f"sigfzeroworse{tag}", nsig(A, "f0 (text only)", -1), "{:d}")
        num(f"sigcasc{tag}", nsig(A, "NeedNbr-TF@20% -> GCN", -1), "{:d}")
    # which helped / hurt nodes fall into the routed 20% (CNS neighbours, 7B)
    gb = [A7[d]["gain_breakdown"]["cns"] for d in DS]
    num("helpedinrouted", pct(mean([x["helped_in_routed20"] for x in gb])), "{:.0f}")
    num("hurtinrouted", pct(mean([x["hurt_in_routed20"] for x in gb])), "{:.0f}")
    num("helpedcns", pct(mean([x["helped"] for x in gb])))
    num("hurtcns", pct(mean([x["hurt"] for x in gb])))
    # static (non-routing) combination of f0 and f1 for every node
    num("staticens", pct(mean([A7[d]["static_baselines"]["ensemble f0+f1(CNS)"]["acc"] for d in DS])))
    num("staticmaxc", pct(mean([A7[d]["static_baselines"]["max-confidence f0 vs f1(CNS)"]["acc"] for d in DS])))
    # profile of the routed 20%
    prof = [A7[d]["routed_profile"]["f0_correct"] for d in DS]
    num("routedfzero", pct(mean([x["routed"] for x in prof])), "{:.0f}")
    num("notroutedfzero", pct(mean([x["not_routed"] for x in prof])), "{:.0f}")
    # GCN alone and the best GNN, on the same 1,000 test nodes
    num("sageacc", pct(mean([GNN[d]["sage"]["test_eval_acc_mean"] for d in DS])))


OVERVIEW_CASES = [("cora", 1848, "helped"), ("cora", 2666, "hurt")]  # real test nodes, Qwen2.5-7B, random nbrs


def overview_case(ds, node, strat="random"):
    import re
    import textwrap
    r = raw(M7, ds)
    p = pickle.load(open(PROC / f"{ds}.pkl", "rb"))
    i = int(np.where(r["targets"] == node)[0][0])
    names = [x.replace("_", " ") for x in p["label_names"]]
    title = p["texts"][node].split(" : ")[0].strip()
    title = re.sub(r"^[A-Z][a-z]+\.\s+", "", title)  # drop a stray author surname left by the raw-text parser
    nb = r["sel"][strat][node]
    return dict(title=textwrap.fill(title[0].upper() + title[1:], 34), y=int(r["y"][i]), p0=np.asarray(r["P0"][i]),
                p1=np.asarray(r["P1"][strat])[i], nby=[int(p["y"][u]) for u in nb], names=names,
                A=float(r["feats"]["est"][i]["A"]))


def fig_overview():
    from matplotlib.offsetbox import OffsetImage, AnnotationBbox
    from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle
    ICON = OUT_F / "icons"
    W, H = 100.0, 40.0
    fig = plt.figure(figsize=(TEXT_W, TEXT_W * H / W))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, W)
    ax.set_ylim(0, H)
    ax.set_facecolor("white")
    ax.axis("off")

    def icon(name, x, y, size=3.2, color=INK):
        im = plt.imread(ICON / f"{name}.png").copy()
        im[..., :3] = matplotlib.colors.to_rgb(color)
        zoom = size / W * TEXT_W * 72 / im.shape[1]  # size in data units -> points
        ax.add_artist(AnnotationBbox(OffsetImage(im, zoom=zoom, dpi_cor=False), (x, y), frameon=False, zorder=7))

    def box(x, y, w, h, fc, ec=None, lw=0.8, r=1.2, shadow=True, z=2):
        if shadow:
            ax.add_patch(FancyBboxPatch((x + 0.35, y - 0.45), w, h, boxstyle=f"round,pad=0,rounding_size={r}",
                                        fc="#00000014", ec="none", zorder=z - 0.5))
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={r}", fc=fc,
                                    ec=ec or fc, lw=lw, zorder=z))

    def arrow(p, q, col=INK2, rad=0.0, lw=1.0, ls="-"):
        ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=8, color=col, lw=lw, ls=ls,
                                     connectionstyle=f"arc3,rad={rad}", shrinkA=0, shrinkB=0, zorder=4))

    def txt(x, y, s, size=6.4, col=INK, w="normal", ha="center", va="center", style="normal", z=8):
        ax.text(x, y, s, fontsize=size, color=col, weight=w, ha=ha, va=va, style=style, zorder=z, linespacing=1.15)

    def badge(x, y, ok):
        ax.add_patch(Circle((x, y), 0.95, fc=AQUA if ok else ORANGE, ec="white", lw=0.6, zorder=6))
        icon("check" if ok else "cross", x, y, 1.1, "white")

    # ------------------------------------------------------------------ panel (a): two real nodes
    box(0.4, 0.6, 33.2, 38.8, PANEL, shadow=False, z=0)
    txt(1.6, 37.6, "(a) Same five neighbours, opposite outcomes", 7.4, w="bold", ha="left")
    txt(1.6, 35.4, "two Cora test papers, Qwen2.5-7B, five random 1-hop neighbours", 5.9, INK2, ha="left",
        style="italic")
    for row, (ds, node, kind) in enumerate(OVERVIEW_CASES):
        c = overview_case(ds, node)
        y0 = 18.5 - 15.8 * row  # bottom of the card
        box(1.4, y0, 31.2, 15.0, "white", lw=0)
        # ego graph: target in the middle, neighbours coloured by whether they share its label
        gx, gy = 7.2, y0 + 6.4
        angs = np.linspace(np.pi / 2, np.pi / 2 + 2 * np.pi, len(c["nby"]), endpoint=False)
        for a, yn in zip(angs, c["nby"]):
            nx, ny = gx + 4.2 * np.cos(a), gy + 4.2 * np.sin(a)
            ax.plot([gx, nx], [gy, ny], color=GRAY2, lw=0.9, zorder=3)
            same = yn == c["y"]
            ax.add_patch(Circle((nx, ny), 1.1, fc=BLUE if same else GRAY2, ec="white", lw=0.8, zorder=5))
        ax.add_patch(Circle((gx, gy), 2.3, fc="white", ec=INK, lw=1.0, zorder=5))
        icon("document", gx, gy, 2.4)
        txt(gx, y0 + 0.75, f"ambiguity $A$ = {c['A']:.2f}", 5.5, INK2)
        tag = "HELPED" if kind == "helped" else "HURT"
        ax.add_patch(FancyBboxPatch((2.1, y0 + 12.6), 6.0, 1.7, boxstyle="round,pad=0,rounding_size=0.8",
                                    fc=AQUA if kind == "helped" else ORANGE, ec="none", zorder=6))
        txt(5.1, y0 + 13.45, tag, 5.6, "white", w="bold")
        # title and true label of the target
        txt(13.4, y0 + 14.1, c["title"], 6.0, INK, ha="left", va="top", style="italic")
        txt(13.4, y0 + 9.2, "true label:", 5.6, INK2, ha="left")
        txt(18.6, y0 + 9.2, c["names"][c["y"]], 5.9, BLUE, ha="left", w="bold")
        # probability of the true label (blue) and of the strongest wrong label (orange)
        for j, (lab, P) in enumerate([("text only $f_0$", c["p0"]), ("+ neighbours $f_1$", c["p1"])]):
            yl = y0 + 7.0 - 3.7 * j
            yb = yl - 1.75
            pred = int(P.argmax())
            wrong = int(np.argsort(P)[-1] if pred != c["y"] else np.argsort(P)[-2])
            right = pred == c["y"]
            short = c["names"][pred].replace("Probabilistic", "Prob.").replace("Reinforcement", "Reinf.")
            txt(13.4, yl, lab, 5.6, INK2, ha="left")
            txt(29.6, yl, f"{chr(ord('A') + pred)}: {short}", 5.6, AQUA if right else ORANGE, ha="right", w="bold")
            badge(31.3, yl - 0.85, right)
            bx, bw = 13.4, 15.2
            ax.add_patch(FancyBboxPatch((bx, yb - 0.6), bw, 1.2, boxstyle="round,pad=0,rounding_size=0.3",
                                        fc=PANEL2, ec="none", zorder=4))
            ax.add_patch(FancyBboxPatch((bx, yb - 0.6), bw * P[c["y"]], 1.2,
                                        boxstyle="round,pad=0,rounding_size=0.3", fc=BLUE, ec="none", zorder=5))
            ax.add_patch(FancyBboxPatch((bx + bw * (1 - P[wrong]), yb - 0.6), bw * P[wrong], 1.2,
                                        boxstyle="round,pad=0,rounding_size=0.3", fc=ORANGE, ec="none", zorder=5))
            txt(bx + bw + 0.4, yb, f"{P[pred]:.2f}", 5.3, INK2, ha="left")
    # legend for panel (a)
    ax.add_patch(Circle((2.4, 1.45), 0.55, fc=BLUE, ec="white", zorder=6))
    txt(3.3, 1.45, "neighbour shares the target's label", 5.3, INK2, ha="left")
    ax.add_patch(FancyBboxPatch((20.0, 1.0), 1.4, 0.9, boxstyle="square,pad=0", fc=BLUE, ec="none", zorder=6))
    txt(21.8, 1.45, "p(true)", 5.3, INK2, ha="left")
    ax.add_patch(FancyBboxPatch((25.4, 1.0), 1.4, 0.9, boxstyle="square,pad=0", fc=ORANGE, ec="none", zorder=6))
    txt(27.2, 1.45, "p(wrong)", 5.3, INK2, ha="left")

    # ------------------------------------------------------------------ panel (b): pipeline
    X0 = 34.6
    box(X0, 0.6, 41.2, 38.8, PANEL, shadow=False, z=0)
    txt(X0 + 1.2, 37.6, "(b) Node-adaptive use of structure", 7.4, w="bold", ha="left")
    txt(X0 + 1.2, 35.4, "the router sees only text-only answer distributions", 5.9, INK2, ha="left", style="italic")

    def step(cx, cy, n, name, title, sub, col=INK, ring=GRAY2):
        ax.add_patch(Circle((cx, cy), 3.3, fc="white", ec=ring, lw=1.0, zorder=5))
        icon(name, cx, cy, 3.4, col)
        ax.add_patch(Circle((cx - 2.6, cy + 2.4), 0.95, fc=INK, ec="white", lw=0.5, zorder=6))
        txt(cx - 2.6, cy + 2.4, str(n), 5.4, "white", w="bold")
        txt(cx, cy - 4.4, title, 6.2, col, w="bold")
        txt(cx, cy - 6.1, sub, 5.5, INK2)

    sy = 27.6
    step(X0 + 5.6, sy, 1, "document", "node text $X_v$", "16 TAGs, ≤1k test nodes each")
    arrow((X0 + 9.2, sy), (X0 + 12.6, sy))
    step(X0 + 16.2, sy, 2, "llm", "text-only $f_0$", f"{NUM['tokzero']} tok / node")
    arrow((X0 + 19.8, sy), (X0 + 23.0, sy))
    # router card with the SIG definition
    box(X0 + 23.0, 18.4, 17.6, 15.4, "white", ec=BLUE, lw=1.0)
    ax.add_patch(Circle((X0 + 26.4, 30.6), 2.2, fc=BLUE_L, ec="none", zorder=5))
    icon("router", X0 + 26.4, 30.6, 2.6, BLUE)
    ax.add_patch(Circle((X0 + 24.2, 32.5), 0.95, fc=INK, ec="white", lw=0.5, zorder=6))
    txt(X0 + 24.2, 32.5, "3", 5.4, "white", w="bold")
    txt(X0 + 29.3, 31.4, "router $s(v)$", 6.4, BLUE, w="bold", ha="left")
    txt(X0 + 29.3, 29.6, "keep the top $b$", 5.5, INK2, ha="left")
    txt(X0 + 31.8, 26.4, r"$\mathrm{SIG}(v)=I(Y_v;X_{\mathcal{N}(v)}\,|\,X_v)$", 6.3, INK)
    txt(X0 + 31.8, 24.2, "ambiguity $A$ · discrim. $D$", 5.4, INK2)
    txt(X0 + 31.8, 22.4, r"naive Bayes with $\Pi$ · self-assess.", 5.4, INK2)
    txt(X0 + 31.8, 20.6, "learned · 12 routers in total", 5.4, INK2)
    # three exits
    by = 3.0
    cards = [(X0 + 1.2, "neighbours", BLUE, "$f_1$ + 5 neighbours", f"{NUM['tokcns']} tok (CNS)",
              [(f"helped {NUM['helpedrandom']}%", BLUE), (f"hurt {NUM['hurtrandom']}%", ORANGE)]),
             (X0 + 14.6, "gnn", AQUA, "GCN on 20 labels/class", "0 LLM tok",
              [(f"cascade {NUM['cascacc']}%", AQUA), (f"vs. {NUM['avgfzero']}% $f_0$", INK2)]),
             (X0 + 28.0, "check", INK2, "keep $\\arg\\max p^0_v$", "no extra cost",
              [("unselected nodes", INK2), ("(80% at $b$=20%)", INK2)])]
    for i, (cx, ic, col, t1, t2, lines) in enumerate(cards):
        box(cx, by, 12.0, 12.6, "white", ec=col if col != INK2 else GRAY2, lw=0.9)
        ax.add_patch(Circle((cx + 6.0, by + 9.7), 2.0, fc={BLUE: BLUE_L, AQUA: AQUA_L}.get(col, PANEL2), ec="none",
                            zorder=5))
        icon(ic, cx + 6.0, by + 9.7, 2.5, col)
        txt(cx + 6.0, by + 6.5, t1, 5.8, col if col != INK2 else INK, w="bold")
        txt(cx + 6.0, by + 4.9, t2, 5.4, INK2)
        for j, (s, cc) in enumerate(lines):
            txt(cx + 6.0, by + 3.0 - 1.6 * j, s, 5.5, cc, w="bold" if j == 0 else "normal")
    arrow((X0 + 26.0, 18.4), (X0 + 7.2, by + 12.6), col=BLUE, rad=0.12)
    arrow((X0 + 31.8, 18.4), (X0 + 20.6, by + 12.6), col=AQUA, rad=0.06)
    arrow((X0 + 37.6, 18.4), (X0 + 34.0, by + 12.6), col=GRAY, rad=-0.05)
    for x_, y_, t_, c_ in [(X0 + 11.0, 18.9, "top-$b$ nodes", BLUE), (X0 + 25.6, 16.6, "or", AQUA),
                           (X0 + 38.6, 17.0, "rest", GRAY)]:
        ax.text(x_, y_, t_, fontsize=5.6, color=c_, style="italic", ha="center", va="center", zorder=9,
                bbox=dict(boxstyle="round,pad=0.2", fc=PANEL, ec="none"))

    # ------------------------------------------------------------------ panel (c): what the averages hide
    X1 = 76.6
    box(X1, 0.6, 23.0, 38.8, PANEL, shadow=False, z=0)
    txt(X1 + 1.2, 37.6, "(c) What averages hide", 7.4, w="bold", ha="left")
    txt(X1 + 1.2, 35.4, "Qwen2.5-7B, mean over 16 TAGs", 5.9, INK2, ha="left", style="italic")
    # node outcomes with five random neighbours (from raw per-node predictions)
    shares = np.zeros(4)
    for d in DS:
        r = raw(M7, d)
        n = r["n_test"]
        yv = r["y"][:n]
        c0 = r["P0"][:n].argmax(1) == yv
        c1 = np.asarray(r["P1"]["random"])[:n].argmax(1) == yv
        shares += np.array([(c0 & c1).mean(), (~c0 & c1).mean(), (c0 & ~c1).mean(), (~c0 & ~c1).mean()]) / len(DS)
    bx, bw, byy = X1 + 1.4, 20.2, 28.0
    labs = [("both right", GRAY2, INK), ("helped", BLUE, "white"), ("hurt", ORANGE, "white"),
            ("both wrong", GRAY3, INK)]
    xc = bx
    for s, (lab, col, tc) in zip(shares, labs):
        ax.add_patch(FancyBboxPatch((xc, byy), bw * s, 3.0, boxstyle="square,pad=0", fc=col, ec="white", lw=0.8,
                                    zorder=5))
        if s > 0.08:
            txt(xc + bw * s / 2, byy + 1.5, f"{100 * s:.0f}%", 5.6, tc, w="bold")
        xc += bw * s
    txt(bx, byy + 4.3, "test-node outcomes with $f_1$", 5.8, INK2, ha="left")
    lx = bx
    for lab, col, _ in labs:
        ax.add_patch(FancyBboxPatch((lx, byy - 2.2), 1.0, 1.0, boxstyle="square,pad=0", fc=col, ec="none", zorder=5))
        txt(lx + 1.4, byy - 1.7, lab, 5.0, INK2, ha="left")
        lx += {"both right": 6.0, "helped": 4.6, "hurt": 3.6, "both wrong": 5.0}[lab]
    # accuracy lollipops, every value from numbers.tex
    rows = [("text only $f_0$", float(NUM["avgfzero"]), INK2),
            ("$f_1$ for all nodes", float(NUM["avgrandom"]), BLUE),
            ("NeedNbr @20%", float(NUM["avgtftwozero"]), BLUE),
            ("oracle @20%", float(NUM["avgorctwozero"]), VIOLET),
            ("cascade to GCN", float(NUM["cascacc"]), AQUA)]
    lo, hi = 46.0, 62.0
    ax0, ax1 = X1 + 10.2, X1 + 21.8
    sx = lambda v: ax0 + (v - lo) / (hi - lo) * (ax1 - ax0)  # noqa: E731
    txt(bx, 22.4, "mean accuracy (%)", 5.8, INK2, ha="left")
    for v in (48, 52, 56, 60):
        ax.plot([sx(v)] * 2, [4.0, 20.8], color="white", lw=0.9, zorder=3)
        txt(sx(v), 3.0, f"{v}", 5.2, INK2)
    f0x = sx(float(NUM["avgfzero"]))
    ax.plot([f0x] * 2, [4.0, 20.8], color=INK2, lw=0.7, ls=(0, (2, 2)), zorder=4)
    for j, (lab, v, col) in enumerate(rows):
        yy = 19.0 - 3.5 * j
        txt(bx, yy, lab, 5.6, col if col != INK2 else INK, ha="left", w="bold" if col in (AQUA, VIOLET) else "normal")
        ax.plot([f0x, sx(v)], [yy, yy], color=col, lw=1.4, zorder=5, solid_capstyle="butt")
        ax.add_patch(Circle((sx(v), yy), 0.75, fc=col, ec="white", lw=0.6, zorder=6))
        txt(sx(v) + (1.2 if v >= float(NUM["avgfzero"]) else -1.2), yy + 1.25, f"{v:.1f}", 5.2, col,
            ha="center")
    save(fig, "overview")


def write_numbers():
    NUM["cascgain"] = "{:+.1f}".format(float(NUM["cascacc"]) - float(NUM["avgfzero"]))
    L = ["% auto-generated by paper/make_figures.py -- do not edit"]
    for k, v in sorted(NUM.items()):
        assert k.isalpha(), k
        L.append(f"\\newcommand{{\\N{k}}}{{{v}}}")
    (ROOT / "paper" / "numbers.tex").write_text("\n".join(L) + "\n", encoding="utf-8")


if __name__ == "__main__":
    dataset_table()
    main_table()
    fig_helped_hurt()
    fig_ambiguity()
    fig_budget()
    router_table()
    fig_scale_homophily()
    scale_table()
    fig_prior_shift()
    fig_tokens()
    selection_table()
    gnn_table()
    misc_numbers()
    write_numbers()
    fig_overview()
    print(f"wrote {len(list(OUT_F.glob('*.pdf')))} figures, {len(list(OUT_T.glob('*.tex')))} tables, "
          f"{len(NUM)} numbers")
