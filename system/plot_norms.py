import h5py, numpy as np, os, glob, csv
os.environ["MPLBACKEND"] = "Agg"
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(_SCRIPT_DIR, "..", "results")
PLOTS_DIR = os.path.join(RESULTS_DIR, "plots")

os.makedirs(PLOTS_DIR, exist_ok=True)

plt.rcParams.update({
    "axes.grid": True,
    "grid.alpha": 0.3,
    "font.size": 11,
})

def get_val(parts, prefix, default="?"):
    for p in parts:
        if p.startswith(prefix):
            return p[len(prefix):]
    return default

def group_files(files):
    groups = {}
    for fpath in files:
        fname = os.path.basename(fpath)
        parts = fname.split("_")
        try:
            cc = get_val(parts, "cc")
            nmal = get_val(parts, "nmal")
            rfake = get_val(parts, "rfake")
            atk = get_val(parts, "atk-")
            key = (cc, nmal, rfake, atk)
        except Exception:
            key = fname
        groups.setdefault(key, []).append(fpath)
    return groups

norms_files = sorted(glob.glob(os.path.join(RESULTS_DIR, "norms_*.csv")))

print(f"Found {len(norms_files)} norms CSVs\n")

if not norms_files:
    print("No norms CSV files found in results/")
    exit(0)

norms_groups = group_files(norms_files)

colors = ["#2196F3", "#FF5722", "#4CAF50", "#9C27B0", "#FF9800", "#00BCD4"]

for i, (key, fpaths) in enumerate(norms_groups.items()):
    cc, nmal, rfake, atk = key
    label = f"cc={cc}, nmal={nmal}, atk={atk}"
    color = colors[i % len(colors)]

    print(f"\n=== {label} ({len(fpaths)} run(s)) ===")

    all_rounds_data = {}
    for fpath in fpaths:
        with open(fpath) as f:
            reader = csv.DictReader(f)
            for row in reader:
                r = int(row['round'])
                cid = int(row['client_id'])
                all_rounds_data.setdefault(r, {})
                all_rounds_data[r][cid] = {
                    'cos': float(row['cosine_score']),
                    'l2': float(row['l2_norm']),
                    'l3': float(row['l3_norm']),
                    'mal': int(row['is_malicious']),
                    'thr_cos': float(row['threshold_cos']),
                    'thr_l3': float(row['threshold_l3']),
                }

    if not all_rounds_data:
        continue

    rounds = sorted(all_rounds_data.keys())
    client_ids = sorted({cid for rdata in all_rounds_data.values() for cid in rdata})

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle(f"Norms - {label}", fontsize=14, fontweight="bold")

    ax_cos = axes[0, 0]
    ax_l2 = axes[0, 1]
    ax_l3 = axes[1, 0]
    ax_scatter = axes[1, 1]

    for cid in client_ids:
        xs, cos_vals, l2_vals, l3_vals, mal = [], [], [], [], False
        for r in rounds:
            if cid in all_rounds_data[r]:
                d = all_rounds_data[r][cid]
                xs.append(r)
                cos_vals.append(d['cos'])
                l2_vals.append(d['l2'])
                l3_vals.append(d['l3'])
                mal = d['mal']

        style = '--' if mal else '-'
        lbl = f"C{cid}{'*' if mal else ''}"
        alpha = 0.8 if mal else 0.5
        lw = 1.5 if mal else 1.0

        ax_cos.plot(xs, cos_vals, style, color=color, label=lbl, alpha=alpha, linewidth=lw)
        ax_l2.plot(xs, l2_vals, style, color=color, label=lbl, alpha=alpha, linewidth=lw)
        ax_l3.plot(xs, l3_vals, style, color=color, label=lbl, alpha=alpha, linewidth=lw)

    ax_cos.set_title("Cosine Similarity per Client")
    ax_cos.set_xlabel("Round")
    ax_cos.legend(fontsize=6, ncol=5)

    ax_l2.set_title("L2 Norm per Client")
    ax_l2.set_xlabel("Round")
    ax_l2.legend(fontsize=6, ncol=5)

    ax_l3.set_title("L3 Norm per Client")
    ax_l3.set_xlabel("Round")
    ax_l3.legend(fontsize=6, ncol=5)

    last_r = rounds[-1]
    lr_data = all_rounds_data[last_r]
    for cid, d in lr_data.items():
        marker = 'x' if d['mal'] else 'o'
        s = 80 if d['mal'] else 40
        ax_scatter.scatter(d['cos'], d['l3'], c=color, marker=marker, s=s, label=f"C{cid}{'*' if d['mal'] else ''}")
    ax_scatter.axvline(d['thr_cos'], color='gray', linestyle='--', alpha=0.5, label=f"Threshold Cos ({d['thr_cos']:.3f})")
    ax_scatter.axhline(d['thr_l3'], color='orange', linestyle='--', alpha=0.5, label=f"Threshold L3 ({d['thr_l3']:.3f})")
    ax_scatter.set_title(f"Cosine vs L3 (Round {last_r})")
    ax_scatter.set_xlabel("Cosine Similarity")
    ax_scatter.set_ylabel("L3 Norm")
    ax_scatter.legend(fontsize=7)

    plt.tight_layout()
    safe_name = f"norms_Cifar10_FedAvg_cc{cc}_nmal{nmal}"
    figname = os.path.join(PLOTS_DIR, safe_name + ".png")
    fig.savefig(figname, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {figname}")

print(f"\nAll plots saved to: {PLOTS_DIR}")
