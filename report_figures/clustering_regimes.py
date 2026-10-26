"""The clustering grid and the three regime centroids.

(a) k-means and (b) Ward: ARI stability and silhouette against k on 3, 5 and 8 PCs (ledger nb05.grid, as
notebook 08 section 8.5 draws 08_unsupervised_summary.png); (c) DBSCAN: share of days labelled noise against eps,
with the number of clusters found (same ledger grid); (d) the regime centroids: mean of the within-day-normalised
shapes by cluster (data/processed/unsupervised/daily_shapes.parquet joined to cluster_labels.parquet, the
`shapes.groupby(labels).mean()` of notebook 05 section 5.4), legend values from ledger nb05.centroid_summary.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
from matplotlib.lines import Line2D

try:
    from . import style as S
except ImportError:  # run as a script
    import style as S

REGIME_COLOR = {0: "#1f77b4", 1: "#ff7f0e", 2: "#2ca02c"}   # matplotlib C0..C2, as notebook 05 colours clusters
PC_STYLE = {3: ":", 5: "--", 8: "-"}
ARI_C, SIL_C = "#1d3557", "#e8833a"


def _centroids() -> pd.DataFrame:
    d = S.ROOT / "data" / "processed" / "unsupervised"
    shapes = pd.read_parquet(d / "daily_shapes.parquet")
    labels = pd.read_parquet(d / "cluster_labels.parquet")
    shapes["date"] = pd.to_datetime(shapes["date"])
    labels["date"] = pd.to_datetime(labels["date"])
    m = shapes.merge(labels, on=["zone", "date"], how="inner", validate="one_to_one")
    assert len(m) == len(shapes) == len(labels)
    hours = [c for c in shapes.columns if c not in ("zone", "date")]
    return m.groupby("cluster")[hours].mean()


def draw() -> Path:
    led = S.ledger()["nb05"]
    g = pd.DataFrame(led["grid"])
    fig, _ax = S.figure(3.3)
    _ax.remove()
    gs = fig.add_gridspec(2, 3, height_ratios=[1, 1.05], hspace=0.62, wspace=0.36, left=0.1, right=0.95, top=0.97, bottom=0.1)
    ax_k, ax_w, ax_d = (fig.add_subplot(gs[0, i]) for i in range(3))
    bottom = gs[1, :].subgridspec(1, 2, width_ratios=[2.35, 1], wspace=0.02)
    ax_c = fig.add_subplot(bottom[0, 0])
    ax_leg = fig.add_subplot(bottom[0, 1])
    ax_leg.axis("off")

    # (a), (b): ARI stability and silhouette per algorithm
    for ax, algo, letter, name in [(ax_k, "kmeans", "a", "k-means"), (ax_w, "ward", "b", "Ward")]:
        for pcs, ls in PC_STYLE.items():
            sub = g[(g.algorithm == algo) & (g.n_components == pcs)].sort_values("n_clusters")
            ax.plot(sub.n_clusters, sub.ari_stability, ls=ls, marker="o", ms=2.8, lw=1.1, color=ARI_C)
            ax.plot(sub.n_clusters, sub.silhouette, ls=ls, marker="s", ms=2.6, lw=1.1, color=SIL_C)
        ax.axhline(0.6, color="#7f7f7f", ls="--", lw=0.8)
        ax.set_ylim(0.15, 1.05)
        ax.set_xticks([3, 4, 5, 6, 7])
        ax.set_xlabel("k (clusters)")
        ax.set_title(name, fontsize=8, weight="normal", loc="left", pad=2)
        S.panel_label(ax, letter)
    ax_k.set_ylabel("score (unitless)")
    ax_k.text(7, 0.62, "goal G4: ARI ≥ 0.6", fontsize=6.5, color="#555", ha="right", va="bottom")
    handles = [Line2D([], [], color=ARI_C, marker="o", ms=2.8, lw=1.1, label="ARI stability"),
               Line2D([], [], color=SIL_C, marker="s", ms=2.6, lw=1.1, label="silhouette")]
    handles += [Line2D([], [], color="0.3", ls=ls, lw=1.1, label=f"{p} PCs") for p, ls in PC_STYLE.items()]
    ax_w.legend(handles=handles, loc="upper center", ncol=2, fontsize=6.3, handlelength=2.0,
                columnspacing=0.8, borderaxespad=0.2, labelspacing=0.25)

    # (c): DBSCAN noise share and clusters found
    db = g[g.algorithm == "dbscan"].sort_values(["n_components", "param"])
    for pcs, ls in PC_STYLE.items():
        sub = db[db.n_components == pcs]
        ax_d.plot(sub.param, 100 * sub.noise_fraction, ls=ls, marker="s", ms=2.8, lw=1.1, color="#2ca02c",
                  label=f"{pcs} PCs")
        for _, r in sub.iterrows():
            ax_d.annotate(f"k={int(r.n_clusters)}", (r.param, 100 * r.noise_fraction), textcoords="offset points",
                          xytext=(3, -7 if pcs == 3 else 2), fontsize=6)
    ax_d.set_xlim(0.25, 0.445)
    ax_d.set_ylim(-5, 34)
    ax_d.set_xlabel("eps (radius in PCA space)")
    ax_d.set_ylabel("days labelled noise (%)")
    ax_d.set_title("DBSCAN", fontsize=8, weight="normal", loc="left", pad=2)
    ax_d.legend(loc="upper right", fontsize=6.3, handlelength=2.0, labelspacing=0.25, borderaxespad=0.2)
    S.panel_label(ax_d, "c")

    # (d): centroids
    cen = _centroids()
    summ = {r["cluster"]: r for r in led["centroid_summary"]}
    names = {int(k): v for k, v in led["regime_names"].items()}
    total = sum(r["n"] for r in summ.values())
    ax_c.axvspan(10, 16, color="#f2c14e", alpha=0.14, lw=0)
    ax_c.axvspan(16, 22, color="#c1121f", alpha=0.06, lw=0)
    hrs = [int(h) for h in cen.columns]
    for c, row in cen.iterrows():
        r = summ[int(c)]
        assert r["n"] == int((pd.read_parquet(S.ROOT / "data" / "processed" / "unsupervised"
                                              / "cluster_labels.parquet")["cluster"] == c).sum())
        ax_c.plot(hrs, row.values, lw=1.6, marker="o", ms=2.6, color=REGIME_COLOR[int(c)],
                  label=(f"cluster {c} = {names[int(c)]}\nn={r['n']:,} ({r['n'] / total:.0%}); "
                         f"belly {r['midday_belly']:.2f}, ramp {r['evening_ramp']:.2f}"))
    ax_c.text(13, 1.02, "belly h10–15", ha="center", va="bottom", fontsize=6.3, color="0.35")
    ax_c.text(19, 1.02, "ramp h16–21", ha="center", va="bottom", fontsize=6.3, color="0.35")
    ax_c.set_ylim(-0.03, 1.13)
    ax_c.set_xlim(-0.5, 23.5)
    ax_c.set_xticks(range(0, 24, 2))
    ax_c.set_xlabel("local hour of day (per-zone time zone)")
    ax_c.set_ylabel("net load, normalised\nwithin the day (0–1)")
    h, lab = ax_c.get_legend_handles_labels()
    ax_leg.legend(h, lab, loc="center left", bbox_to_anchor=(0.0, 0.5), fontsize=6.5, handlelength=1.6,
                  labelspacing=0.9, borderaxespad=0.0)
    S.panel_label(ax_c, "d")
    return S.save(fig, "clustering_regimes")


if __name__ == "__main__":
    print(draw())
