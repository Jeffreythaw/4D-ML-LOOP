from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
WORKER_DIR = PROJECT_ROOT / "reports" / "step_150b_workers"
REPORT_PATH = PROJECT_ROOT / "reports" / "step_150b_reconstructed_pool_backtest.txt"


def load_rows():
    rows = []
    for path in sorted(WORKER_DIR.glob("step150b_worker_*.jsonl")):
        with path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
    rows.sort(key=lambda r: (r["source"], r["target"], r["worker"]))
    return rows


def pct(part, total):
    if not total:
        return 0.0
    return part / total * 100.0


def main() -> int:
    rows = load_rows()

    lines = []
    lines.append("=" * 110)
    lines.append("STEP 150B — RECONSTRUCTED CANDIDATE POOL + RESIDUAL PROTOTYPE BACKTEST")
    lines.append("=" * 110)
    lines.append("Mode: REPORT ONLY")
    lines.append("ProductionMathChanged: NO")
    lines.append("ProductionSwitchRecommendedNow: NO")
    lines.append("")

    total = len(rows)
    baseline_hit_draws = sum(1 for r in rows if r["baseline_hit_count"] > 0)
    proto_hit_draws = sum(1 for r in rows if r["prototype_hit_count"] > 0)
    baseline_raw_hits = sum(int(r["baseline_hit_count"]) for r in rows)
    proto_raw_hits = sum(int(r["prototype_hit_count"]) for r in rows)

    pool10 = sum(1 for r in rows if r["pool_hit_top10"] > 0)
    pool25 = sum(1 for r in rows if r["pool_hit_top25"] > 0)
    pool50 = sum(1 for r in rows if r["pool_hit_top50"] > 0)
    pool100 = sum(1 for r in rows if r["pool_hit_top100"] > 0)

    lines.append("GLOBAL BACKTEST SUMMARY")
    lines.append("-" * 110)
    lines.append(f"RowsChecked: {total}")
    lines.append(f"BaselineDrawsWithHit: {baseline_hit_draws} ({pct(baseline_hit_draws,total):.4f}%)")
    lines.append(f"PrototypeDrawsWithHit: {proto_hit_draws} ({pct(proto_hit_draws,total):.4f}%)")
    lines.append(f"BaselineRawHits: {baseline_raw_hits}")
    lines.append(f"PrototypeRawHits: {proto_raw_hits}")
    lines.append(f"ReconstructedPoolTop10Coverage: {pool10} ({pct(pool10,total):.4f}%)")
    lines.append(f"ReconstructedPoolTop25Coverage: {pool25} ({pct(pool25,total):.4f}%)")
    lines.append(f"ReconstructedPoolTop50Coverage: {pool50} ({pct(pool50,total):.4f}%)")
    lines.append(f"ReconstructedPoolTop100Coverage: {pool100} ({pct(pool100,total):.4f}%)")
    lines.append("")

    by_day = defaultdict(list)
    for row in rows:
        by_day[row["day_type"]].append(row)

    lines.append("DAYTYPE BREAKDOWN")
    lines.append("-" * 110)
    for day_type, items in sorted(by_day.items()):
        n = len(items)
        b = sum(1 for r in items if r["baseline_hit_count"] > 0)
        p = sum(1 for r in items if r["prototype_hit_count"] > 0)
        c100 = sum(1 for r in items if r["pool_hit_top100"] > 0)
        lines.append(
            f"{day_type:<12} Rows={n:<5} "
            f"BaselineHitDraws={b:<4} ({pct(b,n):6.3f}%) "
            f"PrototypeHitDraws={p:<4} ({pct(p,n):6.3f}%) "
            f"PoolTop100Coverage={c100:<4} ({pct(c100,n):6.3f}%)"
        )
    lines.append("")

    family_counter = Counter()
    family_hit_counter = Counter()
    for row in rows:
        for family in row["prototype_families"]:
            family_counter[family] += 1
        if row["prototype_hit_count"] > 0:
            for family in row["prototype_families"]:
                family_hit_counter[family] += 1

    lines.append("PROTOTYPE FAMILY USAGE")
    lines.append("-" * 110)
    for family, count in family_counter.most_common():
        lines.append(
            f"{family:<24} Used={count:<6} "
            f"AppearedInHitDraws={family_hit_counter[family]:<5}"
        )
    lines.append("")

    live_rows = [r for r in rows if r["source"] in {5494, 5495, 5496}]
    lines.append("LIVE MISS WINDOW — 5494..5496")
    lines.append("-" * 110)
    for row in live_rows:
        lines.append(
            f"{row['source']}->{row['target']} "
            f"DayType={row['day_type']} "
            f"Baseline={','.join(row['baseline_top5'])} Hit={row['baseline_hit_count']} "
            f"Prototype={','.join(row['prototype_top5'])} Families={','.join(row['prototype_families'])} "
            f"ProtoHit={row['prototype_hit_count']} "
            f"Pool10/25/50/100={row['pool_hit_top10']}/{row['pool_hit_top25']}/{row['pool_hit_top50']}/{row['pool_hit_top100']}"
        )
    lines.append("")

    lines.append("DIAGNOSIS")
    lines.append("-" * 110)
    if proto_hit_draws > baseline_hit_draws:
        lines.append("Prototype beats baseline on exact Top5 hit-draw count in this reconstructed audit.")
    elif proto_hit_draws == baseline_hit_draws:
        lines.append("Prototype ties baseline on exact Top5 hit-draw count in this reconstructed audit.")
    else:
        lines.append("Prototype underperforms baseline on exact Top5 hit-draw count in this reconstructed audit.")

    if pool100 > baseline_hit_draws:
        lines.append("Reconstructed Top100 coverage exceeds baseline Top5 coverage, indicating ranking/repair opportunity.")
    else:
        lines.append("Reconstructed Top100 coverage does not materially exceed baseline Top5 coverage; generation weakness remains possible.")

    lines.append("")
    lines.append("PromotionRecommendation:")
    lines.append("  Do NOT switch production Current mode yet.")
    lines.append("  Use this report to decide whether Step 150C should harden the residual ranker.")
    lines.append("")

    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    print("")
    print(f"REPORT_WRITTEN: {REPORT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
