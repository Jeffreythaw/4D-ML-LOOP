#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import csv
import math
import re
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import DefaultDict, Dict, List, Optional, Sequence, Tuple

import pandas as pd


MASTER_PARQUET = Path("data/processed/candidate_features_indexed.parquet")
OUTPUT_CSV = Path("data/research/repair_experiments/jeffrey_quad_engine_backtest.csv")

DEFAULT_WARM_START_DRAWS = 100
VALID_SIM_DRAW_MIN = 4051
VALID_SIM_DRAW_MAX = 5494

DAY_TYPES = ("Wednesday", "Saturday", "Sunday", "Special")
REGISTRY_DAY_TYPES = ("Wednesday", "Saturday", "Sunday", "Special")

FOUR_D_RE = re.compile(r"^\d{4}$")


@dataclass(frozen=True)
class DrawRecord:
    draw_no: int
    draw_index: int
    draw_date: datetime
    day_type: str
    winners: Tuple[str, ...]


@dataclass(frozen=True)
class ScoreTrace:
    candidate: str
    total_score: float
    digit_sum: int
    in_day_sum_bounds: bool
    e1: float
    e2: float
    e3: float
    e4: float
    verifier: float
    matrix_trace: str


@dataclass(frozen=True)
class BacktestRecord:
    draw_no_from: int
    draw_no_target: int
    draw_index_target: int
    target_date: str
    target_day_type: str
    sum_p10: int
    sum_p90: int
    top5: Tuple[str, ...]
    actual_winners: Tuple[str, ...]
    hits: Tuple[str, ...]


def normalize_col(name: object) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(name).strip().lower())


def normalize_4d(value: object) -> Optional[str]:
    if value is None:
        return None

    try:
        if pd.isna(value):
            return None
    except Exception:
        pass

    if isinstance(value, float):
        if math.isnan(value):
            return None
        if value.is_integer():
            value = int(value)

    s = str(value).strip()
    if not s or s.lower() in {"nan", "none", "null", "<na>", "nat"}:
        return None

    s = s.replace(" ", "").replace("-", "").replace(",", "")

    if s.endswith(".0") and s[:-2].isdigit():
        s = s[:-2]

    if s.isdigit() and len(s) <= 4:
        s = s.zfill(4)

    if FOUR_D_RE.match(s):
        return s

    return None


def parse_int(value: object, field_name: str, fallback: Optional[int] = None) -> int:
    if value is None:
        if fallback is not None:
            return int(fallback)
        raise ValueError(f"missing {field_name}")

    try:
        if pd.isna(value):
            if fallback is not None:
                return int(fallback)
            raise ValueError(f"missing {field_name}")
    except Exception:
        pass

    if isinstance(value, float):
        if math.isnan(value):
            if fallback is not None:
                return int(fallback)
            raise ValueError(f"NaN {field_name}")
        return int(value)

    s = str(value).strip()
    if s.lower() in {"", "nan", "none", "null", "<na>", "nat"}:
        if fallback is not None:
            return int(fallback)
        raise ValueError(f"missing {field_name}")

    if s.endswith(".0"):
        s = s[:-2]

    return int(s)


def parse_date(value: object) -> datetime:
    if value is None:
        raise ValueError("missing DrawDate")

    try:
        if pd.isna(value):
            raise ValueError("missing DrawDate")
    except TypeError:
        pass

    if isinstance(value, datetime):
        return value

    if hasattr(value, "to_pydatetime"):
        return value.to_pydatetime()

    s = str(value).strip()
    if not s:
        raise ValueError("blank DrawDate")

    for fmt in (
        "%Y-%m-%d",
        "%Y-%m-%d %H:%M:%S",
        "%d/%m/%Y",
        "%m/%d/%Y",
        "%d-%m-%Y",
        "%Y/%m/%d",
        "%d %b %Y",
        "%d %B %Y",
    ):
        try:
            return datetime.strptime(s, fmt)
        except ValueError:
            pass

    return pd.to_datetime(s).to_pydatetime()


def day_type_from_date(dt: datetime) -> str:
    """
    candidate_features_indexed.parquet stores feature date one calendar day before
    the official draw-day bucket used by the verifier.

    Tuesday  -> Wednesday
    Friday   -> Saturday
    Saturday -> Sunday
    """
    official_dt = dt + timedelta(days=1)
    weekday = official_dt.weekday()

    if weekday == 2:
        return "Wednesday"
    if weekday == 5:
        return "Saturday"
    if weekday == 6:
        return "Sunday"

    return "Special"


def digits4(num: str) -> Tuple[int, int, int, int]:
    return int(num[0]), int(num[1]), int(num[2]), int(num[3])


def digit_sum(num: str) -> int:
    return int(num[0]) + int(num[1]) + int(num[2]) + int(num[3])


def nearest_rank_percentile(values: Sequence[int], pct: float) -> int:
    if not values:
        raise ValueError("empty percentile sample")

    sorted_values = sorted(values)
    rank = max(1, min(len(sorted_values), math.ceil(pct * len(sorted_values))))
    return sorted_values[rank - 1]


def detect_required_columns(df: pd.DataFrame) -> Tuple[str, str, str, str, str]:
    normalized = {normalize_col(c): c for c in df.columns}

    draw_no_col = None
    draw_index_col = None
    draw_date_col = None
    label_col = None
    number_col = None

    for c in df.columns:
        n = normalize_col(c)

        if n in {"drawno", "drawnumber", "drawnum"}:
            draw_no_col = c
        elif n in {"drawindex", "index"}:
            draw_index_col = c
        elif n in {"drawdate", "date", "drawdatetime"}:
            draw_date_col = c
        elif n == "label":
            label_col = c

    for wanted in (
        "candidatenumber",
        "candidate",
        "number",
        "num",
        "fourdigits",
        "digit",
        "drawnumber4d",
        "value",
    ):
        if wanted in normalized:
            number_col = normalized[wanted]
            break

    if number_col is None:
        meta = {draw_no_col, draw_index_col, draw_date_col, label_col}
        best_col = None
        best_count = -1

        for c in df.columns:
            if c in meta:
                continue

            sample = df[c].dropna().head(500)
            if sample.empty:
                continue

            count = sum(1 for v in sample if normalize_4d(v) is not None)

            if count > best_count:
                best_col = c
                best_count = count

        if best_col is not None and best_count > 0:
            number_col = best_col

    missing = []
    if draw_no_col is None:
        missing.append("DrawNo")
    if draw_index_col is None:
        missing.append("DrawIndex")
    if draw_date_col is None:
        missing.append("DrawDate")
    if label_col is None:
        missing.append("Label")
    if number_col is None:
        missing.append("CandidateNumber")

    if missing:
        raise ValueError(
            "master parquet missing required columns: "
            + ", ".join(missing)
            + f". Available columns: {list(df.columns)}"
        )

    return (
        str(draw_no_col),
        str(draw_index_col),
        str(draw_date_col),
        str(label_col),
        str(number_col),
    )


def load_master_winning_draws(parquet_path: Path, trace: bool = False) -> List[DrawRecord]:
    if not parquet_path.exists():
        raise FileNotFoundError(f"master parquet not found: {parquet_path}")

    df = pd.read_parquet(parquet_path)

    draw_no_col, draw_index_col, draw_date_col, label_col, number_col = detect_required_columns(df)

    if trace:
        print("[LOAD]")
        print(f"  master parquet: {parquet_path}")
        print(f"  rows: {len(df):,}")
        print(f"  DrawNo column: {draw_no_col}")
        print(f"  DrawIndex column: {draw_index_col}")
        print(f"  DrawDate column: {draw_date_col}")
        print(f"  Label column: {label_col}")
        print(f"  Number column: {number_col}")

    label_numeric = pd.to_numeric(df[label_col], errors="coerce").fillna(0)
    winners_df = df[label_numeric > 0].copy()
    raw_winner_rows = len(winners_df)

    winners_df = winners_df[
        winners_df[draw_no_col].notna()
        & winners_df[draw_date_col].notna()
        & winners_df[number_col].notna()
    ].copy()

    winners_df["_DrawNoNumeric"] = pd.to_numeric(winners_df[draw_no_col], errors="coerce")
    winners_df = winners_df[
        winners_df["_DrawNoNumeric"].notna()
        & (winners_df["_DrawNoNumeric"] >= VALID_SIM_DRAW_MIN)
        & (winners_df["_DrawNoNumeric"] <= VALID_SIM_DRAW_MAX)
    ].copy()

    dropped_invalid_winner_rows = raw_winner_rows - len(winners_df)

    if winners_df.empty:
        raise ValueError("no valid winning rows found in strict DrawNo 4051..5494 window")

    records_by_no: Dict[int, Dict[str, object]] = {}

    for row_idx, row in winners_df.iterrows():
        draw_no = parse_int(row[draw_no_col], "DrawNo")
        draw_index = parse_int(row[draw_index_col], "DrawIndex", fallback=draw_no)
        draw_date = parse_date(row[draw_date_col])
        num = normalize_4d(row[number_col])

        if num is None:
            continue

        if draw_no not in records_by_no:
            records_by_no[draw_no] = {
                "draw_no": draw_no,
                "draw_index": draw_index,
                "draw_date": draw_date,
                "winners": [],
                "seen": set(),
            }

        bucket = records_by_no[draw_no]

        if draw_date.date() != bucket["draw_date"].date():
            raise ValueError(f"inconsistent DrawDate for DrawNo {draw_no}")

        seen = bucket["seen"]
        winners = bucket["winners"]

        if num not in seen:
            winners.append(num)
            seen.add(num)

    records: List[DrawRecord] = []

    for payload in records_by_no.values():
        draw_date = payload["draw_date"]
        winners = tuple(payload["winners"])

        if not winners:
            continue

        records.append(
            DrawRecord(
                draw_no=int(payload["draw_no"]),
                draw_index=int(payload["draw_index"]),
                draw_date=draw_date,
                day_type=day_type_from_date(draw_date),
                winners=winners,
            )
        )

    records.sort(key=lambda r: r.draw_no)

    if not records:
        raise ValueError("no draw records extracted from parquet")

    if trace:
        print(f"  winning rows before metadata filter: {raw_winner_rows:,}")
        print(f"  dropped invalid / out-of-window Label>0 rows: {dropped_invalid_winner_rows:,}")
        print(f"  valid winning rows: {len(winners_df):,}")
        print(f"  extracted draws: {len(records):,}")
        print(f"  draw range: {records[0].draw_no} -> {records[-1].draw_no}")
        print()

    return records


class TemporalVerifierProfile:
    def __init__(self) -> None:
        self.samples_by_day: DefaultDict[str, List[int]] = defaultdict(list)
        self.bounds_by_day: Dict[str, Tuple[int, int]] = {}

    def observe_draw(self, draw: DrawRecord) -> None:
        for n in draw.winners:
            self.samples_by_day[draw.day_type].append(digit_sum(n))

    def lock(self) -> None:
        all_samples: List[int] = []

        for samples in self.samples_by_day.values():
            all_samples.extend(samples)

        if not all_samples:
            raise ValueError("cannot lock temporal verifier with no samples")

        global_p10 = nearest_rank_percentile(all_samples, 0.10)
        global_p90 = nearest_rank_percentile(all_samples, 0.90)

        for day in DAY_TYPES:
            samples = self.samples_by_day.get(day, [])

            if samples:
                self.bounds_by_day[day] = (
                    nearest_rank_percentile(samples, 0.10),
                    nearest_rank_percentile(samples, 0.90),
                )
            else:
                self.bounds_by_day[day] = (global_p10, global_p90)

    def bounds_for(self, day_type: str) -> Tuple[int, int]:
        if not self.bounds_by_day:
            raise RuntimeError("temporal verifier profile has not been locked")

        return self.bounds_by_day.get(day_type, self.bounds_by_day["Special"])


class FormulaRegistry:
    """
    Day-isolated Formula Registry.

    All discovered formulas are stored under the TARGET day type of the transition.
    A Wednesday transition updates only Wednesday branch.
    A Saturday miss/learn does not alter Wednesday branch.
    """

    def __init__(self) -> None:
        self.e1: Dict[str, DefaultDict[Tuple[int, int], int]] = {
            day: defaultdict(int) for day in REGISTRY_DAY_TYPES
        }
        self.e2: Dict[str, DefaultDict[Tuple[int, int, int], int]] = {
            day: defaultdict(int) for day in REGISTRY_DAY_TYPES
        }
        self.e3: Dict[str, DefaultDict[Tuple[int, int, int, int, int], int]] = {
            day: defaultdict(int) for day in REGISTRY_DAY_TYPES
        }

    def update_transition(
        self,
        from_winners: Sequence[str],
        to_winners: Sequence[str],
        target_day_type: str,
    ) -> None:
        branch = target_day_type if target_day_type in self.e1 else "Special"

        e1 = self.e1[branch]
        e2 = self.e2[branch]
        e3 = self.e3[branch]

        for src in from_winners:
            sd = digits4(src)

            for dst in to_winners:
                td = digits4(dst)

                for pos in range(4):
                    delta = (td[pos] - sd[pos]) % 10
                    e1[(pos, delta)] += 1

                for sp in range(4):
                    for tp in range(4):
                        delta = (td[tp] - sd[sp]) % 10
                        e2[(sp, tp, delta)] += 1

                for sa in range(4):
                    for sb in range(sa + 1, 4):
                        src_pair = (sd[sa] + sd[sb]) % 10

                        for ta in range(4):
                            for tb in range(ta + 1, 4):
                                dst_pair = (td[ta] + td[tb]) % 10
                                pair_delta = (dst_pair - src_pair) % 10
                                e3[(sa, sb, ta, tb, pair_delta)] += 1

    @staticmethod
    def top_counter_items(counter: Dict[Tuple, int], k: int) -> List[Tuple[Tuple, int]]:
        return sorted(counter.items(), key=lambda kv: (-kv[1], kv[0]))[:k]

    def generate_blind_candidates(
        self,
        current_winners: Sequence[str],
        target_day_type: str,
        max_candidates: int = 3000,
        e1_top_k: int = 40,
        e2_top_k: int = 80,
        e3_top_k: int = 60,
        per_position_options: int = 4,
    ) -> Tuple[Dict[str, Dict[str, float]], Dict[str, str]]:
        scores: DefaultDict[str, Dict[str, float]] = defaultdict(
            lambda: {"E1": 0.0, "E2": 0.0, "E3": 0.0}
        )
        matrix_traces: Dict[str, str] = {}

        if not current_winners:
            return {}, {}

        branch = target_day_type if target_day_type in self.e1 else "Special"

        e1_top = self.top_counter_items(self.e1[branch], e1_top_k)
        e2_top = self.top_counter_items(self.e2[branch], e2_top_k)
        e3_top = self.top_counter_items(self.e3[branch], e3_top_k)

        e1_by_pos: DefaultDict[int, List[Tuple[int, int]]] = defaultdict(list)
        for (pos, delta), cnt in e1_top:
            e1_by_pos[pos].append((delta, cnt))

        for src in current_winners:
            sd = digits4(src)
            opts: List[List[Tuple[int, int]]] = []

            for pos in range(4):
                pos_opts = sorted(e1_by_pos.get(pos, []), key=lambda x: (-x[1], x[0]))
                if not pos_opts:
                    pos_opts = [(0, 1)]
                opts.append(pos_opts[:per_position_options])

            for d0, w0 in opts[0]:
                for d1, w1 in opts[1]:
                    for d2, w2 in opts[2]:
                        for d3, w3 in opts[3]:
                            cand = (
                                f"{(sd[0] + d0) % 10}"
                                f"{(sd[1] + d1) % 10}"
                                f"{(sd[2] + d2) % 10}"
                                f"{(sd[3] + d3) % 10}"
                            )
                            scores[cand]["E1"] += (
                                math.log1p(w0)
                                + math.log1p(w1)
                                + math.log1p(w2)
                                + math.log1p(w3)
                            )

        for src in current_winners:
            sd = digits4(src)
            votes: List[Counter[int]] = [Counter() for _ in range(4)]

            for (sp, tp, delta), cnt in e2_top:
                predicted_digit = (sd[sp] + delta) % 10
                votes[tp][predicted_digit] += cnt

            opts2: List[List[Tuple[int, int]]] = []

            for pos in range(4):
                common = votes[pos].most_common(per_position_options)
                if not common:
                    common = [(sd[pos], 1)]
                opts2.append(common)

            for d0, w0 in opts2[0]:
                for d1, w1 in opts2[1]:
                    for d2, w2 in opts2[2]:
                        for d3, w3 in opts2[3]:
                            cand = f"{d0}{d1}{d2}{d3}"
                            scores[cand]["E2"] += (
                                math.log1p(w0)
                                + math.log1p(w1)
                                + math.log1p(w2)
                                + math.log1p(w3)
                            )

        for src in current_winners:
            sd = digits4(src)
            votes3: List[Counter[int]] = [Counter() for _ in range(4)]
            strongest_formula = None
            strongest_count = -1

            for key, cnt in e3_top:
                sa, sb, ta, tb, pair_delta = key
                src_pair = (sd[sa] + sd[sb]) % 10
                target_pair = (src_pair + pair_delta) % 10

                if cnt > strongest_count:
                    strongest_formula = key
                    strongest_count = cnt

                for x in range(10):
                    y = (target_pair - x) % 10
                    votes3[ta][x] += cnt
                    votes3[tb][y] += cnt

            opts3: List[List[Tuple[int, int]]] = []

            for pos in range(4):
                common = votes3[pos].most_common(per_position_options)
                if not common:
                    common = [(sd[pos], 1)]
                opts3.append(common)

            for d0, w0 in opts3[0]:
                for d1, w1 in opts3[1]:
                    for d2, w2 in opts3[2]:
                        for d3, w3 in opts3[3]:
                            cand = f"{d0}{d1}{d2}{d3}"
                            row = (w0, w1, w2, w3)
                            dot = w0 * w1 + w1 * w2 + w2 * w3 + w0 * w3
                            product = max(1, w0) * max(1, w1) * max(1, w2) * max(1, w3)

                            scores[cand]["E3"] += (
                                math.log1p(w0)
                                + math.log1p(w1)
                                + math.log1p(w2)
                                + math.log1p(w3)
                            )

                            if cand not in matrix_traces:
                                matrix_traces[cand] = (
                                    f"day_branch={branch}; src={src}; row={row}; "
                                    f"dot={dot}; product={product}; "
                                    f"strongest_formula={strongest_formula}; "
                                    f"strongest_count={strongest_count}"
                                )

        if len(scores) > max_candidates:
            ranked = sorted(
                scores.items(),
                key=lambda kv: (-sum(kv[1].values()), kv[0]),
            )[:max_candidates]
            capped: DefaultDict[str, Dict[str, float]] = defaultdict(
                lambda: {"E1": 0.0, "E2": 0.0, "E3": 0.0}
            )
            for cand, sc in ranked:
                capped[cand] = sc
            scores = capped

        return dict(scores), matrix_traces


class MarkovEngine:
    """
    Day-isolated Markov engine.

    All transitions are stored under the TARGET day type.
    """

    def __init__(self) -> None:
        self.full: Dict[str, DefaultDict[Tuple[str, str], int]] = {
            day: defaultdict(int) for day in REGISTRY_DAY_TYPES
        }
        self.prefix: Dict[str, DefaultDict[Tuple[str, str], int]] = {
            day: defaultdict(int) for day in REGISTRY_DAY_TYPES
        }
        self.suffix: Dict[str, DefaultDict[Tuple[str, str], int]] = {
            day: defaultdict(int) for day in REGISTRY_DAY_TYPES
        }
        self.pos_digit: Dict[str, DefaultDict[Tuple[int, int, int], int]] = {
            day: defaultdict(int) for day in REGISTRY_DAY_TYPES
        }
        self.sum_band_counts: Dict[str, DefaultDict[Tuple[int, int], int]] = {
            day: defaultdict(int) for day in REGISTRY_DAY_TYPES
        }

    @staticmethod
    def sum_band(num: str) -> int:
        return digit_sum(num) // 5

    def update_transition(
        self,
        from_winners: Sequence[str],
        to_winners: Sequence[str],
        target_day_type: str,
    ) -> None:
        branch = target_day_type if target_day_type in self.full else "Special"

        full = self.full[branch]
        prefix = self.prefix[branch]
        suffix = self.suffix[branch]
        pos_digit = self.pos_digit[branch]
        sum_band_counts = self.sum_band_counts[branch]

        for src in from_winners:
            sd = digits4(src)
            src_band = self.sum_band(src)

            for dst in to_winners:
                td = digits4(dst)
                dst_band = self.sum_band(dst)

                full[(src, dst)] += 1
                prefix[(src[:2], dst[:2])] += 1
                suffix[(src[2:], dst[2:])] += 1
                sum_band_counts[(src_band, dst_band)] += 1

                for pos in range(4):
                    pos_digit[(pos, sd[pos], td[pos])] += 1

    def score(self, current_winners: Sequence[str], candidate: str, target_day_type: str) -> float:
        branch = target_day_type if target_day_type in self.full else "Special"

        full = self.full[branch]
        prefix = self.prefix[branch]
        suffix = self.suffix[branch]
        pos_digit = self.pos_digit[branch]
        sum_band_counts = self.sum_band_counts[branch]

        cd = digits4(candidate)
        cand_band = self.sum_band(candidate)
        total = 0.0

        for src in current_winners:
            sd = digits4(src)
            src_band = self.sum_band(src)

            total += 4.00 * math.log1p(full.get((src, candidate), 0))
            total += 1.35 * math.log1p(prefix.get((src[:2], candidate[:2]), 0))
            total += 1.35 * math.log1p(suffix.get((src[2:], candidate[2:]), 0))
            total += 0.80 * math.log1p(sum_band_counts.get((src_band, cand_band), 0))

            for pos in range(4):
                total += 0.55 * math.log1p(pos_digit.get((pos, sd[pos], cd[pos]), 0))

        return total


class DaySpecificWeights:
    """
    Independent operational engine profiles.

    A Saturday miss mutates only Saturday weights.
    Wednesday weights remain untouched.
    """

    def __init__(self) -> None:
        self.weights: Dict[str, Dict[str, float]] = {
            day: {
                "E1": 1.00,
                "E2": 1.12,
                "E3": 1.28,
                "E4": 1.18,
            }
            for day in REGISTRY_DAY_TYPES
        }

    def get(self, day_type: str) -> Dict[str, float]:
        return self.weights.get(day_type, self.weights["Special"])

    @staticmethod
    def _clamp(x: float, lo: float = 0.55, hi: float = 1.75) -> float:
        return max(lo, min(hi, x))

    def update_from_result(
        self,
        day_type: str,
        selected_rows: Sequence[ScoreTrace],
        hits: Sequence[str],
    ) -> None:
        branch = day_type if day_type in self.weights else "Special"
        w = self.weights[branch]

        hit_set = set(hits)

        if hit_set:
            # Reward engines that contributed to actual hit candidates.
            for row in selected_rows:
                if row.candidate not in hit_set:
                    continue

                contributions = {
                    "E1": row.e1,
                    "E2": row.e2,
                    "E3": row.e3,
                    "E4": row.e4,
                }
                total = sum(max(0.0, v) for v in contributions.values()) or 1.0

                for eng, val in contributions.items():
                    share = max(0.0, val) / total
                    w[eng] = self._clamp(w[eng] * (1.0 + 0.035 * share))

            # Mildly protect working profile on hit.
            for eng in w:
                w[eng] = self._clamp(w[eng] * 1.002)

        else:
            # Penalize dominant engines responsible for missed Top 5.
            primary_counts: Counter[str] = Counter(primary_engine(row) for row in selected_rows)

            for eng, count in primary_counts.items():
                penalty = 1.0 - min(0.045, 0.0125 * count)
                w[eng] = self._clamp(w[eng] * penalty)

            # Tiny exploration recovery for underused engines.
            used = set(primary_counts)
            for eng in ("E1", "E2", "E3", "E4"):
                if eng not in used:
                    w[eng] = self._clamp(w[eng] * 1.006)


def verifier_score(candidate: str, lower: int, upper: int) -> Tuple[float, bool]:
    ds = digit_sum(candidate)

    if lower <= ds <= upper:
        sum_score = 2.75
        in_bounds = True
    else:
        in_bounds = False
        distance = lower - ds if ds < lower else ds - upper
        sum_score = max(-3.00, 0.50 - distance * 0.50)

    d = digits4(candidate)
    spread = max(d) - min(d)
    unique = len(set(candidate))
    even_count = sum(1 for x in d if x % 2 == 0)

    if 4 <= spread <= 8:
        spread_score = 0.65
    elif spread in {3, 9}:
        spread_score = 0.35
    else:
        spread_score = -0.15

    if unique == 4:
        repeat_score = 0.35
    elif unique == 3:
        repeat_score = 0.10
    elif unique == 2:
        repeat_score = -0.45
    else:
        repeat_score = -1.00

    if even_count == 2:
        parity_score = 0.45
    elif even_count in {1, 3}:
        parity_score = 0.20
    else:
        parity_score = -0.15

    return sum_score + spread_score + repeat_score + parity_score, in_bounds


def resolve_windows(records: Sequence[DrawRecord], warm_start_draws: int) -> Tuple[List[int], List[int]]:
    draw_nos = sorted(r.draw_no for r in records)

    if len(draw_nos) <= warm_start_draws:
        raise ValueError(
            f"not enough draws for warm-start={warm_start_draws}; "
            f"available draws={len(draw_nos)}"
        )

    warm_start_nos = draw_nos[:warm_start_draws]
    sim_target_nos = draw_nos[warm_start_draws:]

    return warm_start_nos, sim_target_nos


def build_phase1_warm_start(
    records_by_draw: Dict[int, DrawRecord],
    warm_start_nos: Sequence[int],
    trace: bool = False,
) -> Tuple[FormulaRegistry, MarkovEngine, TemporalVerifierProfile, DaySpecificWeights]:
    if len(warm_start_nos) < 2:
        raise ValueError("warm-start requires at least 2 draws")

    registry = FormulaRegistry()
    markov = MarkovEngine()
    temporal = TemporalVerifierProfile()
    weights = DaySpecificWeights()

    for draw_no in warm_start_nos:
        temporal.observe_draw(records_by_draw[draw_no])

    temporal.lock()

    for prev_no, next_no in zip(warm_start_nos[:-1], warm_start_nos[1:]):
        prev_draw = records_by_draw[prev_no]
        next_draw = records_by_draw[next_no]
        registry.update_transition(prev_draw.winners, next_draw.winners, next_draw.day_type)
        markov.update_transition(prev_draw.winners, next_draw.winners, next_draw.day_type)

    if trace:
        print("[PHASE 1] DAY-ISOLATED BASE WARM-START")
        print(f"  warm-start draw range: {warm_start_nos[0]} -> {warm_start_nos[-1]}")
        print(f"  warm-start draws processed exactly once: {len(warm_start_nos):,}")
        print("  isolated registry branch sizes:")
        for day in DAY_TYPES:
            print(
                f"    {day:<10} "
                f"E1={len(registry.e1[day]):<5} "
                f"E2={len(registry.e2[day]):<5} "
                f"E3={len(registry.e3[day]):<5} "
                f"E4_full={len(markov.full[day]):<7}"
            )
        print("  locked day-type DigitSum bounds:")
        for day in DAY_TYPES:
            lo, hi = temporal.bounds_for(day)
            sample_count = len(temporal.samples_by_day.get(day, []))
            print(f"    {day:<10} samples={sample_count:<7} p10={lo:<2} p90={hi:<2}")
        print()

    return registry, markov, temporal, weights


def rank_candidates(
    current_winners: Sequence[str],
    candidates: Dict[str, Dict[str, float]],
    matrix_traces: Dict[str, str],
    markov: MarkovEngine,
    lower: int,
    upper: int,
    target_day_type: str,
    weights: DaySpecificWeights,
) -> List[ScoreTrace]:
    scored: List[ScoreTrace] = []
    w = weights.get(target_day_type)

    for candidate, engine_scores in candidates.items():
        e1 = float(engine_scores.get("E1", 0.0))
        e2 = float(engine_scores.get("E2", 0.0))
        e3 = float(engine_scores.get("E3", 0.0))
        e4 = markov.score(current_winners, candidate, target_day_type)
        vf, in_bounds = verifier_score(candidate, lower, upper)

        total = (
            w["E1"] * e1
            + w["E2"] * e2
            + w["E3"] * e3
            + w["E4"] * e4
            + vf
        )

        scored.append(
            ScoreTrace(
                candidate=candidate,
                total_score=total,
                digit_sum=digit_sum(candidate),
                in_day_sum_bounds=in_bounds,
                e1=e1,
                e2=e2,
                e3=e3,
                e4=e4,
                verifier=vf,
                matrix_trace=matrix_traces.get(candidate, "-"),
            )
        )

    scored.sort(key=lambda r: (-r.total_score, r.candidate))
    return scored


def primary_engine(row: ScoreTrace) -> str:
    parts = {
        "E1": row.e1,
        "E2": row.e2,
        "E3": row.e3,
        "E4": row.e4,
    }
    return max(parts.items(), key=lambda kv: (kv[1], kv[0]))[0]


def engine_score(row: ScoreTrace, engine: str) -> float:
    if engine == "E1":
        return row.e1
    if engine == "E2":
        return row.e2
    if engine == "E3":
        return row.e3
    if engine == "E4":
        return row.e4
    return 0.0


def select_top5_diverse(ranked: Sequence[ScoreTrace], min_engines: int = 3) -> Tuple[str, ...]:
    selected: List[ScoreTrace] = []
    selected_nums = set()
    source_counts: Counter[str] = Counter()
    engines = ("E1", "E2", "E3", "E4")

    engine_ranked: Dict[str, List[ScoreTrace]] = {}
    for eng in engines:
        engine_ranked[eng] = sorted(
            ranked,
            key=lambda r: (-engine_score(r, eng), -r.total_score, r.candidate),
        )

    engine_priority = sorted(
        engines,
        key=lambda eng: (
            -engine_score(engine_ranked[eng][0], eng) if engine_ranked[eng] else 0.0,
            eng,
        ),
    )

    for eng in engine_priority:
        if len(source_counts) >= min_engines:
            break

        for row in engine_ranked[eng]:
            if row.candidate in selected_nums:
                continue
            if engine_score(row, eng) <= 0:
                continue

            selected.append(row)
            selected_nums.add(row.candidate)
            source_counts[eng] += 1
            break

    for row in ranked:
        if len(selected) >= 5:
            break

        if row.candidate in selected_nums:
            continue

        src = primary_engine(row)

        if len(source_counts) < min_engines and source_counts[src] >= 1:
            continue

        if source_counts[src] >= 2 and len(selected) < 4:
            continue

        selected.append(row)
        selected_nums.add(row.candidate)
        source_counts[src] += 1

    for row in ranked:
        if len(selected) >= 5:
            break

        if row.candidate not in selected_nums:
            selected.append(row)
            selected_nums.add(row.candidate)

    return tuple(r.candidate for r in selected[:5])


def run_phase2_incremental_evolution(
    records: Sequence[DrawRecord],
    output_csv: Path,
    warm_start_draws: int,
    trace: bool = False,
    trace_limit: int = 5,
) -> List[BacktestRecord]:
    records_by_draw = {r.draw_no: r for r in records}
    warm_start_nos, sim_target_nos = resolve_windows(records, warm_start_draws)

    registry, markov, temporal, weights = build_phase1_warm_start(
        records_by_draw=records_by_draw,
        warm_start_nos=warm_start_nos,
        trace=trace,
    )

    results: List[BacktestRecord] = []

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    out_f = output_csv.open("w", encoding="utf-8", newline="")
    streaming_writer = csv.DictWriter(
        out_f,
        fieldnames=[
            "DrawNo_From",
            "DrawNo_Target",
            "DrawIndex_Target",
            "TargetDate",
            "TargetDayType",
            "DigitSum_P10",
            "DigitSum_P90",
            "Top1",
            "Top2",
            "Top3",
            "Top4",
            "Top5",
            "ActualWinners",
            "HitCount",
            "Hits",
        ],
    )
    streaming_writer.writeheader()
    out_f.flush()

    if trace:
        print("[PHASE 2] DAY-ISOLATED INCREMENTAL EVOLUTION LOOP")
        print(f"  simulation target range: {sim_target_nos[0]} -> {sim_target_nos[-1]}")
        print(f"  simulation rows: {len(sim_target_nos):,}")
        print("  blind rule: target-day branch only")
        print("  day-specific weights: enabled")
        print("  legacy feature scores: DISABLED")
        print()

    for target_no in sim_target_nos:
        from_no = target_no - 1

        if from_no not in records_by_draw:
            raise ValueError(
                f"cannot predict DrawNo {target_no}: previous DrawNo {from_no} not found"
            )

        current_draw = records_by_draw[from_no]
        target_draw = records_by_draw[target_no]
        target_day = target_draw.day_type

        lower, upper = temporal.bounds_for(target_day)

        candidates, matrix_traces = registry.generate_blind_candidates(
            current_draw.winners,
            target_day_type=target_day,
        )

        for n in current_draw.winners:
            candidates.setdefault(n, {"E1": 0.0, "E2": 0.0, "E3": 0.0})

        ranked = rank_candidates(
            current_winners=current_draw.winners,
            candidates=candidates,
            matrix_traces=matrix_traces,
            markov=markov,
            lower=lower,
            upper=upper,
            target_day_type=target_day,
            weights=weights,
        )

        top5 = select_top5_diverse(ranked)
        ranked_by_candidate = {r.candidate: r for r in ranked}
        top5_traces = [ranked_by_candidate[c] for c in top5]

        if len(top5) != 5:
            raise RuntimeError(f"failed to produce exactly Top 5 for DrawNo {target_no}")

        actual_winners = target_draw.winners
        actual_set = set(actual_winners)
        hits = tuple(n for n in top5 if n in actual_set)

        result = BacktestRecord(
            draw_no_from=from_no,
            draw_no_target=target_no,
            draw_index_target=target_draw.draw_index,
            target_date=target_draw.draw_date.strftime("%Y-%m-%d"),
            target_day_type=target_day,
            sum_p10=lower,
            sum_p90=upper,
            top5=top5,
            actual_winners=actual_winners,
            hits=hits,
        )
        results.append(result)

        top = list(result.top5)
        streaming_writer.writerow(
            {
                "DrawNo_From": result.draw_no_from,
                "DrawNo_Target": result.draw_no_target,
                "DrawIndex_Target": result.draw_index_target,
                "TargetDate": result.target_date,
                "TargetDayType": result.target_day_type,
                "DigitSum_P10": result.sum_p10,
                "DigitSum_P90": result.sum_p90,
                "Top1": top[0],
                "Top2": top[1],
                "Top3": top[2],
                "Top4": top[3],
                "Top5": top[4],
                "ActualWinners": "|".join(result.actual_winners),
                "HitCount": len(result.hits),
                "Hits": "|".join(result.hits),
            }
        )
        out_f.flush()

        if trace and len(results) <= trace_limit:
            current_weights = weights.get(target_day)
            print("=" * 120)
            print(
                f"[TRACE DAY-ISOLATED BLIND TRANSITION] Draw {from_no} -> {target_no} "
                f"| DrawIndex={target_draw.draw_index} "
                f"| {target_draw.draw_date.strftime('%Y-%m-%d')} "
                f"| {target_day}"
            )
            print(f"  day-specific DigitSum bounds: p10={lower}, p90={upper}")
            print(
                "  active weights: "
                f"E1={current_weights['E1']:.4f}, "
                f"E2={current_weights['E2']:.4f}, "
                f"E3={current_weights['E3']:.4f}, "
                f"E4={current_weights['E4']:.4f}"
            )
            print(f"  current Draw {from_no} vectors: {', '.join(current_draw.winners)}")
            print("  diverse blind scoring scoreboard:")
            print(
                "    rank | cand | src | score       | sum | in_rng | "
                "E1         | E2         | E3         | E4         | verifier"
            )

            for rank, tr in enumerate(top5_traces, start=1):
                print(
                    f"    {rank:<4} | "
                    f"{tr.candidate:<4} | "
                    f"{primary_engine(tr):<3} | "
                    f"{tr.total_score:<11.4f} | "
                    f"{tr.digit_sum:<3} | "
                    f"{str(tr.in_day_sum_bounds):<6} | "
                    f"{tr.e1:<10.4f} | "
                    f"{tr.e2:<10.4f} | "
                    f"{tr.e3:<10.4f} | "
                    f"{tr.e4:<10.4f} | "
                    f"{tr.verifier:<10.4f}"
                )
                print(f"         matrix product trace: {tr.matrix_trace}")

            print(f"  blind Top 5 extracted: {', '.join(top5)}")
            print(f"  verification actual winners: {', '.join(actual_winners)}")
            print(f"  honest HitCount: {len(hits)} | Hits: {', '.join(hits) if hits else '-'}")

        weights.update_from_result(target_day, top5_traces, hits)

        registry.update_transition(current_draw.winners, target_draw.winners, target_day)
        markov.update_transition(current_draw.winners, target_draw.winners, target_day)

    out_f.close()

    if trace:
        total_hits = sum(len(r.hits) for r in results)
        hit_rows = sum(1 for r in results if r.hits)
        print()
        print("[COMPLETE]")
        print(f"  saved backtest matrix: {output_csv}")
        print(f"  rows: {len(results):,}")
        print(f"  rows with >=1 hit: {hit_rows:,}")
        print(f"  total Top5 hits: {total_hits:,}")
        print("  final day-specific weights:")
        for day in DAY_TYPES:
            w = weights.get(day)
            print(
                f"    {day:<10} "
                f"E1={w['E1']:.4f} E2={w['E2']:.4f} "
                f"E3={w['E3']:.4f} E4={w['E4']:.4f}"
            )

    return results


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Jeffrey Quad Algebraic Harness: day-isolated registry + "
            "day-specific adaptive engine weights"
        )
    )

    parser.add_argument(
        "--input",
        default=str(MASTER_PARQUET),
        help=f"Master parquet path. Default: {MASTER_PARQUET}",
    )

    parser.add_argument(
        "--output-csv",
        default=str(OUTPUT_CSV),
        help=f"Output backtest CSV path. Default: {OUTPUT_CSV}",
    )

    parser.add_argument(
        "--warm-start-draws",
        type=int,
        default=DEFAULT_WARM_START_DRAWS,
        help="Number of first available draws to use for Phase 1 warm-start. Default: 100.",
    )

    parser.add_argument(
        "--trace-blind-transition",
        action="store_true",
        help="Print day-isolated bounds, matrix product trace, weights, and blind scoreboard.",
    )

    parser.add_argument(
        "--trace-limit",
        type=int,
        default=5,
        help="Number of blind transitions to print when tracing.",
    )

    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        input_path = Path(args.input)
        output_csv = Path(args.output_csv)

        records = load_master_winning_draws(
            parquet_path=input_path,
            trace=args.trace_blind_transition,
        )

        warm_start_nos, sim_target_nos = resolve_windows(records, args.warm_start_draws)

        if args.trace_blind_transition:
            print("[BOOT]")
            print(f"  extracted winning draw range: {records[0].draw_no} -> {records[-1].draw_no}")
            print(f"  warm-start window: {warm_start_nos[0]} -> {warm_start_nos[-1]}")
            print(f"  simulation window: {sim_target_nos[0]} -> {sim_target_nos[-1]}")
            print(f"  output csv: {output_csv}")
            print()

        run_phase2_incremental_evolution(
            records=records,
            output_csv=output_csv,
            warm_start_draws=args.warm_start_draws,
            trace=args.trace_blind_transition,
            trace_limit=args.trace_limit,
        )

        return 0

    except Exception as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
