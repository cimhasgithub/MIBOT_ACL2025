"""Compute aggregate statistics across the generated conversation transcripts."""
from __future__ import annotations

import json
import statistics
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

REPO_ROOT = Path(__file__).resolve().parent
TRANSCRIPTS_DIR = REPO_ROOT / "transcripts"


def load_transcripts(directory: Path) -> Iterable[Dict[str, object]]:
    for path in sorted(directory.glob("*.json")):
        with path.open(encoding="utf-8") as handle:
            yield json.load(handle)


def safe_stdev(values: List[float]) -> float:
    if len(values) < 2:
        return 0.0
    return statistics.stdev(values)


def summarise(values: List[float]) -> Tuple[float, float, float, float]:
    if not values:
        return 0.0, 0.0, 0.0, 0.0
    mean = statistics.mean(values)
    sd = safe_stdev(values)
    minimum = min(values)
    maximum = max(values)
    return mean, sd, minimum, maximum


def format_metric(mean: float, sd: float, minimum: float, maximum: float, *, decimals: int) -> Tuple[str, str]:
    fmt = f"{{:.{decimals}f}}"
    mean_sd = f"{fmt.format(mean)} ({fmt.format(sd)})"
    range_str = f"{fmt.format(minimum)}–{fmt.format(maximum)}"
    return mean_sd, range_str


def compute_metrics(transcripts: Iterable[Dict[str, object]]):
    total_utts: List[float] = []
    counsellor_utts: List[float] = []
    client_utts: List[float] = []
    counsellor_wpu: List[float] = []
    client_wpu: List[float] = []
    session_minutes: List[float] = []

    for transcript in transcripts:
        stats = transcript["stats"]
        total_utts.append(float(stats["total_utterances"]))
        counsellor_utts.append(float(stats["counsellor_utterances"]))
        client_utts.append(float(stats["client_utterances"]))

        counsellor_words = float(stats["counsellor_words"])
        client_words = float(stats["client_words"])
        counsellor_turns = stats["counsellor_utterances"] or 1
        client_turns = stats["client_utterances"] or 1
        counsellor_wpu.append(counsellor_words / counsellor_turns)
        client_wpu.append(client_words / client_turns)

        session_minutes.append(float(stats["session_duration_minutes"]))

    metrics = {
        "Total utterances": (total_utts, 0),
        "Counsellor utterances": (counsellor_utts, 0),
        "Client utterances": (client_utts, 0),
        "Words per counsellor utterance": (counsellor_wpu, 2),
        "Words per client utterance": (client_wpu, 2),
        "Session duration (minutes)": (session_minutes, 2),
    }

    print("| **Metric** | **Mean (SD)** | **Range** |")
    print("|---------------|---------------|------------|")

    for name, (values, decimals) in metrics.items():
        mean, sd, minimum, maximum = summarise(values)
        mean_sd, range_str = format_metric(mean, sd, minimum, maximum, decimals=decimals)
        print(f"| {name} | {mean_sd} | {range_str} |")


def main() -> None:
    if not TRANSCRIPTS_DIR.exists():
        raise FileNotFoundError(
            "The transcripts directory does not exist. Run convert_to_transcripts.py first."
        )

    transcripts = list(load_transcripts(TRANSCRIPTS_DIR))
    if not transcripts:
        raise RuntimeError("No transcript files were found.")

    compute_metrics(transcripts)


if __name__ == "__main__":
    main()
