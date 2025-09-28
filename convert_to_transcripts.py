"""Convert the raw CSV conversation logs into per-participant JSON transcripts.

The script expects to be executed from the repository root.  It reads
``conversations.csv`` and writes structured conversation transcripts to the
``transcripts/`` directory.  Each JSON file corresponds to a single participant
and contains both the utterance level data as well as some light-weight summary
statistics that can be re-used by downstream analysis scripts.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Dict, List

# Conversations are text-based chat interactions.  We approximate an effective
# speaking rate to produce a rough estimate of the session duration.
WORDS_PER_MINUTE = 130.0

REPO_ROOT = Path(__file__).resolve().parent
INPUT_CSV = REPO_ROOT / "conversations.csv"
OUTPUT_DIR = REPO_ROOT / "transcripts"


def count_words(text: str) -> int:
    """Return the number of word-like tokens in ``text``."""
    text = text.strip()
    if not text:
        return 0
    # The simple split on whitespace is robust enough for this dataset and
    # avoids pulling in heavier tokenisation dependencies.
    return sum(1 for token in text.split() if token)


def build_transcripts(rows: List[Dict[str, str]]) -> Dict[str, Dict[str, object]]:
    """Group the CSV rows by participant and prepare their transcript objects."""
    transcripts: Dict[str, Dict[str, object]] = {}

    for row in rows:
        participant_id = row["ParticipantID"]
        utterance_text = row["Utterance"].strip()
        word_count = count_words(utterance_text)

        record = transcripts.setdefault(
            participant_id,
            {
                "participant_id": participant_id,
                "utterances": [],
            },
        )

        record["utterances"].append(
            {
                "speaker": row["Speaker"].strip().lower(),
                "volley_number": int(row["Volley#"]),
                "utterance_number": int(row["Utterance#"]),
                "text": utterance_text,
                "auto_misc_label": row["AutoMISCLabel"].strip(),
                "auto_misc_explanation": row["AutoMISCExplanation"].strip(),
                "word_count": word_count,
            }
        )

    # Populate summary statistics per participant.
    for record in transcripts.values():
        utterances = record["utterances"]
        total_utterances = len(utterances)
        counsellor_utterances = sum(
            1 for utt in utterances if utt["speaker"] == "counsellor"
        )
        client_utterances = sum(
            1 for utt in utterances if utt["speaker"] == "client"
        )
        counsellor_words = sum(
            utt["word_count"] for utt in utterances if utt["speaker"] == "counsellor"
        )
        client_words = sum(
            utt["word_count"] for utt in utterances if utt["speaker"] == "client"
        )
        total_words = counsellor_words + client_words
        session_duration_minutes = total_words / WORDS_PER_MINUTE if total_words else 0.0

        record["stats"] = {
            "total_utterances": total_utterances,
            "counsellor_utterances": counsellor_utterances,
            "client_utterances": client_utterances,
            "counsellor_words": counsellor_words,
            "client_words": client_words,
            "session_duration_minutes": session_duration_minutes,
        }

    return transcripts


def main() -> None:
    if not INPUT_CSV.exists():
        raise FileNotFoundError(f"Could not locate input CSV: {INPUT_CSV}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    with INPUT_CSV.open(newline="", encoding="utf-8") as csv_file:
        reader = csv.DictReader(csv_file)
        rows = list(reader)

    transcripts = build_transcripts(rows)

    for participant_id, transcript in transcripts.items():
        output_path = OUTPUT_DIR / f"{participant_id}.json"
        with output_path.open("w", encoding="utf-8") as json_file:
            json.dump(transcript, json_file, ensure_ascii=False, indent=2)

    expected_count = len(transcripts)
    actual_count = len(list(OUTPUT_DIR.glob("*.json")))
    if expected_count != actual_count:
        raise RuntimeError(
            "Mismatch between participants and written transcripts: "
            f"expected {expected_count}, found {actual_count}"
        )


if __name__ == "__main__":
    main()
