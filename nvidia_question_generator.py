#!/usr/bin/env python3
"""
NVIDIA nemotron-3-super-120b parallel question generator — Task 20.
2 threads, one per API key, both in-flight simultaneously → 2× throughput.

Usage:
    python scripts/nvidia_question_generator.py
    python scripts/nvidia_question_generator.py --topic T4_FD_Normalization --types MCQ MSQ
    python scripts/nvidia_question_generator.py --topic B1_Transactions_Concurrency_Recovery --types NAT
"""

import argparse
import json
import logging
import os
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent / ".env")

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

ROOT = Path(__file__).parent.parent
NOTES_DIR = ROOT / "notes" / "master_notes"
PYQ_FILE = ROOT / "pyq" / "gate_da" / "weekly_tests_tagged.json"
OUTPUT_DIR = ROOT / "questions"
CKPT_DIR = ROOT / "questions" / "_checkpoints_nvidia"

for d in [OUTPUT_DIR / t for t in ("mcq", "msq", "nat", "pyq_style")] + [CKPT_DIR]:
    d.mkdir(parents=True, exist_ok=True)

BATCH_SIZE = 2
MAX_RETRIES = 4
CKPT_EVERY = 15
MODEL = "nvidia/llama-3.3-nemotron-super-49b-v1"
NVIDIA_URL = "https://integrate.api.nvidia.com/v1/chat/completions"

SYSTEM = """You are a GATE DA exam question author specializing in DBMS.
Generate technically rigorous, tricky, conceptual exam-ready questions.
Focus on misconceptions, edge cases, and multi-step reasoning.

Output ONLY a JSON array — no markdown fences, no commentary, no explanation outside JSON.

Each question object:
{
  "id": "string",
  "topic": "string",
  "subtopic": "string",
  "type": "MCQ" | "MSQ" | "NAT",
  "marks": 1 or 2,
  "difficulty": "medium" | "hard",
  "question": "string",
  "options": ["A. ...", "B. ...", "C. ...", "D. ..."],
  "answer": "B",
  "explanation": "string",
  "source": {"file": "string", "page": 0},
  "traps": ["string"]
}
For NAT: omit options, answer = numeric string, add nat_tolerance: 0.
For MSQ: answer = comma-separated e.g. "A,C"."""

TOPICS = {
    "T4_FD_Normalization": {
        "note_file": "T4_FD_Normalization.md",
        "id_prefix": "T4",
        "subtopics": [
            "functional_dependency_definition_and_trivial_FDs",
            "attribute_closure_and_FD_inference_rules_Armstrong",
            "candidate_keys_superkeys_prime_nonprime_attributes",
            "minimal_cover_canonical_cover",
            "1NF_2NF_3NF_definitions_and_violations",
            "BCNF_definition_and_BCNF_vs_3NF_tradeoffs",
            "lossless_decomposition_test",
            "dependency_preserving_decomposition_test",
            "BCNF_decomposition_algorithm",
            "3NF_synthesis_algorithm",
            "multivalued_dependencies_4NF",
        ],
        "targets": {"MCQ": 120, "MSQ": 60, "NAT": 40, "pyq_style": 40},
    },
    "B1_Transactions_Concurrency_Recovery": {
        "note_file": "B1_Transactions_Concurrency_Recovery.md",
        "id_prefix": "B1",
        "subtopics": [
            "ACID_properties_atomicity_consistency_isolation_durability",
            "serializability_conflict_serializability_precedence_graph",
            "view_serializability_vs_conflict_serializability",
            "2PL_strict_2PL_conservative_2PL_differences",
            "deadlock_detection_prevention_wound_wait_wait_die",
            "timestamp_ordering_protocol",
            "WAL_protocol_and_recovery_undo_redo_undo_redo_logs",
            "checkpoint_and_recovery_after_crash",
            "cascading_rollback_cascadeless_schedules",
            "isolation_levels_read_uncommitted_to_serializable",
        ],
        "targets": {"MCQ": 120, "MSQ": 60, "NAT": 30, "pyq_style": 40},
    },
}


def load_keys():
    keys = [v.strip() for k in ("NVIDIA_API_KEY", "NVIDIA_API_KEY_1", "NVIDIA_API_KEY_2")
            if (v := os.environ.get(k, "")).strip()]
    log.info("Loaded %d NVIDIA API keys → %d parallel workers", len(keys), len(keys))
    return keys


def nvidia_call(key: str, system: str, user: str) -> str:
    resp = requests.post(
        NVIDIA_URL,
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
        json={
            "model": MODEL,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            "max_tokens": 4096,
            "temperature": 0.6,
            "top_p": 0.9,
        },
        timeout=180,
    )
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"]


def extract_json(raw: str):
    raw = re.sub(r"```(?:json)?|```", "", raw).strip()
    try:
        parsed = json.loads(raw)
        if isinstance(parsed, list):
            return parsed
    except json.JSONDecodeError:
        pass
    start, end = raw.find("["), raw.rfind("]")
    if start == -1 or end <= start:
        return None
    try:
        return json.loads(raw[start:end + 1])
    except json.JSONDecodeError:
        return None


def validate_question(q: dict, qtype: str) -> bool:
    if not isinstance(q, dict):
        return False
    if "question" not in q and "stem" not in q:
        return False
    if "answer" not in q and "correct_answer" not in q:
        return False
    if qtype in ("MCQ", "MSQ") and "options" not in q:
        return False
    return True


def backfill(q: dict, topic_key: str, note_file: str, qtype: str, idx: int, id_prefix: str) -> dict:
    if "stem" in q and "question" not in q:
        q["question"] = q.pop("stem")
    if "correct_answer" in q and "answer" not in q:
        q["answer"] = q.pop("correct_answer")
    q.setdefault("id", f"{id_prefix}_{qtype}_{idx:03d}")
    q.setdefault("topic", topic_key)
    q.setdefault("subtopic", "general")
    q.setdefault("type", qtype)
    q.setdefault("difficulty", "hard")
    q.setdefault("marks", 2 if qtype == "NAT" else 1)
    q.setdefault("explanation", "")
    q.setdefault("source", {"file": note_file, "page": 0})
    if isinstance(q.get("source"), str):
        q["source"] = {"file": q["source"], "page": 0}
    q.setdefault("traps", [])
    return q


def ckpt_path(topic, qtype):
    return CKPT_DIR / f"{topic}_{qtype}.json"


def load_checkpoint(topic, qtype):
    p = ckpt_path(topic, qtype)
    if p.exists():
        data = json.loads(p.read_text(encoding="utf-8"))
        log.info("Resumed checkpoint %s/%s: %d questions", topic, qtype, len(data))
        return data
    return []


def save_checkpoint(questions, topic, qtype, lock):
    with lock:
        ckpt_path(topic, qtype).write_text(
            json.dumps(questions, indent=2, ensure_ascii=False), encoding="utf-8"
        )


def save_output(questions, topic, qtype):
    subdir = "pyq_style" if qtype == "pyq_style" else qtype.lower()
    out = OUTPUT_DIR / subdir / f"{topic}_nvidia.json"
    out.write_text(json.dumps(questions, indent=2, ensure_ascii=False), encoding="utf-8")
    log.info("Saved %d questions -> %s", len(questions), out)


def build_prompt(qtype, topic_key, subtopic, note_excerpt, note_file, n, id_prefix, start_id, pyq_examples=None):
    if qtype == "MCQ":
        return f"""Topic: {topic_key}\nSubtopic: {subtopic}\n\nMaster notes:\n{note_excerpt}\n\nGenerate exactly {n} GATE-style TRICKY MCQ questions on "{subtopic}".\n- Hard distractors based on real misconceptions\n- At least one multi-step reasoning question\n- explanation: WHY correct + WHY each distractor wrong (1 line each)\n- traps: misconception tag strings\n- IDs: {id_prefix}_MCQ_{start_id:03d} to {id_prefix}_MCQ_{start_id+n-1:03d}\n- source.file = "{note_file}"\nOutput JSON array of {n} objects."""
    if qtype == "MSQ":
        return f"""Topic: {topic_key}\nSubtopic: {subtopic}\n\nMaster notes:\n{note_excerpt}\n\nGenerate exactly {n} GATE-style MSQ questions on "{subtopic}".\n- 2-3 correct answers; wrong options must be genuinely tempting\n- answer: comma-separated e.g. "A,C"\n- IDs: {id_prefix}_MSQ_{start_id:03d} to {id_prefix}_MSQ_{start_id+n-1:03d}\nOutput JSON array of {n} objects."""
    if qtype == "NAT":
        return f"""Topic: {topic_key}\nSubtopic: {subtopic}\n\nMaster notes:\n{note_excerpt}\n\nGenerate exactly {n} GATE-style NAT questions on "{subtopic}".\n- Concrete numbers required (FD counts, normal form checks, lock steps, log records)\n- explanation: step-by-step calculation\n- marks: 2 for all NAT\n- IDs: {id_prefix}_NAT_{start_id:03d} to {id_prefix}_NAT_{start_id+n-1:03d}\nOutput JSON array of {n} objects."""
    if qtype == "pyq_style":
        ex = json.dumps((pyq_examples or [])[:4], indent=2)
        return f"""Topic: {topic_key}\n\nPYQ examples (style guide):\n{ex}\n\nMaster notes:\n{note_excerpt}\n\nGenerate exactly {n} GATE-style questions matching PYQ depth.\n- Mix MCQ and NAT\n- Change all values/relations — do not copy\n- IDs: {id_prefix}_PYQ_{start_id:03d} to {id_prefix}_PYQ_{start_id+n-1:03d}\nOutput JSON array of {n} objects."""


def run_batch(key: str, system: str, prompt: str, qtype: str) -> list:
    """Single batch call — runs in a worker thread."""
    for attempt in range(MAX_RETRIES):
        try:
            raw = nvidia_call(key, system, prompt)
            parsed = extract_json(raw)
            if parsed is None:
                log.warning("JSON parse fail (attempt %d/%d)", attempt + 1, MAX_RETRIES)
                time.sleep(2)
                continue
            accept = "MCQ" if qtype != "NAT" else "NAT"
            valid = [q for q in parsed if validate_question(q, accept) or validate_question(q, "NAT")]
            if not valid:
                log.warning("No valid questions (attempt %d/%d)", attempt + 1, MAX_RETRIES)
                time.sleep(2)
                continue
            return valid
        except Exception as e:
            log.error("Batch error attempt %d: %s", attempt + 1, e)
            time.sleep(5)
    return []


def generate_for_type(topic_key, cfg, note_text, qtype, pyq_data, types_requested, keys):
    if qtype not in types_requested:
        return []
    target = cfg["targets"].get(qtype, 0)
    if target == 0:
        return []

    questions = load_checkpoint(topic_key, qtype)
    if len(questions) >= target:
        log.info("%s/%s already complete (%d/%d)", topic_key, qtype, len(questions), target)
        return questions[:target]

    note_file = cfg["note_file"]
    note_excerpt = note_text[:6000]
    id_prefix = cfg["id_prefix"]
    subtopics = cfg["subtopics"]
    topic_pyqs = [p for p in pyq_data if p.get("topic_bucket") == topic_key]

    # Pre-build all remaining batch tasks
    needed = target - len(questions)
    subtopic_cycle = (subtopics * ((needed // (len(subtopics) * BATCH_SIZE)) + 2))
    batches = []
    sid = len(questions) + 1
    for i in range(0, needed, BATCH_SIZE):
        n = min(BATCH_SIZE, needed - i)
        subtopic = subtopic_cycle[i // BATCH_SIZE % len(subtopic_cycle)]
        prompt = build_prompt(qtype, topic_key, subtopic, note_excerpt, note_file,
                              n, id_prefix, sid, topic_pyqs)
        batches.append((sid, n, prompt))
        sid += n

    results = list(questions)  # start from checkpoint
    lock = threading.Lock()
    n_workers = len(keys)

    log.info("%s/%s: %d questions needed → %d batches → %d workers",
             topic_key, qtype, needed, len(batches), n_workers)

    with ThreadPoolExecutor(max_workers=n_workers) as pool:
        # pin each batch to a key by index
        futures = {
            pool.submit(run_batch, keys[i % n_workers], SYSTEM, prompt, qtype): (start_id, batch_n)
            for i, (start_id, batch_n, prompt) in enumerate(batches)
        }
        for future in as_completed(futures):
            start_id, batch_n = futures[future]
            raw_qs = future.result()
            if not raw_qs:
                log.warning("Batch starting id=%d returned nothing", start_id)
                continue
            valid = [backfill(q, topic_key, note_file, qtype, start_id + j, id_prefix)
                     for j, q in enumerate(raw_qs)]
            with lock:
                results.extend(valid)
                count = len(results)
                log.info("[%s/%s] %d/%d", topic_key, qtype, count, target)
            if count % CKPT_EVERY < BATCH_SIZE:
                save_checkpoint(results, topic_key, qtype, lock)

    save_checkpoint(results, topic_key, qtype, lock)
    return results[:target]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--topic", default=None)
    parser.add_argument("--types", nargs="+", default=["MCQ", "MSQ", "NAT", "pyq_style"])
    args = parser.parse_args()

    keys = load_keys()
    if not keys:
        log.error("No NVIDIA API keys found in .env")
        sys.exit(1)

    pyq_data = []
    if PYQ_FILE.exists():
        pyq_data = json.loads(PYQ_FILE.read_text(encoding="utf-8"))
        log.info("Loaded %d PYQ questions", len(pyq_data))

    topics = {k: v for k, v in TOPICS.items() if args.topic is None or k == args.topic}
    if not topics:
        log.error("Topic not found: %s", args.topic)
        sys.exit(1)

    total = 0
    for topic_key, cfg in topics.items():
        note_path = NOTES_DIR / cfg["note_file"]
        if not note_path.exists():
            log.error("Note missing: %s — skipping", note_path)
            continue
        note_text = note_path.read_text(encoding="utf-8")
        log.info("=== %s ===", topic_key)
        for qtype in ("MCQ", "MSQ", "NAT", "pyq_style"):
            qs = generate_for_type(topic_key, cfg, note_text, qtype, pyq_data, args.types, keys)
            if qs:
                save_output(qs, topic_key, qtype)
                total += len(qs)

    log.info("Done. Total: %d questions", total)
    print(f"\nTotal generated: {total}")
    print(f"Output: {OUTPUT_DIR}/<type>/<topic>_nvidia.json")


if __name__ == "__main__":
    main()
