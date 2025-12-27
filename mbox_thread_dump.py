#!/usr/bin/env python3
"""CLI tool skeleton for filtering and exporting threads from an mbox."""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from datetime import datetime, timezone
from email import policy
from email.message import Message
from email.utils import getaddresses, parsedate_to_datetime
import hashlib
import mailbox
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple


def load_targets(path: Path) -> Set[str]:
    """Load target email addresses from a file; ignore blank lines and # comments."""
    targets: Set[str] = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        targets.add(stripped.lower())
    return targets


def parse_mbox(
    path: Path,
    *,
    verbose: bool = False,
    progress_every: int = 5000,
    max_body_hash_bytes: int = 65536,
    high_memory: bool = False,
) -> Dict[str, object]:
    """Parse the mbox file and build indexes for threading and dedupe."""
    id_to_msg: Dict[str, dict] = {}
    child_to_parents: Dict[str, Set[str]] = {}
    parent_to_children: Dict[str, Set[str]] = {}
    content_hash_to_ids: Dict[str, Set[str]] = {}
    messages: List[dict] = []

    factory = lambda f: mailbox.mboxMessage(f)
    mbox = mailbox.mbox(path, factory=factory)
    try:
        for count, (key, message) in enumerate(mbox.iteritems(), start=1):
            record = _build_record(message, key, max_body_hash_bytes, high_memory)
            messages.append(record)

            message_id = record["message_id"]
            if message_id:
                id_to_msg[message_id] = record

                parents = set(record["parent_ids"])
                if parents:
                    child_to_parents.setdefault(message_id, set()).update(parents)
                    for parent_id in parents:
                        parent_to_children.setdefault(parent_id, set()).add(message_id)

                content_hash_to_ids.setdefault(record["content_hash"], set()).add(message_id)
            else:
                content_hash_to_ids.setdefault(record["content_hash"], set()).add(
                    record["fallback_id"]
                )

            if verbose and count % progress_every == 0:
                logging.getLogger(__name__).info("Parsed %d messages...", count)
    finally:
        mbox.close()

    return {
        "messages": messages,
        "id_to_msg": id_to_msg,
        "child_to_parents": child_to_parents,
        "parent_to_children": parent_to_children,
        "content_hash_to_ids": content_hash_to_ids,
    }


def extract_addresses(message: Message, include_cc: bool) -> Set[str]:
    """Extract addresses from message headers (To/From/Cc as configured)."""
    headers = ["From", "To"]
    if include_cc:
        headers.append("Cc")
    return _extract_addresses_from_headers(message, headers)


def build_thread_graph(parsed: Dict[str, object]) -> dict:
    """Build a thread graph keyed by message-id with parent/child references."""
    return {
        "child_to_parents": parsed["child_to_parents"],
        "parent_to_children": parsed["parent_to_children"],
    }


def find_seed_ids(
    messages: Iterable[dict],
    targets: Set[str],
    include_cc: bool,
    date_min: Optional[datetime] = None,
    date_max: Optional[datetime] = None,
) -> Tuple[Set[str], List[dict]]:
    """Find seed message ids (and standalone no-id matches) that match targets."""
    seed_ids: Set[str] = set()
    standalone_records: List[dict] = []

    for record in messages:
        if not _date_in_range(record.get("date"), date_min, date_max):
            continue

        from_addresses = _extract_addresses_from_values([record.get("from", "")])
        headers = [record.get("to", "")]
        if include_cc:
            headers.append(record.get("cc", ""))
        to_cc_addresses = _extract_addresses_from_values(headers)

        if targets.intersection(from_addresses) or targets.intersection(to_cc_addresses):
            if record["message_id"]:
                seed_ids.add(record["message_id"])
            else:
                standalone_records.append(record)

    return seed_ids, standalone_records


def expand_to_thread(
    seed_ids: Set[str],
    thread_graph: dict,
) -> Tuple[Set[str], Dict[str, Set[str]], Dict[str, Set[str]]]:
    """Expand seeds to connected components; return ids, thread->seed, and thread->ids."""
    child_to_parents = thread_graph["child_to_parents"]
    parent_to_children = thread_graph["parent_to_children"]

    def neighbors(message_id: str) -> Set[str]:
        return set(child_to_parents.get(message_id, set())) | set(
            parent_to_children.get(message_id, set())
        )

    visited: Set[str] = set()
    selected_ids: Set[str] = set()
    thread_to_seeds: Dict[str, Set[str]] = {}
    thread_to_ids: Dict[str, Set[str]] = {}

    for seed_id in seed_ids:
        if seed_id in visited:
            continue

        stack = [seed_id]
        component: Set[str] = set()
        while stack:
            current = stack.pop()
            if current in visited:
                continue
            visited.add(current)
            component.add(current)
            stack.extend(neighbors(current))

        component_seeds = component.intersection(seed_ids)
        if component_seeds:
            selected_ids.update(component)
            thread_id = _compute_thread_id(message_ids=component, content_hashes=None)
            thread_to_seeds[thread_id] = component_seeds
            thread_to_ids[thread_id] = component

    return selected_ids, thread_to_seeds, thread_to_ids


def dedupe_messages(messages: Iterable[dict]) -> Tuple[List[dict], List[dict]]:
    """Deduplicate messages by message-id, falling back to content hash when needed."""
    deduped: List[dict] = []
    duplicates: List[dict] = []
    seen_message_ids: Dict[str, str] = {}
    seen_hashes: Set[str] = set()

    for record in messages:
        message_id = record["message_id"]
        content_hash = record["content_hash"]
        if message_id:
            if message_id not in seen_message_ids:
                seen_message_ids[message_id] = content_hash
                seen_hashes.add(content_hash)
                deduped.append(record)
                continue
            if seen_message_ids[message_id] == content_hash:
                duplicates.append(
                    _duplicate_entry(record, "same message-id")
                )
                continue

        if content_hash in seen_hashes:
            duplicates.append(
                _duplicate_entry(record, "same hash")
            )
            continue

        seen_hashes.add(content_hash)
        deduped.append(record)

    return deduped, duplicates


def export_messages(
    messages: Iterable[dict],
    outdir: Path,
    out_mbox: Path,
    mode: str,
    mbox_path: Path,
) -> None:
    """Export messages to .eml files, an mbox, or both based on mode."""
    if mode in ("mbox", "both"):
        _export_mbox(messages, out_mbox, mbox_path)
    if mode in ("eml", "both"):
        _export_eml(messages, outdir, mbox_path)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Filter mbox threads by target addresses")
    parser.add_argument("--mbox", required=True, help="Path to source mbox")
    parser.add_argument("--targets", default="targets.txt", help="Path to targets file")
    parser.add_argument("--outdir", default="out/", help="Directory for .eml output")
    parser.add_argument("--out-mbox", default="out/filtered.mbox", help="Path for filtered mbox")
    parser.add_argument(
        "--mode",
        choices=("mbox", "eml", "both"),
        default="both",
        help="Export mode",
    )
    parser.add_argument(
        "--include-cc",
        default=True,
        action=argparse.BooleanOptionalAction,
        help="Include Cc headers when matching targets",
    )
    parser.add_argument("--date-min", help="Minimum date (YYYY-MM-DD)")
    parser.add_argument("--date-max", help="Maximum date (YYYY-MM-DD)")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    parser.add_argument(
        "--log",
        help="Write verbose logs to a file",
    )
    memory_group = parser.add_mutually_exclusive_group()
    memory_group.add_argument(
        "--high-memory",
        action="store_true",
        help="Load full messages and hash full bytes",
    )
    memory_group.add_argument(
        "--low-memory",
        action="store_true",
        help="Minimize memory by storing metadata only",
    )
    parser.add_argument(
        "--max-body-hash-bytes",
        type=int,
        default=65536,
        help="Max body bytes for content hash",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=5000,
        help="Progress print interval for --verbose",
    )
    parser.add_argument("--dry-run", action="store_true", help="Only print counts")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)

    _configure_logging(args.verbose, args.log)

    mbox_path = Path(args.mbox)
    if not mbox_path.exists():
        print(
            f"mbox not found at {mbox_path}. "
            "Use --mbox PATH to point to your .mbox file."
        )
        return 2

    targets_path = Path(args.targets)
    targets = load_targets(targets_path)
    if not targets:
        print(
            f"No targets found in {targets_path}. "
            "Add one email address per line to proceed."
        )
        return 2

    high_memory = _resolve_high_memory(args.high_memory, args.low_memory)
    parsed = parse_mbox(
        mbox_path,
        verbose=args.verbose,
        progress_every=args.progress_every,
        max_body_hash_bytes=args.max_body_hash_bytes,
        high_memory=high_memory,
    )
    messages = parsed["messages"]
    thread_graph = build_thread_graph(parsed)
    date_min = _parse_date_bound(args.date_min)
    date_max = _parse_date_bound(args.date_max)
    seed_ids, seed_noid_records = find_seed_ids(
        messages, targets, args.include_cc, date_min, date_max
    )
    selected_ids, thread_to_seeds, thread_to_ids = expand_to_thread(seed_ids, thread_graph)
    id_to_msg = parsed["id_to_msg"]
    selected_records = [
        id_to_msg[message_id]
        for message_id in selected_ids
        if message_id in id_to_msg
    ]
    selected_records.extend(seed_noid_records)
    thread_id_by_message_id = {
        message_id: thread_id
        for thread_id, message_ids in thread_to_ids.items()
        for message_id in message_ids
    }
    for record in selected_records:
        message_id = record.get("message_id")
        if message_id and message_id in thread_id_by_message_id:
            record["thread_id"] = thread_id_by_message_id[message_id]
        else:
            record["thread_id"] = _compute_thread_id(
                message_ids=None,
                content_hashes={record["content_hash"]},
            )
    deduped, duplicates = dedupe_messages(selected_records)

    total_messages = len(messages)
    seed_matches = len(seed_ids) + len(seed_noid_records)
    threads_matched = len(thread_to_seeds)
    selected_count = len(deduped)
    duplicates_removed = len(duplicates)

    if args.dry_run:
        print(
            "total_messages={total} seed_matches={seeds} threads_matched={threads} "
            "selected={selected} duplicates_removed={duplicates}".format(
                total=total_messages,
                seeds=seed_matches,
                threads=threads_matched,
                selected=selected_count,
                duplicates=duplicates_removed,
            )
        )
        print(
            "summary: parsed={total} seeds={seeds} threads={threads} "
            "selected={selected} duplicates={duplicates}".format(
                total=total_messages,
                seeds=seed_matches,
                threads=threads_matched,
                selected=selected_count,
                duplicates=duplicates_removed,
            )
        )
        return 0

    report = _build_report(
        deduped=deduped,
        duplicates=duplicates,
        thread_to_ids=thread_to_ids,
        thread_to_seeds=thread_to_seeds,
        id_to_msg=id_to_msg,
        standalone_records=seed_noid_records,
        targets=targets,
        include_cc=args.include_cc,
    )
    _write_report(Path(args.outdir), report)

    export_messages(deduped, Path(args.outdir), Path(args.out_mbox), args.mode, mbox_path)
    print(
        "summary: parsed={total} seeds={seeds} threads={threads} "
        "selected={selected} duplicates={duplicates}".format(
            total=total_messages,
            seeds=seed_matches,
            threads=threads_matched,
            selected=selected_count,
            duplicates=duplicates_removed,
        )
    )
    return 0


def _normalize_message_id(value: Optional[str]) -> Optional[str]:
    if not value:
        return None
    trimmed = value.strip()
    if trimmed.startswith("<") and trimmed.endswith(">"):
        trimmed = trimmed[1:-1].strip()
    return trimmed or None


def _extract_addresses_from_headers(message: Message, headers: Iterable[str]) -> Set[str]:
    header_values: List[str] = []
    for header in headers:
        header_values.extend(message.get_all(header, []))

    addresses = set()
    for _, addr in getaddresses(header_values):
        normalized = addr.strip().lower()
        if normalized:
            addresses.add(normalized)
    return addresses


def _extract_addresses_from_values(values: Iterable[str]) -> Set[str]:
    header_values: List[str] = []
    for value in values:
        if value:
            header_values.append(value)
    addresses = set()
    for _, addr in getaddresses(header_values):
        normalized = addr.strip().lower()
        if normalized:
            addresses.add(normalized)
    return addresses


def _duplicate_entry(record: dict, reason: str) -> dict:
    return {
        "message_id": record["message_id"],
        "fallback_id": record["fallback_id"],
        "content_hash": record["content_hash"],
        "reason": reason,
    }


def _compute_thread_id(
    *,
    message_ids: Optional[Iterable[str]],
    content_hashes: Optional[Iterable[str]],
) -> str:
    items: List[str] = []
    if message_ids:
        items = sorted(message_ids)
    elif content_hashes:
        items = sorted(content_hashes)

    digest = hashlib.sha256()
    digest.update("\n".join(items).encode("utf-8", errors="ignore"))
    return digest.hexdigest()


def _build_report(
    *,
    deduped: List[dict],
    duplicates: List[dict],
    thread_to_ids: Dict[str, Set[str]],
    thread_to_seeds: Dict[str, Set[str]],
    id_to_msg: Dict[str, dict],
    standalone_records: List[dict],
    targets: Set[str],
    include_cc: bool,
) -> dict:
    selected_ids = [
        record["message_id"] or record["fallback_id"] for record in deduped
    ]
    thread_summaries: List[dict] = []

    for thread_id, message_ids in thread_to_ids.items():
        records = [id_to_msg[mid] for mid in message_ids if mid in id_to_msg]
        subject = next((r["subject"] for r in records if r.get("subject")), None)
        involved_targets = _collect_involved_targets(records, targets, include_cc)
        thread_summaries.append(
            {
                "thread_id": thread_id,
                "thread_size": len(message_ids),
                "sample_subject": subject,
                "involved_targets": sorted(involved_targets),
                "seed_message_ids": sorted(thread_to_seeds.get(thread_id, set())),
            }
        )

    for record in standalone_records:
        involved_targets = _collect_involved_targets([record], targets, include_cc)
        thread_summaries.append(
            {
                "thread_id": _compute_thread_id(
                    message_ids=None,
                    content_hashes={record["content_hash"]},
                ),
                "thread_size": 1,
                "sample_subject": record.get("subject"),
                "involved_targets": sorted(involved_targets),
                "seed_message_ids": [],
            }
        )

    return {
        "selected_message_ids": selected_ids,
        "duplicates_dropped": duplicates,
        "thread_summaries": thread_summaries,
    }


def _collect_involved_targets(
    records: Iterable[dict],
    targets: Set[str],
    include_cc: bool,
) -> Set[str]:
    involved: Set[str] = set()
    for record in records:
        addresses = _extract_addresses_from_values(
            [record.get("from", ""), record.get("to", ""), record.get("cc", "")]
            if include_cc
            else [record.get("from", ""), record.get("to", "")]
        )
        involved.update(addresses.intersection(targets))
    return involved


def _write_report(outdir: Path, report: dict) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    report_path = outdir / "report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")


def _export_mbox(records: Iterable[dict], out_mbox: Path, mbox_path: Path) -> None:
    out_mbox.parent.mkdir(parents=True, exist_ok=True)
    mbox = mailbox.mbox(out_mbox)
    source = mailbox.mbox(mbox_path, factory=lambda f: mailbox.mboxMessage(f))
    try:
        for record in _sorted_records(records):
            message = _record_message(record, source)
            if message is None:
                continue
            mbox.add(message)
        mbox.flush()
    finally:
        mbox.close()
        source.close()


def _export_eml(records: Iterable[dict], outdir: Path, mbox_path: Path) -> None:
    threads_root = outdir / "threads"
    threads_root.mkdir(parents=True, exist_ok=True)

    threads: Dict[str, List[dict]] = {}
    for record in records:
        thread_id = record.get("thread_id")
        if not thread_id:
            thread_id = _compute_thread_id(
                message_ids=None, content_hashes={record["content_hash"]}
            )
        threads.setdefault(thread_id, []).append(record)

    source = mailbox.mbox(mbox_path, factory=lambda f: mailbox.mboxMessage(f))
    for thread_id, thread_records in threads.items():
        thread_dir = threads_root / thread_id
        thread_dir.mkdir(parents=True, exist_ok=True)
        ordered = _sorted_records(thread_records)
        index_entries: List[dict] = []

        for seq, record in enumerate(ordered, start=1):
            message = _record_message(record, source)
            if message is None:
                continue
            safe_id = _sanitize_filename(
                record["message_id"] or record["content_hash"] or record["fallback_id"]
            )
            filename = f"{seq:04d}_{safe_id}.eml"
            path = thread_dir / filename
            path.write_bytes(message.as_bytes(policy=policy.compat32))

            index_entries.append(
                {
                    "Date": record.get("date_header"),
                    "From": record.get("from"),
                    "To": record.get("to"),
                    "Subject": record.get("subject"),
                    "Message-ID": record.get("message_id"),
                }
            )

        index_path = thread_dir / "thread_index.json"
        index_path.write_text(
            json.dumps(index_entries, indent=2, sort_keys=True),
            encoding="utf-8",
        )
    source.close()


def _sorted_records(records: Iterable[dict]) -> List[dict]:
    return sorted(records, key=_record_sort_key)


def _record_sort_key(record: dict) -> Tuple[int, float, int]:
    date_value = record.get("date")
    if date_value is None:
        return (1, 0.0, _mbox_key_to_int(record.get("mbox_key")))

    if date_value.tzinfo is None:
        timestamp = date_value.replace(tzinfo=timezone.utc).timestamp()
    else:
        timestamp = date_value.timestamp()
    return (0, timestamp, _mbox_key_to_int(record.get("mbox_key")))


def _sanitize_filename(value: str, max_length: int = 80) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("._-")
    if not cleaned:
        cleaned = "message"
    return cleaned[:max_length]


def _mbox_key_to_int(value: object) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _record_message(record: dict, source: mailbox.mbox) -> Optional[Message]:
    message = record.get("message")
    if message is not None:
        return message
    return source.get(record["mbox_key"])


def _extract_message_ids(header_value: Optional[str]) -> List[str]:
    if not header_value:
        return []
    matches = re.findall(r"<([^>]+)>", header_value)
    if matches:
        return [_normalize_message_id(match) for match in matches if _normalize_message_id(match)]
    tokens = re.split(r"[\s,]+", header_value.strip())
    return [_normalize_message_id(token) for token in tokens if _normalize_message_id(token)]


def _parse_message_date(value: Optional[str]) -> Optional[datetime]:
    if not value:
        return None
    try:
        return parsedate_to_datetime(value)
    except (TypeError, ValueError):
        return None


def _compute_content_hash(
    message: Message,
    body_bytes_limit: int = 65536,
    full_message_bytes: bool = False,
) -> str:
    if full_message_bytes:
        digest = hashlib.sha256()
        digest.update(message.as_bytes(policy=policy.compat32))
        return digest.hexdigest()

    header_names = ("Message-ID", "Date", "From", "To", "Cc", "Subject")
    header_values: List[str] = []
    for name in header_names:
        header_values.extend(message.get_all(name, []))
    header_blob = "\n".join(header_values)

    body_bytes = b""
    if message.is_multipart():
        for part in message.walk():
            if part.is_multipart():
                continue
            payload = part.get_payload(decode=True)
            if payload:
                body_bytes = payload
                break
    else:
        payload = message.get_payload(decode=True)
        if payload:
            body_bytes = payload

    if not body_bytes:
        payload_text = message.get_payload()
        if isinstance(payload_text, str):
            body_bytes = payload_text.encode(errors="ignore")

    digest = hashlib.sha256()
    digest.update(header_blob.encode("utf-8", errors="ignore"))
    digest.update(b"\n\n")
    digest.update(body_bytes[:body_bytes_limit])
    return digest.hexdigest()


def _build_record(
    message: Message,
    key: object,
    max_body_hash_bytes: int,
    high_memory: bool,
) -> dict:
    message_id = _normalize_message_id(message.get("Message-ID"))
    in_reply_to = message.get("In-Reply-To")
    references = message.get("References")
    parent_ids = _extract_message_ids(in_reply_to)
    parent_ids.extend(_extract_message_ids(references))
    date_header = message.get("Date")
    date_value = _parse_message_date(date_header)
    content_hash = _compute_content_hash(
        message,
        body_bytes_limit=max_body_hash_bytes,
        full_message_bytes=high_memory,
    )
    fallback_id = f"__noid__{key}"

    record = {
        "mbox_key": key,
        "message_id": message_id,
        "date": date_value,
        "date_header": date_header,
        "subject": message.get("Subject"),
        "from": message.get("From"),
        "to": message.get("To"),
        "cc": message.get("Cc"),
        "in_reply_to": in_reply_to,
        "references": references,
        "parent_ids": [pid for pid in parent_ids if pid],
        "content_hash": content_hash,
        "fallback_id": fallback_id,
    }
    if high_memory:
        record["message"] = message
    return record


def _date_in_range(
    value: Optional[datetime],
    date_min: Optional[datetime],
    date_max: Optional[datetime],
) -> bool:
    if value is None:
        return True
    current = value.date()
    if date_min and current < date_min.date():
        return False
    if date_max and current > date_max.date():
        return False
    return True


def _parse_date_bound(value: Optional[str]) -> Optional[datetime]:
    if not value:
        return None
    try:
        return datetime.strptime(value, "%Y-%m-%d")
    except ValueError:
        return None


def _resolve_high_memory(high_memory_flag: bool, low_memory_flag: bool) -> bool:
    if low_memory_flag:
        return False
    if high_memory_flag:
        return True
    return _detect_high_memory()


def _detect_high_memory(threshold_gb: int = 64) -> bool:
    total_bytes = _get_total_memory_bytes()
    if total_bytes is None:
        return False
    return total_bytes >= threshold_gb * 1024**3


def _get_total_memory_bytes() -> Optional[int]:
    if hasattr(os, "sysconf"):
        try:
            page_size = os.sysconf("SC_PAGE_SIZE")
            pages = os.sysconf("SC_PHYS_PAGES")
            if isinstance(page_size, int) and isinstance(pages, int):
                return page_size * pages
        except (ValueError, OSError):
            pass
    meminfo = Path("/proc/meminfo")
    if meminfo.exists():
        for line in meminfo.read_text(encoding="utf-8").splitlines():
            if line.startswith("MemTotal:"):
                parts = line.split()
                if len(parts) >= 2 and parts[1].isdigit():
                    return int(parts[1]) * 1024
    return None


def _configure_logging(verbose: bool, log_path: Optional[str]) -> None:
    handlers: List[logging.Handler] = []
    if verbose:
        handlers.append(logging.StreamHandler(sys.stderr))
    if log_path:
        handlers.append(logging.FileHandler(log_path))

    if handlers:
        logging.basicConfig(
            level=logging.INFO if (verbose or log_path) else logging.WARNING,
            format="%(asctime)s %(levelname)s %(message)s",
            handlers=handlers,
        )


if __name__ == "__main__":
    raise SystemExit(main())
