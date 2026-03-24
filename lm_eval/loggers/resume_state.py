from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from lm_eval.utils import handle_non_serializable, hash_string


class ResumeStateManager:
    """Persist and restore per-sample evaluation progress for resumable runs."""

    STATE_VERSION = 1

    def __init__(
        self,
        output_path: str,
        *,
        run_config: dict[str, Any],
        task_filters: dict[str, list[str]],
    ) -> None:
        output = Path(output_path)
        state_root = output.parent if output.suffix == ".json" else output
        self.state_dir = state_root / ".lm_eval_resume"
        self.manifest_path = self.state_dir / "manifest.json"
        self.run_config = run_config
        self.task_filters = task_filters
        dumped = json.dumps(
            run_config,
            sort_keys=True,
            default=handle_non_serializable,
            ensure_ascii=False,
        )
        self.fingerprint = hash_string(dumped)

    def prepare(self) -> None:
        """Create or validate the manifest for the resumable run."""
        self.state_dir.mkdir(parents=True, exist_ok=True)
        expected = self._manifest_data()
        if self.manifest_path.exists():
            current = json.loads(self.manifest_path.read_text(encoding="utf-8"))
            if current.get("state_version") != self.STATE_VERSION:
                raise ValueError(
                    "Existing resume state uses an unsupported version. "
                    "Delete the '.lm_eval_resume' directory or choose a new "
                    "output_path."
                )
            if current.get("fingerprint") != self.fingerprint:
                raise ValueError(
                    "Existing resume state does not match this run's configuration. "
                    "Delete the '.lm_eval_resume' directory or choose a new "
                    "output_path."
                )
            if current.get("task_filters") != self.task_filters:
                raise ValueError(
                    "Existing resume state has different task filters. "
                    "Delete the '.lm_eval_resume' directory or choose a new "
                    "output_path."
                )
            return

        self.manifest_path.write_text(
            json.dumps(expected, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    def restore_task_history(
        self,
        task_name: str,
        *,
        acc: dict[str, Any],
        log_samples: bool,
    ) -> set[int]:
        """Populate accumulators from persisted sample records."""
        samples = self.load_task_samples(task_name)
        for sample in samples:
            filter_name = sample["filter"]
            for metric_name in sample["metrics"]:
                acc["raw_metrics"][(metric_name, filter_name)].append(
                    sample[metric_name]
                )
            if log_samples:
                acc["logged_samples"].append(sample)

        expected_filters = set(self.task_filters.get(task_name, ["none"]))
        seen_filters: dict[int, set[str]] = {}
        for sample in samples:
            seen_filters.setdefault(int(sample["doc_id"]), set()).add(sample["filter"])

        return {
            doc_id
            for doc_id, filters in seen_filters.items()
            if expected_filters.issubset(filters)
        }

    def load_task_samples(self, task_name: str) -> list[dict[str, Any]]:
        """Load and deduplicate stored samples for a task."""
        path = self._task_path(task_name)
        if not path.exists():
            return []

        deduped: dict[tuple[int, str], dict[str, Any]] = {}
        with path.open(encoding="utf-8") as handle:
            for lineno, line in enumerate(handle, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    sample = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        f"Corrupted resume state in {path} at line {lineno}."
                    ) from exc
                deduped[(int(sample["doc_id"]), sample["filter"])] = sample

        return [
            deduped[key]
            for key in sorted(deduped, key=lambda item: (item[0], item[1]))
        ]

    def append_sample(self, task_name: str, sample: dict[str, Any]) -> None:
        """Append a single sample record to the task journal."""
        path = self._task_path(task_name)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(
                json.dumps(
                    sample,
                    default=handle_non_serializable,
                    ensure_ascii=False,
                )
                + "\n"
            )
            handle.flush()
            os.fsync(handle.fileno())

    def _manifest_data(self) -> dict[str, Any]:
        return {
            "state_version": self.STATE_VERSION,
            "fingerprint": self.fingerprint,
            "run_config": self.run_config,
            "task_filters": self.task_filters,
        }

    def _task_path(self, task_name: str) -> Path:
        return self.state_dir / f"samples_{task_name}.jsonl"
