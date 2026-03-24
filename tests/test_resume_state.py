from __future__ import annotations

from functools import partial
from types import SimpleNamespace

from lm_eval.api.instance import Instance
from lm_eval.api.model import LM
from lm_eval.evaluator import evaluate
from lm_eval.filters import build_filter_ensemble
from lm_eval.loggers.resume_state import ResumeStateManager


class DummyGenerateLM(LM):
    def __init__(self, outputs: dict[str, str]) -> None:
        super().__init__()
        self.outputs = outputs
        self.calls: list[str] = []

    def loglikelihood(self, requests):
        raise NotImplementedError

    def loglikelihood_rolling(self, requests):
        raise NotImplementedError

    def generate_until(self, requests):
        prompts = [req.args[0] for req in requests]
        self.calls.extend(prompts)
        return [self.outputs[prompt] for prompt in prompts]


class DummyGenerateTask:
    VERSION = "test"
    OUTPUT_TYPE = "generate_until"

    def __init__(self) -> None:
        self.config = SimpleNamespace(task="dummy_generate", num_fewshot=0)
        self._filters = [build_filter_ensemble("none", [["take_first", None]])]
        self._instances = []
        self._docs = [
            {"question": "q0", "answer": "A"},
            {"question": "q1", "answer": "B"},
            {"question": "q2", "answer": "C"},
            {"question": "q3", "answer": "D"},
        ]

    @property
    def instances(self):
        return self._instances

    @property
    def eval_docs(self):
        return self._docs

    @property
    def task_name(self):
        return self.config.task

    def get_config(self, key):
        if key == "output_type":
            return self.OUTPUT_TYPE
        return getattr(self.config, key, None)

    def dump_config(self):
        return {
            "task": self.config.task,
            "output_type": self.OUTPUT_TYPE,
            "num_fewshot": self.config.num_fewshot,
        }

    def set_fewshot_seed(self, seed=None):
        return None

    def build_all_requests(
        self,
        *,
        limit=None,
        samples=None,
        rank=0,
        world_size=1,
        **kwargs,
    ):
        self._instances = []
        if samples is not None:
            doc_ids = list(samples)
        else:
            end = len(self._docs) if limit is None else int(limit)
            doc_ids = list(range(end))

        for local_doc_id, absolute_doc_id in enumerate(doc_ids):
            doc = self._docs[absolute_doc_id]
            self._instances.append(
                Instance(
                    request_type="generate_until",
                    doc=doc,
                    arguments=(self.doc_to_text(doc), {"until": ["\n"]}),
                    idx=0,
                    metadata=(self.task_name, local_doc_id, 1),
                )
            )

    def apply_filters(self):
        for filter_pipeline in self._filters:
            filter_pipeline.apply(self._instances)

    def doc_iterator(self, *, rank=0, limit=None, world_size=1, samples=None):
        if samples is not None:
            docs = [self._docs[idx] for idx in samples]
        else:
            end = len(self._docs) if limit is None else int(limit)
            docs = self._docs[:end]
        return iter(enumerate(docs))

    def doc_to_text(self, doc):
        return doc["question"]

    def doc_to_target(self, doc):
        return doc["answer"]

    def process_results(self, doc, results):
        return {"exact_match": 1.0 if results[0] == doc["answer"] else 0.0}

    def aggregation(self):
        return {"exact_match": lambda xs: sum(xs) / len(xs)}

    def higher_is_better(self):
        return {"exact_match": True}


def test_resume_state_skips_completed_docs(tmp_path):
    resume_state = ResumeStateManager(
        str(tmp_path),
        run_config={"task": "dummy_generate", "mode": "resume"},
        task_filters={"dummy_generate": ["none"]},
    )
    resume_state.prepare()
    outputs = {"q0": "A", "q1": "B", "q2": "C", "q3": "D"}

    first_lm = DummyGenerateLM(outputs)
    first_results = evaluate(
        lm=first_lm,
        task_dict={"tasks": {"dummy_generate": DummyGenerateTask()}, "groups": {}},
        limit=2,
        bootstrap_iters=0,
        log_samples=False,
        resume_state=resume_state,
    )
    assert first_results["results"]["dummy_generate"]["sample_len"] == 2
    assert first_lm.calls == ["q0", "q1"]

    second_lm = DummyGenerateLM(outputs)
    second_results = evaluate(
        lm=second_lm,
        task_dict={"tasks": {"dummy_generate": DummyGenerateTask()}, "groups": {}},
        limit=4,
        bootstrap_iters=0,
        log_samples=False,
        resume_state=resume_state,
    )
    assert second_results["results"]["dummy_generate"]["sample_len"] == 4
    assert second_results["results"]["dummy_generate"]["exact_match,none"] == 1.0
    assert second_lm.calls == ["q2", "q3"]

    third_lm = DummyGenerateLM(outputs)
    third_results = evaluate(
        lm=third_lm,
        task_dict={"tasks": {"dummy_generate": DummyGenerateTask()}, "groups": {}},
        limit=4,
        bootstrap_iters=0,
        log_samples=False,
        resume_state=resume_state,
    )
    assert third_results["results"]["dummy_generate"]["sample_len"] == 4
    assert third_results["results"]["dummy_generate"]["exact_match,none"] == 1.0
    assert third_lm.calls == []


def test_resume_state_rejects_mismatched_manifest(tmp_path):
    first_state = ResumeStateManager(
        str(tmp_path),
        run_config={"task": "dummy_generate", "mode": "first"},
        task_filters={"dummy_generate": ["none"]},
    )
    first_state.prepare()

    second_state = ResumeStateManager(
        str(tmp_path),
        run_config={"task": "dummy_generate", "mode": "second"},
        task_filters={"dummy_generate": ["none"]},
    )

    try:
        second_state.prepare()
    except ValueError as exc:
        assert "does not match" in str(exc)
    else:
        raise AssertionError("Expected mismatched resume manifests to fail")


def test_resume_state_manifest_serializes_non_json_values(tmp_path):
    resume_state = ResumeStateManager(
        str(tmp_path),
        run_config={
            "task": "dummy_generate",
            "callable": partial(str.upper),
        },
        task_filters={"dummy_generate": ["none"]},
    )

    resume_state.prepare()

    manifest = resume_state.manifest_path.read_text(encoding="utf-8")
    assert "functools.partial" in manifest
