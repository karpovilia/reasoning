import copy
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from reasoning_control.data import initialize, validate_annotation, validate_trace
from reasoning_control.detector import Detector, context, train
from reasoning_control.gate import detect_and_decide, load
from reasoning_control.gate_runtime.features import encode
from reasoning_control.gate_runtime.labels import build_examples, load_branch_labels, load_observations, split
from reasoning_control.gate_runtime.models import load_checkpoint, save_checkpoint
from reasoning_control.gate_runtime.policy import observation_from_dict
from reasoning_control.gate_runtime.training import Example, aggregate_draws, fit, predict
from reasoning_control.segmentation import segment_records

ROOT = Path(__file__).resolve().parents[1]
torch.set_num_threads(1)


class ReleasedModels(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.gate = load(ROOT / "configs/gate/counts_spans.json")
        cls.detector = Detector(ROOT / "models/detector/lexical-event-span-v1.pkl")
        cls.row = json.loads((ROOT / "examples/gate_observation.jsonl").read_text())

    def test_frozen_original_prediction(self):
        expected = json.loads((ROOT / "examples/gate_expected.json").read_text())
        result = self.gate.decide(self.row)
        self.assertEqual(result["action"], expected["action"])
        self.assertFalse(result["fallback"])
        for field in ("p_success", "cost", "predicted_additional_tokens"):
            np.testing.assert_allclose(result[field], expected[field], rtol=1e-6, atol=1e-6)

    def test_no_future_observation(self):
        row = copy.deepcopy(self.row)
        row["segments"][-1]["available_at"] = row["now"] + 1
        result = self.gate.decide(row)
        self.assertTrue(result["fallback"])
        self.assertEqual(result["action"], "continue_2048")
        self.assertIsNone(result["p_success"])

    def test_missing_confidence_is_explicit(self):
        row = {k: v for k, v in self.row.items() if k != "confidence"}
        self.assertTrue(self.gate.decide(row)["fallback"])

    def test_answers_and_future_targets_never_enter_model(self):
        row = {**self.row, "question": "UNSEEN TEXT", "correct": False,
               "remaining_tokens": 987654, "reference_answer": "NO LEAK"}
        self.assertEqual(self.gate.decide(row), self.gate.decide(self.row))

    def test_counts_are_aggregate_and_joint_is_sequence(self):
        obs = observation_from_dict(self.row)
        self.assertEqual(encode(obs, "counts_spans").shape[0], 1)
        self.assertEqual(encode(obs, "joint").shape[0], len(obs.segments) + 1)

    def test_all_frozen_configs_load_and_decide(self):
        for config in sorted((ROOT / "configs/gate").glob("*.json")):
            with self.subTest(config=config.name):
                self.assertFalse(load(config).decide(self.row)["fallback"])

    def test_text_to_action_pipeline(self):
        row = json.loads((ROOT / "examples/pipeline.jsonl").read_text())
        output = detect_and_decide(self.detector, self.gate, row)
        self.assertFalse(output["decision"]["fallback"])
        self.assertEqual(len(output["observation"]["segments"]), 2)

    def test_segment_predictions_do_not_read_future(self):
        first = self.detector.predict_segments(["Try addition."])
        longer = self.detector.predict_segments(["Try addition.", "Ignore this later sentence."])
        self.assertEqual(first[0], longer[0])
        self.assertEqual(context("a" * 2000, "b" * 1000), "CURRENT " + "a" * 1600 + " PREVIOUS " + "b" * 800)


class TrainingContracts(unittest.TestCase):
    def test_unknown_correctness_keeps_observed_cost(self):
        y, c, coverage = aggregate_draws([[None, True]] * 5, [[20, 30]] * 5)
        np.testing.assert_array_equal(y, np.ones(5))
        np.testing.assert_array_equal(c, np.full(5, 25))
        self.assertEqual(coverage["known_correctness_draws"], [1] * 5)

    def test_small_training_roundtrip_and_disjoint_questions(self):
        obs = observation_from_dict(json.loads((ROOT / "examples/gate_observation.jsonl").read_text()))
        def example(key, part):
            return Example(key, key, part, obs, np.array([0., 1., .5, 1., .5]), np.array([20., 50., 100., 55., 45.]))
        train_rows = [example("train_a", "train"), example("train_b", "train")]
        validation = [example("val_a", "validation")]
        model, metadata = fit(train_rows, validation, variant="counts_spans", architecture="gru", epochs=2)
        self.assertFalse(metadata["selection_uses_test"])
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "observer.pt"
            save_checkpoint(model, path, metadata)
            restored, _ = load_checkpoint(path)
            a, b = predict(model, [obs]), predict(restored, [obs])
            np.testing.assert_array_equal(a["p_action_correct"], b["p_action_correct"])
        with self.assertRaisesRegex(ValueError, "leakage"):
            fit(train_rows, [example("train_a", "validation")], variant="counts_spans", architecture="gru", epochs=1)

    def test_detector_training_masks_and_fold_selection(self):
        segments = [f"Segment {i}: {'verify' if i % 2 else 'derive'} a result." for i in range(28)]
        base = {"group": "synthetic", "fold": 1, "segments": segments,
                "events": [[i % 2] * 6 for i in range(28)], "spans": [i % 7 for i in range(28)],
                "event_known": [True] * 27 + [False]}
        base["spans"][-1] = -1
        with tempfile.TemporaryDirectory() as tmp:
            path, output = Path(tmp) / "train.jsonl", Path(tmp) / "model.pkl"
            excluded = {**base, "group": "held_out", "fold": 0}
            path.write_text(json.dumps(base) + "\n" + json.dumps(excluded) + "\n")
            train(path, output)
            metadata = json.loads(output.with_suffix(".json").read_text())
            self.assertEqual(metadata["questions"], 1)
            self.assertEqual(metadata["segments"], 28)
            self.assertEqual(metadata["known_event_segments"], 27)
            self.assertEqual(metadata["known_span_segments"], 27)
            self.assertEqual(len(Detector(output).predict_segments(segments[:2])), 2)


class DataContracts(unittest.TestCase):
    def setUp(self):
        self.trace = json.loads((ROOT / "examples/trace.jsonl").read_text())
        self.annotation = json.loads((ROOT / "examples/annotation.jsonl").read_text())

    def test_join_rejects_another_text_revision(self):
        validate_trace(self.trace)
        validate_annotation(self.annotation, self.trace)
        self.annotation["reasoning_sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "revision"):
            validate_annotation(self.annotation, self.trace)

    def test_context_availability_and_incomplete_labels(self):
        self.annotation["available_at_char"] = 1
        with self.assertRaisesRegex(ValueError, "availability"):
            validate_annotation(self.annotation, self.trace)
        self.annotation["available_at_char"] = len(self.trace["reasoning"])
        self.annotation["spans"].pop()
        with self.assertRaisesRegex(ValueError, "coverage"):
            validate_annotation(self.annotation, self.trace)
        self.annotation["status"] = "partial"
        validate_annotation(self.annotation, self.trace)

    def test_catalog_init_creates_no_data(self):
        with tempfile.TemporaryDirectory() as tmp:
            catalog = Path(tmp) / "catalog.json"
            config = Path(tmp) / "run.json"
            config.write_text(json.dumps({"annotations": [{"annotation_run_id": "synthetic"}]}))
            catalog.write_text(json.dumps({"runs": [{"run_id": "synthetic", "config": "run.json"}]}))
            count = initialize(catalog, Path(tmp) / "layout")
            self.assertEqual(count, 1)
            self.assertEqual(initialize(catalog, Path(tmp) / "layout"), count)
            self.assertFalse(list(Path(tmp).rglob("*.jsonl*")))

    def test_unicode_and_repeated_segment_offsets(self):
        text = "  Проверим.\nSame sentence. Same sentence.\n✓ done"
        segments = segment_records(text)
        self.assertEqual([text[s.char_start:s.char_end] for s in segments], [s.text for s in segments])
        self.assertLess(segments[1].char_start, segments[2].char_start)


if __name__ == "__main__":
    unittest.main()
