"""Small CPU checks; no external data, network requests, or model downloads."""
from __future__ import annotations

import csv
import gzip
import json
from pathlib import Path
import tempfile
import types
import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch.nn import functional as F
import yaml

from llmtransform.data import examples, item_card, pack_histories, prepare, load_split
from llmtransform.encode import encode
from llmtransform.io import read_json, read_jsonl, write_json, write_jsonl, data_fingerprint, fingerprint
from llmtransform.model import LLMTransform
from llmtransform.train import train, evaluate, evaluate_checkpoint, load_catalog, load_config, metrics

torch.set_num_threads(1)


class ProtocolTests(unittest.TestCase):
    def test_prefix_training_and_last_click_evaluation(self):
        sessions = [{"item_id_list": ["a", "b", "c"]}]
        self.assertEqual(list(examples(sessions, training=True)), [
            {"history": ["a"], "target": "b"}, {"history": ["a", "b"], "target": "c"}])
        self.assertEqual(list(examples(sessions, training=False)), [{"history": ["a", "b"], "target": "c"}])

    def test_history_retains_repeats_and_chronology(self):
        rows = pack_histories([["a", "b", "a"], ["b"]], {"a": 0, "b": 1}, 4)
        np.testing.assert_array_equal(rows, [[-1, 0, 1, 0], [-1, -1, -1, 1]])
        with self.assertRaises(KeyError):
            pack_histories([["missing"]], {"a": 0})

    def test_item_card_leaf_and_missing_fields(self):
        card = item_card({"title": "A &amp; <b>B</b>", "store": "Maker", "price": "1.99",
                          "categories": '["Top", ["Middle", "Leaf"]]'})
        self.assertEqual(card, "Title: A & B | Brand: Maker | Category: Leaf | Price: 1.99")
        self.assertEqual(item_card({"title": "A", "main_category": "Top"}), "Title: A | Category: Top")
        self.assertLessEqual(len(item_card({"title": "x" * 200, "store": "b" * 500})), 400)

    def test_compressed_split_matches_plain_inputs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rows = [{"history": ["a", "b", "a"], "target": "c"},
                    {"history": ["b"], "target": "a"}]
            path = root / "train.jsonl.gz"
            with gzip.open(path, "wt", encoding="utf-8") as handle:
                for row in rows:
                    handle.write(json.dumps(row) + "\n")
            seq, targets = load_split(root, "train", ["a", "b", "c"], max_length=2)
            np.testing.assert_array_equal(seq, [[1, 0], [-1, 1]])
            np.testing.assert_array_equal(targets, [2, 0])
            self.assertEqual(read_jsonl(path), rows)
            write_jsonl(root / "train.jsonl", rows)
            with self.assertRaisesRegex(ValueError, "exactly one"):
                load_split(root, "train", ["a", "b", "c"])

    def test_last_click_metrics(self):
        values = metrics(np.array([[0, 1, 2], [2, 1, 0], [0, 1, 2]]), np.array([0, 1, 3]))
        self.assertAlmostEqual(values["HR@20"], 2 / 3)
        self.assertAlmostEqual(values["MRR@20"], 0.5)

    def test_first_day_head_truncation_and_split(self):
        from llmtransform._sessions import assign_sbr_sessions, build_session_sequences, chronological_session_split
        import pandas as pd
        day = 86400000
        base = 1609459200000
        rows = [{"user_id": "u", "item_id": str(i), "timestamp": base + i} for i in range(25)]
        rows.append({"user_id": "u", "item_id": "next-day", "timestamp": base + day})
        first = assign_sbr_sessions(pd.DataFrame(rows))
        seq = build_session_sequences(first)
        self.assertEqual(seq.iloc[0].item_id_list, [str(i) for i in range(20)])
        sessions = pd.DataFrame([{"user_id": str(i), "session_id": i, "session_time": i,
                                  "item_id_list": ["a", "b"] if i < 8 else ["new", "a", "b"]}
                                 for i in range(10)])
        splits = chronological_session_split(sessions)
        self.assertEqual([len(splits[s]) for s in ("train", "valid", "test")], [8, 1, 1])
        self.assertEqual(splits["test"][0]["history"], ["a"])


class ModelTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        self.vectors = F.normalize(torch.randn(25, 12), dim=-1)
        self.history = torch.tensor([[-1, -1, 0], [-1, 1, 2], [3, 4, 5]])
        self.targets = torch.tensor([1, 3, 6])

    def test_last_item_shortcut_formula_padding_and_backward(self):
        model = LLMTransform(input_dim=12, output_dim=8, hidden_dim=16, max_length=3,
                             heads=2, dropout=0.0)
        items = model.transform_items(self.vectors)
        valid = self.history >= 0
        embeddings = items[self.history.clamp(min=0)] * valid.unsqueeze(-1)
        states = model.encoder(embeddings, valid)
        self.assertTrue(torch.equal(states[~valid], torch.zeros_like(states[~valid])))
        expected = F.normalize(embeddings[:, -1] + model.gamma * F.normalize(states[:, -1], dim=-1), dim=-1)
        torch.testing.assert_close(model.session_embedding(items, self.history), expected)
        scores = model(self.vectors, self.history)
        F.cross_entropy(scores, self.targets).backward()
        self.assertTrue(all(p.grad is not None and bool(torch.isfinite(p.grad).all()) for p in model.parameters()))

    def test_scoring_excludes_all_history_items(self):
        repeated = torch.tensor([[-1, 0, 0], [-1, 1, 2], [3, 4, 5]])
        model = LLMTransform(input_dim=12, output_dim=8,
                             hidden_dim=16, heads=2, dropout=0.0).eval()
        scores = model(self.vectors, repeated)
        self.assertTrue(torch.equal(scores[0, 0], torch.tensor(-1e9)))
        self.assertTrue(bool((scores[1, 1:3] == -1e9).all()))
        self.assertTrue(bool((scores[2, 3:6] == -1e9).all()))
        self.assertGreater(float(scores[0, -1]), -1e8)  # padding is not item -1
        with self.assertRaises(TypeError):
            model(self.vectors, repeated, self.targets)

    def test_evaluation_labels_do_not_change_candidate_scores(self):
        model = LLMTransform(input_dim=12, output_dim=8).eval()
        observed = []
        hook = model.register_forward_hook(lambda _model, _inputs, output: observed.append(output.clone()))
        try:
            evaluate(model, self.vectors.numpy(), self.history.numpy(), self.targets.numpy(), device="cpu")
            repeated_targets = np.array([0, 1, 3])
            result = evaluate(model, self.vectors.numpy(), self.history.numpy(), repeated_targets, device="cpu")
        finally:
            hook.remove()
        torch.testing.assert_close(observed[0], observed[1], rtol=0, atol=0)
        self.assertEqual(result["HR@20"], 0.0)

    def test_attention_is_causal(self):
        model = LLMTransform(input_dim=12, output_dim=8, hidden_dim=16, max_length=3, heads=2, dropout=0.0).eval()
        embeddings = torch.randn(1, 3, 8)
        valid = torch.ones(1, 3, dtype=torch.bool)
        before = model.encoder(embeddings, valid)
        embeddings[:, -1] += 20
        after = model.encoder(embeddings, valid)
        torch.testing.assert_close(before[:, :-1], after[:, :-1])


class EndToEndTests(unittest.TestCase):
    def test_prepare_raw_inputs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with (root / "ratings.csv").open("w", newline="") as handle:
                writer = csv.writer(handle)
                writer.writerow(["user_id", "parent_asin", "rating", "timestamp"])
                for user in range(30):
                    for index, item in enumerate(["a", "a", "b", "c"]):
                        writer.writerow([f"u{user:02}", item, 5, 1609459200000 + user * 86400000 + index])
            write_jsonl(root / "meta.jsonl", [{"parent_asin": x, "title": x, "categories": ["Top", "Leaf"]}
                                             for x in ("a", "b", "c")])
            output = root / "processed"
            prepare(root / "ratings.csv", root / "meta.jsonl", output, "Cell_Phones_and_Accessories")
            self.assertEqual(len(read_jsonl(output / "train.jsonl")), 48)
            self.assertEqual(len(read_jsonl(output / "test.jsonl")), 3)
            self.assertEqual(read_jsonl(output / "test.jsonl")[0], {"history": ["a", "b"], "target": "c"})
            self.assertEqual(read_json(output / "data_config.json")["fingerprint"], data_fingerprint(output))
            with self.assertRaises(FileExistsError):
                prepare(root / "ratings.csv", root / "meta.jsonl", output, "Cell_Phones_and_Accessories")

    def test_encoding_uses_frozen_unique_cards(self):
        calls = []

        class FakeEncoder(torch.nn.Module):
            def __init__(self, *args, **kwargs):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.ones(1))

            def encode(self, texts, **kwargs):
                calls.append(texts)
                assert not self.training and not self.weight.requires_grad
                assert self.max_seq_length == 1024
                return np.full((len(texts), 2560), 1 / np.sqrt(2560), dtype=np.float32)

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data = root / "data"
            data.mkdir()
            write_json(data / "item_cards.json", {"a": "Title: Shared", "b": "Title: Shared"})
            for split in ("train", "valid", "test"):
                write_jsonl(data / f"{split}.jsonl", [{"history": ["a"], "target": "b"}])
            with patch.dict("sys.modules", {"sentence_transformers": types.SimpleNamespace(SentenceTransformer=FakeEncoder)}):
                encode(data, root / "encoded", device="cpu")
            self.assertEqual(calls, [["Title: Shared"]])
            ids, vectors, _ = load_catalog(data, root / "encoded")
            self.assertEqual(ids, ["a", "b"])
            np.testing.assert_array_equal(vectors[0], vectors[1])

    def test_train_reload_and_mismatched_artifacts(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data, language = root / "data", root / "language"
            data.mkdir()
            language.mkdir()
            ids = [str(i) for i in range(25)]
            write_json(data / "item_cards.json", {item: f"Title: {item}" for item in ids})
            rows = [{"history": [str(i), str((i + 1) % 25)], "target": str((i + 2) % 25)} for i in range(25)]
            for split in ("train", "valid", "test"):
                write_jsonl(data / f"{split}.jsonl", rows)
            with gzip.open(data / "train.jsonl.gz", "wt", encoding="utf-8") as handle:
                handle.write((data / "train.jsonl").read_text())
            (data / "train.jsonl").unlink()
            vectors = np.random.default_rng(42).normal(size=(25, 2560)).astype(np.float32)
            vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
            np.save(language / "language_representations.npy", vectors)
            write_json(language / "item_ids.json", ids)
            write_json(language / "encoder_config.json", {
                "data_fingerprint": data_fingerprint(data),
                "language_fingerprint": fingerprint([language / "item_ids.json", language / "language_representations.npy"])})
            cfg = {"seed": 42, "model": {"input_dim": 2560, "output_dim": 8,
                   "hidden_dim": 16, "max_length": 3, "heads": 2, "dropout": 0.0},
                   "train": {"epochs": 2, "batch_size": 8, "lr": 0.001, "weight_decay": 0.01,
                             "patience": 5, "max_grad_norm": 1.0}}
            config = root / "llm_transform.yaml"
            config.write_text(yaml.safe_dump(cfg))
            output = root / "llm_transform"
            train(config, data, language, output, device="cpu")
            metrics_out = evaluate_checkpoint(output / "model.pt", data, language, device="cpu")
            self.assertEqual(metrics_out["n_examples"], 25)
            self.assertTrue(0 <= metrics_out["MRR@20"] <= metrics_out["HR@20"] <= 1)
            with self.assertRaises(FileExistsError):
                train(config, data, language, output, device="cpu")
            # Changed data must never silently reuse a checkpoint/cache.
            write_jsonl(data / "test.jsonl", rows[:-1])
            with self.assertRaisesRegex(ValueError, "Data changed"):
                evaluate_checkpoint(root / "llm_transform/model.pt", data, language, device="cpu")

    def test_configs_have_no_history_switch(self):
        base = Path(__file__).resolve().parents[1]
        cfg = load_config(base / "configs" / "llm_transform.yaml")
        self.assertNotIn("mask_history", cfg["train"])
        cfg["train"]["mask_history"] = False
        with tempfile.TemporaryDirectory() as directory:
            config = Path(directory) / "old.yaml"
            config.write_text(yaml.safe_dump(cfg))
            with self.assertRaisesRegex(ValueError, "train must contain exactly"):
                load_config(config)


if __name__ == "__main__":
    unittest.main()
