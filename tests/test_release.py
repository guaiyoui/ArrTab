import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from agentictqa.cli import DEFAULT_FUSION_CHECKPOINT
from agentictqa.data import FeTaQA10, legacy_predictions_path
from agentictqa.metrics import evaluate_rouge, tokenize
from agentictqa.pipeline import _validate_read_only_sql, clean_answer
from agentictqa.retriever import CachedOpenDomainRetriever, LegacyRetrieverConfig

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
FETAQA_CHECKPOINT_SHA256 = "533d583a1a4357bd5464aebf77da0e4571cd1c366d187ac6098edfe7593da047"


class ReleaseTest(unittest.TestCase):
    def test_fetaqa_fusion_checkpoint_is_released_intact(self):
        checkpoint = REPOSITORY_ROOT / "checkpoints" / "fetaqa_fusion.pt"
        self.assertEqual(DEFAULT_FUSION_CHECKPOINT, checkpoint)
        self.assertEqual(checkpoint.stat().st_size, 37_780_125)
        digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
        self.assertEqual(digest, FETAQA_CHECKPOINT_SHA256)

    def test_dataset_is_complete(self):
        dataset = FeTaQA10()
        examples = dataset.examples()
        retriever = CachedOpenDomainRetriever()
        self.assertEqual([row.id for row in examples], list(range(21, 31)))
        self.assertEqual(len(list(dataset.tables_dir.glob("*.csv"))), 95)
        for example in examples:
            table_ids = retriever.retrieve_ids(example, top_k=10)
            for table_id in table_ids:
                dataset.load_table(table_id)

    def test_cache_is_question_keyed_and_open_domain_only(self):
        dataset = FeTaQA10()
        retriever = CachedOpenDomainRetriever()
        example = dataset.examples(ids=[21])[0]
        self.assertEqual(retriever.retrieve_ids(example, 2), ["31", "32"])
        with (dataset.root / "questions.jsonl").open(encoding="utf-8") as handle:
            for line in handle:
                self.assertEqual(set(json.loads(line)), {"id", "question", "answer"})

    def test_live_retriever_checks_external_assets_without_importing_ml_stack(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            config = LegacyRetrieverConfig(
                assets_root=root / "assets",
                dataset_root=root / "datasets",
                fusion_checkpoint=root / "fusion.pt",
            )
            with self.assertRaisesRegex(FileNotFoundError, "Missing retriever assets"):
                config.validate()

    def test_archived_metrics_are_reproduced(self):
        dataset = FeTaQA10()
        predictions = {}
        with legacy_predictions_path().open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                predictions[row["id"]] = row["prediction"]
        examples = dataset.examples()
        scores = evaluate_rouge(
            [row.answer for row in examples],
            [predictions[row.id] for row in examples],
        )
        self.assertAlmostEqual(scores["ROUGE-1"]["f1"], 0.6183023105416839)
        self.assertAlmostEqual(scores["ROUGE-2"]["f1"], 0.431839372631996)
        self.assertAlmostEqual(scores["ROUGE-L"]["f1"], 0.5160935719888268)

    def test_metric_preserves_legacy_unicode_tokenizer(self):
        self.assertEqual(tokenize("Chávez 55.07%"), ["chávez", "55", "07"])

    def test_answer_cleaning_and_sql_guard(self):
        self.assertEqual(clean_answer("Answer: 29th."), "29th.")
        self.assertEqual(_validate_read_only_sql("SELECT * FROM t0;"), "SELECT * FROM t0")
        with self.assertRaises(ValueError):
            _validate_read_only_sql("DROP TABLE t0")
        with self.assertRaises(ValueError):
            _validate_read_only_sql("SELECT 'remembered answer' AS answer FROM t0")


if __name__ == "__main__":
    unittest.main()
