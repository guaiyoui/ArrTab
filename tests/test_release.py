import json
import unittest

from agentictqa.data import FeTaQA10, legacy_predictions_path
from agentictqa.metrics import evaluate_rouge, tokenize
from agentictqa.pipeline import _validate_read_only_sql, clean_answer


class ReleaseTest(unittest.TestCase):
    def test_dataset_is_complete(self):
        dataset = FeTaQA10()
        examples = dataset.examples()
        self.assertEqual([row.id for row in examples], list(range(21, 31)))
        self.assertEqual(len(list(dataset.tables_dir.glob("*.csv"))), 100)
        for example in examples:
            for table_id in example.candidate_ids:
                dataset.load_table(table_id)
            self.assertEqual(
                [table.id for table in dataset.retrieve(example, mode="oracle")],
                list(example.reference_table_ids),
            )

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
