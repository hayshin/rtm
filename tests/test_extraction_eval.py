from __future__ import annotations

import unittest

from evaluation.extraction_eval import ExtractionItem, evaluate_consultation, parse_segment_indices


class ExtractionEvaluationTests(unittest.TestCase):
    def test_parse_segment_indices(self) -> None:
        self.assertEqual(parse_segment_indices("8, 14,21"), [8, 14, 21])
        self.assertEqual(parse_segment_indices(""), [])

    def test_category_match_counts_as_true_positive(self) -> None:
        gold = [
            ExtractionItem(
                category="Condition",
                text="Loose watery diarrhea",
                segment_indices=(8, 14),
                attributes={"verification_status": "confirmed"},
                source="gold",
            )
        ]
        predicted = [
            ExtractionItem(
                category="Condition",
                text="Diarrhoea watery loose stools",
                segment_indices=(8, 14, 21),
                attributes={"verification_status": "confirmed"},
                source="pred",
            )
        ]

        result = evaluate_consultation("demo", gold, predicted, threshold=0.45)
        summary = result.categories["Condition"]
        self.assertEqual(summary.tp, 1)
        self.assertEqual(summary.fp, 0)
        self.assertEqual(summary.fn, 0)
        self.assertEqual(summary.wrong_attributes, 0)

    def test_wrong_category_is_bucketed(self) -> None:
        gold = [
            ExtractionItem(
                category="Condition",
                text="Chest pain",
                segment_indices=(4,),
                attributes={},
                source="gold",
            )
        ]
        predicted = [
            ExtractionItem(
                category="Observation",
                text="Chest pain",
                segment_indices=(4,),
                attributes={},
                source="pred",
            )
        ]

        result = evaluate_consultation("demo", gold, predicted, threshold=0.65)
        condition_summary = result.categories["Condition"]
        observation_summary = result.categories["Observation"]
        self.assertEqual(condition_summary.wrong_category, 1)
        self.assertEqual(condition_summary.missed, 0)
        self.assertEqual(observation_summary.hallucinated, 0)

    def test_attribute_mismatch_is_counted(self) -> None:
        gold = [
            ExtractionItem(
                category="MedicationStatement",
                text="Amoxicillin",
                segment_indices=(9,),
                attributes={"status": "active"},
                source="gold",
            )
        ]
        predicted = [
            ExtractionItem(
                category="MedicationStatement",
                text="Amoxicillin",
                segment_indices=(9,),
                attributes={"status": "stopped"},
                source="pred",
            )
        ]

        result = evaluate_consultation("demo", gold, predicted, threshold=0.65)
        summary = result.categories["MedicationStatement"]
        self.assertEqual(summary.tp, 1)
        self.assertEqual(summary.wrong_attributes, 1)


if __name__ == "__main__":
    unittest.main()
