import unittest
from optimizer import (
    analyze_prompt, optimize_prompt, estimate_tokens, compare_prompts,
    extract_requirements, validate_prompt, estimate_cost
)

class PromptForgeTests(unittest.TestCase):
    def test_empty_token_count(self):
        self.assertEqual(estimate_tokens(""), 0)

    def test_empty_prompt_rejected(self):
        analysis = analyze_prompt("   ")
        self.assertIn("non-empty", analysis["warnings"][0].lower())
        with self.assertRaises(ValueError):
            optimize_prompt("  ")

    def test_outputs_nonempty_and_preserve_task(self):
        p = "Explain photosynthesis in simple terms and give two examples."
        a = analyze_prompt(p)
        high, lean = optimize_prompt(p, a)
        self.assertTrue(high.strip())
        self.assertTrue(lean.strip())
        self.assertIn("photosynthesis", high.lower())
        self.assertIn("photosynthesis", lean.lower())

    def test_comparison_has_metrics(self):
        p = "Write a short email asking for a meeting."
        a = analyze_prompt(p)
        high, lean = optimize_prompt(p, a)
        c = compare_prompts(p, high, lean)
        self.assertIn("original_tokens", c)
        self.assertIn("lean_savings_pct", c)

    def test_numeric_limit_detected_and_preserved(self):
        p = "Summarize in exactly 3 bullet points. Keep it under 60 words."
        reqs = extract_requirements(p)
        self.assertTrue(any("exactly 3" in x["value"].lower() for x in reqs))
        self.assertTrue(any("under 60 words" in x["value"].lower() for x in reqs))
        self.assertEqual(validate_prompt(p, p)["status"], "PASS")
        broken = "Summarize in 3 bullet points."
        result = validate_prompt(p, broken)
        self.assertEqual(result["status"], "FAIL")
        self.assertFalse(result["safe_to_recommend"])

    def test_changed_number_fails(self):
        p = "Return exactly 5 items."
        broken = "Return exactly 4 items."
        result = validate_prompt(p, broken)
        self.assertEqual(result["status"], "FAIL")

    def test_removed_prohibition_fails(self):
        p = "Summarize the file. Do not include personal data."
        result = validate_prompt(p, "Summarize the file.")
        self.assertEqual(result["status"], "FAIL")
        self.assertFalse(result["safe_to_recommend"])

    def test_json_keys_detected(self):
        p = "Return JSON. Keys: summary, source_id."
        reqs = extract_requirements(p)
        keys = {r["value"] for r in reqs if r["kind"] == "json_key"}
        self.assertIn("summary", keys)
        self.assertIn("source_id", keys)
        result = validate_prompt(p, 'Return JSON with key "summary".')
        self.assertEqual(result["status"], "FAIL")

    def test_empty_candidate_fails(self):
        result = validate_prompt("Return exactly 3 items.", "")
        self.assertEqual(result["status"], "FAIL")
        self.assertFalse(result["safe_to_recommend"])

    def test_uncertain_without_machine_checks(self):
        result = validate_prompt("Tell me something interesting.", "Tell me something interesting.")
        self.assertEqual(result["status"], "UNCERTAIN")
        self.assertFalse(result["safe_to_recommend"])

    def test_optimization_preserves_explicit_constraints(self):
        p = 'Please summarize in exactly 3 bullet points. Keep it under 60 words. Do not include personal data. Include "$4.2M".'
        analysis = analyze_prompt(p)
        high, lean = optimize_prompt(p, analysis)
        self.assertTrue(validate_prompt(p, high)["safe_to_recommend"])
        self.assertTrue(validate_prompt(p, lean)["safe_to_recommend"])

    def test_cost_formula(self):
        self.assertAlmostEqual(estimate_cost(1_000_000, 2.5, 2), 5.0)
        with self.assertRaises(ValueError):
            estimate_cost(1, -1, 1)

if __name__ == "__main__":
    unittest.main(verbosity=2)
