"""Repeatable local benchmark for PromptForge's deterministic pipeline."""
import csv
import statistics
import math
import time
from pathlib import Path
from optimizer import analyze_prompt, optimize_prompt, compare_prompts, validate_prompt

CASES = [
    ("summary_limits", "Please summarize this report in exactly 3 bullet points. Keep it under 60 words. Include $4.2M."),
    ("prohibition", "Could you please explain the dataset in 5 steps? Do not include personal data."),
    ("json_keys", "Extract the result as JSON. Keys: summary, source_id, confidence."),
    ("coding", "Write Python code in exactly 2 functions. Do not use external libraries. Include a short example."),
    ("writing", "Please write a professional email under 120 words. Include the phrase \"Project Falcon\"."),
    ("research", "Could you please compare exactly 4 options in a table? Include at least 3 advantages."),
    ("education", "Please explain recursion in exactly 5 bullet points. Use simple English."),
    ("data", "Return CSV with columns: name, score, status. Do not invent missing values."),
    ("format", "Could you please create a numbered list of exactly 6 ideas? Avoid repeating any idea."),
    ("mixed", "Summarize the findings in under 80 words. Include 12.5% and do not claim causation."),
]

def run_benchmark():
    rows = []
    total_checks = passed_checks = 0
    savings = []
    latencies = []
    for case_id, prompt in CASES:
        start = time.perf_counter()
        analysis = analyze_prompt(prompt)
        high, lean = optimize_prompt(prompt, analysis)
        metrics = compare_prompts(prompt, high, lean)
        high_v = validate_prompt(prompt, high)
        lean_v = validate_prompt(prompt, lean)
        elapsed_ms = (time.perf_counter() - start) * 1000
        latencies.append(elapsed_ms)
        savings.append(metrics["lean_savings_pct"])
        for val in (high_v, lean_v):
            for check in val["checks"]:
                total_checks += 1
                passed_checks += check["status"] == "PASS"
        rows.append({
            "case_id": case_id,
            "original_tokens_est": metrics["original_tokens"],
            "contextmax_tokens_est": metrics["high_tokens"],
            "tokenlite_tokens_est": metrics["lean_tokens"],
            "tokenlite_savings_pct": round(metrics["lean_savings_pct"], 2),
            "contextmax_validation": high_v["status"],
            "tokenlite_validation": lean_v["status"],
            "validation_checks_passed": high_v["pass_count"] + lean_v["pass_count"],
            "validation_checks_failed": high_v["fail_count"] + lean_v["fail_count"],
            "validation_checks_uncertain": high_v["uncertain_count"] + lean_v["uncertain_count"],
            "pipeline_latency_ms": round(elapsed_ms, 3),
        })
    preservation = 100 * passed_checks / total_checks if total_checks else 0
    summary = {
        "cases": len(CASES),
        "total_validation_checks": total_checks,
        "passed_validation_checks": passed_checks,
        "check_pass_rate_pct": round(preservation, 2),
        "median_tokenlite_savings_pct": round(statistics.median(savings), 2),
        "median_pipeline_latency_ms": round(statistics.median(latencies), 3),
        "p95_pipeline_latency_ms": round(sorted(latencies)[max(0, math.ceil(len(latencies)*0.95)-1)], 3),
        "note": "Local deterministic benchmark; token counts may be approximate and check pass rate is not semantic equivalence or answer quality."
    }
    out = Path("benchmark_results.csv")
    with out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print("PromptForge AI benchmark")
    for k, v in summary.items():
        print(f"{k}: {v}")
    print(f"Detailed results: {out.resolve()}")
    return summary

if __name__ == "__main__":
    run_benchmark()
