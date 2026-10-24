"""PromptForge AI core: deterministic optimization, requirement extraction and conservative validation."""
from __future__ import annotations
import re
from typing import Any

try:
    import tiktoken
except Exception:
    tiktoken = None

ACTION_WORDS = re.compile(
    r"\b(return|output|provide|write|explain|create|generate|summarize|list|build|design|analyze|compare|find|calculate|translate|classify|extract|debug|review|describe|convert|draft|recommend|identify|evaluate)\b",
    re.I,
)
STOP_WORDS = {"please","could","would","should","that","this","with","from","your","have","what","when","where","which","their","there","then","into","about","just","want","need","make"}

def estimate_tokens(text: str, tokenizer_profile: str = "cl100k_base") -> int:
    """Use cl100k_base when available; otherwise return a clearly approximate estimate."""
    if not text or not text.strip():
        return 0
    if tiktoken and tokenizer_profile != "Approximate offline fallback":
        try:
            enc = tiktoken.get_encoding(tokenizer_profile)
            return len(enc.encode(text))
        except Exception:
            pass
    # Approximation only; tokenization varies by model, language and punctuation.
    pieces = re.findall(r"\w+|[^\w\s]", text, flags=re.UNICODE)
    return max(1, round(len(pieces) * 1.25))

def _sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[.!?])\s+|\n+", text.strip())
    return [p.strip(" \t-•") for p in parts if p.strip(" \t-•")]

def _unique_sentences(sentences: list[str]) -> list[str]:
    seen, out = set(), []
    for sentence in sentences:
        key = re.sub(r"\W+", " ", sentence.lower()).strip()
        if key and key not in seen:
            seen.add(key)
            out.append(sentence)
    return out

def _infer_intent(low: str) -> str:
    if any(x in low for x in ["python", "java", "javascript", "code", "debug", "function"]): return "Coding / implementation"
    if any(x in low for x in ["summarize", "summary", "key points"]): return "Summarization"
    if any(x in low for x in ["research", "compare", "evaluate", "analyze"]): return "Research / analysis"
    if any(x in low for x in ["email", "essay", "write", "caption", "story"]): return "Writing"
    if any(x in low for x in ["explain", "teach", "what is", "illustrate"]): return "Education / explanation"
    if any(x in low for x in ["json", "extract", "schema", "keys"]): return "Data extraction"
    return "General task"

def analyze_prompt(prompt: str, task_type: str = "Auto-detect") -> dict[str, Any]:
    if not isinstance(prompt, str) or not prompt.strip():
        return {"intent": "Unknown", "signals": [], "warnings": ["Enter a non-empty prompt."],
                "complexity": 0, "constraint_count": 0, "duplicate_count": 0}
    low = prompt.lower()
    signals = []
    checks = [
        (r"\b(please|kindly|could you|can you)\b", "politeness / filler"),
        (r"\b(detailed|step by step|comprehensive|thorough|in depth)\b", "detail requirement"),
        (r"\b(format|table|json|bullet|headings|structure)\b", "output format"),
        (r"\b(do not|don't|must not|never|avoid|must|only|exactly|under \d+ words)\b", "constraints"),
        (r"\b(example|examples|case study|illustrate)\b", "examples requested"),
        (r"\b(compare|evaluate|trade-off|pros and cons|analyze|reason)\b", "analysis / comparison"),
        (r"\b(code|python|java|javascript|function|debug|algorithm)\b", "coding signal"),
    ]
    for pattern, label in checks:
        if re.search(pattern, low): signals.append(label)
    sentences = _sentences(prompt)
    duplicate_count = len(sentences) - len(_unique_sentences(sentences))
    if duplicate_count: signals.append(f"{duplicate_count} repeated sentence(s)")
    constraints = len(re.findall(r"\b(must|should|avoid|do not|don't|must not|never|only|exactly|include|exclude|format as|under|at least|at most|no more than)\b", low))
    complexity = int(any(x in low for x in ["analyze", "compare", "evaluate", "research", "design", "architecture", "step by step", "edge case"]))
    complexity += int(any(x in low for x in ["constraints", "requirements", "multiple", "trade-off", "security", "performance"]))
    intent = task_type if task_type and task_type != "Auto-detect" else _infer_intent(low)
    warnings = []
    if len(prompt.split()) < 5:
        warnings.append("This is a very short prompt; confirm the desired output and key constraints.")
    if not ACTION_WORDS.search(prompt):
        warnings.append("The requested action may be implicit. Consider stating the desired action directly.")
    if constraints == 0:
        warnings.append("No explicit constraints were detected. Add audience, format, length, or scope only when needed.")
    requirements = extract_requirements(prompt)
    if not requirements:
        warnings.append("No machine-checkable critical requirements were detected; human review may be needed.")
    return {"intent": intent, "signals": signals, "warnings": warnings, "complexity": complexity,
            "constraint_count": constraints, "duplicate_count": duplicate_count,
            "requirements": requirements}

def _add_requirement(items: list[dict], kind: str, value: str, label: str, critical: bool = True):
    key = (kind, value.lower())
    if not any((x["kind"], x["value"].lower()) == key for x in items):
        items.append({"id": f"R{len(items)+1}", "kind": kind, "value": value,
                      "label": label, "critical": critical})

def extract_requirements(prompt: str) -> list[dict[str, Any]]:
    """Extract conservative, machine-checkable constraints; this is not semantic proof."""
    if not isinstance(prompt, str) or not prompt.strip():
        return []
    items: list[dict] = []
    # Numeric literals and constraints; exact literal preservation is deliberately conservative.
    for m in re.finditer(r"\bexactly\s+(\d+)\s+([a-zA-Z-]+(?:\s+[a-zA-Z-]+)?)", prompt, re.I):
        _add_requirement(items, "exact_count", m.group(0), f"Preserve exact count: {m.group(0)}")
    for m in re.finditer(r"\b(under|below|less than|at most|no more than|maximum(?: of)?)\s+(\d+)\s*(words?|characters?|sentences?|items?|bullets?|tokens?)\b", prompt, re.I):
        _add_requirement(items, "limit", m.group(0), f"Preserve limit: {m.group(0)}")
    for m in re.finditer(r"\b(at least|minimum(?: of)?)\s+(\d+)\s*(words?|characters?|sentences?|items?|bullets?|examples?|steps?)\b", prompt, re.I):
        _add_requirement(items, "limit", m.group(0), f"Preserve minimum: {m.group(0)}")
    # Explicit numbers and quoted values are important, even where their meaning is domain-specific.
    for m in re.finditer(r'(?<!\w)(?:[$€£]\s?\d[\d,]*(?:\.\d+)?|\b\d+(?:\.\d+)?%?)(?!\w)', prompt):
        _add_requirement(items, "literal", m.group(0), f"Preserve literal value: {m.group(0)}")
    for m in re.finditer(r'["“]([^"”\n]{1,100})["”]', prompt):
        _add_requirement(items, "quoted", m.group(1), f"Preserve quoted text: {m.group(1)}")
    # Prohibitions and required inclusions, retaining the clause for transparent review.
    clause_pattern = r"\b(?:do not|don't|must not|never|avoid|must exclude|cannot|can't)\b[^.;\n]*(?:[.;]|$)"
    for m in re.finditer(clause_pattern, prompt, re.I):
        clause = m.group(0).strip()
        _add_requirement(items, "prohibition", clause, f"Preserve prohibition: {clause}")
    for m in re.finditer(r"\b(?:include|must include|make sure to include|contain|must contain)\b[^.;\n]*(?:[.;]|$)", prompt, re.I):
        clause = m.group(0).strip()
        _add_requirement(items, "inclusion", clause, f"Preserve required inclusion: {clause}")
    # JSON keys stated in natural language.
    for m in re.finditer(r"\b(?:json\s+)?keys?\s*:\s*([A-Za-z0-9_,\s\"']+)", prompt, re.I):
        for key in re.findall(r"[A-Za-z_][A-Za-z0-9_]*", m.group(1)):
            if key.lower() not in {"json", "key", "keys"}:
                _add_requirement(items, "json_key", key.strip("\"'"), f"Preserve JSON key: {key.strip(chr(34)+chr(39))}")
    # Keys in embedded JSON examples.
    for block in re.findall(r"\{[^{}]{1,2000}\}", prompt):
        for key in re.findall(r'["\']([A-Za-z_][A-Za-z0-9_]*)["\']\s*:', block):
            _add_requirement(items, "json_key", key, f"Preserve JSON key: {key}")
    # Output formats are critical when explicit.
    for m in re.finditer(r"\b(JSON|CSV|XML|Markdown|table|bullet points?|numbered list)\b", prompt, re.I):
        _add_requirement(items, "format", m.group(0), f"Preserve output format: {m.group(0)}")
    return items

def optimize_prompt(prompt: str, analysis: dict | None = None, task_type: str = "Auto-detect"):
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("Prompt must not be empty.")
    analysis = analysis or analyze_prompt(prompt, task_type)
    intent = analysis.get("intent", "General task")
    # Preserve the complete source text; only remove a small list of low-risk filler phrases.
    # This deliberately favors safety over aggressive shortening.
    high = (
        f"Task: {intent}.\n"
        "Original request (preserve its requirements exactly):\n"
        f"{prompt.strip()}\n\n"
        "Execution guidance:\n"
        "- Follow every explicit requirement and prohibition in the original request.\n"
        "- Do not invent facts or silently relax constraints.\n"
        "- If requirements conflict or are ambiguous, ask for clarification.\n"
        "- Return the requested output in the specified format."
    )
    lean = prompt.strip()
    replacements = [
        (r"\bcould you please\b", "", re.I),
        (r"\bcan you please\b", "", re.I),
        (r"\bI would like you to\b", "", re.I),
        (r"\bI want you to\b", "", re.I),
        (r"\bin order to\b", "to", re.I),
        (r"\bdue to the fact that\b", "because", re.I),
        (r"\bat this point in time\b", "now", re.I),
        (r"\bkindly\b", "", re.I),
        (r"\bplease\b", "", re.I),
    ]
    for pattern, replacement, flags in replacements:
        lean = re.sub(pattern, replacement, lean, flags=flags)
    lean = re.sub(r"[ \t]{2,}", " ", lean)
    lean = re.sub(r" +([,.!?;:])", r"\1", lean)
    lean = re.sub(r"\n{3,}", "\n\n", lean).strip()
    # Do not append extra instructions to TokenLite; that would inflate token count.
    return high, lean

def _requirement_status(req: dict, candidate: str) -> tuple[str, str]:
    kind, value = req["kind"], req["value"]
    low, cand_low = candidate.lower(), candidate.lower()
    if kind in {"literal", "quoted"}:
        ok = value.lower() in cand_low
        return ("PASS" if ok else "FAIL", "Required literal is present." if ok else f"Missing required literal: {value}")
    if kind in {"exact_count", "limit"}:
        # Preserve the complete extracted phrase, so changed counts or units fail conservatively.
        ok = value.lower() in cand_low
        return ("PASS" if ok else "FAIL", "Numeric constraint phrase is present." if ok else f"Missing or changed constraint: {value}")
    if kind == "prohibition":
        # Require the original prohibition clause, not merely its target noun.
        ok = value.lower().strip(" .;") in cand_low
        return ("PASS" if ok else "FAIL", "Prohibition clause is preserved." if ok else f"Prohibition may have been removed or changed: {value}")
    if kind == "inclusion":
        # Require the clause to survive as written; conservative by design.
        ok = value.lower().strip(" .;") in cand_low
        return ("PASS" if ok else "FAIL", "Required-inclusion clause is preserved." if ok else f"Required inclusion may have changed: {value}")
    if kind == "json_key":
        pattern = r'(?<![A-Za-z0-9_])["\']?' + re.escape(value) + r'["\']?(?![A-Za-z0-9_])'
        ok = bool(re.search(pattern, candidate))
        return ("PASS" if ok else "FAIL", "JSON key/name is present." if ok else f"Missing JSON key/name: {value}")
    if kind == "format":
        ok = value.lower() in cand_low
        return ("PASS" if ok else "FAIL", "Output format mention is present." if ok else f"Output format may have been removed: {value}")
    return "UNCERTAIN", "No validator is available for this requirement type."

def validate_prompt(original: str, candidate: str, requirements: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    """Validate extracted requirements and fail closed on critical failures."""
    requirements = extract_requirements(original) if requirements is None else requirements
    if not isinstance(candidate, str) or not candidate.strip():
        return {"status": "FAIL", "safe_to_recommend": False,
                "pass_count": 0, "fail_count": len(requirements) or 1, "uncertain_count": 0,
                "checks": [{"id": "R0", "label": "Candidate is non-empty", "critical": True,
                            "status": "FAIL", "details": "Candidate prompt is empty."}],
                "message": "Reject: candidate is empty."}
    checks = []
    for req in requirements:
        status, details = _requirement_status(req, candidate)
        checks.append({**req, "status": status, "details": details})
    if not requirements:
        checks.append({"id": "R0", "label": "Machine-checkable requirements", "critical": True,
                       "status": "UNCERTAIN", "details": "No explicit machine-checkable requirements extracted; human review required."})
    passed = sum(x["status"] == "PASS" for x in checks)
    failed = sum(x["status"] == "FAIL" for x in checks)
    uncertain = sum(x["status"] == "UNCERTAIN" for x in checks)
    critical_failed = any(x["status"] == "FAIL" and x.get("critical", True) for x in checks)
    critical_uncertain = any(x["status"] == "UNCERTAIN" and x.get("critical", True) for x in checks)
    safe = not critical_failed and not critical_uncertain and bool(requirements)
    status = "PASS" if safe else ("FAIL" if critical_failed else "UNCERTAIN")
    message = ("Validated: critical requirements passed." if safe and failed == 0 else
               "Validated with non-critical warnings; review the ledger." if safe else
               "Reject: one or more critical requirements failed." if critical_failed else
               "Review required: critical checks are uncertain.")
    return {"status": status, "safe_to_recommend": safe, "pass_count": passed,
            "fail_count": failed, "uncertain_count": uncertain, "checks": checks, "message": message}

def compare_prompts(original: str, high: str, lean: str, tokenizer_profile: str = "cl100k_base") -> dict[str, Any]:
    ot = estimate_tokens(original, tokenizer_profile)
    ht = estimate_tokens(high, tokenizer_profile)
    lt = estimate_tokens(lean, tokenizer_profile)
    return {
        "original_tokens": ot, "high_tokens": ht, "lean_tokens": lt,
        "high_delta_pct": ((ht-ot)/ot*100) if ot else 0,
        "lean_savings_pct": ((ot-lt)/ot*100) if ot else 0,
        "high_coverage": _coverage_score(original, high),
        "lean_coverage": _coverage_score(original, lean),
    }

def _coverage_score(original: str, candidate: str) -> int:
    a = {w.lower() for w in re.findall(r"\b[a-zA-Z0-9_]+\b", original) if len(w) > 2 and w.lower() not in STOP_WORDS}
    b = {w.lower() for w in re.findall(r"\b[a-zA-Z0-9_]+\b", candidate) if len(w) > 2}
    if not a: return 100
    return min(100, round(100 * len(a & b) / len(a)))

def estimate_cost(tokens: int, price_per_million: float, requests: int = 1) -> float:
    if tokens < 0 or requests < 0 or price_per_million < 0:
        raise ValueError("Tokens, price, and request count must be non-negative.")
    return tokens * price_per_million / 1_000_000 * requests
