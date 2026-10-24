import streamlit as st
import pandas as pd
import re
import json
from optimizer import (
    analyze_prompt, optimize_prompt, compare_prompts, estimate_tokens,
    validate_prompt, estimate_cost, extract_requirements
)

st.set_page_config(page_title="PromptForge AI", page_icon="⚡", layout="wide")
st.markdown("""
<style>
.block-container {padding-top: 1.4rem; padding-bottom: 2.5rem; max-width: 1450px;}
.hero {padding: 1.4rem 1.6rem; border-radius: 18px; background: linear-gradient(115deg,#101a38,#243b70); color: white; margin-bottom: 1rem;}
.hero h1 {margin: 0; color: white; font-size: 2.2rem;}
.hero p {margin: .45rem 0 0; color: #dce7ff; font-size: 1.03rem;}
div[data-testid="stMetric"] {border: 1px solid rgba(128,128,128,.22); padding: .8rem; border-radius: 12px;}
</style>
<div class="hero"><h1>⚡ PromptForge AI</h1>
<p>Optimize with evidence. Preserve requirements. Estimate token savings.</p></div>
""", unsafe_allow_html=True)

st.write("PromptForge generates two alternatives, checks extracted requirements, and blocks recommendations when a critical check fails or is uncertain.")
with st.sidebar:
    st.header("Optimization settings")
    task_type = st.selectbox("Task type", ["Auto-detect", "Coding", "Research / analysis", "Writing", "Summarization", "Education", "Data extraction", "General"])
    tokenizer_profile = st.selectbox("Tokenizer profile", ["cl100k_base", "Approximate offline fallback"])
    st.caption("cl100k_base is used when tiktoken is installed. Otherwise counts use an approximate fallback.")
    st.divider()
    st.subheader("Cost estimator")
    price = st.number_input("Input price per 1M tokens (your model's price)", min_value=0.0, value=2.50, step=0.25, help="Enter the current input-token price for the model you use. This is not fetched automatically.")
    price_source = st.text_input("Price source / model", value="User-entered; verify provider pricing")
    price_checked = st.date_input("Date price checked", value=date.today())
    requests_per_day = st.number_input("Requests per day", min_value=0, value=10000, step=100)
    days_per_month = st.number_input("Days per month", min_value=1, value=30, step=1)
    st.caption("Estimates cover input tokens only. Output tokens, retries and provider-specific billing may add cost.")

examples = {
    "Safe optimization example": 'Please summarize the quarterly report for my manager in exactly 3 bullet points. Keep it under 60 words. Include "$4.2M". Use plain English and no jargon.',
    "Intentional failure demo": 'Summarize the report in exactly 3 bullet points. Keep it under 60 words. Do not include personal data. Include JSON keys: summary, source_id.',
}
example = st.selectbox("Load a demo prompt (optional)", ["— Select —", *examples.keys()])
default_prompt = examples.get(example, "")
raw_prompt = st.text_area("Original prompt", value=default_prompt, placeholder="Paste the exact prompt you plan to use with an AI model…", height=180)
run = st.button("Analyze, optimize & validate", type="primary", use_container_width=True)

if run:
    if not raw_prompt or not raw_prompt.strip():
        st.error("Prompt is empty. Enter a prompt before running the pipeline.")
        st.stop()
    if estimate_tokens(raw_prompt, tokenizer_profile) > 8000:
        st.error("Prompt exceeds the 8,000-token MVP limit. Shorten it before running the pipeline.")
        st.stop()
    try:
        analysis = analyze_prompt(raw_prompt, task_type)
        high, lean = optimize_prompt(raw_prompt, analysis, task_type)
        comparison = compare_prompts(raw_prompt, high, lean, tokenizer_profile)
        high_validation = validate_prompt(raw_prompt, high)
        lean_validation = validate_prompt(raw_prompt, lean)
        st.session_state["last_run"] = {
            "original": raw_prompt, "analysis": analysis, "high": high, "lean": lean,
            "comparison": comparison, "high_validation": high_validation, "lean_validation": lean_validation,
            "price": price, "price_source": price_source, "price_checked": str(price_checked),
            "requests_per_day": int(requests_per_day), "days_per_month": int(days_per_month),
        }
    except Exception as exc:
        st.error(f"The pipeline could not complete safely: {exc}")
        st.stop()

result = st.session_state.get("last_run")
if result:
    original, analysis, high, lean = result["original"], result["analysis"], result["high"], result["lean"]
    comparison = result["comparison"]
    hv, lv = result["high_validation"], result["lean_validation"]
    st.subheader("1 · At-a-glance results")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Original tokens", comparison["original_tokens"])
    m2.metric("ContextMax tokens", comparison["high_tokens"], f'{comparison["high_delta_pct"]:+.1f}% vs original')
    m3.metric("TokenLite tokens", comparison["lean_tokens"], f'{comparison["lean_savings_pct"]:.1f}% saved')
    m4.metric("Extracted requirements", len(analysis.get("requirements", [])))
    st.caption("Token counts are tokenizer-specific or approximate. A negative savings percentage means the candidate is longer.")

    st.subheader("2 · Requirement ledger (IntentLock)")
    reqs = analysis.get("requirements", [])
    if reqs:
        ledger_frame = pd.DataFrame(reqs, columns=["id", "kind", "value", "label", "critical"])
        edited_ledger = st.data_editor(
            ledger_frame, key="intentlock_ledger", num_rows="dynamic",
            column_config={
                "id": st.column_config.TextColumn("ID", disabled=True),
                "kind": st.column_config.SelectboxColumn("Check type", options=["exact_count", "limit", "literal", "quoted", "prohibition", "inclusion", "json_key", "format"], required=True),
                "value": st.column_config.TextColumn("Required value / clause", required=True),
                "label": st.column_config.TextColumn("Requirement description", required=True),
                "critical": st.column_config.CheckboxColumn("Critical", default=True),
            },
            use_container_width=True, hide_index=True
        )
        if st.button("Apply ledger edits and revalidate", key="apply_ledger"):
            custom_reqs = []
            for i, row in enumerate(edited_ledger.to_dict("records"), 1):
                value = str(row.get("value", "")).strip()
                kind = str(row.get("kind", "")).strip()
                if not value or not kind:
                    continue
                custom_reqs.append({
                    "id": str(row.get("id") or f"R{i}"), "kind": kind, "value": value,
                    "label": str(row.get("label") or f"Check: {value}"),
                    "critical": bool(row.get("critical", True))
                })
            st.session_state["last_run"]["analysis"]["requirements"] = custom_reqs
            st.session_state["last_run"]["high_validation"] = validate_prompt(original, high, custom_reqs)
            st.session_state["last_run"]["lean_validation"] = validate_prompt(original, lean, custom_reqs)
            st.rerun()
        st.caption("Edit a requirement, add a row, remove a row, or toggle Critical. Apply changes to rerun validation and the recommendation gate.")
    else:
        st.warning("No machine-checkable requirements found. Automated preservation cannot be confirmed; human review is required.")

    st.subheader("3 · Validation gate")
    vcols = st.columns(2)
    for col, name, val in [(vcols[0], "ContextMax", hv), (vcols[1], "TokenLite", lv)]:
        with col:
            st.markdown(f"**{name}**")
            if val["status"] == "PASS":
                st.success(val["message"])
            elif val["status"] == "FAIL":
                st.error(val["message"])
            else:
                st.warning(val["message"])
            st.caption(f'{val["pass_count"]} passed · {val["fail_count"]} failed · {val["uncertain_count"]} uncertain')
            st.dataframe(pd.DataFrame([{"ID": x["id"], "Check": x["label"], "Status": x["status"], "Details": x["details"]} for x in val["checks"]]), use_container_width=True, hide_index=True)

    st.subheader("4 · Optimized candidates")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("### 🧠 ContextMax")
        st.caption("Adds execution guidance while preserving the full original request.")
        st.text_area("ContextMax prompt", value=high, height=250, key="shown_high")
        st.button("Prepare ContextMax for copying", key="copy_high", on_click=lambda: st.session_state.update(copy_text=high))
        st.caption(f'Estimated tokens: {comparison["high_tokens"]}')
    with c2:
        st.markdown("### ⚡ TokenLite")
        st.caption("Applies conservative filler-word substitutions; preserves original constraints.")
        st.text_area("TokenLite prompt", value=lean, height=250, key="shown_lean")
        st.button("Prepare TokenLite for copying", key="copy_lean", on_click=lambda: st.session_state.update(copy_text=lean))
        st.caption(f'Estimated tokens: {comparison["lean_tokens"]}')
    if st.session_state.get("copy_text"):
        st.info("Copy-ready text (select and copy):")
        st.code(st.session_state["copy_text"], language=None)

    st.subheader("4A · Change diff")
    st.caption("Word-level differences between the original prompt and TokenLite. This explains text removal/addition; it is not a semantic proof.")
    import difflib
    diff_html = difflib.HtmlDiff(wrapcolumn=90).make_table(
        original.splitlines(), lean.splitlines(),
        fromdesc="Original", todesc="TokenLite", context=True, numlines=2
    )
    st.components.v1.html(diff_html, height=260, scrolling=True)

    st.subheader("5 · Recommendation")
    # Fail-closed: only candidates passing every machine-checkable check can be recommended.
    validated_candidates = []
    if hv["safe_to_recommend"]: validated_candidates.append(("ContextMax", comparison["high_tokens"], high))
    if lv["safe_to_recommend"]: validated_candidates.append(("TokenLite", comparison["lean_tokens"], lean))
    safe_candidates = [
        item for item in validated_candidates
        if comparison["original_tokens"] > 0 and
        (comparison["original_tokens"] - item[1]) / comparison["original_tokens"] * 100 >= 10
    ]
    if not safe_candidates:
        st.error("No candidate meets the recommendation gate: every critical check must pass AND estimated input-token savings must be at least 10%. Keep the original prompt and review the ledger manually.")
        if validated_candidates:
            st.info("A candidate passed the extracted checks but did not meet the 10% savings threshold. It is not recommended automatically.")
    else:
        best = min(safe_candidates, key=lambda x: x[1])
        st.success(f"Recommendation: {best[0]} — critical checks passed and estimated input-token savings meet the 10% gate.")
        st.caption("This is a provisional recommendation. Passing lexical checks does not guarantee semantic equivalence or answer quality.")

    st.subheader("5A · Intentional failure demonstration")
    st.caption("This deliberately damages a candidate to demonstrate that IntentLock blocks a recommendation when a critical requirement disappears.")
    critical_to_break = next((r for r in reqs if r["kind"] in {"limit", "exact_count", "prohibition", "json_key", "inclusion"}), None)
    if critical_to_break:
        if st.button(f"Simulate failure: remove {critical_to_break['label']}", key="simulate_failure"):
            damaged = lean
            if critical_to_break["kind"] == "json_key":
                damaged = re.sub(r"\\b" + re.escape(critical_to_break["value"]) + r"\\b", "", damaged)
            else:
                damaged = re.sub(r"\\b" + re.escape(critical_to_break["value"]) + r"\\b", "", damaged)
            failure_result = validate_prompt(original, damaged)
            st.session_state["failure_demo"] = {"candidate": damaged, "result": failure_result, "requirement": critical_to_break["label"]}
    else:
        st.info("This prompt has no supported critical requirement to remove. Load the safe example or a prompt with explicit constraints.")
    failure_demo = st.session_state.get("failure_demo")
    if failure_demo:
        st.write("Removed requirement:", failure_demo["requirement"])
        st.code(failure_demo["candidate"], language=None)
        if failure_demo["result"]["status"] == "FAIL":
            st.error("PASS: the safety gate correctly rejected the deliberately damaged candidate.")
        else:
            st.warning(f"The automated checks returned {failure_demo['result']['status']}; inspect the ledger and do not claim a successful rejection.")

    st.subheader("6 · Cost and savings estimate")
    price_now = result["price"]
    st.caption(f"Price source/model: {result['price_source']} · checked: {result['price_checked']}")
    daily = int(result["requests_per_day"])
    monthly_requests = daily * int(result["days_per_month"])
    cost_cols = st.columns(3)
    for col, name, tokens in [(cost_cols[0], "Original", comparison["original_tokens"]), (cost_cols[1], "ContextMax", comparison["high_tokens"]), (cost_cols[2], "TokenLite", comparison["lean_tokens"])]:
        col.metric(f"{name} · daily input cost", f"${estimate_cost(tokens, price_now, daily):,.4f}")
        col.caption(f"Per call: ${estimate_cost(tokens, price_now):,.6f}")
        col.caption(f"Monthly estimate: ${estimate_cost(tokens, price_now, monthly_requests):,.2f}")
    saved_daily = estimate_cost(comparison["original_tokens"] - comparison["lean_tokens"], price_now, daily)
    st.write(f"**Estimated daily difference (Original − TokenLite):** ${saved_daily:,.4f}. Negative means TokenLite costs more.")
    st.caption(f"Formula: tokens × input price per 1,000,000 × requests. Configured price: ${price_now:g} per 1M input tokens.")

    with st.expander("Analysis details and limitations"):
        st.write("**Likely task:**", analysis["intent"])
        st.write("**Detected signals:**", ", ".join(analysis["signals"]) if analysis["signals"] else "No strong signals detected.")
        for warning in analysis["warnings"]: st.warning(warning)
        st.markdown("""
- This version uses deterministic, rules-based optimization; it does not call an LLM.
- IntentLock checks extracted literals, numerical limits, prohibitions, required inclusions, JSON keys and output-format mentions.
- Passing these checks does not guarantee semantic equivalence or the same model answer.
- Token counts are exact only for the selected tokenizer, not necessarily the eventual provider/model.
- Prices are user-entered and estimates cover input tokens only.
""")
    report = f"""PromptForge AI — Run Report

ORIGINAL
{original}

CONTEXTMAX
{high}

TOKENLITE
{lean}

VALIDATION
ContextMax: {hv['status']} ({hv['pass_count']} pass, {hv['fail_count']} fail, {hv['uncertain_count']} uncertain)
TokenLite: {lv['status']} ({lv['pass_count']} pass, {lv['fail_count']} fail, {lv['uncertain_count']} uncertain)

TOKENS
Original: {comparison['original_tokens']}
ContextMax: {comparison['high_tokens']}
TokenLite: {comparison['lean_tokens']}
TokenLite savings: {comparison['lean_savings_pct']:.2f}%

COST ASSUMPTIONS
Input price per million: ${price_now:g}
Price source/model: {result['price_source']}
Price checked: {result['price_checked']}
Requests/day: {daily}
Days/month: {result['days_per_month']}
"""
    st.download_button("Download run report (TXT)", report, file_name="promptforge_run_report.txt", mime="text/plain")
    json_report = {
        "original_prompt": original,
        "contextmax_prompt": high,
        "tokenlite_prompt": lean,
        "token_metrics": comparison,
        "validation": {"contextmax": hv, "tokenlite": lv},
        "cost_assumptions": {"input_price_per_million": price_now, "price_source": result["price_source"], "price_checked": result["price_checked"], "requests_per_day": daily, "days_per_month": result["days_per_month"]},
        "recommendation_gate": "pass" if safe_candidates else "blocked",
    }
    st.download_button("Download structured report (JSON)", json.dumps(json_report, indent=2), file_name="promptforge_run_report.json", mime="application/json")
else:
    st.subheader("What this version does")
    a, b, c = st.columns(3)
    a.info("**Extract**\n\nCreate an explicit requirement ledger from the original prompt.")
    b.info("**Validate**\n\nCheck numbers, limits, prohibitions, quoted values, JSON keys and formats.")
    c.info("**Gate recommendations**\n\nDo not recommend a candidate if any extracted check fails or remains uncertain.")
    st.markdown("### Suggested live demo")
    st.markdown("1. Load **Safe optimization example** and run the pipeline.\n2. Show the requirement ledger and PASS results.\n3. In your test script, remove `under 60 words` or a JSON key and show the validator rejecting the altered candidate.")
