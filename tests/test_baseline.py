"""O1: the reference file is well formed and the baseline is reproducible."""

from baseline import greedy_generate, tokens_digest


def test_reference_file_is_complete(reference):
    entries = reference["prompts"]
    meta = reference["meta"]
    assert len(entries) >= 20
    lengths = {e["prompt_len"] for e in entries}
    assert min(lengths) <= 3 and max(lengths) >= 50, "prompt lengths should vary"
    for e in entries:
        assert len(e["output_token_ids"]) == meta["max_new_tokens"]
        assert len(e["per_token_latency_ms"]) == meta["max_new_tokens"]
    assert tokens_digest(entries) == meta["tokens_sha256"]


def test_baseline_rerun_is_byte_identical(hf_model, reference):
    n = reference["meta"]["max_new_tokens"]
    for e in reference["prompts"]:
        out, _ = greedy_generate(hf_model, e["prompt_token_ids"], n)
        assert out == e["output_token_ids"], f"prompt {e['id']} differs on re-run"
