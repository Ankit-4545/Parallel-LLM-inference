"""O2: running stage 0 then stage 1 reproduces the full model."""

import pytest
import torch

from pllm.partition import PipelineStage, SegmentInput, load_stages


def split_logits(stages, ids):
    """Full-sequence forward through all stages; logits for every position."""
    x = stages[0].embed(torch.tensor(ids), torch.arange(len(ids)))
    for st in stages:
        x, _ = st.forward_segments(x, [SegmentInput(len(ids), None)])
    return stages[-1].head(x)


def test_split_logits_match_full_model(hf_model, stages, reference):
    worst = 0.0
    for e in reference["prompts"]:
        ids = e["prompt_token_ids"]
        full = hf_model(input_ids=torch.tensor([ids])).logits[0]
        split = split_logits(stages, ids)
        diff = (full - split).abs().max().item()
        worst = max(worst, diff)
        assert diff <= 1e-5, f"prompt {e['id']}: max |logit diff| = {diff:.3e}"
        assert torch.equal(full.argmax(-1), split.argmax(-1))
    print(f"\nmax |full - split| logit difference over all prompts/positions: {worst:.3e}")


def test_split_with_kv_cache_reproduces_reference_tokens(stages, reference):
    s0, s1 = PipelineStage(stages[0]), PipelineStage(stages[1])
    n = reference["meta"]["max_new_tokens"]
    for e in reference["prompts"]:
        rid, out = f"r{e['id']}", []
        inp = torch.tensor(e["prompt_token_ids"])
        for step in range(n):
            h = s0.forward(inp, rid, step)
            logits = s1.forward(h, rid, step)
            tok = int(logits.argmax())
            out.append(tok)
            inp = torch.tensor([tok])
        assert out == e["output_token_ids"], f"prompt {e['id']} differs"


@pytest.mark.parametrize("num_stages,split", [(2, "3"), (2, "9"), (3, "4,8"), (4, "auto"), (2, "auto")])
def test_other_partitions_are_equivalent(model_dir, stages, reference, num_stages, split):
    other = load_stages(model_dir, num_stages=num_stages, split=split)
    ids = reference["prompts"][-1]["prompt_token_ids"]
    a, b = split_logits(stages, ids), split_logits(other, ids)
    assert (a - b).abs().max().item() <= 1e-5


def test_each_stage_holds_about_half_the_weights(stages, hf_model):
    full = sum(p.numel() * p.element_size() for p in hf_model.parameters())  # tied wte counted once
    s0, s1 = (s.param_bytes() for s in stages)
    print(f"\nfull model {full / 2**20:.1f} MiB | stage0 {s0 / 2**20:.1f} MiB ({s0 / full:.0%}) "
          f"| stage1 {s1 / 2**20:.1f} MiB ({s1 / full:.0%})")
    # Each stage holds 6 of 12 blocks; the tied 147 MiB embedding is needed by both the
    # input embedding (stage 0) and the LM head (stage 1), so each is ~2/3 of the full model.
    for s in (s0, s1):
        assert 0.45 * full < s < 0.70 * full
    assert abs(s0 - s1) / full < 0.02  # balanced
    # Only the needed transformer blocks are loaded.
    assert stages[0].layer_start == 0 and stages[0].layer_end == 6 and stages[1].layer_end == 12
    assert len(stages[0].blocks) == 6 and len(stages[1].blocks) == 6
    assert not hasattr(stages[1], "wpe")
