"""Step 2: logit lens -- layer-wise readout of the order asymmetry.

For each model/layer: cache the residual stream at the final token position,
apply the model's final LayerNorm + unembedding ("logit lens"), compute E[r]
per order at that layer, and record Delta(L) = E[r|order1] - E[r|order2] as a
function of normalized depth. This is the core novel result (MECH_INTERP_PLAN
Step 2) -- it locates *where* an effect already established by behavioral.py
(Step 1) emerges in the residual stream, it does not establish the effect
itself.

Handles both single-token-rating models (Llama-3.x) and two-token-rating
models (Qwen, Gemma, Mistral: "10".."20" split into two digit tokens). For
two-token ratings, only the *first* token's decision is examined layer-by-
layer -- an intermediate layer has one well-defined next-token distribution
for the position right after the prompt, but no analogue for a continuation
token that hasn't been generated yet. The second-token conditional is always
drawn from a real, full-depth forward pass (behavioral.py's
compute_second_token_probs) and held fixed across all layers. This means
Delta(L) for these models reflects the layer at which the *first-digit*
decision (e.g. single digit vs. "1x"/"2x") diverges between orders, which is
coarser than the single-token case but still layer-localized.
"""

import argparse

import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from mech_interp.behavioral import (
    combine_first_and_second_probs,
    compute_second_token_probs,
    select_answer_prefix,
)
from mech_interp.config import (
    DEBUG_MODEL,
    MODEL_REPO_IDS,
    RATING_MAX,
    RATING_MIN,
    RESULTS_DIR,
    encode_prompt,
    get_device,
    get_num_layers,
    get_text_decoder,
    load_model_and_tokenizer,
)
from mech_interp.stimuli import build_stimuli


def get_final_norm(model):
    """Return the model's final pre-unembedding LayerNorm/RMSNorm module."""
    decoder = get_text_decoder(model)
    if hasattr(decoder, "norm"):
        return decoder.norm
    raise AttributeError(
        f"Could not find a final norm module on {type(model).__name__}; "
        "inspect the model architecture and adjust get_final_norm()."
    )


def logit_lens_expected_ratings(
    model, tokenizer, prompt: str, rating_token_seqs: dict[int, list[int]], device: str,
    answer_prefix: str | None = None,
) -> dict[int, float]:
    """E[r] at every layer via the logit lens, for a single prompt.

    Returns {layer_index: expected_rating}, layer 0 = embeddings,
    layer n_layers = final residual stream (post final block, same as
    behavioral.py's direct read-out, useful as a consistency check).
    """
    prompt_ids = encode_prompt(tokenizer, prompt, device, answer_prefix)

    with torch.no_grad():
        out = model(prompt_ids, output_hidden_states=True)
    hidden_states = out.hidden_states  # tuple: (embeddings, layer_1, ..., layer_n)
    second_token_probs = compute_second_token_probs(model, prompt_ids, rating_token_seqs, device)

    final_norm = get_final_norm(model)
    lm_head = model.get_output_embeddings()

    # HF appends hidden states *before* each block, then appends one more after
    # the final norm -- so hidden_states[-1] is already normed and must not be
    # normed again (doing so shifts its logits by ~8, silently corrupting the
    # final point of every Delta(L) curve, which is also the consistency check
    # against behavioral.py's direct read-out).
    last_idx = len(hidden_states) - 1

    results = {}
    with torch.no_grad():
        for layer_idx, h in enumerate(hidden_states):
            last_token_h = h[0, -1]
            resid = last_token_h if layer_idx == last_idx else final_norm(last_token_h)
            first_probs = lm_head(resid).softmax(dim=0)
            results[layer_idx] = combine_first_and_second_probs(
                first_probs, rating_token_seqs, second_token_probs
            )
    return results


def run_logit_lens(model_key: str, device: str | None = None) -> pd.DataFrame:
    device = get_device(device)

    model, tokenizer = load_model_and_tokenizer(model_key, device)

    stimuli = build_stimuli(RATING_MIN, RATING_MAX)
    answer_prefix, rating_token_seqs = select_answer_prefix(
        model, tokenizer, stimuli[0].prompt, device
    )
    n_layers = get_num_layers(model)

    rows = []
    for stim in stimuli:
        per_layer = logit_lens_expected_ratings(
            model, tokenizer, stim.prompt, rating_token_seqs, device, answer_prefix
        )
        for layer_idx, e_r in per_layer.items():
            rows.append(
                {
                    "model": model_key,
                    "pair": f"{stim.pair[0]}-{stim.pair[1]}",
                    "is_control": stim.is_control,
                    "template": stim.template_name,
                    "task_type": stim.task_type,
                    "order": stim.order,
                    "layer": layer_idx,
                    "layer_frac": layer_idx / n_layers,
                    "expected_rating": e_r,
                }
            )

    df = pd.DataFrame(rows)
    out_path = RESULTS_DIR / f"logit_lens_{model_key}.csv"
    df.to_csv(out_path, index=False)
    print(f"Saved {len(df)} rows to {out_path}")
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=DEBUG_MODEL, choices=list(MODEL_REPO_IDS))
    parser.add_argument("--device", default=None)
    args = parser.parse_args()
    run_logit_lens(args.model, args.device)
