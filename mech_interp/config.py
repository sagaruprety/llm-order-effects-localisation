"""Models, paths, and tracing setup shared across mech_interp scripts."""

import os
from pathlib import Path

import torch
from dotenv import load_dotenv

load_dotenv()

ROOT_DIR = Path(__file__).resolve().parent
RESULTS_DIR = ROOT_DIR / "results"
RESULTS_DIR.mkdir(exist_ok=True)

# Tversky & Gati (1978) used a 1-20 scale ("1 (no similarity) to 20 (maximal
# similarity)", p. 87). The 2024 paper's prompts say "0 to 20" and this
# pipeline inherits that, so the scale is off by one point at the floor
# relative to the original instrument. Kept for continuity with the published
# results; noted as a deviation rather than silently corrected, since changing
# it would invalidate comparison with the 2024 numbers.
RATING_MIN = 0
RATING_MAX = 20

# --- Constrained answer format ----------------------------------------------

# The 2024 experiments asked for JSON ("Shape: score: int") and parsed whatever
# the model generated. Here we instead *write the JSON scaffold ourselves* and
# append it to the prompt, so the rating is the immediate next token. Since the
# schema has a single field, nothing is left for the model to choose before the
# number -- this is equivalent to constrained decoding, but keeps three
# properties the pipeline depends on:
#   1. every sample shares an identical prefix, so one forward pass fully
#      determines the rating distribution (n sampled ratings can then be drawn
#      from it directly, rather than costing n forward passes);
#   2. the read-out position stays a simple "last position", which the
#      logit-lens and patching hooks both assume;
#   3. no decoding wrapper sits between us and the forward hooks in patching.py.
#
# Whether the number tokenizes as "14" or " 14" depends on the trailing space,
# and which is right differs per tokenizer -- so both are tried and the one
# putting more probability mass on rating tokens wins (see
# behavioral.select_answer_prefix).
ANSWER_PREFIX_VARIANTS = ('{"score": ', '{"score":')

# --- Models -----------------------------------------------------------------

MODEL_REPO_IDS = {
    "qwen2.5-0.5b": "Qwen/Qwen2.5-0.5B-Instruct",  # debug model, MPS/CPU
    "llama3.2-1b": "meta-llama/Llama-3.2-1B-Instruct",
    "llama3.2-3b": "meta-llama/Llama-3.2-3B-Instruct",
    "llama3.1-8b": "meta-llama/Llama-3.1-8B-Instruct",
    "mistral-7b": "mistralai/Mistral-7B-Instruct-v0.2",
    "qwen2.5-7b": "Qwen/Qwen2.5-7B-Instruct",
    # Winners from the OpenRouter behavioral screen (mech_interp/openrouter_screen.py):
    # gemma3-4b and qwen2.5-7b matched the human/Tversky direction consistently
    # across templates; qwen3-8b and gemma3-12b showed a consistent *reversed*
    # asymmetry. Added here to run the local logit-based pipeline on all three.
    "gemma3-4b": "google/gemma-3-4b-it",
    "qwen3-8b": "Qwen/Qwen3-8B",
    "gemma3-12b": "google/gemma-3-12b-it",
}

DEBUG_MODEL = "qwen2.5-0.5b"

# Models whose bf16 weights alone (~2 bytes/param) leave no headroom for
# activations/KV cache on a 24GB L4 (gemma3-12b: 12B params ~= 24GB). Loaded
# in 4-bit via bitsandbytes instead; forward hooks and manual norm+lm_head
# calls (logit_lens.py) both work fine against a quantized model since hooks
# attach to module calls, not parameter dtype.
QUANTIZE_4BIT_MODELS = {"gemma3-12b"}


def load_model_and_tokenizer(
    model_key: str, device: str, attn_implementation: str | None = None
):
    """AutoModelForCausalLM + AutoTokenizer for `model_key`, 4-bit-quantized
    if it's in QUANTIZE_4BIT_MODELS, else bf16.

    `attn_implementation` is passed through to from_pretrained. It must be
    "eager" to read attention weights back via output_attentions=True: the
    default sdpa/flash kernels fuse the softmax and never materialise the
    attention matrix, so they return None for it (see jury_attention.py). Left
    at None (HF's default) for the logit-lens/patching pipelines, which never
    inspect attention and are faster without eager.
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer

    repo_id = MODEL_REPO_IDS[model_key]
    tokenizer = AutoTokenizer.from_pretrained(repo_id)

    kwargs = {}
    if attn_implementation is not None:
        kwargs["attn_implementation"] = attn_implementation

    if model_key in QUANTIZE_4BIT_MODELS:
        from transformers import BitsAndBytesConfig

        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4",
        )
        model = AutoModelForCausalLM.from_pretrained(
            repo_id, quantization_config=quant_config, device_map=device, **kwargs
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            repo_id, torch_dtype=torch.bfloat16, device_map=device, **kwargs
        )
    model.eval()
    return model, tokenizer


def get_text_decoder(model):
    """The transformer decoder stack -- the module owning `.layers` and `.norm`.

    Text-only models (Llama, Qwen) expose it as `model.model`. Multimodal
    models (Gemma-3, whose class is *ForConditionalGeneration* even for the
    text-only checkpoints) nest it under `model.model.language_model`.
    """
    base = getattr(model, "model", model)
    if hasattr(base, "language_model"):
        base = base.language_model
    if not hasattr(base, "layers"):
        raise AttributeError(
            f"Could not find the decoder layer stack on {type(model).__name__}; "
            "inspect the architecture and adjust get_text_decoder()."
        )
    return base


def get_num_layers(model) -> int:
    """Number of transformer blocks, for flat and nested (multimodal) configs."""
    cfg = model.config
    if hasattr(cfg, "num_hidden_layers"):
        return cfg.num_hidden_layers
    if hasattr(cfg, "text_config") and hasattr(cfg.text_config, "num_hidden_layers"):
        return cfg.text_config.num_hidden_layers
    raise AttributeError(
        f"Could not determine layer count from {type(cfg).__name__}; "
        "inspect the config and adjust get_num_layers()."
    )


def encode_prompt(tokenizer, prompt: str, device: str, answer_prefix: str | None = None):
    """apply_chat_template as an input_ids tensor on `device`.

    Some tokenizers (e.g. Qwen3's) return a BatchEncoding from
    apply_chat_template(..., return_tensors="pt") instead of a raw tensor;
    this normalizes both cases.

    `answer_prefix` (e.g. '{"score": ') is appended after the generation prompt
    so the model's next token is the rating itself -- see ANSWER_PREFIX_VARIANTS.
    """
    messages = [{"role": "user", "content": prompt}]
    encoded = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, return_tensors="pt"
    )
    input_ids = encoded["input_ids"] if hasattr(encoded, "keys") else encoded
    if answer_prefix:
        prefix_ids = torch.tensor(
            [tokenizer.encode(answer_prefix, add_special_tokens=False)], dtype=input_ids.dtype
        )
        input_ids = torch.cat([input_ids, prefix_ids], dim=1)
    return input_ids.to(device)


def get_device(preferred: str | None = None) -> str:
    """Resolve --device: explicit choice, else best available."""
    if preferred:
        return preferred
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


# --- Tracing (Arize Phoenix, replaces LangSmith) -----------------------------

def maybe_setup_phoenix_tracing() -> None:
    """Start a local Phoenix session and instrument HF calls, if installed.

    No-op if `arize-phoenix` isn't installed or PHOENIX_ENABLED=0, so scripts
    run fine without tracing during quick debug loops.
    """
    if os.environ.get("PHOENIX_ENABLED", "1") == "0":
        return
    try:
        import phoenix as px
        from openinference.instrumentation.transformers import TransformersInstrumentor

        px.launch_app()
        TransformersInstrumentor().instrument()
    except ImportError:
        pass
