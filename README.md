# Specific or Generic? Localising Order Effects in LLM Judgements Across Two Cognitive Paradigms

Code and results for the Findings of EACL 2027 paper.

The paper asks where in a network an order effect arises, and argues that no
intervention can be said to *locate* a cognitive bias until it has been shown to
do nothing where the bias should be absent. Every experiment here is therefore
paired with a control condition for which theory predicts no effect, and the
informative quantity throughout is the **gap** between the two curves, not the
height of either one.

Two paradigms:

- **Similarity judgements** (Tversky and Gati, 1978): human ratings shift when
  the two countries being compared are swapped. Prominence pairs are read
  against equal-prominence control pairs.
- **Jury decision-making** (Trueblood and Busemeyer, 2011): the same defence
  evidence raises or lowers confidence in guilt depending on the prosecution
  case that precedes it. The defence conditions are read against a neutral-item
  condition with no defence present.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

Llama-3.1 and Gemma-3 are gated on HuggingFace: request access on each model
page, then put a read token in `.env` as `HUGGINGFACEHUB_API_TOKEN`.

Everything was run on a single 24GB NVIDIA L4. Gemma-3-12B is the one model
loaded in 4-bit, because its bfloat16 weights alone fill that card; the other
three run in bfloat16. See `mech_interp/config.py`.

## Datasets

There are three kinds of data here, and it is worth being clear about which is
which.

**1. The stimuli** are defined in code, in `mech_interp/stimuli.py` and
`mech_interp/stimuli_jury.py`, and those modules are authoritative: the
experiment scripts build their prompts from them directly. For inspection
without reading Python, `data/` holds a generated export of exactly the same
content:

| File | Contents |
|---|---|
| `data/similarity_pairs.csv` | 21 prominence pairs and 10 equal-prominence controls |
| `data/similarity_templates.csv` | The six prompt templates |
| `data/similarity_prompts.csv` | All 372 rendered prompts (31 pairs × 6 templates × 2 orders) |
| `data/jury_evidence.csv` | The four evidence blocks, verbatim |
| `data/jury_conditions.csv` | The nine conditions and their evidence order |
| `data/jury_prompts.csv` | The nine rendered condition prompts |
| `data/stimuli.json` | All of the above in one nested document |

Regenerate with `python mech_interp/export_stimuli.py`. A test fails if the
export drifts from the stimulus modules, so what is in `data/` is always what
the experiments ran.

The similarity stimuli are the country pairs of Tversky and Gati (1978, Study
2). The 10 equal-prominence control pairs are ours: Tversky's account predicts
no order asymmetry for them, which is what makes them a specificity control
rather than a second treatment. The jury materials are the vignette of
Trueblood and Busemeyer (2011); the three neutral-item conditions (`NE`,
`SP_NE`, `WP_NE`) are ours, and serve the same purpose.

**2. Human ground truth** is published aggregate data reproduced from the source
studies, not data collected here:

- `results/results_human.csv` — Tversky and Gati (1978) ratings for the 21 pairs
- `results/results_jury.csv` — Trueblood and Busemeyer (2011) Table 3

No data was collected from human participants for this work.

**3. Model outputs** are in `mech_interp/results/`: 51 CSVs, one per
(experiment, model), containing the measurements behind every figure and table
in the paper. These are what the plotting scripts read.

## Reproducing the figures

The committed CSVs under `mech_interp/results/` are the exact results behind the
paper, so **the figures can be rebuilt without a GPU**:

```bash
python mech_interp/plot_logit_lens_delta.py     --out plots/logit_lens_delta.pdf
python mech_interp/plot_similarity_patching.py  --out plots/patching_prominence_vs_control.pdf
python mech_interp/plot_jury_logit_lens.py      --out plots/jury_logit_lens_recency.pdf
python mech_interp/plot_jury_patching.py        --out plots/jury_patching_recency.pdf
python mech_interp/plot_jury_steering.py        --out plots/jury_steering.pdf
```

| Paper figure | Script | Reads |
|---|---|---|
| Logit-lens Δ(L), similarity | `plot_logit_lens_delta.py` | `logit_lens_<model>.csv` |
| Patching, prominence vs control | `plot_similarity_patching.py` | `patching_<model>.csv` |
| Jury logit lens | `plot_jury_logit_lens.py` | `jury_logit_lens_<model>.csv` |
| Jury patching + KL | `plot_jury_patching.py` | `jury_patching_<model>.csv` |
| Jury steering | `plot_jury_steering.py` | `jury_steering_<model>.csv` |

## Re-running the experiments

Needs a GPU and will overwrite `mech_interp/results/`. Model keys are
`llama3.1-8b`, `gemma3-4b`, `gemma3-12b`, `qwen3-8b`.

```bash
# Behavioural ratings, both paradigms
python mech_interp/behavioral.py --model llama3.1-8b --task similarity
python mech_interp/behavioral.py --model llama3.1-8b --task jury

# Similarity: logit lens and activation patching
python mech_interp/logit_lens.py --model llama3.1-8b
python mech_interp/patching.py   --model llama3.1-8b

# Jury: logit lens, patching, and steering the prosecution-strength direction
python mech_interp/jury_interp.py   --model llama3.1-8b --step logit_lens
python mech_interp/jury_interp.py   --model llama3.1-8b --step patching
python mech_interp/jury_steering.py --model llama3.1-8b

# Statistics over the behavioural results
python mech_interp/analysis.py --model llama3.1-8b
```

`mech_interp/archive_run.py` snapshots `mech_interp/results/` into a timestamped
directory, so a published claim never rests on a CSV that the next sweep can
overwrite in place.

### Determinism

No sampling is involved anywhere. Each prompt is followed by a fixed JSON answer
scaffold (`{"score": `) so the rating is the immediate next token, and the rating
is the expected value of the model's next-token distribution over rating tokens
at that position, computed in a single forward pass. No temperature, seed or
decoding parameter enters the measurement. Ratings 10–20 occupy two tokens in
the Gemma and Qwen tokenizers but one in Llama-3.1; for two-token ratings the
second token's conditional probability is scored with an additional forward pass
and multiplied.

## Layout

```
mech_interp/
  config.py             Model registry, paths, answer scaffold, 4-bit policy
  stimuli.py            Similarity stimuli and prompt templates (authoritative)
  stimuli_jury.py       Jury evidence blocks and the nine conditions (authoritative)
  export_stimuli.py     Dumps both of the above to data/ as CSV and JSON
  build_jury_data.py    Writes the human jury ground truth to results/
  behavioral.py         E[r] measurement for both paradigms
  logit_lens.py         Per-layer Δ(L) for similarity
  patching.py           Activation patching for similarity
  jury_interp.py        Jury logit lens and patching
  jury_steering.py      Steering the prosecution-strength direction
  analysis.py           Wilcoxon tests and summary tables
  archive_run.py        Snapshot results/ into a timestamped run
  plot_*.py             One script per paper figure
  tests/                Regression cover for the rating-probability combination
  results/              The CSVs behind the paper's figures and tables
data/                   Generated export of the stimuli (see Datasets above)
results/
  results_human.csv     Tversky and Gati (1978) published aggregate ratings
  results_jury.csv      Trueblood and Busemeyer (2011) published aggregates
```

No data was collected from human participants for this work, and no model was
trained or fine-tuned.

## Tests

```bash
python -m pytest mech_interp/tests/
```

## Citation

```bibtex
@inproceedings{uprety2027specific,
  title     = {Specific or Generic? Localising Order Effects in {LLM} Judgements Across Two Cognitive Paradigms},
  author    = {Uprety, Sagar and Jaiswal, Amit Kumar and Tiwari, Prayag and Liu, Haiming and Song, Dawei},
  booktitle = {Findings of the Association for Computational Linguistics: EACL 2027},
  year      = {2027}
}
```

## Licence

Code released under the MIT Licence (see `LICENSE`). The model checkpoints are
used under their own terms: the Llama 3.1 Community License, the Gemma Terms of
Use, and Apache 2.0 for Qwen3. The stimuli are reproduced from the published
psychology studies cited above.
