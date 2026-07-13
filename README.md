# Argus

**An LLM computes the gradients.** Argus treats a RAG pipeline like a neural network and
runs a *training pass* on it — except the optimizer isn't SGD, it's a language model. One
pipeline run is a **forward pass**; a strong model reads the intermediate artifacts plus an
F1-based **loss** and emits a **textual gradient** — which knob to move, which way, and why.
Apply, re-run, re-grade, repeat, until the loss plateaus.

The demo takes a deliberately mis-configured retrieval pipeline (`mean F1 ≈ 0.17`) and
converges every knob to a good setting (`mean F1 = 1.0`) in a handful of legible steps —
each one mapping to a named retrieval failure mode.

Argus optimizes **only** via an LLM — there is no rule-based fallback. Set your key once
(loaded automatically from `.env`), then run:

```
cp .env.example .env          # then put your ANTHROPIC_API_KEY in it
pip install -e .

python run.py --brain anthropic   # Anthropic API brain (key from .env)
python run.py --no-cache          # ablation: identical result, more stage computes
```

---

## Why this is interesting

Tuning a RAG pipeline is a search over a dozen coupled knobs — chunk size, retrieval mode,
candidate count, rerank cut-off, MMR diversity, the synthesis prompt — and the usual process
is a human reading logs, forming a hypothesis, changing one thing, and re-running an eval by
hand. Argus automates exactly that loop, with an LLM in the role a human tuner plays.

It stands on real prior art and names its combination honestly:

| Prior art | Idea Argus reuses |
|---|---|
| **OPRO** (*LLMs as Optimizers*) | feed the optimizer the trajectory of past `(params → score)` attempts |
| **TextGrad** (*differentiation via text*) | the model's rationale *is* the gradient |
| **DSPy / MIPRO** | optimize the prompt as a first-class parameter against a metric |
| **Reflexion** | a short self-critique before each step |

What's new in the combination: optimizing a **full retrieval cascade's heterogeneous knobs**
(numeric retrieval params *and* the free-text synthesis prompt) jointly, using the
**intermediate pipeline artifacts as activations**, against an **F1-based loss**, with
**staged forward-pass caching** so most steps are cheap.

## The analogy

| Neural-net training | Argus |
|---|---|
| forward pass | one pipeline run: chunk → embed → retrieve → rerank → MMR → synthesize |
| activations | per-stage trace: candidates, rerank scores, what got selected/collapsed, the answer |
| parameters | the knobs (incl. the synthesis prompt as a free-text weight) |
| loss | `1 − composite F1` (recall + precision, with a hallucination penalty) |
| gradient (backprop) | the LLM's "move knob X this way because…" |
| optimizer step | apply the move under a trust region (accept iff loss doesn't worsen) |
| optimizer state / momentum | the in-context history of past attempts |
| learning rate | how far a knob may move per round |
| convergence | loss plateau over a patience window |

## What you'll see

```
round 1   retrieval reach   bm25→hybrid, candidate_count 2→18      loss 0.87 → 0.62
round 2   diversity         enable MMR, budget 2→8                 loss 0.62 → 0.37
round 3   context budget    doc_char_limit 600→8000                loss 0.37 → 0.24
round 4   grounding         rewrite synthesis prompt (cite-or-refuse) loss 0.24 → 0.00
round 5   converged
```

Each round fixes the *current dominant* failure, in order: you can't ground a prompt to drop
a hallucination until retrieval actually surfaces the correct fact to ground on.

## The engineered corpus

Every question is built so a *specific* broken knob is the reason it fails — which makes the
optimizer's reasoning checkable:

| Question | Designed failure | The knob that fixes it |
|---|---|---|
| flight time / warranty | fact buried in a long padded doc | `chunk_chars`, `candidate_count`, `doc_char_limit` |
| admin authentication | gold fact crowded out by 9 near-identical policy docs | **`mmr_*`** (collapse duplicates) |
| enterprise price | a superseded doc carries a plausible wrong price | grounded `synthesis_prompt` |
| password reset | answer phrased with different words than the query | `retrieval_mode` (hybrid/vector) |

A held-out split (warranty, on-call) is graded every round but never optimized against — if
holdout F1 tracks train F1, the gains generalize rather than over-fitting the judge.

## Design notes

- **Three real LLMs, each in a distinct role.** A cheap model **writes** the answer (synthesis;
  its system prompt is the `synthesis_prompt` knob), a strong model **grades** it against the
  gold key (the judge — robust, semantic, not string-matching), and the strongest **optimizes**
  the knobs. Retrieval is a deliberately *simulated* loss surface — its fidelity isn't the point;
  it only needs real knob→behaviour dynamics and to emit a realistic retrieval log.
- **Symptom-level feedback, not a diagnosis.** The optimizer sees only the question, the answer,
  the grader's score, and the retrieval log — never where a correct fact lives or which knob to
  turn. It must infer cause→knob, so it's reasoning, not obeying. There is no rule-based fallback;
  with no brain configured, Argus errors rather than degrading to fixed rules.
- **Staged + response cache.** Each stage is memoized under a hash of only the knobs upstream of
  it, so a cheap downstream change never re-runs the expensive stages — and identical LLM calls
  (synthesis, judge) are reused, keeping reruns cheap and stable.

## Layout

```
argus/
  params.py      knob registry + the broken starting point
  dataset.py     engineered corpus, questions, gold answer key
  pipeline.py    the toy RAG cascade (chunk→embed→retrieve→rerank→mmr→synth)
  cache.py       staged forward-pass memoization
  judge.py       recall / precision / F1 → composite loss
  optimizer.py   the textual-gradient optimizer (LLM-only; no rule-based fallback)
  llm.py         brain backends: anthropic API / local (Ollama) / agent handoff
  loop.py        the trust-region convergence driver
  ui.py          rich console view (loss curve, gradients, before/after)
run.py           entry point
```

## From toy to production

The engine is backend-agnostic. To point it at a real pipeline you swap two adapters: a
*forward pass* (run the pipeline / re-probe a frozen index with candidate params) and a *loss*
(an LLM judge grading answers against a ground-truth key into recall/precision/F1). The knob
taxonomy here mirrors a production RAG + enrichment pipeline 1:1 — chunk sizes, candidate
count, rerank top-k and floor, MMR λ/threshold, per-doc context budget, and the synthesis
prompt — so the same gradients transfer.
