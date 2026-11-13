# Interpretable sparse features in AI-designed protein binders

**Applied Math Seminar (APMA 4903) — Vignesh Karthik**

We take 36,710 computationally designed protein binders, re-express each amino
acid's dense neural representation as a sparse combination of learned features,
and ask which of those features correspond to the physical binding interface.

The project has **three components and one experiment**:

| | component | role |
|---|---|---|
| 1 | **Boltz-Prot-1** | generative design model — produced the binders |
| 2 | **ESM-C 300M** | protein language model — produced the embeddings |
| 3 | **TopK sparse autoencoder** | our contribution — decomposes the embeddings |

**The experiment:** extract features that fire at the binding interface, and
quantify their interface specificity via *enrichment*.

**The finding:** the features are legible — specific sequence motifs docking
onto specific pockets. But those pockets are **not the ones the design
campaigns targeted.**

**The negative result:** the legible features are **not causally steerable.**

---

## 1. Why this problem

Generative models now design proteins that bind a chosen target. You request a
binder, and tens of thousands of candidates come back, each with a predicted 3D
structure and a confidence score. What does *not* come back is any account of
**why** a design is predicted to work.

The standard workaround is to rank by confidence and test the top few
experimentally. That tells you *which* designs work. It never tells you *what
makes* them work — so nothing transfers to the next target.

---

## 2. Component 1 — Boltz-Prot-1 (the design model)

**Target.** Reg3A (regenerating islet-derived protein 3-alpha), signal peptide
removed: 138 residues, chain A in every predicted complex.

> *TODO (Vignesh): why Reg3A was chosen for the original iGEM campaign. Not
> inferable from anything in this repository, and the first thing an audience
> will ask.*

**Binders.** 36,710 de novo designs across 10 campaigns:

| campaign | designs | directed at target residues |
|---|---|---|
| `REG3A_final_25_33` | 18,000 | 25–33 |
| `REG3A_final_45_57` | 18,000 | 45–57 |
| 7 x `probe100_*` | ~710 | various hotspots, plus one unguided |

Binder lengths 50–139 residues (median 65) → **2,387,652 amino acids**, which
is the real sample size: the SAE trains on one data point per amino acid.

Each design carries a predicted complex structure (mmCIF, binder = chain B), a
PAE uncertainty matrix, and Boltz-Prot-1's own metrics. **12,060 designs
(32.9%) score ipTM > 0.7**, the usual "probably a real interface" threshold.

---

## 3. Component 2 — ESM-C 300M (the embedding model)

A **protein language model**: it reads an amino acid sequence and builds a
numerical representation of each residue in context, trained by masking
residues and predicting them from their surroundings.

`biohub/ESMC-300M`: **30 transformer blocks, 960 numbers per residue.**

**The embedding** is that 960-number vector — the model's compressed summary of
one position given its whole sequence. Two properties matter:

- it is **contextual** — the same amino acid at two positions gets different
  vectors
- **depth matters** — early layers encode local properties, deeper layers more
  abstract ones. We read **layer 23 of 30**.

### The key trick: co-attending binder with target

Binding is a property of a *pair* of proteins. A residue that matters when the
target is nearby may be unremarkable in isolation. So we feed both chains:

```
binder  +  |  +  Reg3A target
```

where `|` is ESM-C's **native chain-break token** (vocabulary id 31) — the
model's own trained multi-chain convention, not a linker hack.

Then **we keep only the binder's residue vectors and discard the target's.**
Attention has already let every binder residue see the target, so we get
target-aware representations of binder residues without carrying 138 target
residues per design.

Output: a **2,387,652 x 960** float16 matrix (4.58 GB).

---

## 4. Component 3 — the TopK sparse autoencoder

### The problem it solves

Those 960 numbers are **dense** (nearly all nonzero) and **polysemantic** (no
single coordinate means one thing). Each concept is smeared across many
coordinates; each coordinate participates in many concepts.

The reason is **superposition**: there are far more useful things to know about
a residue than 960, so the model packs thousands of concepts into a
960-dimensional space by giving each its own direction and activating any given
one only rarely. Efficient — and unreadable.

### The construction

Learn an **over-complete dictionary** and force each residue to use only a few
entries:

$$z = \mathrm{TopK}_k\!\left(W_{\mathrm{enc}}^\top x + b_{\mathrm{enc}}\right), \qquad \hat{x} = W_{\mathrm{dec}}^\top z + b_{\mathrm{dec}}$$

minimizing $\|x-\hat{x}\|_2^2$ subject to $\|z\|_0 = k$, with

$$d_{\text{model}} = 960, \qquad d_{\text{hidden}} = 16{,}384, \qquad k = 64$$

Three choices worth defending:

- **Hard $\ell_0$ via TopK, not an $\ell_1$ relaxation.** TopK keeps the $k$
  largest pre-activations and zeroes the rest: exact, dial-able sparsity with
  no shrinkage bias on the survivors. Cost: a non-convex, non-smooth objective
  where gradients flow only through selected coordinates.
- **Over-completeness is the point.** 16,384 directions in a 960-dimensional
  space. Under superposition, recovering concepts one-per-direction *requires* a
  dictionary larger than the ambient dimension.
- **Unit-norm decoder rows.** Without the constraint the model can shrink codes
  and inflate decoder norms — satisfying sparsity in form, not substance. This
  makes training constrained optimization on a product of 16,384 unit spheres.

### Result

| measure | value |
|---|---|
| **held-out FVE** | **0.9687** |
| **dead features** | **0 / 16,384** |
| active features per residue | 64.0 exactly |
| training time | 3 h 02 min, Apple M3 Pro (MPS) |

**64 of 16,384 features reconstruct 96.9% of the variance, with every
dictionary entry in use.** Held-out FVE is uniform across campaigns
(0.94–0.99), so the dictionary is not specialized to one binding mode.

FVE is the standard SAE reconstruction metric, and on its own it is a dial you
turn up by raising $k$. What makes it a *measurement* is the surrounding
protocol — hold out before training, freeze the normalization statistics,
report $k$ and $d_{\text{hidden}}$, name the evaluation distribution. That
protocol, the cross-target replication (0.9687 here against 0.9688–0.9730 on a
different target and corpus), and the limits of the metric are written up in
[`docs/FVE-PROTOCOL.md`](docs/FVE-PROTOCOL.md).

**K-annealing and AuxK, observed working.** Dead features spiked to **2,810 at
epoch 10** — exactly where $k$ finishes annealing 256→64 — then AuxK drove it
to **0**. The designed failure mode and its remedy, both visible in
`results/training_log_layer23_paired.csv`.

---

## 5. The experiment — interface features and enrichment

### Defining the interface

For each of the 12,060 gated designs, measure every binder residue's **minimum
heavy-atom distance to any target atom**. Under 5 Å = a **contact residue**
(standard CAPRI/PISA convention).

> **164,285 contact residues of 771,228 binder residues = 21.3%**, 0 designs
> skipped.

### Enrichment

For feature $f$:

$$E_f \;=\; \frac{P(f \text{ fires} \mid \text{residue is a contact})}{P(f \text{ fires} \mid \text{residue is far})}$$

| $E_f$ | meaning |
|---|---|
| 1.0 | fires equally everywhere — no interface preference |
| 5.0 | fires 5x more at the interface — interface-specific |
| 0.2 | fires 5x *less* — avoids the interface (buried core) |

Across 13,399 supported features: **median 1.009** (the typical feature has no
preference, as expected), **maximum 5.49**.

Enrichment is an **annotation, never a filter.** If several valid binding modes
exist, a feature causal within one mode is diluted against every residue of
every other mode — so gating on enrichment would preferentially discard
mode-specific features, which are the interesting ones.

### A statistical trap worth knowing

Ranking features by their *strongest* interface activation surfaces features
that fire on **two residues in the whole corpus** at high magnitude, while
features with 1,173 residues of support peak around 12. A maximum is an
**extreme-value statistic** — it rewards flukes. Fix: require >= 50 contact
residues of support *before* ranking.

---

## 6. The finding — legible motifs, at the wrong hotspots

### The features are readable

Feature **14364**, hardest-firing interface residues:

```
QVKEKIEE[L]TGIAPEDQ   touches TYR124, VAL128          pLDDT 95.6
DVKKKIEE[L]TGLDPAEQ   touches TYR124, VAL128, ARG129
QVKEKISE[K]TGLDPKYQ   touches TYR124, VAL128
EVKERIEE[L]TGIAPEDQ   touches TYR124, VAL128, ARG129
```

An amphipathic helical run, the firing residue, then an invariant **`TG`**
(glycine being a classic helix breaker) — and **11 of 15** examples dock the
same Tyr124/Val128 pocket. The feature plausibly means *"helix ending in a TG
turn, packed against that pocket."*

Feature **2672**: near-invariant `KDY.A..[D]EL.L..GD`, fires almost only on
Asp/Glu, $E_f = 5.41$.

### But the pockets are not the targeted ones

| campaign | % contacts 25–33 | % 45–57 | % 58–70 | % 112–130 |
|---|---|---|---|---|
| `final_25_33` | **17.9** | 1.6 | 20.4 | 26.2 |
| `final_45_57` | 1.2 | **11.7** | 18.1 | 37.6 |
| `probe100_124_128` | 0.0 | 2.1 | 32.5 | **46.9** |

The guidance **is real** — each campaign is ~11x enriched for its own hotspot
relative to the other's. But **the majority of contacts for every campaign land
on a shared surface around residues 58–70 and 112–130** that designs gravitate
to regardless of instruction.

Across all 430,358 contact pairs, the most-contacted target residues are
**Ser63, Tyr124, Ser61, Ser115, Arg98, Tyr62, Val128, Tyr64.**

So most "interface features" describe that shared sticky epitope, not the
designed hotspot — a quantitative case of a generative model's behaviour
diverging from its specification, visible only by measuring geometry rather
than trusting the instruction.

---

## 7. The negative result — legible is not steerable

If a feature *means* something, can we *use* it? ESM-C is a masked language
model, so its hidden states decode back to amino acids through its own
`lm_head` — a causal test needing no extra model. Add a feature's direction at
layer 23 and see whether the model produces that feature's pattern.

**It works mechanically.** Injecting feature 14364's direction drives its own
activation 15.9 → 143 (9x) while leaving other features near baseline.

**It is not feature-specific.** The decisive test — inject all 32 candidate
directions at one site and ask where the one that *actually fires there* ranks:

| | |
|---|---|
| chance expectation | 16.5 of 32 |
| **observed mean rank** | **15.72** (Wilcoxon p = 0.41) |
| ranked first | **3 of 96 — chance is exactly 3.0** |

The correct feature is no better than the other 31 at producing its own motif.

A real dictionary direction *does* beat matched-norm random noise highly
significantly (p = 3.3e-6, n = 270, pre-registered on held-out features) — but
that is explained by **on-manifold geometry**: real directions perturb the
representation less destructively than random vectors. Nothing about identity.

**Scope of the claim.** Tested: decoder-direction injection, layers 23–29,
constant and annealed schedules, alpha from 2 to 150, masked-span and
synthetic-canvas readouts, with cross-injection as the specificity criterion.
Untested: encoder directions, SAE-space optimization during generation,
non-language-model readouts. A result about this intervention, not a proof that
SAE steering is impossible.

### Why the controls were load-bearing

Six measurement artifacts were found during this work. **Five of six initially
pointed toward "steering works."**

| artifact | direction |
|---|---|
| stale `hidden_states` under a modifying hook | false null |
| motif context left in the input | false null |
| cherry-picked alpha on a single case | **false positive** |
| clamp magnitude correlated with being already-active (rho = 0.879) | **false positive** |
| homopolymer collapse x feature-correlated composition | **false positive** |
| poly-A canvas: rank tracked canvas alanine content (rho = -0.947) | **false positive** |

When a readout is subtly broken, the easiest way to get a number resembling
"the right feature wins" is for the intervention to damage something in a
feature-correlated way. This is the most transferable lesson in the project.

---

## 8. What this does and does not enable

**Does:**
- **Target selection.** Designs overwhelmingly prefer the 58–70 / 112–130
  surface. Targeting it deliberately is a testable prediction: success rates
  should beat the original campaigns' 28.3% and 37.6% ipTM > 0.7.
- **Structurally-grounded filtering.** "Does this design carry the 14364 motif
  docking the Tyr124 pocket?" — a criterion ipTM cannot express.
- **A bindability map** of the target surface.

**Does not:** provide a generative control knob. That was the steering
hypothesis, and it failed.

So: **a diagnostic and analytical tool, not yet a generative one.**

---

## 9. Repository

```
00_data/build_manifest.py              design table from the Boltz-Prot-1 output tree
01_embed/embed_esmc_paired.py          co-attended ESM-C embedding
        layer_sweep.py                 layer selection
02_prepare_data/data.py                train/held-out split, centering + scaling
03_train/sae_model.py                  the SAE: TopK, AuxK, sphere constraint
        train.py                       training loop, k-annealing, resumable
05_interpret/interface_activations.py  interface geometry + per-feature activations
             label_features.py         LLM labeling from the evidence CSVs
06_steering/cross_injection_test.py    the specificity test (section 7)
             steer_recover_motif.py    motif-recovery readout
             steer_v2.py               calibrated clamp, multi-layer schedules
             steer_canvas.py           synthetic-canvas profile matching
sentinel/cli.py                        console entry point, dispatches to stages
run_overnight.sh                       one-command pipeline driver
REPORT.md                              full results, timings, every artifact
docs/STANDARDS.md                      formats and conventions the pipeline uses
docs/FVE-PROTOCOL.md                   how to measure a latent representation, reproducibly
results/                               summary CSVs (small, committed)
```

The numbered stage directories are named for the order the pipeline runs in.
Those names are not importable Python identifiers, so the `sentinel` package
holds the command-line interface and dispatches to each stage, leaving the
scientific code in place and unchanged.

**Not committed** (large, regenerable — rebuild commands in `.gitignore`): the
4.58 GB embedding matrix, 362 MB checkpoints, 17 MB design table.

## 10. Install

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e .
```

That installs the dependencies and puts a `sentinel` command on your path.
Install from a checkout rather than a wheel: the stage scripts live in the
repository, and the command runs them where they are.

```bash
sentinel --help              # the stage list
sentinel train --help        # that stage's own options
```

## 11. Reproducing

```bash
sentinel manifest \
    --campaigns-dir <your Boltz-Prot-1 results tree> \
    --glob 'REG3A*' --output results/manifest_combined.csv

caffeinate -i ./run_overnight.sh    # layer sweep -> embedding -> SAE, ~3.5 h on an M3 Pro
```

Or drive the stages one at a time:

| | command |
|---|---|
| design table | `sentinel manifest --campaigns-dir ... --output ...` |
| layer selection | `sentinel layer-sweep` |
| co-attended embedding | `sentinel embed` |
| train the SAE | `sentinel train --data-dir ... --output-dir ...` |
| interface enrichment | `sentinel interface` |
| the specificity test (§7) | `sentinel cross-injection` |

Arguments after the stage name pass through untouched, so anything the
underlying script accepts, the command accepts.

Needs ~12 GB free disk and a GPU (CUDA or Apple Silicon). Every stage is
resumable: a failed run restarts at the stage that failed.

## 12. Standards and formats

The pipeline reads and writes community formats at every boundary — mmCIF, PDB
and FASTA in, CSV, JSON and NPY out — and adopts existing conventions
rather than local ones: the CAPRI/PISA 5 Å heavy-atom contact definition, ESM-C's
native chain-break token for multi-chain input, and the AlphaFold-lineage
confidence metrics with their standard meanings. See
[`docs/STANDARDS.md`](docs/STANDARDS.md), which also states plainly where SBOL
does and does not apply to work at this level.

[`docs/FVE-PROTOCOL.md`](docs/FVE-PROTOCOL.md) covers the measurement side: a
label-free, reproducible way to evaluate a latent protein representation, what
has to be reported alongside the number for it to mean anything, and what the
metric does *not* tell you.

## 13. Licence

MIT. See [`LICENSE`](LICENSE).

## 14. References

- Gao, L., la Tour, T. D., Tillman, H., Goh, G., Troll, R., Radford, A.,
  Sutskever, I., Leike, J., & Wu, J. (2024). *Scaling and evaluating sparse
  autoencoders.* arXiv:2406.04093. — the TopK/AuxK recipe and dead-feature
  analysis this SAE implements.
- Bricken, T., Templeton, A., Batson, J., Chen, B., Jermyn, A., et al. (2023).
  *Towards monosemanticity: decomposing language models with dictionary
  learning.* Transformer Circuits Thread. — superposition, and the
  max-activating-examples method for reading features.

**Citations for Boltz-Prot-1, ESM-C, and ipSAE must be verified against the
primary sources before use** — see `PRESENTATION_OUTLINE.md`, References.

## 15. Generative AI disclosure

The pipeline code, analysis scripts, and drafts of this documentation were
written with substantial assistance from **Claude (Anthropic), via Claude
Code.** Experimental design decisions, the steering hypotheses, and the
critical catch that an unmasked motif context invalidated the first steering
result were the author's. All results were produced by code in this repository
and are reproducible from it.
