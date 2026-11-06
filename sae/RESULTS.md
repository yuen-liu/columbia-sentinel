# Vilip1 SAE results log

Running record of every training/benchmark run in the vilip1 SAE roadmap,
for pulling numbers into the New England Comp Bio conference abstract
later (https://newenglandcompbio.org/) without re-digging through chat
history. Update this whenever a new run/benchmark lands.

All runs: ESM-C layer 23, per-token TopK SAE (`sae/03_train/sae_model.py`),
`k=64`, `auxk=256`, `batch_size=4096`, `epochs=50`, `lr=4e-4`. "Natural FVE"
= `natural_binders_qualitative` pooled FVE from `benchmark.py` (ours vs.
Biohub's official ESMC-300M-sae-k64-codebook16384, both compared against
the same shared baseline). Biohub's own natural FVE is consistently
~0.70 regardless of our checkpoint (~0.6972-0.7036 across benchmark runs).
**Read the note directly below before quoting any "Natural FVE" number.**

## IMPORTANT (2026-09-26): the "natural binder" eval set is mostly NOT binders

The 82-sequence natural eval set (`EVAL_ONLY_SOURCES` in
`sae/02_prepare_data/data.py`) is two different things, and only one of
them is real binders:

| Source | n | Actually binds Vilip-1? | Trained on? |
|---|---|---|---|
| `natural_binders` (hand-picked, `data/VILIP-1_natural_binders.fasta`) | 13 | **yes** | never, in any run (`ALWAYS_EVAL_ONLY_SOURCES`) |
| `binder_dataset_vilip1` (STRING-derived, from `data/binder_dataset_raw.csv`) | 69 | **no** -- STRING `functional_or_physical_association` only, not validated binding | eval-only by default; 55/69 folded into training in run4/run6 |

Follow-up curation (reported 2026-09-26) established that the 69
STRING-derived sequences are not Vilip-1 binders. They are real human proteins
co-associated with VSNL1 in STRING (confidence 0.40-0.885), which is a
functional-association score, not evidence of physical binding. Only the
13 hand-picked UniProt sequences bind.

Consequences, in order of how much they matter:

1. **Every "Natural FVE" number below is dominated by non-binders.** The
   69 are 35,869 of 41,231 eval residues (87%) in the runs that hold out
   all 82 (run1/run2/run3/run5), since FVE is residue-weighted. Read that
   column as *reconstruction of natural human protein sequence*, NOT
   *reconstruction of Vilip-1 binders*.
2. **The 55 sequences mixed into training in run4/run6/run_paired (every
   `--natural-train-frac 0.8` run) were all non-binders.** Finding #2 below is still a real effect, but the
   mechanism is "mix in natural proteome sequence", not "mix in binders"
   -- see the rewritten finding.
3. **No binder-specific training leak.** Since `natural_binders` (the 13)
   is `ALWAYS_EVAL_ONLY`, no run ever trained on a real binder. A
   13-only FVE is therefore apples-to-apples across *every* row in the
   table, including run4/run6 -- which the pooled 82-vs-~27 column never
   was. See "Re-benchmark plan" below.
4. The NECB abstract (`necb_conference/necb_abstract.tex`) already says
   "natural-protein sequences" and "natural-protein reconstruction
   fidelity" rather than "binders" for the 0.39->0.60 result, so that
   claim holds as written. Do not reword it to say "binders".

The 13 real binders: O94919 (ENDOD1), P04155 (TFF1), Q03692 (COL10A1),
Q9BXJ5 (C1QTNF2), O43610 (SPRY3), Q7Z698 (SPRED2), Q99732 (LITAF),
Q86UW9 (DTX2), P54253 (ATXN1), Q86TD4 (SRL), Q7Z699 (SPRED1), Q96AQ9
(FAM131C), Q9BXU2 (TEX13B).

Counting note: `binder_dataset_raw.csv` has 70 VSNL1 rows, not 69.
Q96AQ9 (FAM131C) appears in both source files and was deduped out of the
STRING set, so it is counted in the 13, not the 69.

"Natural FVE (pooled)" is the number as originally reported: pooled over
whichever of the 82 were held out for that run, so 87%-by-residue
non-binder. "Binder-only FVE (13)" is the comparable-across-all-rows
number from the same `benchmark_summary.csv`'s `source=natural_binders`
row -- only run1's is in the repo; the rest need the re-benchmark below.

| Run | Training data | d_hidden | k-anneal | natural-train-frac | Design FVE (val, pooled) | Natural FVE (pooled, mostly non-binder) | **Binder-only FVE (13)** | Dead (natural eval) | Eval n (natural) = binders + non-binders |
|---|---|---|---|---|---|---|---|---|---|
| run1 (baseline, pre-session) | vilip1 65k | 4096 | no | 0 | 0.9656 | 0.3851 | **0.4404** | 1/4096 | 82 = 13 + 69 |
| run2_kanneal16384 (norm bug) | vilip1 65k | 16384 | 128->64 | 0 | 0.9715 | 0.4278 | not computed | 69/16384 | 82 = 13 + 69 |
| run3_normfix | vilip1 65k | 16384 | 128->64 | 0 | 0.9730 | 0.4156 | not computed | 65/16384 | 82 = 13 + 69 |
| run4_natural_mix | vilip1 65k | 16384 | 128->64 | 0.8 | 0.9660 | **0.5987** | not computed | 1017/16384 | ~27 = 13 + ~14 |
| run5_multi_target | vilip1+uchl1+fabp7+reg3a (141k) | 16384 | 128->64 | 0 | 0.9723 | 0.4495 | not computed | 25/16384 | 82 = 13 + 69 |
| run6_combined | vilip1+uchl1+fabp7+reg3a (141k) | 16384 | 128->64 | 0.8 | 0.9686 | 0.5973 | not computed | 683/16384 | ~27 = 13 + ~14 |
| run_paired (co-attention) | vilip1 65k, binder+target via ESM-C's native `\|` chain-break | 16384 | 128->64 | 0.8 | 0.9688 | not benchmarked vs. biohub (input-distribution mismatch, see below) | -- | -- | -- |
| Andrew's own run (external, not reproduced) | vilip1 65k + other-target mix (exact ratio unconfirmed) | 16384 | 128->64 (his own schedule) | unknown | -- | 0.5157 (reported) | not computed (his own eval set, "85 sequences" -- composition unverified) | -- | 82 (assumed) |

**Key findings, in order discovered:**
1. Decoder-norm axis bug (`w_dec` normalized per-output-dim instead of
   per-feature) found + fixed comparing against Andrew's pushed notebook --
   real bug, but retested (run3 vs run2) showed it did NOT explain the FVE
   gap (0.4156 vs 0.4278, a wash). Kept anyway -- mathematically correct.
2. **Natural-*sequence* mixing (`--natural-train-frac`) is the single
   biggest lever found** -- run4 alone (0.5987) beats Andrew's reported
   0.5157. Two caveats, the second added 2026-09-26:
   - run4/run6's natural FVE is measured on a smaller eval set (~27
     held-out, vs. 82 for every other row) since 55/69 STRING-derived
     sequences got folded into training -- not perfectly apples-to-apples,
     but the eval-set-size doesn't explain a jump this large.
   - **The 55 sequences folded into training are NOT Vilip-1 binders**
     (see the note at the top). So what this lever actually shows is that
     mixing *real human proteome sequence* into a mostly-synthetic design
     corpus improves reconstruction of held-out real protein sequence.
     That is still the biggest lever measured, and it is still a
     legitimate generalization result -- but it is not evidence about
     binders, and the eval set it is measured on is itself ~half
     non-binder (13 binders + ~14 non-binders). Describe it as
     natural-protein mixing, never as natural-binder mixing.
3. Multi-target diversity (run5 vs run3: 0.4495 vs 0.4156) helps on its
   own, but does NOT meaningfully stack with natural-sequence mixing
   (run6 vs run4: 0.5973 vs 0.5987, statistically a wash). Real
   side-effect: run6 has far fewer dead features than run4 (683 vs. 1017)
   despite the tied FVE -- larger/more diverse corpus keeps more of the
   dictionary alive.
4. Andrew's reported 0.5157 apparently came from a different run (not
   pushed to this repo) that also mixed in other-target binders --
   unconfirmed exact ratio/method.
5. `run_paired` (co-attention/binder+target via ESM-C's native chain-break
   token, confirmed present in `ESMCTokenizer().get_vocab()` as `|`) is
   NOT directly comparable to the others via `benchmark.py` -- that script
   re-runs ESM-C on the plain (binder-alone) sequence text, which doesn't
   match what run_paired's SAE was trained on (target-influenced
   activations). Its held-out design FVE (0.9688) is in line with every
   other run, so reconstruction quality isn't degraded. The actual
   co-attention question ("does it predict binding affinity better") is
   answered by the `feature_analysis.py` probe comparison, not FVE --
   see `sae/results/run4/` vs `sae/results/paired/` results once both are in.

**Decision (2026-08-13): `run4_natural_mix` chosen as the primary
checkpoint for feature analysis** (Vignesh's `feature_analysis.py`) --
tied with run6 on the metric that matters, simpler/cheaper, and analysis
tooling was already built around it.

**2026-08-14, ~1am: linear probe deprioritized for time.** Even after
parallelizing `LassoCV` (`n_jobs=-1`, commit `aaa6b3c`) across 48 cores, it
was still too slow to finish before the submission deadline allowed for --
switched to just the qualitative pass (`feature_analysis.py` without
`--probe-metrics-csv`: `feature_stats.csv` + `feature_top_examples.csv`,
under a minute). This means: no `cv_r2` numbers, and **the co-attention
question (does `run_paired` predict binding affinity better than
binder-alone) is NOT answered** -- that was the probe's job specifically,
nothing else in the pipeline tests it. If there's time before the actual
submission, revisit: profile why the parallel version was still slow
(worth checking whether joblib's process-based parallelism was spending
most of its time re-serializing the ~850MB pooled-code matrix to workers
rather than actually fitting), or just accept a much smaller/faster probe
(fewer alphas, fewer CV folds -- both hardcoded in `feature_analysis.py`,
not exposed via CLI) rather than skipping it entirely.

## Feature analysis results

**Qualitative pass done (2026-08-14), probe not run (see note above).**
`sae/results/run4/` and `sae/results/paired/` (scp'd down from Waluigi's
`~/analysis/`) each have `feature_stats.csv` + `feature_top_examples.csv`
from `feature_analysis.py` without `--probe-metrics-csv`.

- Both dictionaries healthy: 18/16384 dead (run4), 31/16384 dead (paired),
  both <0.2%. Similar density distributions (median ~0.0016-0.0018) and
  activation-magnitude distributions between the two.
- **Cross-dictionary consistency check**: took the top 30 highest-density
  (most generic) features in each of run4/run_paired, looked at each
  one's single hardest-firing residue (protein id + position). 6 of ~28-30
  landed on the EXACT same (protein, position) in both -- independently
  trained SAEs, different training data (binder-alone vs. binder+target),
  converging on the same residues by chance across millions of candidate
  residues would be essentially impossible. Real, non-artifactual signal.
  Several of the overlapping hits are on real natural-protein ids
  (Q9ULV0, Q9ULU8, Q03692, P49810), not just synthetic designs -- the
  shared signal isn't a design-campaign-specific artifact. **2026-09-26:
  of those four, only Q03692 (COL10A1) is a real Vilip-1 binder; Q9ULV0
  (MYO5B), Q9ULU8 (CADPS) and P49810 (PSEN2) are from the 69 STRING
  non-binders.** This finding is unaffected -- it is about two
  independently trained dictionaries converging on the same residues of
  the same real human proteins, which does not depend on those proteins
  binding anything. But do not restate it as "converges on binder
  residues"; the correct claim is "converges on real-protein residues".
- Single most generic feature in BOTH dictionaries peaks at the same
  residue: `binder_dataset_vilip1`, position 1715, context
  `MQLRYN[I]SQLEEW` (run4 feature 14247, density=0.7646; paired feature
  582, density=0.7631).
- **InterPro annotations (2026-08-14)**: ran `fetch_interpro.py` on an
  11-feature hand-picked subset per checkpoint (3 highest-density +
  3 rare-but-strong, per `FEATURE_LABELING_SETUP.md`'s own pilot recipe,
  plus one representative feature per cross-dictionary overlap hotspot
  above). Real domain/family coverage, surprisingly high for a mostly-de-
  novo-design corpus (expected sparse per the setup doc): run4 131/165
  examples annotated (79.4%), run_paired 136/165 (82.4%) -- makes sense in
  hindsight, since this feature subset specifically targets features tied
  to real natural-protein examples, not synthetic designs. Notable
  cross-checkpoint pattern: run4's feature 2214 fires on 9 consecutive
  real-protein positions (365-375 in evalbinder_P36222/P36222), all in
  `GH_hydrolase_sf (IPR017853)`; run_paired's feature 3896 shows the same
  coherent-motif pattern independently -- 5 consecutive positions
  (370-374 in evalbinder_P49810) all in `Peptidase_A22A (IPR001108)`.
  (2026-09-26: both P36222/CHI3L1 and P49810/PSEN2 are from the 69
  non-binders. The `evalbinder_` id prefix is just the manifest naming
  convention for `binder_dataset_vilip1` rows -- it does not mean the
  sequence was ever shown to bind. The domain-coherence result stands on
  its own; it's a statement about InterPro domains, not about binding.)
  Full results + methodology in
  `sae/notebooks/sae_feature_analysis_run4_vs_paired.ipynb`.
- **LLM auto-labeling done** (`label_features.py`, Claude Haiku 4.5 via
  Message Batches API, same 11-feature subset per checkpoint, InterPro
  evidence included). Well-calibrated: correctly returns "No clear
  pattern" when InterPro evidence is genuinely scattered (run4 feature
  1707; run_paired features 5997/9725) rather than forcing a story, and
  gives specific, confident labels when evidence is strong (run4 feature
  2214: "all 15 examples GH_hydrolase_sf"; feature 12918: tubulin/FtsZ
  GTPase domain + a specific recurring motif `TGLQG[F]L`).
- **Checked the shared-`--seed 0` confound on the cross-dictionary
  consistency finding**: both run4 and run_paired used the same default
  seed (identical weight init), so checked whether the SAME feature
  *index* is the hotspot representative in both dictionaries more than
  chance would predict. Only 1/6 hotspots share an index; the other 5 are
  detected by genuinely different features per dictionary -- the
  consistency finding is NOT mainly a seed artifact, a stronger result
  than assumed going in.
- **Probe (`cv_r2`) not run** -- see the "linear probe deprioritized"
  note above. Co-attention question still open.
- Full methodology + reproducible cells:
  `sae/notebooks/sae_feature_analysis_run4_vs_paired.ipynb`.

## Re-benchmark plan: binder-only (13-sequence) FVE

**Why:** the pooled "Natural FVE" column mixes 13 real binders with 69
non-binders, at 87%-by-residue non-binder, and its eval set changes size
between rows (82 vs ~27). The `source=natural_binders` row of the same
output is 13 real binders in every run, never trained on in any run --
the only natural-eval number that is comparable across the whole table.

**Good news: no code change needed.** `benchmark.py:239-241` already
accumulates per-source stats and writes a `source` column, so every
existing `benchmark_summary.csv` on Waluigi *already contains* the
binder-only row. run1's is in the repo:

| source | n_residues | ours FVE | biohub FVE | ours dead |
|---|---|---|---|---|
| `__pooled__` (82) | 41231 | 0.3851 | 0.7036 | 1/4096 |
| `binder_dataset_vilip1` (69 non-binders) | 35869 | 0.3795 | 0.7026 | 1/4096 |
| `natural_binders` (13 real binders) | 5362 | **0.4404** | 0.7134 | 50/4096 |

### Full biohub comparison, run1 (`benchmark_results_65k/benchmark_summary.csv`)

The natural rows above are only half of that file. The design split was
never recorded here, and it is where the comparison actually favours us --
the tradeoff is directional, not a uniform loss:

| split | source | n_residues | ours FVE | biohub FVE | ours dead | biohub dead |
|---|---|---|---|---|---|---|
| held_out_designs | `__pooled__` | 696005 | **0.9656** | 0.6805 | 0/4096 | 921/16384 |
| held_out_designs | `vilip1_full20k` | 131804 | **0.9067** | 0.8149 | 1/4096 | 3023/16384 |
| held_out_designs | `composite_hotspot` | 62922 | **0.9729** | 0.6673 | 3/4096 | 3997/16384 |
| held_out_designs | `composite_hotspot_20260728` | 501279 | **0.9711** | 0.6673 | 0/4096 | 1327/16384 |
| natural_binders_qualitative | `__pooled__` (82) | 41231 | 0.3851 | **0.7036** | 1/4096 | 662/16384 |
| natural_binders_qualitative | `binder_dataset_vilip1` (69) | 35869 | 0.3795 | **0.7026** | 1/4096 | 794/16384 |
| natural_binders_qualitative | `natural_binders` (13) | 5362 | 0.4404 | **0.7134** | 50/4096 | 6522/16384 |

Two things worth quoting from this that are not in the table above:

1. **On the design distribution we beat biohub by a wide margin** (0.9656
   vs. 0.6805 pooled; +0.29 FVE), and still win on `vilip1_full20k`
   (0.9067 vs. 0.8149) where biohub does best. The natural-FVE deficit is
   a generalization gap on out-of-distribution sequence, not a worse
   autoencoder.
2. **Biohub's dead-feature count is highly input-dependent** -- 662/16384
   on pooled naturals but 6522/16384 on the 13 real binders, and
   3997/16384 on `composite_hotspot`. Ours stays at 0-3/4096 on designs
   and 50/4096 on the binders. So "dead" is not a fixed property of a
   dictionary; quote it with the eval source attached or it means nothing.

**Still missing (Waluigi only).** Biohub numbers for run2-run6 are not in
the repo -- only run1's `benchmark_summary.csv` is here. The header's
"~0.6972-0.7036 across benchmark runs" range is therefore not reproducible
from anything committed. Pull the remaining `benchmark_summary.csv` files
per Step 1 below and extend this table rather than re-deriving the range.

There is a second, earlier `benchmark_summary.csv` in
`vilip1_layer23_sae_outputs/benchmark_results/` (ours 0.4196 / biohub
0.7016 on the 13, design pooled 0.9412 / 0.7159). Its eval set is the 13
binders alone, so it predates the 82-sequence setup and is **not** run1 --
provenance unconfirmed, do not merge the two.

So step 1 is just to pull the CSVs that already exist, and only re-run
`benchmark.py` for any run whose CSV was lost.

**Step 1 -- check what's already on Waluigi (cheap, do this first):**

```bash
ls -la ~/notebooks/sae_training/benchmark_results_65k*/benchmark_summary.csv
for f in ~/notebooks/sae_training/benchmark_results_65k*/benchmark_summary.csv; do
  echo "=== $f"
  grep -E 'source|natural_binders' "$f"
done
```

Anything that prints a `natural_binders` row is already done -- no GPU
time needed, just scp it down.

**Step 2 -- re-run only for missing runs.** Per-run, substituting the
checkpoint dir (`run2_kanneal16384`, `run3_normfix`, `run4_natural_mix`,
`run5_multi_target`, `run6_combined`):

```bash
source /tmp/esm_verify_venv/bin/activate   # NOT esmfold2_venv -- meta-device crash
cd ~/notebooks/sae_training
RUN=run4_natural_mix
python benchmark.py \
    --checkpoint checkpoints_65k/$RUN/best.pt \
    --manifest manifest_combined.csv \
    --output-dir benchmark_results_65k_$RUN \
    --smoke-test
python benchmark.py \
    --checkpoint checkpoints_65k/$RUN/best.pt \
    --manifest manifest_combined.csv \
    --output-dir benchmark_results_65k_$RUN
grep -E 'source|natural_binders' benchmark_results_65k_$RUN/benchmark_summary.csv
```

Notes:
- run5/run6 were trained on the combined multi-target pool, so they need
  the combined manifest (the `combine_datasets.py` output used at train
  time), not vilip1's `manifest_combined.csv`. The natural rows are
  identical either way, but the `held_out_designs` split needs the
  manifest that actually contains their val ids.
- The 13 are only 5,362 residues, so this is fast -- the cost is
  re-running ESM-C + Biohub's SAE hook over the other sequences in the
  same pass. If GPU time is tight, the binder-only number alone can be
  had from a manifest filtered to `source == natural_binders`.
- Expect dead-feature counts on the 13-only row to look alarming
  (run1: 50/4096 ours, 6522/16384 biohub) -- that is the small-eval-set
  artifact already noted for run4, not a regression. Compare dead counts
  only between rows with the same eval set.

**Step 3 -- once the numbers are in:** fill the "Binder-only FVE (13)"
column above, and check whether the run3 -> run4 jump (0.4156 -> 0.5987
pooled) still holds on real binders only. If it shrinks a lot, the
0.39 -> 0.60 claim in the NECB abstract is specifically about natural
protein sequence, not binders, and the abstract's current wording
("natural-protein reconstruction fidelity") is already the right claim --
don't strengthen it. If it holds, that's a stronger result than what's
currently written.

## Steering results: feature injection (2026-09-26)

First real run of `sae/06_steer/inject_feature.py` -- injecting each of 11
candidate features' decoder direction back into `run4_natural_mix` at layer
23 and checking (a) SAE re-emergence and (b) MLM-head amino-acid preference
shift. Candidates and their interface positions come from Vignesh's
`extract_interface_features.py`, run in 3 batches (full20k, ch_20260724,
ch_20260728; 3A/8A contact/shell cutoffs) and combined via the new
`merge_interface_yamls.py` into 64,998 designs, 0 conflicts. `--smoke-test`
caught a real, repo-wide bug before any of this ran: `biohub/ESMC-300M`'s
HF `main` revision silently moved to an incompatible checkpoint format,
causing `from_pretrained` to random-init the entire model with no hard
error (see `sae/README.md`'s "Steering" section) -- now pinned to the
working revision everywhere it's loaded.

**Run**: 11 features x 20 randomly-sampled designs each x alpha in
{0.5, 1.0, 2.0} x mean_activation_when_active -> 7,302 rows,
`injection_scaled.csv` (not committed, see `.gitignore`).

**Dose-response confirms a real causal effect, not noise** -- both the
mean SAE code shift and the sequence-level effect rate scale almost
exactly linearly with alpha:

| alpha_multiplier | n | re_emerged rate | aa_argmax_changed rate | mean code_delta |
|---|---|---|---|---|
| 0.5 | 2434 | 98.8% | 2.1% | 2.81 |
| 1.0 | 2434 | 100.0% | 4.0% | 5.61 |
| 2.0 | 2434 | 100.0% | 7.1% | 11.22 |

> **SUPERSEDED (2026-09-27) -- do not quote this table as evidence of a
> causal effect.** This run had no null condition. With norm-matched
> controls added, ~2/3 of the `aa_argmax_changed` dose-response is
> reproduced by a random direction of the same norm; `re_emerged` is
> near-tautological (98.5% of injected sites were already active before
> injection); and `code_delta` linearity is forced by the geometry of
> injecting `w_dec[f]` and then encoding. See "Controlled feature
> injection" at the end of this file.

**Per-feature results**, cross-referenced against `sae/results/run4/feature_labels.csv`'s
LLM-drafted labels, sorted by how often steering actually changed the
model's top amino-acid pick (`aa_argmax_changed`) -- the stricter,
sequence-level bar, vs. `re_emerged` (SAE-level only, 99.6% overall and not
very discriminating):

| Feature | n candidate designs (of 64,998) | aa_argmax_changed rate | Qualifying rows (re_emerged & changed) | Label (`feature_labels.csv`) |
|---|---|---|---|---|
| 6869 | 74 | **27.3%** | 16 | Glycine immediately preceding lysine (G-K motif); P-loop NTPase/GTPase nucleotide-binding domains |
| 2214 | 19 | 11.7% | 7 | All examples in GH_hydrolase_sf (Glycoside Hydrolase superfamily) |
| 12918 | 97 | 6.9% | 5 | Phenylalanine in `TGLQG[F]L` motif; Tubulin/FtsZ GTPase domain |
| 12588 | 18 | 6.7% | 4 | MUN domain (IPR010439) |
| 14247 | 46,995 | 6.2% | **173** | Hydrophobic residues (I/L/F), no dominant InterPro domain |
| 6073 | 273 | 4.0% | 3 | Leucine preceded by acidic residues (D/E) |
| 10586 | 41 | 3.8% | 3 | Hydrophobic residues (G/L/F/I/W); secondary ionotropic glutamate receptor association |
| 11326 | 64,997 | 3.1% | 35 | Basic residues (R/K) + Pro, before acidic/Pro-rich motifs |
| 233 | 46,327 | 2.9% | 65 | Basic/acidic residues after an "LSE"/"LSEE" motif |
| 1707 | 50,765 | **1.1%** | 7 | **"No clear pattern"** -- highly diverse activating residues, scattered InterPro |
| 4657 | 728 | 0.0% | **0** | Serine after acidic residues; RTN1-4/CAPS family |

Overall: 318 qualifying rows, 75 unique designs, 178 unique
(design, position) mutation sites.

**Findings:**
1. **Specificity correlates with effect strength.** The rarest features
   (6869: 74/64,998 designs; 2214: 19; 12918: 97; 12588: 18) all show
   markedly higher `aa_argmax_changed` rates (6.7-27.3%) than the broad,
   near-universal features (11326: fires in 64,997/64,998 designs, 233:
   46,327, 1707: 50,765) -- a feature that only fires in a handful of
   designs and reliably shifts the model's sequence preference when
   pushed is a much sharper causal signal than one firing almost
   everywhere.
2. **Broad features aren't automatically noise, though** -- `14247`
   (hydrophobic residues, fires in 46,995/64,998 designs) has a below-
   average per-design rate (6.2%) but the largest absolute number of
   qualifying hits (173) by sheer volume, and its label matches a
   biologically sensible, common motif (hydrophobic packing) rather than
   nothing. Treat "broad" and "generic/meaningless" as separate axes.
3. **The weakest feature's own LLM label agrees with the weak effect**:
   `1707` has both the lowest `aa_argmax_changed` rate (1.1%) and a label
   of "No clear pattern... no consistent functional or structural motif"
   -- two independent signals (causal steering effect, sequence-context
   interpretation) converging on "this one is probably not real." Good
   candidate to deprioritize.
4. **SAE-level and sequence-level effects can dissociate**: `4657` shows
   100% re-emergence (the SAE code reliably comes back after injection)
   but **zero** qualifying rows -- injecting it changes the internal
   representation the SAE reads but never once flips the MLM head's top
   amino-acid pick at any alpha tested (up to 2x mean activation). Worth
   trying a larger alpha before concluding it's sequence-inert.
5. **InterPro domain labels for the strongest hits (6869: P-loop
   NTPase/GTPase; 2214: Glycoside Hydrolase; 12918: Tubulin/FtsZ GTPase)
   are from protein families with no known relationship to VSNL1/vilip1's
   actual biology** (an EF-hand calcium sensor). These are very likely
   coincidental InterPro matches against unrelated natural reference
   sequences in the training corpus (see the InterPro coverage caveat in
   "Feature analysis results" above), not evidence that the *design
   generator* is doing anything related to nucleotide-binding or
   hydrolase activity. Treat these as "real, causal, interpretable
   sequence motifs" but not yet "confirmed biologically meaningful for
   this target" -- that call needs Vignesh's energy profiling (H-bonds,
   Amber/Rosetta) as an independent check, not just steering + labels.
6. Small residual noise floor, already understood and not a new issue:
   48/7,302 rows (0.66%) had `pre_code_pool`/`pre_code_live` disagree
   beyond the 10% tolerance -- consistent with the pure-PyTorch
   attention/LayerNorm fallback kernels' printed numerical-difference
   warning (no `transformer_engine`/`xformers` installed in
   `esm_verify_venv`) occasionally flipping a borderline feature across
   the SAE's hard TopK cutoff. Confirmed on 2 earlier spot-checked rows:
   discrete 0.0 vs. small-nonzero flips, not gradual drift. Does not
   affect `code_delta`/`re_emerged` (computed entirely from same-methodology
   live forward passes, never from the pool baseline).

## Energy profiling: Schrodinger Prime interaction energy (2026-09-27)

Closes the gap above. For the same 477 designs sampled for steering
(`sae/07_energy/select_energy_sample.py`, up to 50/feature), computed an
approximate MM **interaction energy** (not full binding free energy --
single Prime minimization of the interface region, default 5A cutoff
around the binder, then single-point energy of the isolated target/binder
extracted from that same geometry; `dE_interaction = E_complex - E_target
- E_binder`) plus interface H-bond counts
(`schrodinger.structutils.analyze.hbond`), via `sae/07_energy/profile_binding_energy.py`
+ `run_energy_profiling.bash` (SLURM array, Gates). All 477/477 designs
succeeded, ~45s-3min/design. Real numbers: `dE_interaction` ranges -475 to
-28 kcal/mol (median -184, all favorable, none repulsive), correlates with
H-bond count (r=-0.68, more H-bonds = more favorable, as expected).
Combined output: `sae/results/run4/energy_profile_combined.csv` (2,111
rows), one row per (design, feature) pair.

**Follow-up finding**: activation *strength* (not just presence) tracks
binding quality for the near-universal features -- `233`'s activation
correlates with `dE_interaction` at r=-0.36 (p=5.6e-15, n=444; more
negative dE = more favorable, so stronger firing = better binding) and
with H-bond count at r=0.15 (p=0.0013). `14247`/`11326` show the same
direction, weaker. This reframes `233` from "boring baseline, fires
everywhere" to "fires everywhere, but firing strength is a real,
statistically robust binding-quality signal." The rare features
(`12588`/`6073`/`10586`/`6869`/`2214`/`12918`) show no significant
activation-vs-energy correlation, but likely underpowered (n=18-57 vs.
`233`'s 444) rather than genuinely null -- worth a larger sample before
concluding they lack a dose-dependent signal.

**Checked whether this dose-dependence also shows up on the steering
side, using existing `injection_scaled.csv` data (no new run needed)**:
for feature 233, correlating each design's baseline activation strength
(`design_max_activation`) against the steering outcome. Much weaker than
the energy-side result -- `code_delta` (does re-injection re-activate the
feature) shows ~zero correlation (r=0.000, p=0.99), and `aa_argmax_changed`
(does steering flip the model's amino-acid pick -- the practically
important outcome) shows no significant difference in baseline activation
between designs where it flipped vs. didn't (p=0.62). Only
`native_logit_shift` shows a small significant effect (r=-0.13,
p=1.1e-9, n=2208 rows / 20 unique designs -- all already high-activation
"top" picks from the original sampling, limiting dynamic range). Energy
dose-dependence and steering dose-dependence are measuring genuinely
different things (a correlational structural signal vs. a causal
intervention response) and don't need to track each other -- they don't,
here.

**Cross-referenced against the steering results** (`injection_scaled.csv`), per feature:

| Feature | n designs (energy) | Mean dE_interaction | Mean H-bonds | aa_argmax_changed rate (steering) | Qualifying rows |
|---|---|---|---|---|---|
| 12588 | 18 | **-226.9** | 14.4 | 6.7% | 4 |
| 4657 | 57 | -214.5 | 14.8 | **0.0%** | 0 |
| 6073 | 56 | -212.2 | 12.2 | 4.0% | 3 |
| 10586 | 41 | -207.1 | 13.7 | 3.8% | 3 |
| 11326 (baseline, near-universal) | 477 | -184.6 | 12.2 | 3.1% | 35 |
| 14247 (baseline, near-universal) | 449 | -184.5 | 12.2 | 6.2% | 173 |
| 1707 (baseline, near-universal) | 448 | -184.3 | 12.2 | 1.1% | 7 |
| 233 (baseline, near-universal) | 444 | -183.7 | 12.1 | 2.9% | 65 |
| 12918 | 52 | -173.5 | 11.6 | 6.9% | 5 |
| 6869 | 50 | -164.1 | 12.1 | **27.3%** | 16 |
| 2214 | 19 | **-152.5** | 11.2 | 11.7% | 7 |

**Key finding: energy and steering-effect strength are inversely related
for the two most exciting steering hits.** `6869` and `2214` -- the
rarest, most causally-responsive features under steering (27.3% and 11.7%
`aa_argmax_changed` rates, the two highest of any candidate) -- have the
**two least favorable interaction energies of the entire list**, both
worse than the near-universal baseline features (~-184). This directly
validates the caution already raised above (finding #5): `6869`'s
InterPro label (P-loop NTPase/GTPase) and `2214`'s (Glycoside Hydrolase)
have no known relationship to VSNL1's real biology, and now there's
concrete energetic evidence they aren't "helps binding" features either --
a strong, reproducible causal steering effect is not the same claim as
"contributes to favorable binding energy," exactly the distinction this
gap note existed to make. Conversely, `12588`, `6073`, and `10586` show
the best combination of both signals: favorable energy *and* a real (if
more modest) steering effect. `4657` has the second-best energy of any
feature but **zero** causal steering effect (consistent with finding #4
above) -- a real energetic association without (yet-demonstrated)
sequence-level causality; worth a higher-alpha steering re-test before
concluding it's uninteresting, not immediate hand-off material.

> **RETRACTED (2026-09-27, after controls) -- the steering column above is
> uncontrolled and two of its conclusions are wrong.** Every
> `aa_argmax_changed` rate in that table is an absolute rate with no null
> condition. With norm-matched controls paired at the same design/site/alpha
> (see "Controlled feature injection" at the end of this file):
>
> | Feature | table's claim | controlled result |
> |---|---|---|
> | `6869` | "most causally-responsive", 27.3% | +3.0pp vs random, **p=0.73** -- not distinguishable from a random direction |
> | `2214` | 11.7%, second-highest | +3.3pp, **p=0.77** |
> | `4657` | "zero effect, worth a higher-alpha re-test" | re-tested: null at 4x, nominal only at 8x (p=0.0094, Bonferroni 0.056) |
> | `12588`/`6073`/`10586` | "best combination of both signals" | steering half is null; energy half stands |
> | `233` | "2.9%, near-universal baseline" | **the only feature that survives correction: +2.8pp, p=5.0e-09, n=2184** |
>
> So "the two most exciting steering hits" were not hits, and the priority
> ordering derived from this table should not be used. **The energy column is
> unaffected** -- those are Prime minimizations with no dependence on
> injection, and `233`'s activation-vs-`dE_interaction` correlation
> (r=-0.36, p=5.6e-15, n=444) stands exactly as reported.
>
> **What this table got right:** it flagged that the strongest apparent
> steering hits had the *least* favorable interaction energies, and warned
> that a causal steering effect is not the same claim as "contributes to
> favorable binding." That warning was correct, and the controls explain it
> -- those steering rates were noise. The energy numbers were signalling the
> problem before a null condition existed to prove it.
>
> **Where this leaves candidate selection:** the energy analysis and the
> controlled steering analysis, chosen for unrelated reasons, converge on
> `233` -- the only feature with both a robust correlational link to
> interaction energy and a demonstrated causal steering effect. Retarget the
> structural arm onto `233`, not the sparse features. Its ~46,000 candidate
> sites also avoid the pool-exhaustion that caps `10586` at 41.

**Correction (2026-09-27): the raw per-feature averages above are confounded, and a
proper multi-feature regression changes the priority.** Designs carry
several candidate features at once (the per-feature groups overlap), so a
feature's raw mean `dE_interaction` partly reflects whichever *other*
features/binder-length characteristics its designs happen to also have,
not necessarily its own effect. Ran a regression of `dE_interaction` on
every feature's presence AND activation-strength (per-SD, for designs
where present) simultaneously, controlling for binder length, robust SEs,
BH-FDR correction across all terms (`energy_bench/feature_energy_regression.py`,
`feature_energy_stats.py`). R²=0.26 (adj. 0.22).

**Holds up robustly (q<=0.01)**: `233` activation (-13.5 kcal/mol/SD),
`11326` activation (-12.9 kcal/mol/SD), `6073` presence (-30.1 kcal/mol).
**Borderline (q~0.06)**: `14247` activation, `4657` presence, `10586`
presence -- all still pointing toward more favorable binding.
**No independent effect once confounds are controlled**: `1707`, `2214`,
`6869`, `12588`, `12918` -- including `12588`, which had looked like the
best raw-average candidate above; that raw signal was apparently mostly
overlap with other features/binder length, not `12588`'s own effect.
**No feature was linked to *worse* binding** -- `6869`/`2214`'s
unfavorable raw averages (discussed above) don't hold up as a real
"hurts binding" effect either once confounds are controlled; the honest
read is "no detected effect," not "detected harm." (Caveat: `233`'s and
`14247`'s *presence* coefficients are unreliable -- only ~30 designs lack
either feature, mostly the same 30, r=0.92 between their absence, so the
model can't separate them there; their *activation* coefficients are
reliable, estimated from the ~440 designs that have them.)

**Revised candidate priority for hand-off** (energy regression + steering,
superseding the raw-average-based priority above): **`6073`** (solid
presence effect, -30.1 kcal/mol, q=0.011) is the clearest single candidate
*on the energy arm*. `10586` and `4657` are borderline-significant presence
effects -- reasonable secondary candidates on the same arm.

> **Corrected (2026-09-29).** This paragraph originally credited `6073` with
> a "real steering signal" and called `10586`/`4657` "steering-consistent."
> That was written against the pre-control steering rates and contradicts the
> retraction block above. Under the norm-matched controls all three are null:
> `6073` -2.4pp (p=0.75), `10586` -2.6pp (p=0.75), `4657` -1.1pp (p=0.51);
> `4657`'s 8x retest is Bonferroni 0.056, which is not confirmation. Their
> **energy** effects are unaffected and stand as reported. **`233` is the only
> feature with a real effect on both arms** (activation -13.5 kcal/mol q=0.011;
> steering +2.8pp p=5.0e-09). Consolidated per-feature view, both arms, controls
> only: `sae/results/run4/feature_summary.csv` (built by
> `scripts/build_feature_summary.py`).

`233` and `11326`'s activation
effects are the most statistically robust finding overall, but both are
near-universal (fire in ~93-100% of designs), so they're not useful as
"introduce this feature into a new design" candidates the way rare
features are -- their actionable form is different: steering to *increase*
activation at a position where one of them already fires in an otherwise-good
design, as a refinement lever, not a new mutation to introduce. `12588`
is **dropped** from the priority list (looked best on raw averages, no
independent effect in the regression). `6869`/`2214` remain deprioritized,
now on more precise grounds ("no detected effect" rather than "detected
harm").

**`4657` higher-alpha retest (2026-09-27): confirmed real, was under-dosed,
not sequence-inert.** The original run only tested alpha up to 2x mean
activation and got 0% `aa_argmax_changed`. Retested at 4x/8x
(`injection_4657_highalpha.csv`, 50 designs, 232 rows): clear dose-response
-- 5.2% `aa_argmax_changed` at 4x, 11.2% at 8x (19 qualifying rows total,
re-emergence 100% throughout). `4657` is confirmed in, not pending.

**Next step**: `inject_feature.py` outputs a `mutated_sequence` column
(native sequence with the flagged position swapped to `argmax_aa_post`)
for exactly this handoff. Filter `re_emerged & aa_argmax_changed &
argmax_aa_post != native_aa` **restricted to `6073`/`10586`/`4657`** (not
`12588`, per the correction above; not the full 178-site list; `4657`
confirmed above), and hand those to Andrew for Boltz re-folding +
ProteinMPNN inverse-folding (steps 3-4 of the steering pipeline). Built by
`06_steer/build_handoff.py`; do not filter ad hoc.

**Handoff filter correction (2026-09-27).** The first list sent to Andrew
had 18 sites and **6 of them proposed no mutation at all**. `aa_argmax_changed`
compares the *pre*-steering argmax to the *post*-steering argmax, not the
post-steering argmax to the native residue. When ESM-C already disagreed
with nature at a position, steering could move its pick back *onto* the
native residue: the flag reads True while `mutated_sequence` comes out
byte-identical to the input (e.g. `4657` resnum 52, native F, pre-argmax L,
post-argmax F). Those 6 sites were 5 of the 13 `4657` rows and 1 of the 3
`10586` rows. Andrew folded and inverse-folded the unmodified native protein
for them, so they appear as failures in his post-ProteinMPNN ESM-C re-check
but were never tests. Corrected list is 12 sites (`4657` 8, `10586` 2,
`6073` 2) -- the same 12 valid rows, with the 6 no-ops dropped; nothing was
wrongly excluded.

**The argmax readout proposes mutations that delete the feature's anchor
residue.** Checking `argmax_aa_post` against the residue each feature
actually fires on (bracketed position in `feature_top_examples.csv`):
`4657` fires on serine in **15/15** top examples, yet steering proposes
S->E at resnums 28/58/61 and S->T at 17 -- 4 of 8 proposals *remove* the
serine the feature is defined on, and only 1 of 8 (T7->S) installs one.
`6073` fires on leucine (7/15) and neither of its 2 proposals installs an
L. The label for `4657` is "serine preceded by acidic residues": the
injected direction encodes anchor *plus* acidic context, and the MLM head
resolves it toward the context (E) rather than the anchor (S). So the
post-ProteinMPNN ESM-C re-check fails on these by construction, not
because of a pipeline error -- the mutation removes the residue the
feature requires. A single-position argmax readout cannot express "keep
the serine, acidify the neighbours," which is what these features
actually want. Revisit the readout before drawing conclusions about
whether steering transfers through structure.

## Controlled feature injection (2026-09-27)

**The headline: injection causally steers residue preference for 1 of 11
features tested; every earlier steering number was uncontrolled and ~2/3
of the apparent effect is perturbation magnitude.**

`06_steer/inject_feature.py --control-directions random,other-feature` adds
two paired null conditions, evaluated at the same design/site/alpha as the
real injection: `random` (isotropic Gaussian rescaled to `||w_dec[f]/scale||`,
controls for perturbation magnitude) and `other-feature` (another live
decoder row at matched norm, also controls for lying in the SAE's span).
Conditions are exactly paired, so the right test is **exact McNemar on
discordant pairs**, not a two-proportion test.

**Dose-response, 11 features, 24,831 rows, zero skips:**

| alpha | feature | random | other-feature |
|---|---|---|---|
| 0.5x | 3.7% | -- | -- |
| 1.0x | 5.4% | -- | -- |
| 2.0x | **8.8%** | **6.4%** | -- |

At 2x the feature direction changes 8.8% of sites and a norm-matched random
vector changes 6.4%. **Only ~1/3 of the effect is feature semantics**;
reading the feature curve alone overstates it ~3x. This replicated across
two different pools (25k-pool run: 8.6% vs 5.7%), so it is robust.

**Per-feature, 0.5-2x, feature vs norm-matched random (paired McNemar):**

| Feature | density | n pairs | feature | random | excess | p |
|---|---|---|---|---|---|---|
| **233** | 0.42 | 2184 | 5.0% | 2.2% | **+2.8pp** | **5.0e-09** |
| 14247 | 0.76 | 2865 | 8.2% | 6.9% | +1.3pp | 0.022 |
| 11326 | 0.44 | 1680 | 5.2% | 4.3% | +1.0pp | 0.11 |
| 1707 | 0.15 | 867 | 1.6% | 1.2% | +0.5pp | 0.42 |
| 4657 | 0.0099 | 261 | 1.5% | 2.7% | -1.1pp | 0.51 |
| 6869 | 0.0001 | 66 | 13.6% | 10.6% | +3.0pp | 0.73 |
| 10586 | 0.0006 | 78 | 9.0% | 11.5% | -2.6pp | 0.75 |
| 6073 | 0.0004 | 84 | 8.3% | 10.7% | -2.4pp | 0.75 |
| 2214 | 0.0002 | 60 | 15.0% | 11.7% | +3.3pp | 0.77 |
| 12588 | 0.0002 | 60 | 6.7% | 5.0% | +1.7pp | 1.00 |
| 12918 | 0.0004 | 72 | 12.5% | 13.9% | -1.4pp | 1.00 |

Bonferroni for 11 comparisons is p < 0.0045, so **only `233` survives**.
Pooled across all 11 the gap is 6.00% vs 4.60% random (p=5.1e-07) and 4.76%
other-feature (p=9.0e-06), but that pooled significance is carried almost
entirely by `233`.

**High-alpha arm (4x/8x), 1,830 rows:** `4657` is null at 4x (3.8% vs 3.8%,
p=1.00) and nominally positive at 8x (13.1% vs 5.5%, n=183, 20/6 discordant,
p=0.0094 -- but 1 of 6 feature-by-alpha tests, so Bonferroni puts it at
0.056). `6073` and `10586` are negative or null at both doses. The
uncontrolled `injection_4657_highalpha.csv` magnitudes reproduce (3.8% at
4x, 13.1% at 8x vs the 5.2%/11.2% reported); what they lacked was the
control showing only the 8x half is feature-specific.

### What does NOT hold

1. **`re_emerged` is not evidence.** 90.7% here, 99.6-100% in the earlier
   run -- but **98.5% of injected sites already had the feature active
   before injection** (median pre-injection code 7.26), and control
   directions "re-emerge" at 88% for the same reason. It measures site
   selection, not causal re-firing.
2. **`code_delta` linearity is a mathematical identity.** Injecting
   `alpha*w_dec[f]` and then encoding raises code `f` by
   `~alpha*(w_enc[f].w_dec[f])`, linear in alpha by construction; a random
   direction in 960-d has near-zero overlap with `w_enc[f]`, so controls at
   ~0.00 are forced. Not an empirical result.
3. **There is no demonstrated density effect.** The dense-vs-sparse
   contrast is **p=0.077**. Pooling the six sparse features gives 420 pairs
   but only **54 discordant** ones (b=27, c=27, p=1.00), and McNemar power
   scales with discordant pairs, not total pairs: power to detect a
   `233`-sized (+2.8pp) effect at 54 discordants is **0.33**. So the sparse
   features are **underpowered, not demonstrably null**, and "steerability
   tracks density" is a hypothesis this data cannot test. Claiming it from
   one significant class plus one non-significant class is the
   significant/non-significant comparison error.
4. **`4657` moves the wrong way semantically.** See the anchor-residue
   section above: it fires on serine 15/15 and injection proposes S->E.
   Even granting the 8x effect, the residue it installs is anti-correlated
   with what the feature detects.

### Robustness and caveats

- **Layer-offset warning, now bounded.** 108 rows in the sweep and 42 in
  the high-alpha arm had `pre_code_pool`/`pre_code_live` disagree beyond
  tolerance -- 36 distinct sites x 3 direction modes. Dropping all of them
  plus their paired partners moves `233` from n=2184 to 2181 with
  discordants unchanged (b=86, c=25) and p=5.03e-09. The flagged rows do
  not drive any result, but the underlying per-site indexing issue is still
  unexplained.
- **Absolute rates differ from the published sweep** (3.7/5.4/8.8% vs
  2.1/4.0/7.1%) **because the original sweep used `--sampling random` and
  these runs used the default `--sampling top`.** Not the pool, not the
  interface YAML (md5 `8653fe198ddc4ed090f3c8ee66259d89`, mtime 2026-09-26
  16:55, predates `injection_scaled.csv` at 18:19 and contains 19 of the
  original's 20 `233` designs), and not the RNG stream -- under
  `--sampling top` `select_design_feature_pairs` sorts by `max_activation`
  and never draws from its generator, and control directions use a separate
  generator seeded `--seed + 1`, so they cannot affect design selection.

  The `design_max_activation` of the selected designs settles it: `top`
  picks a tight high band, `random` spans the whole range.

  | Feature | candidates | orig min/mean | new min/mean | designs shared |
  |---|---|---|---|---|
  | `233` | 46,327 | 2.37 / 7.13 | 13.00 / 13.39 | 0/20 |
  | `11326` | 64,997 | 1.89 / 3.99 | 17.29 / 19.35 | 0/20 |
  | `14247` | 46,995 | 8.99 / 12.33 | 15.60 / 15.92 | 0/20 |
  | `1707` | 50,765 | 0.53 / 1.20 | 6.48 / 7.28 | 0/20 |
  | `2214` | 19 | 0.17 / 0.68 | 0.17 / 0.68 | 19/19, identical values |
  | `12588` | 18 | 0.20 / 0.92 | 0.20 / 0.92 | 18/18, identical values |

  Features with <= 19 candidates match to the decimal because both modes
  take everything; every large-pool feature shares nothing. And it explains
  the rate gap with no extra assumption: `top` selects the
  strongest-activating designs, which respond more strongly to injection.
  Restricted to the 99 designs both runs share, 2x agrees at 15.79% vs
  15.76% -- **the method reproduces, the sample does not.**

  **Any rerun meant to compare absolute rates against the published sweep
  must pass `--sampling random` explicitly.** Otherwise do not mix these
  absolute rates with the published ones. The paired comparisons are
  unaffected either way, since every control is drawn at the same design and
  site as its feature row.
- **Exhausted pools.** `2214`, `12588`, `6869`, `10586` and `12918` drew
  every candidate available. Raising `--designs-per-feature` cannot add
  power for them -- a density claim needs a different screen.

### Run configuration (Waluigi)

Ran on Waluigi, where the repo layout is `sae/training/`, not `sae/06_steer/`.
Outputs `injection_with_controls_65kpool.csv` (24,831 rows) and
`injection_controls_highalpha.csv` (1,830 rows) live on that host, not in
this repo.

| Input | Path |
|---|---|
| Checkpoint | `sae/training/checkpoints_65k_1_4/run4_natural_mix/best.pt` |
| Pool | `/home/bridget/notebooks/vilip1_layer23_65k_per_residue` (**must be the 65k pool** -- `layer23_per_residue` holds only 116 of the original sweep's 217 designs and silently skips the rest) |
| Interface YAML | `sae/training/interface_features_combined.yaml` |
| Feature stats | `sae/training/run4_feature_stats.csv` |
| Interpreter | `~/esmfold2_venv/bin/python` (torch 2.7.1+cu118; **not** `esm_verify_venv`, whose torch 2.13.0+cu130 needs an r580+ driver and silently falls back to CPU on this host's 525.125.06) |

Hook alignment verified before both runs: `hook(blocks[22])` vs
`hidden_states[23]` matched at max abs diff 0.

**What is quotable for the abstract:** feature injection causally steers
ESM-C's residue preference for `233` (+2.8pp over norm-matched random,
p=5.0e-09, n=2184), and ~2/3 of an uncontrolled dose-response is
perturbation magnitude. Nothing else above clears correction.

### 233 across five doses, with controls (2026-09-27)

`injection_topk_controls.csv` (13,035 rows = 4,345 sites x 3 direction modes,
`--features 233,4657,6073,10586 --alpha-multipliers 0.5,1.0,2.0,4.0,8.0
--sampling top`, zero skips). Extends the controlled measurement to 4x/8x,
which the earlier arm never covered, and **replicates it**: averaged over the
three shared doses the excess is +2.9pp here against +2.8pp before, on the
same 728 sites per dose.

| alpha | n pairs | feature | random | excess | b/c | p (exact McNemar) |
|---|---|---|---|---|---|---|
| 0.5x | 728 | 2.9% | 1.1% | +1.8pp | 17/4 | 0.0072 |
| 1.0x | 728 | 4.1% | 1.6% | +2.5pp | 24/6 | 0.0014 |
| 2.0x | 728 | 8.1% | 3.6% | +4.5pp | 46/13 | 1.9e-05 |
| 4.0x | 728 | 10.9% | 5.1% | +5.8pp | 64/22 | 6.5e-06 |
| 8.0x | 728 | 22.5% | 7.6% | +15.0pp | 136/27 | 1.1e-18 |
| **all** | **3640** | **9.7%** | **3.8%** | **+5.9pp** | **287/72** | **1.6e-31** |

**Significant at every dose individually**, including the lowest. This is the
strongest steering result in the project and the one to quote.

**But the specificity ratio does not grow with dose -- state this carefully.**
The share of discordant pairs favouring the feature is flat (0.81 / 0.80 /
0.78 / 0.74 / 0.83; Cochran-Armitage z=0.671, **p=0.50**; the narrower 8x
vs 0.5x ratio test independently gives p=0.759). What rises with
alpha is the *number* of sites that change at all (21 -> 163 discordant
pairs), not the feature's share of them. So the claim is an invariance --
**the feature direction beats a norm-matched random direction ~4:1 among
discordant pairs, constant across a 16-fold dose range** -- not "steering
becomes more feature-specific at higher dose." The percentage-point excess
grows only because the base rate does.

**The other three features remain null at every dose**, now including 4x/8x:
`4657` +1.4pp pooled (p=0.21, n=435), `6073` -0.7pp (p=1.00, n=140), `10586`
-2.3pp (p=0.61, n=130). `4657`'s earlier nominal 8x effect does not reappear
here (+2.3pp, p=0.75) -- consistent with it having been one of six tests.

**Caveat on generality.** These runs use `--sampling top`, so every site is
one where the feature already fires strongly (`233`'s selected designs span
`design_max_activation` 13.00-13.39). The controls are drawn at the identical
sites so the comparison is clean, but the claim is bounded: injection steers
residue preference *at sites where the feature fires strongly*. Whether it
holds at typical sites needs one `--sampling random` arm, untested as of now.

### Anchor-aware candidate list (2026-09-27)

`06_steer/select_topk_mutations.py` over `injection_topk.csv` (4,345 rows):

| outcome | n | share |
|---|---|---|
| `native_is_anchor` (excluded) | 2135 | 49% |
| `no_anchor_in_topk` (excluded) | 696 | 16% |
| installs anchor at rank 1 | 548 | 13% |
| installs anchor at ranks 2-5 | 966 | 22% |

**49% of sites have the anchor residue already present**, which is the scale
of the problem the argmax readout was hiding -- at those sites every
substitution destroys what the feature detects. The 966 rank-2-to-5 hits are
proposals an argmax readout cannot reach at all.

Output `sae/results/run4/steering_candidates_topk.csv`: 370 distinct
proposals over 57 designs (`233` 306, `4657` 34, `6073` 16, `10586` 14),
anchor sets `233`->{E,K}, `4657`->{S}, `6073`->{C,L} (C excluded),
`10586`->{F,G,I,L}. **Before handing this off, drop the 4 proposals that
mutate an initiator methionine at resnum 1**; terminal positions are 4% of
the list and ESM-C's predictions there are least reliable.

### Structural arm: handoff to Andrew (2026-09-27)

**Retargeted onto `233`.** The old structural candidate list
(`steering_candidates_final.csv`, features `4657`/`6073`/`10586`) was chosen
before controls existed, on energy association and `re_emerged` -- and
`re_emerged` is near-tautological while those three features show no
controlled steering effect at any dose. Testing structural accommodation of a
mutation from a feature that does not demonstrably steer answers nothing.
`233` is the only feature with both a controlled causal effect (+5.9pp,
p=1.6e-31) and an energy association (r=-0.36, p=5.6e-15).

**One site per design, deliberately.** `233`'s 294 eligible anchor-installing
proposals sit on only 20 designs (7-21 sites each, because density 0.42 means
it fires at many positions per design). Sites sharing a backbone are not
independent observations, so a sign test across all 264 would treat correlated
measurements as independent and overstate significance.
`06_steer/build_structural_handoff.py` takes one site per design: **20
independent paired observations**, fold budget 100 structures (20 native + 20
mutant + 60 random-substitution controls) rather than ~1,400.

Outputs `sae/results/run4/structural_sites_233.csv` and
`structural_folds_233.fasta`. All 20 are `installs_anchor_rank1` (the
readout's top pick was already an anchor residue), 14 install K and 6 install
E, resnum 8-63 (median 32.5). **All 20 native backbones are already on disk**
in `data/vilip1_full20k/results/`, so only the 80 mutant/control folds are
strictly new work.

**Selection-bias bug worth remembering.** The first version tiebroke on
`(rank, alpha, resnum)`. Rank and alpha tie almost everywhere -- all 20 sites
came out rank-1 at 0.5x -- so the resnum term silently selected the most
N-terminal eligible site in every design: 17 of 20 at resnum <= 22, exactly
the positions the same script drops below resnum 3 for being unreliable. Now
a seeded random choice among tied candidates. **When a tiebreak key is
reached this often it is not a tiebreak, it is the selection criterion.**

**Scoring protocol (`06_steer/score_accommodation.py`).** Use ProteinMPNN's
`--unconditional_probs_only` and score the position directly. **Do not
resample** -- that is what produced the earlier all-zero re-check: a single
point mutation gives a backbone nearly identical to native, so MPNN reads a
near-native fold and writes back near-native sequence, deleting the mutation
before ESM-C sees it. Dense features appeared to "survive" only because they
fire on near-native sequence anyway.

    d_mutant      = logP(proposed) - logP(native)   on the mutant backbone
    d_native      = logP(proposed) - logP(native)   on the native backbone
    accommodation = d_mutant - d_native

Positive means the mutant fold prefers the steered residue more than the
native fold does. Never conditioning on the mutant sequence keeps the
mutation out of its own score; the paired form cancels each site's intrinsic
preferences, which is what makes n=20 workable. The 60 controls are not
optional -- MPNN recovers native residues ~40-50% of the time, so there is no
interpretable absolute baseline without them.

**Status: folds not yet run as of this writing.**

### Structural arm results and the matched-baseline check (2026-09-27)

Andrew ran ProteinMPNN `--unconditional_probs_only` on the 100 folds. Two
findings, one negative and one pending.

**1. Accommodation of the steered mutation: null.** The double-difference
(`d_mutant - d_native`) came out +1.318 for steered mutations against +1.786
for the pooled random-substitution controls -- paired difference **-0.468,
p=0.50**. Random substitutions are "accommodated" at least as well as steered
ones. The controls did exactly the job they were added for.

`d_mutant` is also confounded independently of that result, and this was
flagged at the start of the day then designed in anyway: the mutant backbone
was folded by Boltz **from the mutant sequence**, so the substitution is baked
into the geometry before ProteinMPNN reads it. MPNN's
`--unconditional_probs_only` closes the leak through MPNN and leaves the leak
through Boltz open. The random controls cancel that artifact (they are folded
the same way), which is why the null is trustworthy -- but `d_mutant` should
not be quoted as a standalone quantity. `d_native` has no such path: a native
backbone is folded from the native sequence only.

**2. `d_native` looks positive (d=1.15) but the comparison is confounded.**
The claim is that MPNN favours E/K at `233`'s positions more than elsewhere in
the same protein. Two problems:

- **These designs are 45.5% E+K by composition** (E 25.8%, K 19.7% across the
  20 handoff designs). `233` fires at E/K positions 50.2% of the time -- a
  1.1x enrichment, not a discovery.
- The score is `log P(E or K) - log P(native residue)`, so it is **near zero
  wherever the native residue is already E or K**. The anchor-aware selector
  keeps only sites whose native is *not* an anchor (all 20 handoff natives are
  `DHIIILLLLNNQQQRRRSST`, zero E/K), while the baseline "other positions in
  the same protein" retains the ~45% that are. Feature sites can score
  positive against a baseline half-pinned at zero, from the selection alone.

`06_steer/score_matched_baseline.py` settles it by holding native-residue
identity fixed: each site is compared only against non-site positions carrying
the **same** native residue (same design where >=3 exist, pooled across
designs otherwise). It reports both comparisons, so the gap between them is
the size of the artifact. It needs only the native `.npz` files -- no mutant
structures, hence no Boltz leak, and no CIFs.

Validated on synthetic backbones: with a uniform charge preference and **no
feature effect at all**, the unmatched comparison returns **d=+1.50**
(p<0.0001) while the matched one returns d=+0.18 (p=0.50). With a real effect
injected, both stay significant (d=+5.13 and d=+3.29). **The pure-artifact
simulation lands in the same range as the observed d=1.15**, so the observed
value is consistent with zero structural signal until the matched test runs.

Matched pool within the 20 designs is thin per residue (L 20, S 51, D 26,
T 17, N 13, I 12, R 6, Q 5, H 0 -- 150 total), so pool across residue strata
rather than testing each. A much larger baseline is available at no folding
cost from the ~20,000 predicted structures in `data/vilip1_full20k/results/`
on the local machine, though those are not on the cluster.

**Status: matched test not yet run. Needs the 20 native `.npz` files.**
No binding claim either way -- ΔipSAE was flat zero.

### Matched-baseline test on real MPNN output (2026-09-27, late)

Ran on Andrew's 20 native `.npz` (`native_npz_233.tar.gz`). **The prediction
that matching would kill the effect was wrong -- matching strengthened it.**

**Two file-format facts that any rerun must handle.** The npz key is `log_p`,
not `log_probs`, and its shape is `(1, L, 21)`. More importantly the structures
are **binder-target complexes**: Vilip-1 occupies the first **191** residues
and the binder follows, so `log_p` is `(1, 256, 21)` for a 65-residue binder.
Site positions are binder-local, and the non-site **baseline must also be
restricted to the binder** or it silently averages over target-chain
positions. Andrew handled the offset correctly (his reported `native_aa`
matches `S[191+position]` in 20/20 and the absolute index in 1/20, that one by
coincidence). `score_matched_baseline.py` now locates the binder by finding the
native sequence inside the structure's own sequence and refuses to guess.

**Why my predicted mechanism was backwards.** I argued the baseline was
half-pinned near zero because E/K-native positions give
`log P(E or K) - log P(native) = 0`. It is not zero: the anchor set *contains*
the native residue there, so the score is `log(1 + P(K)/P(E)) >= 0` and often
clearly positive. Those 529 E/K-native baseline positions therefore **raise**
the unmatched baseline and **suppress** the unmatched delta. Removing them by
matching increases the effect. Andrew's choice to exclude E/K from his
baseline was right; my unmatched figure is the downward-confounded one.

| comparison | n | mean | +/- | sign p |
|---|---|---|---|---|
| unmatched (all other binder positions) | 20 | +0.670 | 16/4 | 0.0118 |
| **matched on native residue, pooled baseline** | **20** | **+1.015** | **18/2** | **0.0004** |
| matched, in-design baseline only | 17 | +0.951 | 14/3 | 0.0127 |

Magnitude-aware tests on the pooled version agree: one-sample t p=0.0006,
Wilcoxon p=0.0025. **Andrew's report that magnitude-aware tests go
non-significant was an artifact of baseline size**, not a property of the
effect: his in-design-only matching left 12 of 20 sites with <=2 matched
controls and 3 with zero (dropping n to 17). Re-testing his own
`matched_diff` column reproduces his p=0.049 exactly. Pooling across designs
gives a median of 30 matched controls per site.

**But the effect is not E/K-specific, which is the part that matters for the
figure.** Repeating the matched test with the anchor scored against arbitrary
residues instead of the native one (`--reference random`, the quantity the
steered-vs-control comparison computes since `log P(native)` cancels from it):

| comparison | n | mean | +/- | sign p |
|---|---|---|---|---|
| unmatched | 20 | -0.035 | 9/11 | 0.82 |
| matched on native residue | 20 | +0.259 | 10/10 | **1.00** |

So E/K gains no advantage over arbitrary residues at `233`'s positions.
Decomposing the native-reference effect: `log P(native)` is -0.514 lower at
sites (p=0.50) and `log P(E or K)` +0.502 higher (p=0.26) -- neither
individually significant, the effect living in the difference. Probability
mass shifts **off the native residue onto alternatives generally**, not onto
E/K in particular.

**What can honestly be claimed.** At positions feature `233` flags, ProteinMPNN
assigns relatively less probability to the residue actually present than at
native-residue-matched control positions (p=0.0004-0.0025 across three tests).
Since those positions were chosen because ESM-C + the SAE wanted to change the
residue, this is a genuine cross-model convergence: **a sequence model and a
structure model independently agree the incumbent residue is suboptimal
there.** What cannot be claimed is that these are E/K-tolerant or
E/K-preferring environments.

**Unresolved.** MPNN per-position entropy is +0.165 nats higher at `233`'s
sites (14/6, p=0.115) -- not significant, but not excludable at n=20. If it
holds at larger n the result weakens to "`233` fires at structurally ambiguous
positions." Scaling to n=60-80 would settle both this and the E/K question.

**Figure guidance.** Fig 1 as drafted ("E/K advantage vs every other position
in the protein, d=1.15") should not be used: the unmatched comparison is the
confounded one, and the surviving effect is not about E/K. A matched version
plotting `site - mean(same-native-residue positions)` against a zero line is
defensible, labelled as the native residue being disfavoured rather than E/K
being preferred. Fig 2 (accommodation by arm, treatment below its own null) is
sound as drafted. Fig 3 should not assert three independent confirmations: the
energy and steering legs share the SAE and design corpus, and the structural
leg supports a narrower claim than "structural tolerance".

### Interaction energy on steered mutants: run, and uninformative (2026-09-29)

**The energy regression remains observational, and this run does not change
that.** Until 2026-09-29 the Prime numbers covered only the 477 *unmutated*
designs from `07_energy/select_energy_sample.py`; the steered mutants had been
scored for ProteinMPNN accommodation (null, -0.468, p=0.50) and ipSAE (flat
zero) but never for interaction energy. That gap is now closed mechanically --
all 100 structural-arm folds have been profiled -- but the result does not
license a causal claim in either direction, for the reasons below. Anything
built on the energy regression should still say *associational*.

**The experiment is already specified and mostly staged.**
`structural_folds_233.fasta` holds the 100 folds for `233` -- 20 native,
20 steered mutant, 60 random-substitution control, one site per design so the
20 designs are independent observations. Running the existing Prime pipeline
over them yields 20 paired DDE values with a built-in null, on the only feature
that survives the steering controls.

  07_energy/build_mutant_energy_inputs.py      ids + YAML + manifest, --stage-natives
  07_energy/run_mutant_energy_profiling.bash   SLURM array, same shape as the 477 run
  07_energy/analyze_mutant_energy.py           paired test, one effect per design

**Ran 2026-09-29** on Andrew's `9_29_f233_results/structures` (all 100 folds,
naming matched, no prep failures; energies -123 to -435 kcal/mol, comparable to
the 477-design run's ~-184).

| comparison | n | mean | t p | Wilcoxon p |
|---|---|---|---|---|
| mutant - random-substitution controls (primary) | 20 | **+16.98** | 0.207 | 0.123 |
| mutant - native (confounded by the refold) | 20 | +3.11 | 0.849 | 0.898 |
| *native - controls* (diagnostic, should be ~0) | 20 | *+13.87* | *0.204* | -- |

**Do not report this as "steering does not improve binding energy."** It is
uninformative, for three separate reasons, in order of importance:

**1. The mutation barely moves the regression's predictor.** The energy
coefficient is -13.5 kcal/mol *per SD of `max_activation`* -- the MAXIMUM
activation over positions in a design (SD 2.08 across the 444 designs that
carry `233`). Each of these 20 designs already has `233` firing at **7-21
positions**. Installing one more anchor at one position raises the maximum only
if the new site out-activates the existing best, which it generally will not.
**The model's own predicted effect for this mutation is ~0, not -13.5.** The
regression is a between-design statement; this experiment is a within-design
single-residue perturbation. They are different quantities, and a null on the
second is fully consistent with the first being real. (Only 2 of these 20
designs are even in the 477-design energy set the regression was fit on.)

**2. Underpowered regardless.** Paired SD is 58.1 kcal/mol, so at n=20 the
minimum detectable effect is ~37 kcal/mol -- larger than the between-design
association it was meant to probe.

**3. The control set is not residue-matched.** Every mutant installs K (14/20)
or E (6/20); the controls install a broad spread dominated by bulky residues.
Among controls, K/R/E average -198.2 kcal/mol and everything else -257.8, a
59.6 kcal/mol gap (per-residue: W -340.8, I -322.7, Q -291.0 vs K -123.3).
Prime's interaction energy tracks how much residue sits at the interface, so
mutant-vs-control partly measures the amino acid. The `native - controls`
diagnostic (+13.87) is the tell: the native was never steered, so that offset
is a property of the control set. **0 of 20 designs has a control installing
the same residue as its mutant**, so this batch cannot separate the two.
This is the same failure mode the matched-baseline test caught in the MPNN arm
-- a baseline not matched on residue identity.

**To actually test the energy claim** you need designs differing substantially
in `max_activation`, not one extra site on a design that has twenty: steer hard
enough at the top-activating position to move the design's maximum, or compare
across the activation range with matched composition. For the residue confound
specifically, controls installing K/E at *non-`233`* positions in the same
designs -- matched on residue, varied on position, inverting the current
design. n~60-80 for power, which the E/K question needs anyway.

**The comparison to make is mutant vs. its own controls, not mutant vs. native.**
The mutant backbone was folded from the mutant sequence, so the substitution is
baked into the geometry before Prime reads it -- the same leak that made
`d_mutant` unquotable in the accommodation arm. The controls are folded the
same way and cancel it; mutant-minus-native does not and is secondary.

**`dE_interaction` is not a stability measure.** It is a chain-split interaction
energy (interface-restricted minimisation of the complex, then single-point
energies of the isolated chains from that geometry). Lower means binds VILIP-1
better, *not* a more thermostable binder -- that would need folding dG or a
separately minimised apo state. Do not let "features associated with lower
energy" become "features that make binders more stable" on a poster.

## Multi-position steering sweep: k does not move the maximum (2026-10-17)

**The headline: steering feature `233` at up to 38 interface positions at once
does not raise a design's feature maximum any higher than steering one
position does. The maximum tracks alpha alone, identically at every k. At
matched total injected magnitude, raising k strictly *dilutes* the effect. The
`k` axis buys nothing on the quantity the energy regression is about.**

Ran `06_steer/sweep_multiposition.py` on features 233, 4657, 6073, 10586 --
60 designs each (41 for 10586), k=1,2,4,8,all, alpha=1,2,4,8x
`mean_activation_when_active`, `--control-directions random`, 8,840 rows per
arm. Both arms: `sweep_multiposition.csv` (alpha per position) and
`sweep_multiposition_split.csv` (`--alpha-split`, alpha/k, total magnitude held
constant). Numbers below are feature `233`, 2,400 rows per arm.
Resolved inputs: checkpoint `checkpoints_65k_1_4/run4_natural_mix/best.pt`,
pool `vilip1_layer23_65k_per_residue`, `interface_features_combined.yaml`,
`results/run4/feature_stats.csv`.

**1. `design_feature_max_delta` is the knob, not a finding.** For `233`
(`mean_activation_when_active` = 3.599, natural max ~14.4):

| alpha | alpha abs | max_delta | SD of max_activation | max_pre -> max_post |
|---|---|---|---|---|
| 1.0x | 3.60 | 3.544 | +1.70 SD | 14.4 -> 18.0 |
| 2.0x | 7.20 | 7.100 | +3.41 SD | 14.4 -> 21.5 |
| 4.0x | 14.40 | 14.214 | +6.83 SD | 14.4 -> 28.7 |
| 8.0x | 28.79 | 28.441 | +13.67 SD | 14.4 -> 42.9 |

`max_delta` = 0.985 x alpha, with SD 0.092 across all 60 designs, and it is
**identical to six decimals at k=1, 2, 4, 8 and all**. The maximum moves
because the injection puts it there, not because the perturbation propagated.
So this run is *not* the vacuous 2026-09-29 case -- the predictor did move,
and `design_feature_max_moved` is 1.00 everywhere -- but it moved
tautologically, and nothing about it is evidence of a compositional effect.
Note the dose: at 8x the design's maximum is pushed to +13.7 SD on a
regression fit over an observed SD of 2.08 (-13.5 kcal/mol per SD). Anything
above ~2x is extrapolating the energy claim far outside its support.

**2. `--alpha-split` shows the k effect is total magnitude, nothing else.**
Holding total injected magnitude constant, `max_delta` falls as alpha/k, so
k=8 moves the maximum **8x less** than k=1 at the same budget:

| k | max_delta, 8x unsplit | max_delta, 8x split | frac_aa_changed, 8x unsplit | split |
|---|---|---|---|---|
| 1 | 28.441 | 28.441 | 0.250 | 0.250 |
| 2 | 28.441 | 14.214 | 0.183 | 0.125 |
| 4 | 28.441 | 7.100 | 0.279 | 0.067 |
| 8 | 28.441 | 3.544 | 0.306 | 0.052 |
| all (~38) | 28.441 | 0.758 | 0.423 | 0.021 |

In the unsplit arm `frac_aa_changed` rises with k (0.250 at k=1 to 0.423 at
k=all); in the split arm that ordering **inverts** (0.250 to 0.021). The
apparent "k=all beats k=1" is entirely the 38x larger total push. There is no
compositional claim here, which is exactly what the second arm was for.

**3. The count did not move either.** `233` already fires at 27-61 positions
per design (median 43.5) -- far more than the 7-21 the script's docstring
assumed. `n_active_pre` equals `n_active_post` at k=1, 2, 4 and 8, and k=all
(mean 38.1 positions steered) moves it 43.50 -> 43.53. Steering installed
nothing new because the top-k sites were already active. The sweep moved
neither the maximum (except by construction) nor the count.

**4. Random controls are null on the maximum, NOT null on sequence.**

| alpha | max_delta feature | max_delta random | `moved` random | designs with >=1 AA change: feature / random | mean n_aa_changed: feature / random |
|---|---|---|---|---|---|
| 1.0x | 3.544 | -0.002 to +0.025 | 0.40-0.57 | 30.3% / 19.3% | 0.60 / 0.30 |
| 2.0x | 7.100 | -0.041 to +0.011 | 0.42-0.60 | 40.0% / 24.7% | 1.13 / 0.38 |
| 4.0x | 14.214 | -0.152 to +0.017 | 0.37-0.50 | 50.7% / 35.0% | 2.08 / 0.72 |
| 8.0x | 28.441 | -0.090 to +0.051 | 0.42-0.55 | 65.3% / 45.3% | 4.07 / 1.35 |

On the maximum the control is clean: random `max_delta` is ~0 at every k and
alpha and `design_feature_max_moved` is a coin flip (0.37-0.60), so the
maximum result is the feature direction's. On *sequence* it is not clean: a
norm-matched random push still changes a residue in 45.3% of designs at 8x
against the feature's 65.3%. The feature/random ratio on `n_aa_changed` is
2.0-3.0x, the same order as the 2026-09-27 controlled-injection result that
~2/3 of apparent steering effect is perturbation magnitude. Also note random
at k=all, 8x *destroys* active sites (43.5 -> 37.5), which the feature
direction does not.

**5. Where the MLM argmax turns over.** It is already turning over at 1x:
30.3% of designs change at least one residue at alpha=1x (mean 0.60
residues), rising to 65.3% and 4.07 residues at 8x. The earlier
`inject_feature.py` read of "argmax essentially never changes below ~3.6x"
was a single-position-per-pass artifact of counting per-site flips; across the
whole sequence, changes appear at the lowest dose tested. But see 4 -- a third
to a half of that is reproduced by random vectors.

**What this licenses.** Nothing about the -13.5 kcal/mol per SD energy claim.
The `k` axis is dead for that purpose: it does not raise the maximum, and at
matched magnitude it lowers it. The only lever on the maximum is alpha at the
top-activating position, which is single-position steering -- i.e. the design
`inject_feature.py` already had -- and by 4-8x it has pushed the predictor to
+7 to +14 SD, outside the range the regression was fit on. **Do not run an
energy arm off the k axis.** The open question is unchanged from 2026-09-29:
designs differing substantially in `max_activation` with matched composition,
and residue-matched controls at non-`233` positions.

**Scope.** Alphas 1-8x only; a 16x arm has not been run, so where the
extrapolated high-dose curve saturates is unmeasured. Of the 11 annotated
interface features, only these 4 were swept (the ones `inject_feature.py`
already ran, to stay comparable to run4), and only `233` is a feature the
energy regression was fit on.
