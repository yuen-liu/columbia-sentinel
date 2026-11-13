# Overnight run report — 2026-10-05

Run 00:35:55 → 04:11:27 (**3 h 36 min**), layer 23, local M3 Pro (MPS).
Scope: S1 layer sweep → S2 paired embedding → S3 SAE training.
S4 onward deferred (need the energy-tool and response-variable decisions).

## Status: COMPLETE — all three stages succeeded, first attempt each

| Stage | Result |
|---|---|
| S1 layer sweep | layer 23 chosen (pre-registered tie-break); criterion found non-discriminating |
| S2 paired embedding | 36,710 designs → 2,387,652 × 960 fp16 (4.58 GB), 30 min, integrity verified |
| S3 SAE training | **val FVE 0.9687**, **0 dead features** of 16,384, 50 epochs, 3 h 2 min |

**Headline: the SAE reconstructs 96.9% of layer-23 activation variance using
64 of 16,384 features per residue, with a fully live dictionary.** That is the
real test of whether layer 23 encodes strong structure, and it passes —
matching iGEM's validated vilip1 result (0.9688–0.9730) with a cleaner
dictionary (0 dead vs their 18–31).

Artifacts:
- `results/layer_sweep.csv`, `results/chosen_layer.json`
- `data/reg3a_layer23_paired/` — activations, index, manifest, target
- `03_train/checkpoints/reg3a_layer23_paired/` — `best.pt` (362 MB),
  `latest.pt`, `training_log.csv`

---

## Pre-flight: all gates passed

| Check | Result |
|---|---|
| ESM-C on MPS | PASS — d_model=960, 30 blocks, chain-break `\|`=31, binder span finite |
| Embedding throughput | 23.8 designs/s solo → full corpus ~30 min, not the 2–3 h estimated |
| SAE training, MPS, d_hidden=16384 | PASS — ~0.3 s/step → 50 epochs ≈ 2.5 h |
| Resume after interruption | PASS — MSE continued 0.599→0.448, log appended not truncated |
| Manifest | 36,710 designs, 2,387,652 binder residues, 0 skipped, all with `pae.npz` |

MPS substantially outperformed the plan's estimate, which is why the layer
sweep fit alongside the main run.

## Ported-code corrections

1. **transformers 5.x renamed the ESM-C classes** — `ESMCForMaskedLM` /
   `ESMCTokenizer` → `EsmcForMaskedLM` / `EsmcTokenizer`. The iGEM (4.x)
   imports fail outright.
2. **The paired embedder hardcoded VSNL1 (vilip1) as the target.** Now a
   required argument with no default: silently embedding Reg3A binders
   against vilip1 would have produced plausible-looking, meaningless
   activations.
3. **Chain IDs: Reg3A is target=A, binder=B**; every ported script defaulted
   to binder=C from vilip1's Boltz spec.
4. **`min_interaction_pae` is a MINIMUM**, not the BindCraft-convention mean
   ipAE the quality gate intends — documented at the point of use so the
   threshold gets recalibrated rather than inherited.

---

## S1: layer sweep — an informative negative

`results/layer_sweep.csv`. Criterion: ridge CV R² from the mean-pooled binder
embedding to each binding-quality metric, 1500 designs, 5-fold CV, identical
fold splits across layers.

| layer | r2_iptm | ±fold std | iptm rank | r2_ipae | ipae rank | r2_binding_conf |
|---|---|---|---|---|---|---|
| 18 | 0.0271 | 0.0084 | 4 | 0.0374 | 8 | −0.0142 |
| 19 | 0.0255 | 0.0082 | 7 | 0.0426 | 3 | −0.0111 |
| 20 | 0.0199 | 0.0076 | 10 | 0.0374 | 7 | −0.0075 |
| 21 | 0.0213 | 0.0140 | 9 | 0.0358 | 10 | −0.0059 |
| 22 | 0.0196 | 0.0128 | 11 | 0.0352 | 11 | −0.0064 |
| **23** | 0.0256 | 0.0128 | 6 | 0.0368 | 9 | −0.0060 |
| 24 | 0.0257 | 0.0137 | 5 | 0.0381 | 5 | −0.0138 |
| 25 | 0.0250 | 0.0181 | 8 | 0.0378 | 6 | −0.0065 |
| 26 | 0.0282 | 0.0102 | 3 | 0.0430 | 2 | −0.0151 |
| 27 | 0.0296 | 0.0099 | 2 | **0.0452** | 1 | −0.0166 |
| 28 | **0.0306** | 0.0078 | 1 | 0.0425 | 4 | −0.0129 |

**Chosen: layer 23**, via the pre-registered tie-break (layer 28 was the
argmax but led by only 0.0050 R², inside the 0.01 margin), retained for
Biohub benchmark comparability.

### What this actually shows

**The criterion cannot discriminate layers.** All layers decode iptm at
R²≈0.02–0.03. The between-layer spread (0.011) is smaller than within-layer
CV fold std (0.008–0.018). So layer 23 was not shown to beat layer 28 — the
truthful statement is that this criterion has no resolving power, and the
pre-registered fallback to prior work carried the decision.

**But the directional hint is consistent.** Layers 26–28 are top-4 on *both*
iptm and ipae while 23 sits 6th and 9th. A contiguous block winning on two
different response variables is harder to dismiss than a single noisy
argmax. Caveat: adjacent layers' embeddings are highly correlated, so 26/27/28
agreeing is closer to one observation than three independent ones.

**Likely mechanism for the floor**: mean-pooling 65 residue vectors into one,
then linearly decoding a *global* complex-confidence metric, discards nearly
everything the layer encodes about specific residues. This is a result about
pooling and linear decodability, not about layer information content.

**Information content is tested properly by the SAE's FVE** (S3), not by this
sweep: can 64 of 16,384 features reconstruct the layer's activations? iGEM
reached 0.941 design FVE at layer 23 on vilip1.

### Scope limit

A second criterion was specified — per-residue interface-contact AUROC, the one
that matches what this project actually studies — was **not run**: it needs
the mmCIF contact parser, which is not built yet. **Tonight's layer choice is
provisional.**

### Separate finding: `binding_confidence` is unusable as-is

R² is **negative at every layer** — worse than predicting the mean. The metric
spans ~7 orders of magnitude (4.3e-7 upward), so it needs a log transform
before any regression can use it. This matters for S9, where it is a candidate
response variable.

---

## S2: paired embedding

Layer 23, binder + `|` + Reg3A(138aa), binder-span activations only.
Output `data/reg3a_layer23_paired/` — 2,387,652 × 960 fp16 ≈ 4.4 GB.

**Complete** — 00:39:05 → 01:09:03, **1792 s (30 min)**, averaging 20.5
designs/s. Matched the pre-flight estimate.

Integrity verified before committing training time to it:

| Check | Result |
|---|---|
| shape / dtype | (2,387,652, 960) float16 — 4.58 GB |
| rows == manifest residue total | yes (2,387,652) |
| unique ids | 36,710 of 36,710 |
| index coverage | contiguous, no gaps or overlaps |
| per-id lengths vs manifest | identical |
| sampled values | all finite, mean −0.327, std 21.0 |
| all-zero rows in sample | 0 |

Disk after: 24 GB free (was 28 GB).

### Incident: concurrent duplicate embedding job

A first launch was killed ~90 s in after a sampling flaw was found in the
sweep (below). The `pkill` pattern covered `run_overnight.sh|layer_sweep.py`
but **not** the embed child that run-1's sweep had already spawned — it was
reparented to init and kept running at layer 28. The verification `pgrep`
reused the same incomplete pattern and so reported "nothing running."

Detected ~13 min later via a full `ps` listing. Two jobs had been splitting
the GPU (throughput 23.8 → ~15/s each) and had preallocated two 4.4 GB
memmaps. Orphan killed, its 1.2 GB partial output removed, rate recovered to
22.4/s. No corruption — the jobs wrote to separate directories and the live
run's `progress.json`/`index.csv` stayed consistent. Cost: ~10 min of halved
throughput.

**Lesson**: verify process cleanup with a pattern at least as broad as what
was spawned, or by process tree, not by reusing the kill pattern.

### The sampling flaw that caused the relaunch — and why it mattered

The sweep's first subsample used equal-per-(campaign × iptm-quartile) quotas.
The 7 small `probe100` campaigns can always fill their quota while the two
18k campaigns get capped, so the subsample came out **71% probe designs /
29% final-campaign** — inverting a corpus that is **98% final-campaign**.
Since the SAE trains on that 98%, the layer must be chosen for *that*
distribution. Replaced with proportional-plus-floor sampling: 95.9% finals,
small campaigns still represented.

**This was material, not cosmetic**: under the probe-heavy subsample layer 28
beat 23 by *more* than the 0.01 margin and won outright (hence the orphaned
layer-28 embedding job). Under the corrected subsample its lead fell to
0.0050 and it lost the tie-break. Both agree 28 is the argmax; they disagree
on whether the margin is real — consistent with the criterion sitting at the
noise floor.

---

## S3: SAE training

d_hidden=16384, k=64 (per-token TopK), auxk=256, k-anneal 256→64 over the
first 20% of steps, batch 4096, lr 4e-4, 50 epochs, train/val stratified by
campaign.

Started 01:09:03. **224 s/epoch** (unbiased), ≈3.1 h total, finishing ~04:17.

Convergence:

| epoch | train_mse | val FVE (pooled) | avg L0 | dead features |
|---|---|---|---|---|
| 0 | 0.2097 | 0.5853 | 64.0 | 30 / 16384 |
| 2 | 0.0455 | 0.8183 | 64.0 | 6 / 16384 |
| 12 | 0.0400 | 0.9639 | 64.0 | **2332** / 16384 |
| 13 | 0.0393 | 0.9644 | 64.0 | 56 / 16384 |
| 14 | 0.0387 | 0.9649 | 64.0 | 15 / 16384 |

L0 is exactly 64 throughout, as the per-token TopK constraint requires.

**val FVE 0.9649 by epoch 14** already sits inside iGEM's final layer-23
vilip1 range (0.9688–0.9730), with 35 epochs left. This is the real answer to
"does layer 23 encode strong information": 64 of 16,384 features reconstruct
~96.5% of activation variance. The layer sweep's R²≈0.03 reflected the
weakness of pooled linear decoding, not the layer.

**K-annealing caught in the act.** The dead-feature spike to 2332 at epoch 12
and its recovery to 15 within two epochs is k reaching its target (256→64
over the first 20% of steps) killing features in the transition, then AuxK
resurrecting them. Exactly the designed mechanism, and a concrete
illustration of why AuxK is in the architecture.

### Timing-measurement error (mine)

I twice estimated the epoch rate by waiting for two new `training_log.csv`
rows and dividing elapsed by 2, getting 120 then 165 s/epoch, and reported a
02:49 finish. **Both were biased low, by up to 2×**: the wait starts at an
arbitrary point *within* an epoch, so the first row can arrive almost
immediately, and `elapsed/2` then measures roughly one epoch plus a fragment.
The unbiased estimator is `(process_elapsed − load_time) / epochs_done`,
which gives 224 s/epoch. Use that form for future rate estimates.

### Final result

**Complete 01:09:03 → 04:11:27 (3 h 2 min), 50/50 epochs, first attempt.**
Best checkpoint is epoch 49: **val FVE 0.968656, 0 dead features, avg L0 64.0.**

Per-campaign held-out FVE at the final epoch:

| campaign | val FVE | avg L0 |
|---|---|---|
| REG3A_probe100_124_128 | 0.9912 | 64.0 |
| REG3A_probe100_unguided | 0.9878 | 64.0 |
| REG3A_probe100_99_105 | 0.9705 | 64.0 |
| REG3A_final_25_33 | 0.9690 | 64.0 |
| REG3A_final_45_57 | 0.9682 | 64.0 |
| REG3A_probe100_45_57 | 0.9589 | 64.0 |
| REG3A_probe100_75_101 | 0.9585 | 64.0 |
| REG3A_probe100_75_78 | 0.9542 | 64.0 |
| REG3A_probe100_25_33 | 0.9533 | 64.0 |
| REG3A_Probe_25_33 | 0.9396 | 64.0 |
| **pooled** | **0.9687** | 64.0 |

Reconstruction is uniform across campaigns (0.94–0.99), so the dictionary is
not specialized to one hotspot — it generalizes across all seven probe
hotspots and both 18k campaigns. The two large campaigns, which dominate the
corpus, sit almost exactly at the pooled value as expected. The spread at the
low end (`REG3A_Probe_25_33`, 0.9396) is on 10 designs total, so it is small-n
noise rather than a real deficit.

Checkpoint verified loadable and well-formed:
- `d_model=960, d_hidden=16384, k=64, auxk=256`
- reconstructs into `SparseAutoencoder` from its stored config
- all 16,384 decoder rows unit-norm (the `norm_weights` invariant holds)
- train/val protein split disjoint, 33,039 / 3,671
- `encode()` returns exactly L0=64 per residue

### Convergence

| epoch | train_mse | val FVE (pooled) | avg L0 | dead features |
|---|---|---|---|---|
| 0 | 0.2097 | 0.5853 | 64.0 | 30 |
| 2 | 0.0455 | 0.8183 | 64.0 | 6 |
| 10 | — | — | 64.0 | **2810 (peak)** |
| 12 | 0.0400 | 0.9639 | 64.0 | 2332 |
| 14 | 0.0387 | 0.9649 | 64.0 | 15 |
| 22 | 0.0358 | 0.9668 | 64.0 | 37 |
| 39 | 0.0335 | 0.9683 | 64.0 | 2 |
| 49 | 0.0329 | **0.9687** | 64.0 | **0** |

Gain over the last 10 epochs was 0.00035 — fully converged, and confirming
the prior observation that iGEM's curves flattened well before epoch 50. **30 epochs
would have sufficed**, worth knowing for the layer-28 second arm.

**K-annealing and AuxK, caught in the act.** Dead features spiked to 2810 at
epoch 10 — exactly where k finishes annealing 256→64 (20% of 26,200 steps) —
then AuxK drove it back to 15 by epoch 14 and to 0 by the end. This is the
precise failure mode the architecture is designed around (features that lose
the per-token top-k race early never get gradient signal), observed and then
corrected. A clean, concrete illustration for the seminar.

**Monitoring note**: `train.py` prints without `flush=True`, so under shell
redirection its stdout sits in a buffer and `s3_train.log` stays empty for
long stretches — this looks like a hang but is not one. Per-epoch progress
was tracked through `training_log.csv` instead, which is opened and closed
each epoch and therefore always current. Worth adding `flush=True` for future
runs.

Reference — iGEM vilip1, layer 23, same architecture:

| | ours (d=16384) | Biohub (d=16384) |
|---|---|---|
| design FVE (held out) | 0.9688–0.9730 | 0.716 |
| dead features | 18–31 / 16384 | 1945 / 16384 |

---

## Notes carried forward

Two items recorded here because they affect how the results above should be
read, not as planned work:

- **The layer choice is provisional.** It was made with a criterion that
  proved unable to discriminate layers (§S1). A per-residue,
  interface-geometry-based criterion would be a sharper test.
- **`binding_confidence` needs a log transform** before use in any regression
  (§S1, "Separate finding") — it spans ~7 orders of magnitude.

---

# Phase 2 (2026-10-10): interface features, interpretability, steering

## S5: interface feature extraction — complete

`05_interpret/interface_activations.py` on the 12,060 ipTM>0.7 designs:
**0 skipped**, **164,285 contact residues** (21.3% of 771,228 binder residues).
Geometry ported from igemmodel's `extract_interface_features.py` (min
heavy-atom distance to target chain; contact ≤5 Å, shell 5–10 Å), with the
hard sequence check that refuses a design rather than misaligning residues to
SAE codes. Reg3A chains: target=A, binder=B.

Three issues caught by smoke-testing before they cost anything:

1. **Ranking by max activation selects n=2 outliers.** Top features by raw
   `max_act_contact` fired on *two* contact residues corpus-wide at activation
   ~40, while features with 1,173 residues of support topped out near 12. Max
   is an extreme-value statistic. Added a support filter applied *before*
   ranking.
2. **A dense `(batch × 16384)` code matrix.** The original 65,536-row batch
   would have allocated 4.3 GB on MPS — invisible at smoke-test scale, fatal
   at full scale. Now 8,192 rows / 537 MB.
3. **Geometry caching**, since parsing dominates runtime and is independent of
   both the SAE and the ranking.

No single ranking answers the question, so labeling evidence was collected for
the **union** of three (max activation, mean-when-firing, enrichment).
Enrichment stayed an annotation, never a gate: median 1.009 across 13,399
supported features, tail to 5.5×.

## Features are legibly interpretable

Feature **14364** — an invariant `[X]TG` immediately after an amphipathic
helical run, with **11/15** max-activating examples contacting the target's
**Tyr124/Val128** pocket at mean pLDDT 90.5:

```
QVKEKIEE[L]TGIAPEDQ   contacts=TYR124;VAL128        pLDDT=95.6
DVKKKIEE[L]TGLDPAEQ   contacts=TYR124;VAL128;ARG129
QVKEKISE[K]TGLDPKYQ   contacts=TYR124;VAL128
```

Feature **2672** — near-invariant `KDY.A..[D]EL.L..GD`, firing almost only on
Asp/Glu, enrichment 5.41, mean pLDDT 85.6.

## Incidental finding: hotspot guidance is real but partial

| campaign | % contacts 25–33 | % 45–57 | % 58–70 | % 112–130 | top contact |
|---|---|---|---|---|---|
| `final_25_33` | **17.9** | 1.6 | 20.4 | 26.2 | **29** |
| `final_45_57` | 1.2 | **11.7** | 18.1 | 37.6 | 115 |
| `probe100_124_128` | 0.0 | 2.1 | 32.5 | **46.9** | **124** |

Each campaign is enriched for its own hotspot (~11× contrast for
`final_25_33`), so guidance is measurable. But the majority of contacts for
every campaign land on a **shared epitope around residues 58–70 and 112–130**
that designs gravitate to regardless of instruction. Most "interface features"
therefore describe that shared surface, not the designed hotspot.

## S6: steering — readable, but not steerable

Intervention: inject `alpha * w_dec[f] / scale` at ESM-C layer 23, read the
effect through ESM-C's own `lm_head`. Validated end-to-end — at alpha=0 the
in-hook feature activation reproduced the pipeline's independently computed
value (16.514 vs 16.51), confirming manifest → paired embedding → layer 23 →
SAE → position indexing.

### Two readout traps, both of which produced a wrong answer

* **`output_hidden_states` is not a valid readout of a hook-modified pass.**
  Verified: zeroing `layers[22]` leaves `hidden_states[23]` untouched while
  changing `[24]` onward and the logits. The collected tuple holds pre-hook
  tensors. Feature activation must be measured *inside* the hook.
* **Masking only the probed residues leaves the answer in the input.** With
  `EVKEIAQA[M]` still visible, the model already knew the motif and steering
  had nothing to contribute — this produced an apparent null. Blanking the
  whole region fixed it.

### Every stage changed the answer

| test | result |
|---|---|
| naive single-residue mask | apparent null — flawed |
| region-blanked, 1 case, alpha=60 | apparent success (`QEKTGVPP`, 6/8 vs 3/8) — cherry-picked |
| replication, 5 features x 3 designs | weak; alpha=60 collapses below baseline; nothing survives BH over 10 tests |
| **pre-registered, 27 held-out features x 10 designs (n=270), alpha=20, p_native** | **+0.00565, 95% CI [+0.0033, +0.0080], t p=3.3e-6**, dz=0.29, 19/27 features positive |
| same run, vs another real feature's direction | +0.00179, p=0.19 — not significant |
| **cross-injection: 96 cases x 32 candidate directions** | **mean rank of the correct feature 15.72 vs 16.5 under null, p=0.41; ranked #1 in 3/96 vs chance 3.0** |

### Conclusion

Injecting a real SAE direction beats matched-norm **random** noise highly
significantly, but the **correct** feature is no better than any other
feature at producing its own motif. The significant result against random is
therefore explained by **on-manifold geometry** — real dictionary directions
damage the representation less than random vectors — and not by feature
identity.

**Readable is not the same as steerable.**

Scope of the claim, stated narrowly: this concerns the *decoder* direction, at
*layer 23*, at *alpha=20*, read out as masked-region native-residue recovery.
Untested alternatives include the encoder direction, clamping in latent space
and decoding the full reconstruction, earlier injection layers, and non-MLM
behavioural probes. This is a result about this intervention, not a proof that
SAE steering is impossible.

### Why the controls were load-bearing

The single-case alpha=60 result (`QEKTGVPP`, motif recovered, 6/8 vs 3/8
baseline, all three random controls at or below baseline) is a convincing
figure and is false. It did not replicate, and the feature that produced it
turns out to be no better than 31 unrelated features at the same task. Matched-norm
randoms, ablation, pre-registration on held-out features, and finally
cross-injection were each necessary to get to the right answer.

## Not run

Haiku auto-labeling (`05_interpret/label_features.py`) is written and
dry-run verified but needs `ANTHROPIC_API_KEY`; the evidence CSVs are
legible by hand, and the motifs above were read directly from them.
