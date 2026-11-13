# Standards, formats and conventions

This pipeline reads and writes community formats at every boundary, and adopts
existing conventions rather than inventing local ones. Nothing here is a
bespoke serialization: every artifact the pipeline produces can be opened by
tools that know nothing about this repository.

## Structural formats

| Format | Where | Standard |
|---|---|---|
| **PDBx/mmCIF** | predicted complexes from Boltz-Prot-1, one per design | wwPDB exchange format |
| **PDB** | the Reg3A target structure (`REG3A_no_signal_pro.pdb`) | legacy wwPDB |
| **FASTA** | the target sequence (`config/reg3a_target.fasta`) | NCBI/UniProt convention |

Chain assignment follows the wwPDB convention of one identifier per polymer
entity: Reg3A is chain A, the designed binder is chain B, in every complex.
This is checked rather than assumed — a ported version of the embedder
defaulted to chain C (inherited from a different target's Boltz spec) and
would have silently embedded the wrong chain.

## Tabular and array formats

| Format | Direction | Where |
|---|---|---|
| **CSV** | written | every result table in `results/`, the training log, the activation `index.csv` |
| **JSON** | written | run metadata, `chosen_layer.json`, per-run `*_meta.json` sidecars |
| **NPY** | written | `activations.npy` — the per-residue activation matrix, preallocated as a memmap so a run is resumable |
| **NPZ** | read | `pae.npz`, the PAE uncertainty matrix from the design model |
| **NPZ** | written | contact fingerprints from `05_interpret/fingerprints.py` |

Every results CSV ships with a `*_meta.json` sidecar recording the parameters
that produced it, so a table is never separated from its provenance.

## Domain conventions we adopt rather than redefine

**Interface contacts — CAPRI / PISA.** A binder residue is in contact when its
minimum heavy-atom distance to any target atom is under 5 Å. This is the
standard used by the CAPRI assessment and by PISA, so our interface definition
is comparable to published interface analyses instead of being a local
threshold. Applied to the gated corpus it yields 164,285 contact residues of
771,228 (21.3%), with zero designs skipped.

**Multi-chain input — the model's own token.** Binder and target are joined
with ESM-C's native chain-break token (vocabulary id 31), which is the
convention the model was trained with. We deliberately did not use a poly-G or
poly-A linker, which would have put residues into the sequence that the model
would try to explain.

**Confidence metrics — AlphaFold lineage.** ipTM, pLDDT and PAE are used with
their standard meanings, and reported as what they are: a model's confidence
in its own prediction, not a measured affinity. One inherited quantity,
`min_interaction_pae`, is a *minimum* rather than the BindCraft-convention mean
interface PAE; that discrepancy is documented at the point of use so the
threshold gets recalibrated rather than silently inherited.

**Reconstruction quality — FVE.** Fraction of variance explained, the standard
sparse-autoencoder metric, which makes our dictionary directly comparable to
published ones. We benchmark against Biohub's released ESM-C autoencoder
rather than only against our own runs. The reporting protocol that makes two
teams' FVE numbers comparable — held-out split, frozen normalization
statistics, `k` and `d_hidden` quoted with the figure, evaluation distribution
named — is in [`FVE-PROTOCOL.md`](FVE-PROTOCOL.md).

**Sparse autoencoder construction.** TopK activation with an auxiliary loss for
dead features, following Gao et al. (2024), including the unit-norm decoder
constraint. We did not invent a training recipe; we implemented a published one
and report where our numbers sit against it.

**Identifiers.** The target is identified by its UniProt entry, with the signal
peptide removed; residue numbering is stated relative to the mature chain so
that every residue index in our results (Tyr124, Ser63, and the rest) resolves
against the public record.

## On SBOL

We do not emit SBOL, and the reason is scope rather than oversight. SBOL
describes genetic designs — parts, circuits, assemblies. This pipeline operates
one level up, on protein sequence and predicted structure, where the
interchange formats are mmCIF, PDB and FASTA. The point at which our work
becomes a genetic design is the expression construct, and that boundary is
where Registry and SBOL conventions apply.

The honest statement of our position: the pipeline is interoperable in the
formats its own domain uses, and we chose existing conventions over local ones
at every decision point where a community standard existed.

## Packaging and software conventions

- **PEP 621** project metadata in `pyproject.toml`
- **Console entry point** via `project.scripts`, so the pipeline is a command
  rather than a directory of scripts to be run by path
- **MIT**, an OSI-approved licence
- **Semantic versioning**
- Every stage is independently resumable, so a failed run restarts at the stage
  that failed rather than at the beginning
