"""Train a sparse autoencoder on ESM-C layer-23 per-residue activations for
the vilip1 binder-design pool (vilip1_full20k + composite_hotspot, natural
binders held out for eval only -- see data.py's module docstring).

Usage:
    python train.py --data-dir vilip1_layer23_per_residue --output-dir checkpoints/run1
"""

import argparse
import csv
import sys
import time
from pathlib import Path

# data.py lives in ../02_prepare_data, not this directory
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "02_prepare_data"))

# NOTE: import order matters on this machine -- importing torch before numpy
# segfaults in numpy's macOS Accelerate-framework self-check (confirmed via
# `python -c "import torch; import numpy"` vs. the reverse order). data.py
# imports numpy/pandas before torch, so importing it first here guarantees
# numpy loads first process-wide, regardless of what sae_model.py imports.
from data import center_scale, iter_batches, load_dataset  # noqa: E402
from sae_model import SparseAutoencoder, loss_fn  # noqa: E402
import torch  # noqa: E402


def fve(sq_err_sum: float, sq_x_sum: float) -> float:
    """Fraction of variance explained: 1 - MSE(recon, x) / MSE(0, x). The
    "0" baseline is the do-nothing predictor in centered+scaled space, i.e.
    reconstructing the train-set mean in the original space -- standard
    convention for SAE reconstruction quality, and scale-invariant so no
    un-centering/un-scaling is needed to interpret it."""
    return 1.0 - sq_err_sum / max(sq_x_sum, 1e-12)


@torch.no_grad()
def validate(model, ds, mean, scale, batch_size, device):
    model.eval()
    sources = sorted(set(ds.val_source.tolist()))
    sq_err = {s: 0.0 for s in sources}
    sq_x = {s: 0.0 for s in sources}
    l0_sum = {s: 0.0 for s in sources}
    n_rows = {s: 0 for s in sources}

    cursor = 0
    for batch in iter_batches(ds.val_x, batch_size, shuffle=False):
        batch_source = ds.val_source[cursor : cursor + batch.shape[0]]
        cursor += batch.shape[0]

        x_proc = center_scale(batch.to(device), mean, scale)
        recon, _, _, latents = model(x_proc)
        per_row_sq_err = (x_proc - recon).pow(2).sum(dim=-1)
        per_row_sq_x = x_proc.pow(2).sum(dim=-1)
        per_row_l0 = (latents != 0).sum(dim=-1).float()

        for s in sources:
            mask = torch.from_numpy(batch_source == s).to(device)
            sq_err[s] += per_row_sq_err[mask].sum().item()
            sq_x[s] += per_row_sq_x[mask].sum().item()
            l0_sum[s] += per_row_l0[mask].sum().item()
            n_rows[s] += int(mask.sum().item())

    assert cursor == ds.val_x.shape[0]

    per_source = {
        s: {"fve": fve(sq_err[s], sq_x[s]), "avg_l0": l0_sum[s] / max(n_rows[s], 1)}
        for s in sources
    }
    pooled = {
        "fve": fve(sum(sq_err.values()), sum(sq_x.values())),
        "avg_l0": sum(l0_sum.values()) / max(sum(n_rows.values()), 1),
    }
    num_dead = int(model.dead_feature_mask().sum().item())
    model.train()
    return pooled, per_source, num_dead


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--d-hidden", type=int, default=16384)
    parser.add_argument("--k", type=int, default=64)
    parser.add_argument(
        "--k-start",
        type=int,
        default=None,
        help="Initial k for the annealing schedule (defaults to 4x --k). "
        "Linearly decayed down to --k over --k-anneal-frac of total steps, "
        "then held at --k. See sae_model.py's K-ANNEALING docstring.",
    )
    parser.add_argument(
        "--k-anneal-frac",
        type=float,
        default=0.2,
        help="Fraction of total training steps over which k decays from "
        "--k-start to --k. Set to 0 to disable annealing (k fixed at --k).",
    )
    parser.add_argument("--auxk", type=int, default=256)
    parser.add_argument("--dead-tokens-threshold", type=int, default=400_000)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=4e-4)
    parser.add_argument("--warmup-frac", type=float, default=0.05)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument(
        "--natural-train-frac",
        type=float,
        default=0.0,
        help="Fraction of the 69 STRING-derived natural sequences "
        "(data.NATURAL_PARTIAL_SOURCE) to fold into training instead of "
        "holding out entirely. The 13 hand-picked natural_binders sequences "
        "are always 100%% held out regardless. Default 0.0 preserves the "
        "original all-held-out behavior. See data.py's load_dataset docstring.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from <output-dir>/latest.pt if it exists, continuing at the "
        "next epoch. Model/optimizer state are restored; the data split is "
        "reproduced from --seed (unchanged), so val stays val.",
    )
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Device: {device}")
    if device.type == "cpu":
        print("WARNING: no GPU/MPS device found, falling back to CPU (will be slow)")

    ds = load_dataset(
        args.data_dir,
        val_fraction=args.val_fraction,
        seed=args.seed,
        natural_train_frac=args.natural_train_frac,
    )
    d_model = ds.train_x.shape[1]
    mean = ds.mean.to(device)
    scale = ds.scale

    model = SparseAutoencoder(
        d_model=d_model,
        d_hidden=args.d_hidden,
        k=args.k,
        auxk=args.auxk,
        dead_tokens_threshold=args.dead_tokens_threshold,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    steps_per_epoch = ds.train_x.shape[0] // args.batch_size
    total_steps = steps_per_epoch * args.epochs
    warmup_steps = max(1, int(args.warmup_frac * total_steps))

    def lr_lambda(step):
        return min(1.0, step / warmup_steps)

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    k_start = args.k_start if args.k_start is not None else 4 * args.k
    k_anneal_steps = max(1, int(args.k_anneal_frac * total_steps))

    def current_k(step: int) -> int:
        if args.k_anneal_frac <= 0 or k_start <= args.k:
            return args.k
        frac = min(1.0, step / k_anneal_steps)
        return round(k_start + frac * (args.k - k_start))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    log_path = args.output_dir / "training_log.csv"
    log_fields = [
        "epoch", "train_mse", "train_auxk_loss", "val_fve_pooled", "val_avg_l0_pooled",
        "num_dead",
    ] + [f"val_fve_{s}" for s in sorted(set(ds.val_source.tolist()))] + [
        f"val_avg_l0_{s}" for s in sorted(set(ds.val_source.tolist()))
    ]
    if not (args.resume and log_path.is_file()):
        with open(log_path, "w", newline="") as f:
            csv.writer(f).writerow(log_fields)

    print(
        f"{steps_per_epoch} steps/epoch x {args.epochs} epochs = {total_steps} total steps "
        f"(warmup: {warmup_steps} steps)"
    )

    generator = torch.Generator().manual_seed(args.seed)
    best_val_fve = float("-inf")
    step = 0
    start_epoch = 0
    latest_path = args.output_dir / "latest.pt"
    if args.resume and latest_path.is_file():
        ckpt = torch.load(latest_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        if "optimizer_state_dict" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        step = int(ckpt.get("step", 0))
        start_epoch = int(ckpt["epoch"]) + 1
        best_val_fve = float(ckpt.get("val_fve", float("-inf")))
        for _ in range(step):
            scheduler.step()
        print(
            f"resumed from {latest_path}: epoch {start_epoch}, step {step}, "
            f"best_val_fve={best_val_fve:.4f}",
            flush=True,
        )

    for epoch in range(start_epoch, args.epochs):
        start = time.time()
        model.train()
        epoch_mse, epoch_auxk, n_batches = 0.0, 0.0, 0
        for batch in iter_batches(ds.train_x, args.batch_size, shuffle=True, generator=generator):
            x_proc = center_scale(batch.to(device), mean, scale)
            step_k = current_k(step)
            recon, auxk_recon, num_dead, _ = model(x_proc, k=step_k)
            mse_loss, auxk_loss = loss_fn(x_proc, recon, auxk_recon)
            loss = mse_loss + auxk_loss

            optimizer.zero_grad()
            loss.backward()
            model.norm_grad()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            model.norm_weights()
            scheduler.step()

            epoch_mse += mse_loss.item()
            epoch_auxk += auxk_loss.item()
            n_batches += 1
            step += 1
            if step % args.log_every == 0:
                print(
                    f"  step {step}/{total_steps}: mse={mse_loss.item():.4f} "
                    f"auxk={auxk_loss.item():.4f} num_dead={num_dead} k={step_k} "
                    f"lr={scheduler.get_last_lr()[0]:.2e}"
                )

        train_mse = epoch_mse / max(n_batches, 1)
        train_auxk = epoch_auxk / max(n_batches, 1)
        pooled, per_source, num_dead = validate(model, ds, mean, scale, args.batch_size, device)
        elapsed = time.time() - start

        per_source_str = ", ".join(f"{s}: fve={m['fve']:.4f} l0={m['avg_l0']:.1f}" for s, m in per_source.items())
        print(
            f"epoch {epoch}: train_mse={train_mse:.4f} train_auxk={train_auxk:.4f} "
            f"val_fve={pooled['fve']:.4f} val_l0={pooled['avg_l0']:.1f} num_dead={num_dead} "
            f"end_k={current_k(step - 1)} ({elapsed:.1f}s) -- {per_source_str}"
        )

        with open(log_path, "a", newline="") as f:
            row = [epoch, train_mse, train_auxk, pooled["fve"], pooled["avg_l0"], num_dead]
            row += [per_source[s]["fve"] for s in sorted(per_source)]
            row += [per_source[s]["avg_l0"] for s in sorted(per_source)]
            csv.writer(f).writerow(row)

        checkpoint = {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "step": step,
            # NOTE: keep this dict limited to SparseAutoencoder.__init__'s actual
            # params -- benchmark.py does SparseAutoencoder(**ckpt["config"]).
            # k_start/k_anneal_frac are training-only and already captured below
            # in "args" (and resolved k_start in "k_start_resolved").
            "config": {
                "d_model": d_model,
                "d_hidden": args.d_hidden,
                "k": args.k,
                "auxk": args.auxk,
                "dead_tokens_threshold": args.dead_tokens_threshold,
            },
            "k_start_resolved": k_start,
            "mean": ds.mean,
            "scale": scale,
            "train_protein_ids": ds.train_protein_ids,
            "val_protein_ids": ds.val_protein_ids,
            "args": vars(args),
            "epoch": epoch,
            "val_fve": pooled["fve"],
        }
        torch.save(checkpoint, args.output_dir / "latest.pt")
        if pooled["fve"] > best_val_fve:
            best_val_fve = pooled["fve"]
            torch.save(checkpoint, args.output_dir / "best.pt")
            print(f"  new best val_fve={best_val_fve:.4f}, saved best.pt")

    print(f"Done. Best val_fve={best_val_fve:.4f}. Checkpoints in {args.output_dir}")


if __name__ == "__main__":
    main()
