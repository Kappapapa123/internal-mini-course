"""Assemble and upload the notebook-07 Hugging Face model repo.

Run this **on Fir** (the checkpoints are there and are 820 MB each; downloading them to a
laptop first would double the transfer). It stages a directory in the layout nnU-Net's
``$nnUNet_results`` expects, verifies the sha256 sums against the values recorded in
``MODEL_CARD.md``, and uploads.

Usage on Fir, from a login node (this is network I/O, not compute, so no job needed)::

    module load python/3.11
    virtualenv --no-download ~/hf_upload_env && source ~/hf_upload_env/bin/activate
    pip install --no-index huggingface_hub
    huggingface-cli login          # paste a WRITE token from hf.co/settings/tokens

    python publish_checkpoints.py --repo-id <your-hf-username>/<repo-name> \
        --model-card MODEL_CARD.md --dry-run
    python publish_checkpoints.py --repo-id <your-hf-username>/<repo-name> \
        --model-card MODEL_CARD.md

``--dry-run`` stages and verifies everything, printing what would be uploaded, and touches
no network. Do that first.

The mirrored cases are picked from fold 0's **validation** split, so they are cases neither
checkpoint trained on. ``--num-cases`` controls how many (default 2); the script prefers the
smallest files so the download stays quick.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path

RESULTS = Path("/scratch/kcuoft/flare_pancancer/nnUNet_results/Dataset591_liver_lesions")
RAW = Path("/scratch/kcuoft/flare_pancancer/nnUNet_raw/Dataset591_liver_lesions")
PREPROCESSED = Path("/scratch/kcuoft/flare_pancancer/nnUNet_preprocessed/Dataset591_liver_lesions")

DATASET = "Dataset591_liver_lesions"
PLANS = "nnUNetResEncUNetL1x1x1_Plans_znorm_bs24_mig_bs1"
TRAINERS = ("nnUNetTrainer_plain1e3_wandb", "nnUNetTrainer_warmup1e3_wandb")

# Recorded in MODEL_CARD.md; verified on Fir 2026-07-30. A mismatch means the wrong run
# directory (note the sibling mig_bs2 dirs) or a corrupted copy — stop rather than publish.
EXPECTED_SHA256 = {
    "nnUNetTrainer_plain1e3_wandb": "91dd81f0d738885bcb8bf0cf9bb6d839b4012a975902614b256bc9ff42539168",
    "nnUNetTrainer_warmup1e3_wandb": "7ae5546dd1c95cb9fed93bc4719c2a6e0643ff5bbdba119c3ac67101841abf1e",
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(8 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def pick_cases(n: int) -> list[str]:
    """The n smallest fold-0 validation cases, so the mirror stays small."""
    splits = json.loads((PREPROCESSED / "splits_final.json").read_text())
    val = splits[0]["val"]
    sized = []
    for case in val:
        img, lab = RAW / "imagesTr" / f"{case}_0000.nii.gz", RAW / "labelsTr" / f"{case}.nii.gz"
        if img.exists() and lab.exists():
            sized.append((img.stat().st_size + lab.stat().st_size, case))
    if len(sized) < n:
        sys.exit(f"only {len(sized)} of {len(val)} validation cases found under {RAW}")
    sized.sort()
    return [case for _, case in sized[:n]]


def stage(staging: Path, num_cases: int) -> list[str]:
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)

    for trainer in TRAINERS:
        src = RESULTS / f"{trainer}__{PLANS}__3d_fullres"
        if not src.is_dir():
            sys.exit(f"missing run directory: {src}")
        dst = staging / DATASET / src.name
        (dst / "fold_0").mkdir(parents=True)

        for meta in ("plans.json", "dataset.json"):
            shutil.copy2(src / meta, dst / meta)

        ckpt = src / "fold_0" / "checkpoint_final.pth"
        print(f"hashing {ckpt} ({ckpt.stat().st_size:,} B) ...", flush=True)
        got = sha256(ckpt)
        if got != EXPECTED_SHA256[trainer]:
            sys.exit(f"sha256 mismatch for {trainer}\n  expected {EXPECTED_SHA256[trainer]}\n  got      {got}")
        print(f"  sha256 OK  {got}")
        shutil.copy2(ckpt, dst / "fold_0" / "checkpoint_final.pth")

        # nnU-Net writes the validation scores here; ship them as the provenance for the
        # Dice numbers quoted in the card and in the notebook.
        summary = src / "fold_0" / "validation" / "summary.json"
        if summary.exists():
            shutil.copy2(summary, dst / "fold_0" / "validation_summary.json")
            dice = json.loads(summary.read_text())["foreground_mean"]["Dice"]
            print(f"  {trainer}: liver-lesion Dice {dice:.7f}")

    cases = pick_cases(num_cases)
    (staging / "cases" / "images").mkdir(parents=True)
    (staging / "cases" / "labels").mkdir(parents=True)
    for case in cases:
        shutil.copy2(RAW / "imagesTr" / f"{case}_0000.nii.gz",
                     staging / "cases" / "images" / f"{case}_0000.nii.gz")
        shutil.copy2(RAW / "labelsTr" / f"{case}.nii.gz",
                     staging / "cases" / "labels" / f"{case}.nii.gz")
    print(f"mirrored {len(cases)} fold-0 validation cases: {cases}")
    return cases


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--repo-id", required=True, help="e.g. your-username/nnunet-liver-lesion-lr-ablation")
    ap.add_argument("--model-card", required=True, type=Path,
                    help="path to MODEL_CARD.md; uploaded as the repo's README.md")
    ap.add_argument("--staging", type=Path, default=Path.home() / "hf_staging_nb07")
    ap.add_argument("--num-cases", type=int, default=2)
    ap.add_argument("--private", action="store_true", help="create the repo private (flip it public later)")
    ap.add_argument("--dry-run", action="store_true", help="stage and verify only; no network")
    ap.add_argument("--card-only", action="store_true",
                    help="upload just the model card as README.md; skips staging and the 820 MB files")
    args = ap.parse_args()

    if not args.model_card.is_file():
        sys.exit(f"model card not found: {args.model_card}")

    if args.card_only:
        from huggingface_hub import HfApi

        HfApi().upload_file(path_or_fileobj=str(args.model_card), path_in_repo="README.md",
                            repo_id=args.repo_id, repo_type="model")
        print(f"card updated: https://huggingface.co/{args.repo_id}")
        return

    cases = stage(args.staging, args.num_cases)
    shutil.copy2(args.model_card, args.staging / "README.md")

    total = sum(p.stat().st_size for p in args.staging.rglob("*") if p.is_file())
    print(f"\nstaged {args.staging}  ({total / 1e9:.2f} GB)")
    for p in sorted(args.staging.rglob("*")):
        if p.is_file():
            print(f"  {p.relative_to(args.staging).as_posix():70s} {p.stat().st_size:>12,} B")

    if args.dry_run:
        print("\n--dry-run: nothing uploaded. Re-run without it to publish.")
        return

    from huggingface_hub import HfApi

    api = HfApi()
    api.create_repo(args.repo_id, repo_type="model", private=args.private, exist_ok=True)
    print(f"\nuploading to {args.repo_id} ... (820 MB x2, this takes a while)")
    api.upload_folder(folder_path=str(args.staging), repo_id=args.repo_id, repo_type="model")

    print(f"\ndone: https://huggingface.co/{args.repo_id}")
    print("\nNext: set CKPT_SOURCE in scripts/build_notebooks.py nb07() to")
    print(f"    CKPT_SOURCE = '{args.repo_id}'")
    print(f"and note the mirrored cases: {cases}")


if __name__ == "__main__":
    main()
