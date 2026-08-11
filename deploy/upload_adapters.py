"""
Upload LoRA Adapters to the Modal Volume
=========================================
One-time (or whenever you retrain) step that pushes the three LoRA adapters
from your machine into the ``t2s-model-weights`` Volume, where
``deploy/modal_app.py`` expects to find them.

The adapters live in Google Drive, not in git — see ``lora_weights/weights.txt``.
Download all three zips from that folder first; this script unzips them for you.

USAGE
-----
    # point at the directory holding the three downloaded .zip files
    python deploy/upload_adapters.py --local-dir ./lora_zips

    # or, if you already unzipped them yourself, pass explicit dirs
    python deploy/upload_adapters.py \
        --planner  ./lora_adapter_planner \
        --plan2sql ./lora_adapter_plan2sql \
        --single    ./lora_adapter_combined

    # verify what landed in the Volume
    modal volume ls t2s-model-weights adapters

RESULTING LAYOUT
----------------
    /models/adapters/planner/    adapter_config.json + adapter_model.safetensors
    /models/adapters/plan2sql/
    /models/adapters/single/

WHICH ZIP IS WHICH
------------------
Mapped from the ``ADAPTER_DIR`` each experiment notebook extracts into:

    lora_adapter.zip          -> planner    (text2plan  -> lora_adapter_planner)
    lora_adapter_plan2sql.zip -> plan2sql   (plan2sql   -> lora_adapter_plan2sql)
    lora_adapter_single.zip   -> single     (single_*   -> lora_adapter_combined)

The third mapping is inferred from directory naming, not from the archive
contents. If your results look wrong for the single-transformer endpoint,
this is the first thing to check.
"""

import argparse
import os
import shutil
import sys
import tempfile
import zipfile

import modal

VOLUME_NAME  = "t2s-model-weights"
ADAPTER_ROOT = "adapters"

# Adapter key -> candidate zip basenames, most specific first.  The browser's
# duplicate-download suffixes ("lora_adapter (1).zip") are tolerated by the
# stem match in _find_zip.
ZIP_HINTS = {
    "planner":  ["lora_adapter"],
    "plan2sql": ["lora_adapter_plan2sql"],
    "single":   ["lora_adapter_single", "lora_adapter_combined"],
}

# Files PEFT actually needs, plus tokenizer files the notebooks read from the
# adapter directory.  Anything else in the zip is skipped.
WANTED = {
    "adapter_config.json", "adapter_model.safetensors", "adapter_model.bin",
    "tokenizer_config.json", "tokenizer.json", "spiece.model",
    "special_tokens_map.json", "added_tokens.json", "generation_config.json",
}


def _stem(name: str) -> str:
    """'lora_adapter (1) (2).zip' -> 'lora_adapter'"""
    base = os.path.splitext(os.path.basename(name))[0]
    while base.endswith(")") and "(" in base:
        base = base[:base.rindex("(")].rstrip()
    return base.strip()


def _find_zip(local_dir: str, key: str) -> str | None:
    """Match a zip in local_dir to an adapter key, longest hint wins."""
    zips = [os.path.join(local_dir, f)
            for f in os.listdir(local_dir) if f.lower().endswith(".zip")]
    for hint in ZIP_HINTS[key]:
        for z in zips:
            if _stem(z) == hint:
                return z
    return None


def _flatten(src_dir: str) -> str:
    """
    Return the directory actually containing adapter_config.json.

    Some zips wrap everything in a top-level folder; PEFT needs the files at
    the root of the directory it is handed.
    """
    if os.path.isfile(os.path.join(src_dir, "adapter_config.json")):
        return src_dir
    for root, _dirs, files in os.walk(src_dir):
        if "adapter_config.json" in files:
            return root
    raise RuntimeError(f"no adapter_config.json anywhere under {src_dir}")


def _resolve(args) -> dict:
    """Build {adapter_key: local_directory} from the CLI arguments."""
    resolved, tmpdirs = {}, []

    explicit = {"planner": args.planner,
                "plan2sql": args.plan2sql,
                "single": args.single}

    for key, path in explicit.items():
        if path:
            if not os.path.isdir(path):
                sys.exit(f"not a directory: {path}")
            resolved[key] = _flatten(path)

    if args.local_dir:
        if not os.path.isdir(args.local_dir):
            sys.exit(f"not a directory: {args.local_dir}")
        for key in ZIP_HINTS:
            if key in resolved:
                continue
            z = _find_zip(args.local_dir, key)
            if not z:
                continue
            tmp = tempfile.mkdtemp(prefix=f"adapter_{key}_")
            tmpdirs.append(tmp)
            with zipfile.ZipFile(z) as zf:
                zf.extractall(tmp)
            print(f"  unzipped {os.path.basename(z)} -> {key}")
            resolved[key] = _flatten(tmp)

    return resolved, tmpdirs


def main():
    ap = argparse.ArgumentParser(
        description="Upload LoRA adapters into the Modal Volume.")
    ap.add_argument("--local-dir",
                    help="Directory holding the three downloaded .zip files")
    ap.add_argument("--planner",  help="Unzipped planner adapter directory")
    ap.add_argument("--plan2sql", help="Unzipped plan2sql adapter directory")
    ap.add_argument("--single",   help="Unzipped single-transformer directory")
    ap.add_argument("--force", action="store_true",
                    help="Overwrite adapters already present in the Volume")
    args = ap.parse_args()

    if not any([args.local_dir, args.planner, args.plan2sql, args.single]):
        ap.error("pass --local-dir, or at least one explicit adapter path")

    resolved, tmpdirs = _resolve(args)
    if not resolved:
        sys.exit("no adapters found — check --local-dir contents")

    missing = set(ZIP_HINTS) - set(resolved)
    if missing:
        print(f"\nwarning: no adapter matched for {sorted(missing)}. "
              f"Those endpoints will fail until uploaded.")

    volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)

    try:
        existing = {e.path.strip("/") for e in volume.listdir(ADAPTER_ROOT)}
    except Exception:
        existing = set()

    with volume.batch_upload(force=args.force) as batch:
        for key, src in sorted(resolved.items()):
            remote = f"{ADAPTER_ROOT}/{key}"
            if remote in existing and not args.force:
                print(f"  {key}: already in Volume, skipping (--force to replace)")
                continue

            n = 0
            for fname in sorted(os.listdir(src)):
                if fname not in WANTED:
                    continue
                fpath = os.path.join(src, fname)
                if not os.path.isfile(fpath):
                    continue
                batch.put_file(fpath, f"{remote}/{fname}")
                n += 1
            if n == 0:
                sys.exit(f"{key}: found no usable adapter files in {src}")
            print(f"  {key}: uploading {n} files -> /models/{remote}/")

    print(f"\ndone. verify with:  modal volume ls {VOLUME_NAME} {ADAPTER_ROOT}")

    for t in tmpdirs:
        shutil.rmtree(t, ignore_errors=True)


if __name__ == "__main__":
    main()
