from __future__ import annotations

import argparse
import json
from contextlib import nullcontext
from pathlib import Path
from typing import Iterable

import numpy as np
import open_clip
import torch
from PIL import Image, ImageFile
from tqdm import tqdm

ImageFile.LOAD_TRUNCATED_IMAGES = True

IMAGE_EXTENSIONS = {".jpg", ".png"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path, help="Image directory or a single image file.")
    parser.add_argument("--output", required=True, type=Path, help="Directory that will receive .bin files.")
    parser.add_argument("--model", default="ViT-B-32", help="OpenCLIP model name.")
    parser.add_argument("--pretrained", default="laion2b_s34b_b79k", help="OpenCLIP pretrained weights.")
    parser.add_argument("--cache-dir", type=Path, help="Directory used to cache model weights.")
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"], help="Compute device.")
    parser.add_argument("--batch-size", default=128, type=int, help="Images per inference batch.")
    parser.add_argument("--recursive", action="store_true", help="Recursively scan input directory.")
    parser.add_argument("--normalize", action="store_true", help="L2-normalize embeddings before projection.")
    parser.add_argument("--amp", action="store_true", help="Use mixed precision on CUDA.")
    parser.add_argument("--projection-bits", default=128 * 8, type=int, help="Number of SimHash bits to write.")
    parser.add_argument("--projection-seed", default=42, type=int, help="Seed for deterministic projections.")
    parser.add_argument("--extensions", nargs="*", default=sorted(IMAGE_EXTENSIONS), help="Allowed image extensions.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing .bin files.")

    return parser.parse_args()


def list_images(input_path: Path, recursive: bool, extensions: Iterable[str]) -> list[Path]:
    allowed = {ext.lower() if ext.startswith(".") else f".{ext.lower()}" for ext in extensions}
    if input_path.is_file():
        return [input_path] if input_path.suffix.lower() in allowed else []

    pattern = "**/*" if recursive else "*"
    return sorted(path for path in input_path.glob(pattern) if path.is_file() and path.suffix.lower() in allowed)


def output_path_for(image_path: Path, input_path: Path, output_dir: Path) -> Path:
    if input_path.is_file():
        return output_dir / f"{image_path.name}.bin"

    relative_path = image_path.relative_to(input_path)
    return output_dir / relative_path.parent / f"{relative_path.name}.bin"


def resolve_device(requested_device: str) -> str:
    if requested_device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if requested_device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA was requested, but this PyTorch install cannot use CUDA.")
    return requested_device


def make_projection_matrix(embedding_dim: int, bits: int, seed: int, device: str) -> torch.Tensor:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    matrix = torch.randn((embedding_dim, bits), generator=generator, dtype=torch.float32)
    return matrix.to(device, non_blocking=True)


def load_batch(paths: list[Path], preprocess) -> tuple[torch.Tensor | None, list[Path], list[dict[str, str]]]:
    images = []
    loaded_paths = []
    failures = []

    for path in paths:
        try:
            with Image.open(path) as image:
                images.append(preprocess(image.convert("RGB")))
            loaded_paths.append(path)
        except Exception as exc:
            failures.append({"path": str(path), "error": f"{type(exc).__name__}: {exc}"})

    return (torch.stack(images) if images else None), loaded_paths, failures


def save_signature(image_path: Path, signature: np.ndarray, input_path: Path, output_dir: Path) -> dict[str, str | int]:
    bin_path = output_path_for(image_path, input_path, output_dir)
    bin_path.parent.mkdir(parents=True, exist_ok=True)
    bin_path.write_bytes(signature.tobytes())
    return {"path": str(image_path), "bin": str(bin_path), "bytes": int(signature.shape[0])}


def main() -> None:
    args = parse_args()
    input_path = args.input.expanduser().resolve()
    output_dir = args.output.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    device = resolve_device(args.device)
    all_paths = list_images(input_path, args.recursive, args.extensions)
    if not all_paths:
        raise SystemExit(f"No images found at {input_path}")

    paths = [path for path in all_paths if args.overwrite or not output_path_for(path, input_path, output_dir).exists()]
    if not paths:
        print(f"All .bin files already exist in {output_dir}")
        return

    cache_dir = args.cache_dir.expanduser().resolve() if args.cache_dir else None
    if cache_dir:
        cache_dir.mkdir(parents=True, exist_ok=True)

    model, _, preprocess = open_clip.create_model_and_transforms(
        args.model,
        pretrained=args.pretrained,
        device=device,
        cache_dir=str(cache_dir) if cache_dir else None,
    )
    model.eval()

    index_path = output_dir / "index.jsonl"
    failures_path = output_dir / "failures.jsonl"
    projection_matrix = None
    embedding_dim = 0
    embedded_count = 0
    failed_count = 0

    with (
        index_path.open("w", encoding="utf-8") as index_file,
        failures_path.open("w", encoding="utf-8") as failures_file,
    ):
        for start in tqdm(range(0, len(paths), args.batch_size), desc="SimHash"):
            batch_paths = paths[start:start + args.batch_size]
            images, loaded_paths, failures = load_batch(batch_paths, preprocess)

            for failure in failures:
                failed_count += 1
                failures_file.write(json.dumps(failure, ensure_ascii=False) + "\n")
            if images is None:
                continue

            images = images.to(device, non_blocking=True)
            amp_context = torch.autocast(device_type="cuda") if args.amp and device == "cuda" else nullcontext()

            with torch.inference_mode(), amp_context:
                embeddings = model.encode_image(images)
                if args.normalize:
                    embeddings = torch.nn.functional.normalize(embeddings, dim=-1)

                if projection_matrix is None:
                    embedding_dim = int(embeddings.shape[1])
                    projection_matrix = make_projection_matrix(
                        embedding_dim,
                        args.projection_bits,
                        args.projection_seed,
                        device,
                    )

                signatures = (embeddings.float() @ projection_matrix >= 0).cpu().numpy().astype(np.uint8)
                packed_signatures = np.packbits(signatures, axis=1, bitorder="big")

            for image_path, signature in zip(loaded_paths, packed_signatures):
                record = save_signature(image_path, signature, input_path, output_dir)
                index_file.write(json.dumps(record, ensure_ascii=False) + "\n")
                embedded_count += 1

    manifest = {
        "model": args.model,
        "pretrained": args.pretrained,
        "input": str(input_path),
        "output": str(output_dir),
        "image_count": len(all_paths),
        "embedded_count": embedded_count,
        "failed_count": failed_count,
        "embedding_dim": embedding_dim,
        "projection_bits": args.projection_bits,
        "projection_seed": args.projection_seed,
        "normalize": args.normalize,
        "device": device,
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Wrote {embedded_count} SimHash .bin files to {output_dir}")
    if failed_count:
        print(f"Skipped {failed_count} broken images. See {failures_path}")


if __name__ == "__main__":
    main()
