import argparse
import io
import random
import re
import time
import urllib.error
import urllib.request
from pathlib import Path

import pandas as pd
from huggingface_hub import hf_hub_download
from PIL import Image, ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True

REPO = "Spawning/PD12M"
DEFAULT_OUT = Path.home() / "ml_llm_pi/MouseSD-1/mousesd_data"
METADATA_DIR = Path.home() / ".cache/mousesd_pd12m"
SHARD_COUNT = 91

USER_AGENT = "MouseSD-1-PD12M-Downloader/1.0"


def get_metadata(shard):
    name = f"pd12m.{shard:03d}.parquet"

    path = hf_hub_download(
        repo_id=REPO,
        repo_type="dataset",
        filename=f"metadata/{name}",
        cache_dir=str(METADATA_DIR),
    )

    return pd.read_parquet(path)


def clean_caption(text):
    text = str(text).strip().lower()
    text = re.sub(r"\s+", " ", text)
    return text[:500]


def clean_url(value):
    if not isinstance(value, str):
        return None

    value = value.strip()

    # Handle URLs displayed as Markdown links.
    match = re.match(r"\[.*?\]\((https?://.*?)\)", value)

    if match:
        value = match.group(1)

    # Handle URLs wrapped in angle brackets.
    if value.startswith("<") and value.endswith(">"):
        value = value[1:-1]

    if not value.startswith(("http://", "https://")):
        return None

    return value


def make_filename(image_id, index):
    if pd.isna(image_id):
        image_id = index

    filename = f"pd12m_{image_id}"
    return re.sub(r"[^a-zA-Z0-9_-]", "_", str(filename))


def download_image(row, output, res, index):
    url = clean_url(row.get("url"))

    if not url:
        return False

    caption = clean_caption(row.get("caption", ""))

    if not caption:
        return False

    filename = make_filename(row.get("id"), index)

    image_path = output / f"{filename}.jpg"
    caption_path = output / f"{filename}.txt"

    # Already downloaded successfully.
    if image_path.exists() and caption_path.exists():
        return True

    try:
        request = urllib.request.Request(
            url,
            headers={"User-Agent": USER_AGENT},
        )

        with urllib.request.urlopen(request, timeout=60) as response:
            raw = response.read()

        with Image.open(io.BytesIO(raw)) as im:
            im = im.convert("RGB")

            width, height = im.size
            side = min(width, height)

            left = (width - side) // 2
            top = (height - side) // 2

            im = im.crop((
                left,
                top,
                left + side,
                top + side,
            ))

            im = im.resize(
                (res, res),
                Image.Resampling.LANCZOS,
            )

            # Write temporary files first to avoid incomplete outputs.
            temp_image = output / f"{filename}.tmp.jpg"
            temp_caption = output / f"{filename}.tmp.txt"

            im.save(
                temp_image,
                "JPEG",
                quality=92,
                optimize=True,
            )

            temp_caption.write_text(
                caption,
                encoding="utf-8",
            )

            temp_image.replace(image_path)
            temp_caption.replace(caption_path)

        return True

    except Exception:
        # Remove incomplete temporary files.
        for path in (
            output / f"{filename}.tmp.jpg",
            output / f"{filename}.tmp.txt",
        ):
            try:
                path.unlink(missing_ok=True)
            except OSError:
                pass

        return False


def count_existing(output):
    count = 0

    for image_path in output.glob("pd12m_*.jpg"):
        caption_path = image_path.with_suffix(".txt")

        if caption_path.exists():
            count += 1

    return count


def main():
    parser = argparse.ArgumentParser(
        description="Download PD12M images for MouseSD-1"
    )

    parser.add_argument("--count", type=int, default=100)
    parser.add_argument("--res", type=int, default=64)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--shards", type=int, default=10)
    parser.add_argument("--retry", type=int, default=2)

    args = parser.parse_args()

    if args.count < 1:
        parser.error("--count must be positive")

    if args.res < 1:
        parser.error("--res must be positive")

    if not 1 <= args.shards <= SHARD_COUNT:
        parser.error(f"--shards must be between 1 and {SHARD_COUNT}")

    if args.retry < 0:
        parser.error("--retry cannot be negative")

    args.out.mkdir(parents=True, exist_ok=True)

    print("MouseSD-1 PD12M Downloader")
    print("==========================")
    print(f"Target images: {args.count:,}")
    print(f"Resolution: {args.res}x{args.res}")
    print(f"Output: {args.out}")
    print(f"Metadata shards: {args.shards}")
    print()

    successful = count_existing(args.out)
    failed = 0
    skipped = 0
    started = time.time()

    print(f"Already downloaded: {successful:,}")

    if successful >= args.count:
        print("Target already reached!")
        return

    # Process one metadata shard at a time to limit memory usage.
    for shard in range(args.shards):

        if successful >= args.count:
            break

        print(
            f"\nLoading metadata shard "
            f"{shard + 1}/{args.shards}..."
        )

        try:
            data = get_metadata(shard)

        except Exception as e:
            print(f"Metadata shard failed: {e}")
            continue

        required = {"id", "url", "caption"}
        missing = required - set(data.columns)

        if missing:
            print(
                f"Skipping shard: missing columns "
                f"{sorted(missing)}"
            )
            continue

        data = data.dropna(
            subset=["url", "caption"]
        )

        data = data[
            data["caption"].astype(str).str.strip().ne("")
        ]

        data = data.drop_duplicates(subset=["id"])

        # Shuffle each shard independently.
        data = data.sample(
            frac=1,
            random_state=args.seed + shard,
        ).reset_index(drop=True)

        print(f"Candidates in shard: {len(data):,}")

        for index, row in data.iterrows():

            if successful >= args.count:
                break

            filename = make_filename(row.get("id"), index)

            image_path = args.out / f"{filename}.jpg"
            caption_path = args.out / f"{filename}.txt"

            if image_path.exists() and caption_path.exists():
                skipped += 1
                continue

            ok = False

            for attempt in range(args.retry + 1):

                if download_image(
                    row,
                    args.out,
                    args.res,
                    f"{shard}_{index}",
                ):
                    ok = True
                    break

                if attempt < args.retry:
                    time.sleep(min(2 ** attempt, 15))

            if ok:
                successful += 1
            else:
                failed += 1

            elapsed = time.time() - started

            rate = (
                successful / elapsed
                if elapsed > 0
                else 0
            )

            print(
                f"\rDownloaded: {successful:,}/{args.count:,} "
                f"| Failed: {failed:,} "
                f"| Skipped: {skipped:,} "
                f"| Rate: {rate:.1f}/s "
                f"| Time: {elapsed / 60:.1f} min",
                end="",
                flush=True,
            )

        del data

    elapsed = time.time() - started

    print("\n\nFinished!")
    print(f"Total images: {count_existing(args.out):,}")
    print(f"Failed downloads: {failed:,}")
    print(f"Skipped existing: {skipped:,}")
    print(f"Elapsed time: {elapsed / 60:.1f} minutes")
    print(f"Output folder: {args.out}")


if __name__ == "__main__":
    main()
