import argparse
import base64
import shutil
import zlib
from pathlib import Path

def compress_json_to_b64(src_path: Path, dst_path: Path, level: int = 9) -> None:
    raw = src_path.read_bytes()
    compressed = zlib.compress(raw, level=level)
    b64 = base64.b64encode(compressed)
    dst_path.write_bytes(b64)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src",  default="results", help="Source folder containing json/txt/etc.")
    parser.add_argument("--dst", default="results_compressed", help="Destination folder (will be created).")
    parser.add_argument("--level", type=int, default=9, help="zlib compression level: 1-9")
    parser.add_argument("--json_ext", default=".json", help="JSON extension to compress (default: .json)")
    parser.add_argument("--out_suffix", default=".zlib.b64", help="Suffix appended to compressed JSON (default: .zlib.b64)")
    args = parser.parse_args()

    src_dir = Path(args.src).expanduser().resolve()
    dst_dir = Path(args.dst).expanduser().resolve()

    if not src_dir.is_dir():
        raise SystemExit(f"Source is not a directory: {src_dir}")

    dst_dir.mkdir(parents=True, exist_ok=True)

    json_ext = args.json_ext
    out_suffix = args.out_suffix

    # Walk all files; preserve relative paths
    for path in src_dir.rglob("*"):
        if path.is_dir():
            continue

        rel = path.relative_to(src_dir)
        out_parent = dst_dir / rel.parent
        out_parent.mkdir(parents=True, exist_ok=True)

        if path.suffix.lower() == json_ext.lower():
            # e.g. foo.json -> foo.json.zlib.b64
            out_path = out_parent / (path.name + out_suffix)
            compress_json_to_b64(path, out_path, level=args.level)
        else:
            # copy other files as-is (txt, error logs, etc.)
            out_path = dst_dir / rel
            shutil.copy2(path, out_path)

    print(f"Done. Output in: {dst_dir}")

if __name__ == "__main__":
    main()
