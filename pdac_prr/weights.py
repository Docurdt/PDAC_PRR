"""Verify or install the exact model checkpoints listed in the release manifest."""
import argparse
import hashlib
import json
from pathlib import Path
import tempfile
import zipfile


def manifest_files(bundle, families=None):
    bundle = Path(bundle).resolve()
    manifest = json.loads((bundle / "model_manifest.json").read_text())
    selected = families or list(manifest["models"])
    files = {}
    for family in selected:
        if family not in manifest["models"]:
            raise ValueError(f"Unknown model: {family}")
        for item in manifest["models"][family]["expected_checkpoints"]:
            if Path(item["name"]).name != item["name"]:
                raise ValueError("Invalid checkpoint filename in manifest")
            files[f"weights/{family}/{item['name']}"] = item
    return files


def digest_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify(bundle, families=None):
    bundle = Path(bundle).resolve()
    result = {"verified": [], "missing": [], "invalid": []}
    for relative, item in manifest_files(bundle, families).items():
        path = bundle / relative
        if not path.is_file():
            result["missing"].append(relative)
        elif path.stat().st_size != item["size"] or digest_file(path) != item["sha256"]:
            result["invalid"].append(relative)
        else:
            result["verified"].append(relative)
    result["complete"] = not result["missing"] and not result["invalid"]
    return result


def install(archive, bundle):
    """Stream only whitelisted weight files; verify their hashes before installing.

    Valid existing files are retained. An invalid existing file must be resolved
    by the user; installation does not overwrite it.
    """
    bundle = Path(bundle).resolve()
    allowed = manifest_files(bundle)
    installed, retained = [], []
    with zipfile.ZipFile(archive) as source:
        entries = [entry for entry in source.infolist() if not entry.is_dir()]
        if not entries or len({entry.filename for entry in entries}) != len(entries):
            raise ValueError("Empty archive or duplicate archive entries")
        for entry in entries:
            if entry.filename not in allowed:
                raise ValueError(f"Unexpected archive entry: {entry.filename}")
            if entry.file_size != allowed[entry.filename]["size"]:
                raise ValueError(f"Checkpoint size mismatch: {entry.filename}")
        for entry in entries:
            item = allowed[entry.filename]
            destination = bundle / entry.filename
            if not destination.resolve().is_relative_to(bundle):
                raise ValueError("Checkpoint destination is outside the bundle")
            if destination.exists():
                if destination.stat().st_size != item["size"] or digest_file(destination) != item["sha256"]:
                    raise FileExistsError(f"Invalid existing file; resolve it first: {destination}")
                retained.append(entry.filename)
                continue
            destination.parent.mkdir(parents=True, exist_ok=True)
            temporary = None
            try:
                digest = hashlib.sha256()
                with tempfile.NamedTemporaryFile(dir=destination.parent, suffix=".part", delete=False) as target:
                    temporary = Path(target.name)
                    with source.open(entry) as stream:
                        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                            target.write(chunk)
                            digest.update(chunk)
                if digest.hexdigest() != item["sha256"]:
                    raise ValueError(f"Checkpoint SHA-256 mismatch: {entry.filename}")
                # Exclusive destination creation avoids replacing another file.
                with destination.open("xb") as output, temporary.open("rb") as stream:
                    for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                        output.write(chunk)
                installed.append(entry.filename)
            finally:
                if temporary is not None:
                    temporary.unlink(missing_ok=True)
    return {"installed": installed, "retained": retained}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, default=Path.cwd())
    parser.add_argument("--model", nargs="+", choices=["esc", "prrpath", "prrpath_text", "prrfull"])
    parser.add_argument("--install", type=Path, metavar="ARCHIVE.zip")
    args = parser.parse_args()
    try:
        if args.install:
            result = install(args.install, args.bundle)
        else:
            result = verify(args.bundle, args.model)
        print(json.dumps(result, indent=2))
        if not args.install and not result["complete"]:
            raise SystemExit(1)
    except (ValueError, FileExistsError, FileNotFoundError, zipfile.BadZipFile) as exc:
        parser.exit(1, f"{exc}\n")


if __name__ == "__main__":
    main()
