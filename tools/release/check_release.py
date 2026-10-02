#!/usr/bin/env python3
"""Check the release catalog, local links, model hashes and publication scope."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[2]


def sha256(path):
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(2**20), b""):
            result.update(block)
    return result.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", action="store_true", help="Require materialized Git LFS weights")
    args = parser.parse_args()
    catalog_path = ROOT / "data/releases/2026-10-02/catalog.json"
    catalog = json.loads(catalog_path.read_text())
    assert not catalog["trace_bodies_published"] and not catalog["annotation_bodies_published"]
    assert catalog["download_url"] is None
    seen = set()
    for row in catalog["runs"]:
        assert row["run_id"] not in seen
        seen.add(row["run_id"])
        config = json.loads((catalog_path.parent / row["config"]).read_text())
        assert config["run_id"] == row["run_id"] and config["trace_count"] == row["trace_count"]
        assert config["publication_status"] == "structure_only" and config["download_url"] is None
        assert config["benchmark"] and config["solver"]["model"]
    manifest = json.loads((ROOT / "models/MANIFEST.json").read_text())
    pointers = 0
    for relative, item in manifest["artifacts"].items():
        path = ROOT / relative
        assert path.is_file(), relative
        with path.open("rb") as stream:
            prefix = stream.read(200)
        if prefix.startswith(b"version https://git-lfs.github.com/spec/v1\n"):
            assert not args.weights, f"Run git lfs pull: {relative}"
            assert ("oid sha256:" + item["sha256"]).encode() in prefix, relative
            assert ("size " + str(item["bytes"])).encode() in prefix, relative
            pointers += 1
        else:
            assert path.stat().st_size == item["bytes"] and sha256(path) == item["sha256"], relative
    for path in (ROOT / "configs/gate").glob("*.json"):
        config = json.loads(path.read_text())
        assert config["weights"] in manifest["artifacts"], path.name
        assert not Path(config["weights"]).is_absolute(), path.name
    tracked = subprocess.check_output(["git", "-C", str(ROOT), "ls-files", "-z"]).decode().split("\0")
    assert not any(p.startswith(("data/corpora/", "data/archives/", "outputs/")) for p in tracked)
    print(json.dumps({"runs": len(seen), "model_artifacts": len(manifest["artifacts"]),
                      "lfs_pointers_without_local_weights": pointers, "trace_bodies_tracked": False}))


if __name__ == "__main__":
    main()
