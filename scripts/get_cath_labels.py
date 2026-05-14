#!/usr/bin/env python3
"""
Get CATH labels for all 5,399 mdCATH proteins.

Methods tried in order (first success wins):
  1. mdcath_source.h5 root-level attrs or domain-level metadata
  2. CATH domain list flat file (single HTTP GET, ~10 MB)
  3. CATH REST API (parallel requests for any remaining unlabeled domains)

Output: data/mdcath_metadata.csv
  Columns: domain_id, cath_class, architecture, topology, homology, n_residues, pdb_id

Usage:
    python scripts/get_cath_labels.py \
        --out data/mdcath_metadata.csv
"""
import csv
import io
import json
import sys
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import click
import requests


CATH_FLATFILE_URLS = [
    "http://download.cathdb.info/cath/releases/latest-release/cath-classification-data/cath-domain-list.txt",
    "https://ftp.ebi.ac.uk/pub/databases/cath/releases/latest-release/cath-classification-data/cath-domain-list.txt",
]
CATH_REST_URL = "http://www.cathdb.info/version/v4_3_0/api/rest/domain_summary/{domain_id}"
HF_DOMAINS_URL = "https://huggingface.co/datasets/compsciencelab/mdCATH/resolve/main/mdCATH_domains.txt"


def load_mdcath_domain_ids(hf_domains_path: str | None = None) -> list[str]:
    """Load domain IDs from local path or HuggingFace."""
    if hf_domains_path and Path(hf_domains_path).exists():
        with open(hf_domains_path) as f:
            return [l.strip() for l in f if l.strip()]

    # Try /tmp (already downloaded by previous step)
    tmp_path = Path("/tmp/mdCATH_domains.txt")
    if tmp_path.exists():
        click.echo("  Using cached /tmp/mdCATH_domains.txt")
        with open(tmp_path) as f:
            return [l.strip() for l in f if l.strip()]

    # Download from HuggingFace
    click.echo("  Downloading mdCATH_domains.txt from HuggingFace...")
    from huggingface_hub import hf_hub_download
    local = hf_hub_download(
        "compsciencelab/mdCATH", "mdCATH_domains.txt",
        repo_type="dataset", local_dir="/tmp"
    )
    with open(local) as f:
        return [l.strip() for l in f if l.strip()]


def method_h5_source(domain_ids: list[str]) -> dict[str, dict]:
    """
    Method 1: Check mdcath_source.h5 root attrs.
    Returns dict: domain_id -> {cath_class, architecture, topology, homology, n_residues}
    """
    click.echo("Method 1: Checking mdcath_source.h5...")
    results = {}

    # Try to find local copy or download small header
    h5_path = Path("/tmp/mdcath_source.h5")
    if not h5_path.exists():
        click.echo("  Downloading mdcath_source.h5 (may be large)...")
        try:
            from huggingface_hub import hf_hub_download
            local = hf_hub_download(
                "compsciencelab/mdCATH", "mdcath_source.h5",
                repo_type="dataset", local_dir="/tmp"
            )
            h5_path = Path(local)
        except Exception as e:
            click.echo(f"  H5 download failed: {e}")
            return results

    try:
        import h5py
        with h5py.File(str(h5_path), "r") as f:
            # Check root attrs
            root_attrs = dict(f.attrs)
            click.echo(f"  Root attrs: {list(root_attrs.keys())}")

            # Check if any domain has CATH attrs
            sample_domains = list(f.keys())[:3]
            for d in sample_domains:
                if d in domain_ids or d in {x for x in domain_ids[:10]}:
                    domain_attrs = dict(f[d].attrs) if hasattr(f[d], 'attrs') else {}
                    click.echo(f"  Sample domain '{d}' attrs: {list(domain_attrs.keys())}")
                    if "cath_class" in domain_attrs or "CATH" in str(domain_attrs):
                        click.echo("  Found CATH attrs! Extracting all...")
                        for domain_id in domain_ids:
                            if domain_id in f:
                                attrs = dict(f[domain_id].attrs)
                                if "cath_class" in attrs:
                                    results[domain_id] = {
                                        "cath_class": int(attrs["cath_class"]),
                                        "architecture": int(attrs.get("architecture", 0)),
                                        "topology": int(attrs.get("topology", 0)),
                                        "homology": int(attrs.get("homology", 0)),
                                        "n_residues": int(attrs.get("n_residues", 0)),
                                    }
                        click.echo(f"  Extracted {len(results)} labels from H5")
                        return results
                    break

            click.echo("  No CATH attrs found in H5. Moving to Method 2.")
    except Exception as e:
        click.echo(f"  H5 read error: {e}")

    return results


def method_flatfile(domain_ids: list[str]) -> dict[str, dict]:
    """
    Method 2: Download CATH domain list flat file and cross-reference.

    File format (space-separated, lines starting with # are comments):
      domain_id class arch topo homol s35 s60 s95 s100 s100count length resolution

    Returns dict: domain_id -> {cath_class, architecture, topology, homology, n_residues}
    """
    click.echo("Method 2: Downloading CATH domain list flat file...")
    domain_id_set = set(domain_ids)
    results = {}

    for url in CATH_FLATFILE_URLS:
        click.echo(f"  Trying: {url}")
        try:
            resp = requests.get(url, timeout=60, stream=True)
            if resp.status_code != 200:
                click.echo(f"  HTTP {resp.status_code}, trying next URL...")
                continue

            content = resp.text
            lines_parsed = 0
            for line in content.splitlines():
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                parts = line.split()
                if len(parts) < 11:
                    continue
                domain_id = parts[0]
                if domain_id in domain_id_set:
                    try:
                        results[domain_id] = {
                            "cath_class": int(parts[1]),
                            "architecture": int(parts[2]),
                            "topology": int(parts[3]),
                            "homology": int(parts[4]),
                            "n_residues": int(parts[10]),
                        }
                    except (ValueError, IndexError):
                        pass
                lines_parsed += 1

            click.echo(f"  Parsed {lines_parsed} CATH entries, matched {len(results)}/{len(domain_ids)} mdCATH domains")
            if results:
                return results

        except Exception as e:
            click.echo(f"  Error with {url}: {e}")
            continue

    click.echo("  Both flat file URLs failed or returned 0 matches.")
    return results


def _fetch_cath_rest(domain_id: str, session: requests.Session, retries: int = 3) -> tuple[str, dict | None]:
    """Fetch CATH classification for one domain via REST API."""
    url = CATH_REST_URL.format(domain_id=domain_id)
    for attempt in range(retries):
        try:
            resp = session.get(url, timeout=10)
            if resp.status_code == 200:
                data = resp.json()
                # Response structure: {"data": {"domain": {...}, "cath_node_id": "1.10.10.10", ...}}
                # Try to extract CATH classification
                cath_id = None
                n_res = 0

                # Try common response patterns
                if "data" in data:
                    d = data["data"]
                    # Pattern: cath_node_id = "1.10.10.10"
                    if "cath_node_id" in d:
                        cath_id = d["cath_node_id"]
                    elif "cathCode" in d:
                        cath_id = d["cathCode"]
                    n_res = d.get("length", d.get("n_residues", 0))

                if cath_id:
                    parts = str(cath_id).split(".")
                    return domain_id, {
                        "cath_class": int(parts[0]) if len(parts) > 0 else 0,
                        "architecture": int(parts[1]) if len(parts) > 1 else 0,
                        "topology": int(parts[2]) if len(parts) > 2 else 0,
                        "homology": int(parts[3]) if len(parts) > 3 else 0,
                        "n_residues": int(n_res) if n_res else 0,
                    }
                return domain_id, None

            elif resp.status_code == 404:
                return domain_id, None  # Domain not in CATH
            elif resp.status_code == 429:
                time.sleep(2 ** attempt)
                continue
        except Exception:
            if attempt < retries - 1:
                time.sleep(1)
    return domain_id, None


def method_rest_api(domain_ids: list[str], workers: int = 20) -> dict[str, dict]:
    """
    Method 3: CATH REST API for domains not found via flat file.
    Parallel requests with rate limiting.
    """
    click.echo(f"Method 3: CATH REST API for {len(domain_ids)} remaining domains ({workers} workers)...")
    results = {}
    session = requests.Session()
    session.headers.update({"Accept": "application/json"})

    done = 0
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_fetch_cath_rest, d, session): d for d in domain_ids}
        for fut in as_completed(futures):
            domain_id, info = fut.result()
            done += 1
            if info:
                results[domain_id] = info
            if done % 100 == 0:
                elapsed = time.time() - t0
                rate = done / elapsed
                remaining = (len(domain_ids) - done) / rate if rate > 0 else 0
                click.echo(f"  [{done}/{len(domain_ids)}] {len(results)} found, ETA {remaining:.0f}s")

    click.echo(f"  REST API: {len(results)}/{len(domain_ids)} found")
    return results


@click.command()
@click.option("--out", type=click.Path(), default="data/mdcath_metadata.csv",
              help="Output CSV path")
@click.option("--domains_file", type=click.Path(), default=None,
              help="Local mdCATH_domains.txt (optional, downloads from HF if missing)")
@click.option("--skip_h5", is_flag=True, help="Skip Method 1 (H5 source file check)")
@click.option("--rest_workers", type=int, default=20,
              help="Parallel workers for REST API fallback")
def main(out: str, domains_file: str | None, skip_h5: bool, rest_workers: int):
    out_path = Path(out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Load domain IDs
    click.echo("Loading mdCATH domain IDs...")
    domain_ids = load_mdcath_domain_ids(domains_file)
    click.echo(f"  {len(domain_ids)} domains")

    results = {}

    # Method 1: H5 source file
    if not skip_h5:
        results.update(method_h5_source(domain_ids))
        if len(results) >= len(domain_ids) * 0.9:
            click.echo(f"Method 1 sufficient ({len(results)} labels). Skipping Methods 2+3.")
        else:
            if results:
                click.echo(f"Method 1 partial: {len(results)} labels. Continuing...")
            # Method 2: Flat file
            remaining = [d for d in domain_ids if d not in results]
            flatfile_results = method_flatfile(remaining if results else domain_ids)
            results.update(flatfile_results)
    else:
        flatfile_results = method_flatfile(domain_ids)
        results.update(flatfile_results)

    # Method 3: REST API for any remaining
    remaining = [d for d in domain_ids if d not in results]
    if remaining:
        click.echo(f"\n{len(remaining)} domains still unlabeled. Using REST API...")
        if len(remaining) > 2000:
            click.echo(f"  WARNING: {len(remaining)} REST calls may take {len(remaining)//rest_workers//2:.0f}s")
        rest_results = method_rest_api(remaining, workers=rest_workers)
        results.update(rest_results)

    # Final coverage
    still_missing = [d for d in domain_ids if d not in results]
    click.echo(f"\nFinal: {len(results)}/{len(domain_ids)} labeled ({len(still_missing)} unlabeled)")

    if still_missing:
        click.echo(f"  Unlabeled (first 10): {still_missing[:10]}")
        # Add them with class=0 (unknown) so they can be filtered
        for d in still_missing:
            results[d] = {
                "cath_class": 0,
                "architecture": 0,
                "topology": 0,
                "homology": 0,
                "n_residues": 0,
            }

    # Write CSV
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "domain_id", "cath_class", "architecture", "topology",
            "homology", "n_residues", "pdb_id"
        ])
        writer.writeheader()
        for domain_id in domain_ids:
            info = results.get(domain_id, {})
            pdb_id = domain_id[:4].lower() if len(domain_id) >= 4 else domain_id
            writer.writerow({
                "domain_id": domain_id,
                "cath_class": info.get("cath_class", 0),
                "architecture": info.get("architecture", 0),
                "topology": info.get("topology", 0),
                "homology": info.get("homology", 0),
                "n_residues": info.get("n_residues", 0),
                "pdb_id": pdb_id,
            })

    click.echo(f"\nWritten: {out_path}")

    # Print class distribution
    from collections import Counter
    class_counts = Counter(results[d]["cath_class"] for d in domain_ids if d in results and results[d]["cath_class"] > 0)
    click.echo("\nCATH class distribution:")
    class_names = {1: "Mainly α", 2: "Mainly β", 3: "α/β", 4: "Few SS"}
    for cls in sorted(class_counts):
        click.echo(f"  Class {cls} ({class_names.get(cls, 'Unknown')}): {class_counts[cls]}")


if __name__ == "__main__":
    main()
