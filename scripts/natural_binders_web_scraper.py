"""
Binder / Interactor Protein Dataset Collector
=============================================

Builds an amino-acid sequence dataset for ESM-C / SAE baseline models.

Sources:
  - STRING interaction_partners API
  - UniProt REST API for canonical protein metadata and sequences
  - Optional IntAct PSICQUIC collection (disabled by default because the
    public endpoint may intermittently return HTTP 503)

Outputs:
  1. binder_dataset_raw.csv
     One row per POI-partner-source association.

  2. binder_sequences_unique.fasta
     One amino-acid sequence per unique UniProt accession, suitable for ESM-C.

  3. binder_sequences_unique.csv
     One row per unique sequence with metadata.

  4. unresolved_binders.csv
     STRING partners that could not be mapped confidently to UniProt.

"""

from __future__ import annotations

import csv
import json
import re
import time
from pathlib import Path
from typing import Any, Iterable
from urllib.parse import quote

import requests #type: ignore 
from requests.adapters import HTTPAdapter #type: ignore 
from urllib3.util.retry import Retry #type: ignore 


# CONFIGURATION


#protein of interests, ensure correct uniprot ID is mentioned. 
POIS = [
   # {"name": "VILIP-1", "gene": "VSNL1", "uniprot": "P62760"},
   # {"name": "UCHL1", "gene": "UCHL1", "uniprot": "P09936"},
   # {"name": "FABP7", "gene": "FABP7", "uniprot": "O15540"},
   {"name": "REG3A", "gene": "REG3A", "uniprot": "Q06141"},
   #{"name": "REG3A", "gene": "REG3A", "uniprot": "O09037"},
   #{"name": "REG3A", "gene": "REG3A", "uniprot": "P35231"},
]

SPECIES_TAXID = 9606
STRING_LIMIT = 200
STRING_REQUIRED_SCORE = 400  # 0-1000; 400 is STRING "medium confidence"

ENABLE_INTACT = False
REQUEST_DELAY = 0.20

OUTPUT_ASSOCIATIONS = Path("binder_dataset_raw.csv")
OUTPUT_UNIQUE_CSV = Path("binder_sequences_unique.csv")
OUTPUT_FASTA = Path("binder_sequences_unique.fasta")
OUTPUT_UNRESOLVED = Path("unresolved_binders.csv")
CACHE_FILE = Path("uniprot_mapping_cache.json")

USER_AGENT = "binder-esmc-sae-dataset/2.0 (academic research)"

# Keep reviewed Swiss-Prot entries when available.
PREFER_REVIEWED = True

# Exclude POIs from their own partner lists.
EXCLUDE_SELF_INTERACTIONS = True


ASSOCIATION_FIELDS = [
    "poi_uniprot",
    "poi_gene",
    "poi_name",
    "binder_uniprot",
    "binder_entry_name",
    "binder_gene",
    "binder_protein_name",
    "binder_string_id",
    "binder_ensembl_protein_id",
    "source_db",
    "association_class",
    "confidence_score",
    "string_score",
    "string_nscore",
    "string_fscore",
    "string_pscore",
    "string_ascore",
    "string_escore",
    "string_dscore",
    "string_tscore",
    "mapping_method",
    "reviewed",
    "organism_id",
    "sequence_length",
    "binder_sequence",
]

UNIQUE_SEQUENCE_FIELDS = [
    "binder_uniprot",
    "binder_entry_name",
    "binder_gene",
    "binder_protein_name",
    "reviewed",
    "organism_id",
    "sequence_length",
    "binder_sequence",
    "source_pois",
    "source_databases",
    "max_confidence_score",
]

UNRESOLVED_FIELDS = [
    "poi_uniprot",
    "poi_gene",
    "poi_name",
    "binder_string_id",
    "binder_ensembl_protein_id",
    "binder_gene",
    "confidence_score",
    "reason",
]


# ---------------------------------------------------------------------------
# HTTP
# ---------------------------------------------------------------------------

def make_session() -> requests.Session:
    retry = Retry(
        total=6,
        connect=6,
        read=6,
        status=6,
        backoff_factor=1.5,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=frozenset({"GET", "POST"}),
        respect_retry_after_header=True,
        raise_on_status=False,
    )

    session = requests.Session()
    session.headers.update(
        {
            "User-Agent": USER_AGENT,
            "Accept": "application/json",
        }
    )
    adapter = HTTPAdapter(max_retries=retry, pool_connections=10, pool_maxsize=10)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    return session


SESSION = make_session()


def request_json(
    method: str,
    url: str,
    *,
    timeout: tuple[int, int] = (15, 90),
    **kwargs: Any,
) -> Any:
    response = SESSION.request(method, url, timeout=timeout, **kwargs)

    if not response.ok:
        body_preview = response.text[:500].replace("\n", " ")
        raise RuntimeError(
            f"{method} {url} returned HTTP {response.status_code}: "
            f"{body_preview}"
        )

    try:
        return response.json()
    except ValueError as exc:
        raise RuntimeError(f"Non-JSON response from {url}") from exc

# CACHE


def load_cache() -> dict[str, Any]:
    if not CACHE_FILE.exists():
        return {}

    try:
        data = json.loads(CACHE_FILE.read_text())
        return data if isinstance(data, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


def save_cache(cache: dict[str, Any]) -> None:
    temporary = CACHE_FILE.with_suffix(".tmp")
    temporary.write_text(json.dumps(cache, indent=2, sort_keys=True))
    temporary.replace(CACHE_FILE)


UNIPROT_CACHE = load_cache()

# HELPERS


def chunked(values: list[str], size: int) -> Iterable[list[str]]:
    for start in range(0, len(values), size):
        yield values[start : start + size]


def split_string_id(string_id: str) -> str:
    """
    Convert STRING IDs such as 9606.ENSP00000354587 into ENSP00000354587.
    """
    return string_id.split(".", 1)[1] if "." in string_id else string_id


def valid_protein_sequence(sequence: str) -> bool:
    """
    Standard/extended amino-acid alphabet accepted for protein FASTA.
    """
    return bool(sequence) and bool(
        re.fullmatch(r"[ACDEFGHIKLMNPQRSTVWYBXZJUO]+", sequence.upper())
    )


def wrap_fasta(sequence: str, width: int = 60) -> str:
    return "\n".join(sequence[i : i + width] for i in range(0, len(sequence), width))


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default

# STRING

def get_string_interactors(
    identifier: str,
    species: int = SPECIES_TAXID,
    limit: int = STRING_LIMIT,
    required_score: int = STRING_REQUIRED_SCORE,
) -> list[dict[str, Any]]:
    """
    Retrieve STRING interaction partners.

    STRING returns combined and channel-specific scores as values from 0 to 1.
    The API required_score parameter uses 0-1000.
    """
    url = "https://string-db.org/api/json/interaction_partners"
    params = {
        "identifiers": identifier,
        "species": species,
        "limit": limit,
        "required_score": required_score,
        "caller_identity": USER_AGENT,
    }

    data = request_json("GET", url, params=params)
    results: list[dict[str, Any]] = []

    for entry in data:
        string_id = str(entry.get("stringId_B", "")).strip()
        if not string_id:
            continue

        results.append(
            {
                "binder_string_id": string_id,
                "binder_ensembl_protein_id": split_string_id(string_id),
                "binder_gene": str(entry.get("preferredName_B", "")).strip(),
                "string_score": entry.get("score", ""),
                "string_nscore": entry.get("nscore", ""),
                "string_fscore": entry.get("fscore", ""),
                "string_pscore": entry.get("pscore", ""),
                "string_ascore": entry.get("ascore", ""),
                "string_escore": entry.get("escore", ""),
                "string_dscore": entry.get("dscore", ""),
                "string_tscore": entry.get("tscore", ""),
            }
        )

    return results

# UniProt

def parse_uniprot_result(entry: dict[str, Any]) -> dict[str, Any]:
    protein_description = entry.get("proteinDescription") or {}

    recommended_name = (
        protein_description.get("recommendedName", {})
        .get("fullName", {})
        .get("value", "")
    )

    if not recommended_name:
        submission_names = protein_description.get("submissionNames") or []
        if submission_names:
            recommended_name = (
                submission_names[0].get("fullName", {}).get("value", "")
            )

    genes = entry.get("genes") or []
    gene_name = ""
    if genes:
        gene_name = genes[0].get("geneName", {}).get("value", "")

    organism = entry.get("organism") or {}
    sequence_data = entry.get("sequence") or {}
    sequence = str(sequence_data.get("value", "")).replace(" ", "").upper()

    entry_type = str(entry.get("entryType", ""))
    reviewed = "reviewed" in entry_type.lower() and "unreviewed" not in entry_type.lower()

    return {
        "binder_uniprot": str(entry.get("primaryAccession", "")).strip(),
        "binder_entry_name": str(entry.get("uniProtkbId", "")).strip(),
        "binder_gene": gene_name,
        "binder_protein_name": recommended_name,
        "reviewed": reviewed,
        "organism_id": organism.get("taxonId", ""),
        "sequence_length": sequence_data.get("length", len(sequence)),
        "binder_sequence": sequence,
    }


def search_uniprot_by_ensembl_protein(
    ensembl_protein_id: str,
    species: int = SPECIES_TAXID,
) -> dict[str, Any] | None:
    """
    Search UniProtKB directly using its Ensembl cross-reference.

    This avoids launching one asynchronous ID-mapping job per protein.
    """
    cache_key = f"ensembl:{ensembl_protein_id}:{species}"
    if cache_key in UNIPROT_CACHE:
        return UNIPROT_CACHE[cache_key]

    query = f"(xref:ensembl-{ensembl_protein_id}) AND (organism_id:{species})"
    if PREFER_REVIEWED:
        query += " AND (reviewed:true)"

    url = "https://rest.uniprot.org/uniprotkb/search"
    params = {
        "query": query,
        "format": "json",
        "size": 5,
        "fields": (
            "accession,id,reviewed,gene_names,protein_name,"
            "organism_id,length,sequence"
        ),
    }

    data = request_json("GET", url, params=params)
    results = data.get("results", [])

    # Fall back to reviewed+unreviewed if no reviewed cross-reference exists.
    if not results and PREFER_REVIEWED:
        params["query"] = (
            f"(xref:ensembl-{ensembl_protein_id}) AND "
            f"(organism_id:{species})"
        )
        data = request_json("GET", url, params=params)
        results = data.get("results", [])

    parsed = parse_uniprot_result(results[0]) if results else None

    if parsed and not valid_protein_sequence(parsed["binder_sequence"]):
        parsed = None

    UNIPROT_CACHE[cache_key] = parsed
    save_cache(UNIPROT_CACHE)
    time.sleep(REQUEST_DELAY)
    return parsed


def search_uniprot_by_gene(
    gene_name: str,
    species: int = SPECIES_TAXID,
) -> dict[str, Any] | None:
    """
    Fallback mapping using an exact gene symbol.

    This is less specific than mapping the Ensembl protein ID, so the output
    records mapping_method='gene_exact_fallback'.
    """
    if not gene_name:
        return None

    cache_key = f"gene:{gene_name.upper()}:{species}"
    if cache_key in UNIPROT_CACHE:
        return UNIPROT_CACHE[cache_key]

    query = f"(gene_exact:{gene_name}) AND (organism_id:{species})"
    if PREFER_REVIEWED:
        query += " AND (reviewed:true)"

    url = "https://rest.uniprot.org/uniprotkb/search"
    params = {
        "query": query,
        "format": "json",
        "size": 5,
        "fields": (
            "accession,id,reviewed,gene_names,protein_name,"
            "organism_id,length,sequence"
        ),
    }

    data = request_json("GET", url, params=params)
    results = data.get("results", [])

    if not results and PREFER_REVIEWED:
        params["query"] = (
            f"(gene_exact:{gene_name}) AND (organism_id:{species})"
        )
        data = request_json("GET", url, params=params)
        results = data.get("results", [])

    parsed = parse_uniprot_result(results[0]) if results else None

    if parsed and not valid_protein_sequence(parsed["binder_sequence"]):
        parsed = None

    UNIPROT_CACHE[cache_key] = parsed
    save_cache(UNIPROT_CACHE)
    time.sleep(REQUEST_DELAY)
    return parsed


def resolve_string_partner(hit: dict[str, Any]) -> tuple[dict[str, Any] | None, str]:
    """
    Resolve a STRING partner to a UniProt protein sequence.

    Resolution order:
      1. Exact Ensembl protein cross-reference.
      2. Exact human gene-symbol fallback.
    """
    ensembl_id = hit["binder_ensembl_protein_id"]
    gene = hit["binder_gene"]

    record = search_uniprot_by_ensembl_protein(ensembl_id)
    if record:
        return record, "ensembl_protein_xref"

    record = search_uniprot_by_gene(gene)
    if record:
        return record, "gene_exact_fallback"

    return None, "unresolved"


def get_intact_interactors(uniprot_id: str) -> list[dict[str, Any]]:
    """
    Optional PSICQUIC query. This remains isolated so a 503 cannot stop the
    STRING/UniProt dataset build.
    """
    query = quote(f"id:{uniprot_id}", safe=":")
    url = (
        "https://www.ebi.ac.uk/Tools/webservices/psicquic/"
        f"intact/webservices/current/search/query/{query}"
    )
    params = {
        "format": "tab27",
        "firstResult": 0,
        "maxResults": 1000,
    }

    response = SESSION.get(url, params=params, timeout=(15, 120))
    if not response.ok:
        raise RuntimeError(f"IntAct PSICQUIC returned HTTP {response.status_code}")

    hits: list[dict[str, Any]] = []

    for line in response.text.splitlines():
        columns = line.split("\t")
        if len(columns) < 15:
            continue

        id_a, id_b = columns[0], columns[1]
        alias_a, alias_b = columns[4], columns[5]

        a_matches = uniprot_id.upper() in id_a.upper()
        b_matches = uniprot_id.upper() in id_b.upper()

        if a_matches == b_matches:
            continue

        other_id = id_b if a_matches else id_a
        other_alias = alias_b if a_matches else alias_a

        uniprot_match = re.search(
            r"uniprotkb:([A-Z0-9]+(?:-\d+)?)",
            other_id,
            flags=re.IGNORECASE,
        )
        if not uniprot_match:
            continue

        hits.append(
            {
                "binder_uniprot": uniprot_match.group(1).upper(),
                "binder_name_raw": other_alias,
                "confidence_score": columns[14],
                "detection_method": columns[6],
                "publication_id": columns[8],
                "interaction_type": columns[11],
            }
        )

    return hits


# OUTPUT

def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def build_unique_sequence_rows(
    association_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    grouped: dict[str, dict[str, Any]] = {}

    for row in association_rows:
        accession = str(row.get("binder_uniprot", "")).strip()
        sequence = str(row.get("binder_sequence", "")).strip()

        if not accession or not valid_protein_sequence(sequence):
            continue

        confidence = safe_float(row.get("confidence_score"))

        if accession not in grouped:
            grouped[accession] = {
                "binder_uniprot": accession,
                "binder_entry_name": row.get("binder_entry_name", ""),
                "binder_gene": row.get("binder_gene", ""),
                "binder_protein_name": row.get("binder_protein_name", ""),
                "reviewed": row.get("reviewed", ""),
                "organism_id": row.get("organism_id", ""),
                "sequence_length": row.get("sequence_length", len(sequence)),
                "binder_sequence": sequence,
                "_pois": set(),
                "_sources": set(),
                "_max_confidence": confidence,
            }

        grouped[accession]["_pois"].add(str(row.get("poi_gene", "")))
        grouped[accession]["_sources"].add(str(row.get("source_db", "")))
        grouped[accession]["_max_confidence"] = max(
            grouped[accession]["_max_confidence"],
            confidence,
        )

    final_rows: list[dict[str, Any]] = []

    for accession in sorted(grouped):
        item = grouped[accession]
        final_rows.append(
            {
                "binder_uniprot": item["binder_uniprot"],
                "binder_entry_name": item["binder_entry_name"],
                "binder_gene": item["binder_gene"],
                "binder_protein_name": item["binder_protein_name"],
                "reviewed": item["reviewed"],
                "organism_id": item["organism_id"],
                "sequence_length": item["sequence_length"],
                "binder_sequence": item["binder_sequence"],
                "source_pois": ";".join(sorted(item["_pois"])),
                "source_databases": ";".join(sorted(item["_sources"])),
                "max_confidence_score": item["_max_confidence"],
            }
        )

    return final_rows


def write_fasta(path: Path, unique_rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in unique_rows:
            accession = row["binder_uniprot"]
            entry_name = row["binder_entry_name"] or accession
            gene = row["binder_gene"] or "NA"
            protein_name = row["binder_protein_name"] or "Unknown protein"
            sequence = row["binder_sequence"]

            # Avoid spaces and pipe characters in metadata values where practical.
            protein_name = str(protein_name).replace("|", "/")
            header = (
                f">{accession}|{entry_name}|GN={gene}|"
                f"OS=Homo_sapiens|POI={row['source_pois']}|{protein_name}"
            )
            handle.write(header + "\n")
            handle.write(wrap_fasta(sequence) + "\n")

# MAIN

def main() -> None:
    association_rows: list[dict[str, Any]] = []
    unresolved_rows: list[dict[str, Any]] = []

    for poi in POIS:
        print(f"\n=== Processing {poi['name']} ({poi['uniprot']}) ===")

        try:
            string_hits = get_string_interactors(poi["gene"])
        except Exception as exc:
            print(f"  [ERROR] STRING query failed: {exc}")
            continue

        print(f"  STRING partners returned: {len(string_hits)}")

        for index, hit in enumerate(string_hits, start=1):
            if (
                EXCLUDE_SELF_INTERACTIONS
                and hit["binder_gene"].upper() == poi["gene"].upper()
            ):
                continue

            print(
                f"  [{index:>3}/{len(string_hits)}] "
                f"{hit['binder_gene'] or hit['binder_string_id']}",
                end="",
                flush=True,
            )

            try:
                uniprot_record, mapping_method = resolve_string_partner(hit)
            except Exception as exc:
                print(f" -> ERROR: {exc}")
                unresolved_rows.append(
                    {
                        "poi_uniprot": poi["uniprot"],
                        "poi_gene": poi["gene"],
                        "poi_name": poi["name"],
                        "binder_string_id": hit["binder_string_id"],
                        "binder_ensembl_protein_id": hit[
                            "binder_ensembl_protein_id"
                        ],
                        "binder_gene": hit["binder_gene"],
                        "confidence_score": hit["string_score"],
                        "reason": str(exc),
                    }
                )
                continue

            if not uniprot_record:
                print(" -> unresolved")
                unresolved_rows.append(
                    {
                        "poi_uniprot": poi["uniprot"],
                        "poi_gene": poi["gene"],
                        "poi_name": poi["name"],
                        "binder_string_id": hit["binder_string_id"],
                        "binder_ensembl_protein_id": hit[
                            "binder_ensembl_protein_id"
                        ],
                        "binder_gene": hit["binder_gene"],
                        "confidence_score": hit["string_score"],
                        "reason": "No UniProt match by Ensembl protein or exact gene",
                    }
                )
                continue

            print(
                f" -> {uniprot_record['binder_uniprot']} "
                f"({uniprot_record['sequence_length']} aa; {mapping_method})"
            )

            association_rows.append(
                {
                    "poi_uniprot": poi["uniprot"],
                    "poi_gene": poi["gene"],
                    "poi_name": poi["name"],
                    **uniprot_record,
                    "binder_string_id": hit["binder_string_id"],
                    "binder_ensembl_protein_id": hit[
                        "binder_ensembl_protein_id"
                    ],
                    "source_db": "STRING",
                    "association_class": "functional_or_physical_association",
                    "confidence_score": hit["string_score"],
                    "string_score": hit["string_score"],
                    "string_nscore": hit["string_nscore"],
                    "string_fscore": hit["string_fscore"],
                    "string_pscore": hit["string_pscore"],
                    "string_ascore": hit["string_ascore"],
                    "string_escore": hit["string_escore"],
                    "string_dscore": hit["string_dscore"],
                    "string_tscore": hit["string_tscore"],
                    "mapping_method": mapping_method,
                }
            )

        if ENABLE_INTACT:
            try:
                intact_hits = get_intact_interactors(poi["uniprot"])
                print(f"  IntAct partners returned: {len(intact_hits)}")
            except Exception as exc:
                print(f"  [WARN] IntAct unavailable: {exc}")

    unique_rows = build_unique_sequence_rows(association_rows)

    write_csv(OUTPUT_ASSOCIATIONS, association_rows, ASSOCIATION_FIELDS)
    write_csv(OUTPUT_UNIQUE_CSV, unique_rows, UNIQUE_SEQUENCE_FIELDS)
    write_csv(OUTPUT_UNRESOLVED, unresolved_rows, UNRESOLVED_FIELDS)
    write_fasta(OUTPUT_FASTA, unique_rows)

    print("\n=== Complete ===")
    print(f"Associations:       {len(association_rows):>6} -> {OUTPUT_ASSOCIATIONS}")
    print(f"Unique sequences:   {len(unique_rows):>6} -> {OUTPUT_UNIQUE_CSV}")
    print(f"ESM-C FASTA:               -> {OUTPUT_FASTA}")
    print(f"Unresolved mappings:{len(unresolved_rows):>6} -> {OUTPUT_UNRESOLVED}")
    print(f"UniProt cache:             -> {CACHE_FILE}")


if __name__ == "__main__":
    main()
