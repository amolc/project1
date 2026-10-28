"""Load CSV files from the archive folder, chunk them, and upsert them into Pinecone.

The ingestion flow has four main steps:

    1. Read each CSV file from the archive folder.
    2. Group consecutive rows into chunks of roughly CHUNK_SIZE characters.
    3. Embed each chunk's text through OpenRouter.
    4. Store the vector, text, and metadata in Pinecone in batches.

Each chunk holds the CSV header once plus consecutive data rows. The stored
metadata includes the original text so rag_chat.py can retrieve it later.

Usage:
    python archive_to_pinecone.py
    python archive_to_pinecone.py races.csv drivers.csv
    python archive_to_pinecone.py --dry-run
"""

import argparse
import csv
import hashlib
import json
import os
import ssl
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from pathlib import Path

import certifi
from dotenv import load_dotenv
from pinecone import Pinecone

load_dotenv()

ROOT_DIR = Path(__file__).resolve().parent
ARCHIVE_DIR = ROOT_DIR / "archive"

# Pinecone destination for the archive vectors.
INDEX_NAME = os.getenv("PINECONE_INDEX", "f1")
NAMESPACE = os.getenv("PINECONE_NAMESPACE", "archive")

# Embedding provider and model. Rag chat must use the same values.
OPENROUTER_EMBEDDINGS_URL = os.getenv(
    "OPENROUTER_EMBEDDINGS_URL",
    "https://openrouter.ai/api/v1/embeddings",
)
EMBEDDING_MODEL = os.getenv(
    "OPENROUTER_EMBED_MODEL",
    "openai/text-embedding-3-large",
)
EMBEDDING_DIMENSION = int(os.getenv("EMBEDDING_DIMENSION", "1024"))

# Chunking and upload settings.
CHUNK_SIZE = int(os.getenv("ARCHIVE_CHUNK_SIZE", "1200"))
BATCH_SIZE = int(os.getenv("ARCHIVE_BATCH_SIZE", "64"))
CONCURRENCY = int(os.getenv("ARCHIVE_CONCURRENCY", "4"))
MAX_RETRIES = int(os.getenv("ARCHIVE_MAX_RETRIES", "5"))

SSL_CONTEXT = ssl.create_default_context(cafile=certifi.where())

NULL_VALUES = {"", "\\N", "NULL", "null", "None"}


def format_row(row, columns):
    """Turn a CSV row into a compact record while preserving column order."""

    values = []

    for column in columns:
        value = (row.get(column) or "").strip()
        values.append("" if value in NULL_VALUES else value)

    if not any(values):
        return None

    return " | ".join(values)


def clean(value):
    """Return a stripped CSV value, or an empty string for null markers."""

    value = (value or "").strip()
    return "" if value in NULL_VALUES else value


def load_lookups(archive_dir):
    """Load small lookup tables so rows can be turned into readable sentences."""

    specs = {
        "races.csv": "raceId",
        "drivers.csv": "driverId",
        "constructors.csv": "constructorId",
        "circuits.csv": "circuitId",
        "status.csv": "statusId",
    }

    lookups = {}

    for filename, key in specs.items():
        path = archive_dir / filename
        rows = {}

        if path.exists():
            with path.open("r", encoding="utf-8-sig", newline="") as handle:
                for row in csv.DictReader(handle):
                    rows[clean(row.get(key))] = row

        lookups[filename] = rows

    return lookups


def race_label(race):
    if not race:
        return "Unknown race"

    name = clean(race.get("name"))
    year = clean(race.get("year"))
    date = clean(race.get("date"))
    round_number = clean(race.get("round"))

    label = name or "Unknown race"

    if year and not label.startswith(year):
        label = f"{year} {label}"

    if round_number and date:
        label += f" (round {round_number}, {date})"
    elif date:
        label += f" ({date})"

    return label


def driver_label(driver):
    if not driver:
        return "Unknown driver"

    name = " ".join(
        part
        for part in (
            clean(driver.get("forename")),
            clean(driver.get("surname")),
        )
        if part
    )
    code = clean(driver.get("code"))

    if not name:
        return clean(driver.get("driverRef")) or "Unknown driver"

    return f"{name} ({code})" if code else name


def constructor_label(constructor):
    return clean(constructor.get("name")) if constructor else ""


def circuit_label(circuit):
    if not circuit:
        return ""

    name = clean(circuit.get("name"))
    location = clean(circuit.get("location"))
    country = clean(circuit.get("country"))

    place = ", ".join(part for part in (location, country) if part)

    return f"{name}, {place}" if place else name


def enrich_row(filename, row, lookups):
    """Convert one raw CSV row into a sentence describing the record.

    These sentences are much easier for an embedding model to search than
    comma-separated IDs. Unknown tables fall back to the compact raw format.
    """

    races = lookups["races.csv"]
    drivers = lookups["drivers.csv"]
    constructors = lookups["constructors.csv"]
    circuits = lookups["circuits.csv"]
    statuses = lookups["status.csv"]

    race = races.get(clean(row.get("raceId")))
    driver = drivers.get(clean(row.get("driverId")))
    constructor = constructors.get(clean(row.get("constructorId")))
    status = statuses.get(clean(row.get("statusId")))

    if filename == "results.csv":
        position = clean(row.get("positionText")) or clean(row.get("position"))
        return (
            f"{race_label(race)} result: {driver_label(driver)} drove for "
            f"{constructor_label(constructor)}. Finished position {position}, "
            f"started grid {clean(row.get('grid'))}, scored "
            f"{clean(row.get('points'))} points, completed "
            f"{clean(row.get('laps'))} laps. Status: "
            f"{clean(status.get('status')) if status else 'unknown'}. "
            f"Fastest lap rank {clean(row.get('rank'))}, time "
            f"{clean(row.get('fastestLapTime'))}, speed "
            f"{clean(row.get('fastestLapSpeed'))}."
        )

    if filename == "sprint_results.csv":
        position = clean(row.get("positionText")) or clean(row.get("position"))
        return (
            f"{race_label(race)} sprint result: {driver_label(driver)} drove "
            f"for {constructor_label(constructor)}. Finished position "
            f"{position}, started grid {clean(row.get('grid'))}, scored "
            f"{clean(row.get('points'))} points. Status: "
            f"{clean(status.get('status')) if status else 'unknown'}."
        )

    if filename == "driver_standings.csv":
        return (
            f"Drivers championship standings after {race_label(race)}: "
            f"{driver_label(driver)} is position {clean(row.get('position'))} "
            f"with {clean(row.get('points'))} points and "
            f"{clean(row.get('wins'))} wins."
        )

    if filename == "constructor_standings.csv":
        return (
            f"Constructors championship standings after {race_label(race)}: "
            f"{constructor_label(constructor)} is position "
            f"{clean(row.get('position'))} with {clean(row.get('points'))} "
            f"points and {clean(row.get('wins'))} wins."
        )

    if filename == "constructor_results.csv":
        return (
            f"{race_label(race)} constructor result: "
            f"{constructor_label(constructor)} scored "
            f"{clean(row.get('points'))} points. Status: "
            f"{clean(status.get('status')) if status else 'unknown'}."
        )

    if filename == "qualifying.csv":
        return (
            f"{race_label(race)} qualifying: {driver_label(driver)} drove for "
            f"{constructor_label(constructor)} and qualified position "
            f"{clean(row.get('position'))}. Q1 {clean(row.get('q1'))}, "
            f"Q2 {clean(row.get('q2'))}, Q3 {clean(row.get('q3'))}."
        )

    if filename == "pit_stops.csv":
        return (
            f"{race_label(race)} pit stop: {driver_label(driver)} made stop "
            f"{clean(row.get('stop'))} on lap {clean(row.get('lap'))}, "
            f"duration {clean(row.get('duration'))} seconds."
        )

    if filename == "races.csv":
        circuit = circuits.get(clean(row.get("circuitId")))
        return (
            f"{clean(row.get('year'))} {clean(row.get('name'))} was round "
            f"{clean(row.get('round'))} of the championship, held at "
            f"{circuit_label(circuit)} on {clean(row.get('date'))}. "
            f"Race start time {clean(row.get('time'))}."
        )

    if filename == "drivers.csv":
        return (
            f"Driver {driver_label(row)}: number {clean(row.get('number'))}, "
            f"nationality {clean(row.get('nationality'))}, born "
            f"{clean(row.get('dob'))}."
        )

    if filename == "constructors.csv":
        return (
            f"Constructor {constructor_label(row)}: nationality "
            f"{clean(row.get('nationality'))}."
        )

    if filename == "circuits.csv":
        return f"Circuit {circuit_label(row)}."

    return None


def iter_chunks(csv_path, chunk_size, enrich=False, lookups=None):
    """Yield chunks of consecutive CSV rows bounded by chunk_size characters."""

    with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)

        if not reader.fieldnames:
            print(f"Skipping {csv_path.name}: no header row")
            return

        columns = list(reader.fieldnames)

        if enrich:
            header = f"# {csv_path.name}"
        else:
            header = f"# {csv_path.name}\n" + " | ".join(columns)

        buffer = []
        buffer_chars = 0
        row_start = None
        row_end = None
        chunk_index = 0

        def emit():
            body = "\n".join(buffer)
            content = f"{header}\nrows {row_start}-{row_end}\n{body}"
            return {
                "content": content,
                "row_start": row_start,
                "row_end": row_end,
                "chunk_index": chunk_index,
            }

        for row_number, row in enumerate(reader, start=1):
            record = None

            if enrich:
                record = enrich_row(csv_path.name, row, lookups)

            if record is None:
                record = format_row(row, columns)

            if record is None:
                continue

            record_length = len(record) + 1

            if buffer and buffer_chars + record_length > chunk_size:
                yield emit()
                chunk_index += 1
                buffer = []
                buffer_chars = 0

            if not buffer:
                row_start = row_number

            buffer.append(record)
            buffer_chars += record_length
            row_end = row_number

        if buffer:
            yield emit()


def build_chunk_id(document_id, chunk):
    content_hash = hashlib.sha256(
        chunk["content"].encode("utf-8")
    ).hexdigest()[:16]

    return (
        f"{document_id}-{chunk['row_start']}-"
        f"{chunk['row_end']}-{content_hash}"
    )


def iter_documents(archive_dir, csv_files, chunk_size, enrich=False, lookups=None):
    """Yield every chunk for every selected CSV file in the archive."""

    # Step 1: Find the CSV files to process.
    paths = sorted(archive_dir.glob("*.csv"))

    if csv_files:
        wanted = {name.lower() for name in csv_files}
        paths = [path for path in paths if path.name.lower() in wanted]

        missing = wanted - {path.name.lower() for path in paths}
        if missing:
            raise FileNotFoundError(
                f"CSV files not found in {archive_dir}: "
                f"{', '.join(sorted(missing))}"
            )

    if not paths:
        raise FileNotFoundError(f"No CSV files found in {archive_dir}")

    for csv_path in paths:
        # Give each source file a stable ID so repeated runs are idempotent.
        document_id = hashlib.sha256(
            csv_path.name.encode("utf-8")
        ).hexdigest()[:20]

        count = 0

        # Step 2: Split this file into chunks and attach searchable metadata.
        for chunk in iter_chunks(csv_path, chunk_size, enrich, lookups):
            chunk["id"] = build_chunk_id(document_id, chunk)
            chunk["metadata"] = {
                "source": csv_path.name,
                "document_id": document_id,
                "chunk_index": chunk["chunk_index"],
                "row_start": chunk["row_start"],
                "row_end": chunk["row_end"],
                "text": chunk["content"],
            }

            count += 1
            yield chunk

        print(f"Prepared {count} chunks from {csv_path.name}")


def embed_texts(texts):
    """Step 3: Convert a batch of chunk texts into embedding vectors."""

    api_key = os.getenv("OPENROUTER_API_KEY")

    if not api_key:
        raise EnvironmentError(
            "Please set the OPENROUTER_API_KEY environment variable."
        )

    payload = json.dumps(
        {
            "model": EMBEDDING_MODEL,
            "input": texts,
            "dimensions": EMBEDDING_DIMENSION,
        }
    ).encode("utf-8")

    request = urllib.request.Request(
        OPENROUTER_EMBEDDINGS_URL,
        data=payload,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://localhost",
            "X-Title": "archive-to-pinecone",
        },
        method="POST",
    )

    for attempt in range(MAX_RETRIES + 1):
        try:
            with urllib.request.urlopen(
                request, timeout=120, context=SSL_CONTEXT
            ) as response:
                data = json.load(response)
            break
        except urllib.error.HTTPError as error:
            retryable = error.code == 429 or error.code >= 500

            if not retryable or attempt == MAX_RETRIES:
                raise RuntimeError(
                    f"Embedding API error {error.code}: "
                    f"{error.read().decode(errors='replace')[:500]}"
                ) from error

            time.sleep(min(2 ** attempt, 30))
        except urllib.error.URLError as error:
            if attempt == MAX_RETRIES:
                raise RuntimeError(
                    f"Embedding API request failed: {error.reason}"
                ) from error

            time.sleep(min(2 ** attempt, 30))

    items = sorted(data["data"], key=lambda item: item.get("index", 0))
    vectors = [item["embedding"] for item in items]

    if len(vectors) != len(texts):
        raise RuntimeError(
            f"Embedding API returned {len(vectors)} vectors "
            f"for {len(texts)} inputs"
        )

    usage = data.get("usage", {})

    return vectors, usage.get("total_tokens", 0), usage.get("cost", 0.0)


def connect_pinecone():
    api_key = os.getenv("PINECONE_API_KEY")

    if not api_key:
        raise EnvironmentError(
            "Please set the PINECONE_API_KEY environment variable."
        )

    pc = Pinecone(api_key=api_key)

    try:
        description = pc.indexes.describe(INDEX_NAME)
    except Exception as error:
        raise RuntimeError(
            f"Pinecone index '{INDEX_NAME}' is not available: {error}. "
            "Set PINECONE_INDEX to an existing index."
        ) from error

    if description.dimension != EMBEDDING_DIMENSION:
        raise ValueError(
            f"Index '{INDEX_NAME}' has dimension {description.dimension}, "
            f"but EMBEDDING_DIMENSION is {EMBEDDING_DIMENSION}."
        )

    return pc, pc.Index(INDEX_NAME)


def upload_chunks(pc, index, batch_size, concurrency, chunks):
    """Step 4: Embed and upsert chunks in parallel batches."""

    def process(batch):
        texts = [chunk["content"] for chunk in batch]
        vectors, tokens, cost = embed_texts(texts)

        # Each vector is stored with its ID and metadata. The metadata includes
        # the chunk text, which is what the chat retriever reads back later.
        index.upsert(
            vectors=[
                {
                    "id": chunk["id"],
                    "values": vector,
                    "metadata": chunk["metadata"],
                }
                for chunk, vector in zip(batch, vectors)
            ],
            namespace=NAMESPACE,
        )

        return len(batch), tokens, cost

    pending = []
    futures = set()
    uploaded = 0
    total_tokens = 0
    total_cost = 0.0

    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        def handle_done(done):
            nonlocal uploaded, total_tokens, total_cost

            for future in done:
                count, tokens, cost = future.result()
                uploaded += count
                total_tokens += tokens
                total_cost += cost

                print(
                    f"Upserted {uploaded} chunks | "
                    f"{total_tokens:,} tokens | "
                    f"${total_cost:.4f}"
                )

        for chunk in chunks:
            pending.append(chunk)

            if len(pending) >= batch_size:
                futures.add(pool.submit(process, pending))
                pending = []

                if len(futures) >= concurrency * 2:
                    done, _ = wait(futures, return_when=FIRST_COMPLETED)
                    handle_done(done)
                    futures -= done

        if pending:
            futures.add(pool.submit(process, pending))

        while futures:
            done, _ = wait(futures, return_when=FIRST_COMPLETED)
            handle_done(done)
            futures -= done

    print(
        f"\nUpload completed: {uploaded} chunks | "
        f"{total_tokens:,} tokens | ${total_cost:.4f}"
    )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Chunk archive CSVs and upsert them into Pinecone."
    )

    parser.add_argument(
        "files",
        nargs="*",
        help="CSV filenames to ingest (default: all CSVs in the archive)",
    )
    parser.add_argument(
        "--archive",
        default=str(ARCHIVE_DIR),
        help=f"Folder containing CSV files (default: {ARCHIVE_DIR})",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=CHUNK_SIZE,
        help=f"Approximate characters per chunk (default: {CHUNK_SIZE})",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=BATCH_SIZE,
        help=f"Chunks embedded/upserted per batch (default: {BATCH_SIZE})",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=CONCURRENCY,
        help=f"Embed/upsert batches in flight (default: {CONCURRENCY})",
    )
    parser.add_argument(
        "--enrich",
        action="store_true",
        help=(
            "Turn rows into readable sentences with driver, race, and "
            "constructor names before embedding"
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only build and count chunks; do not call Pinecone",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    archive_dir = Path(args.archive).resolve()

    if not archive_dir.is_dir():
        print(f"Archive folder not found: {archive_dir}", file=sys.stderr)
        sys.exit(1)

    if args.chunk_size <= 0 or args.batch_size <= 0:
        print("--chunk-size and --batch-size must be positive", file=sys.stderr)
        sys.exit(1)

    if args.concurrency <= 0:
        print("--concurrency must be positive", file=sys.stderr)
        sys.exit(1)

    lookups = load_lookups(archive_dir) if args.enrich else None
    chunks = iter_documents(
        archive_dir,
        args.files,
        args.chunk_size,
        enrich=args.enrich,
        lookups=lookups,
    )

    if args.dry_run:
        total = sum(1 for _ in chunks)

        print(f"\nDry run: {total} chunks prepared")
        print(f"Index: {INDEX_NAME}")
        print(f"Namespace: {NAMESPACE}")
        print(f"Embedding model: {EMBEDDING_MODEL}")
        print(f"Embedding dimension: {EMBEDDING_DIMENSION}")
        print(f"Enriched: {args.enrich}")
        return

    pc, index = connect_pinecone()

    print(f"Index: {INDEX_NAME}")
    print(f"Namespace: {NAMESPACE}")
    print(f"Embedding model: {EMBEDDING_MODEL}")
    print(f"Embedding dimension: {EMBEDDING_DIMENSION}")
    print(f"Batch size: {args.batch_size}")
    print(f"Concurrency: {args.concurrency}\n")

    upload_chunks(
        pc,
        index,
        args.batch_size,
        args.concurrency,
        chunks,
    )


if __name__ == "__main__":
    main()
