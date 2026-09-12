import pandas as pd


def do_1brc_pandas_iter(
    file_path: str,
    output_data: bool = True,
    *,
    chunksize: int = 1_000_000,
) -> str:
    """Perform the 1BRC in Pandas using bounded-memory chunked CSV reads.

    Each chunk is reduced to four sufficient statistics per station: minimum,
    maximum, sum, and count.  If ``A`` is the aggregate accumulated so far and
    ``B`` is the aggregate for the next chunk, the merged statistics are

        min(A ∪ B)   = min(min(A), min(B))
        max(A ∪ B)   = max(max(A), max(B))
        sum(A ∪ B)   = sum(A) + sum(B)
        count(A ∪ B) = count(A) + count(B)

    and the mean is computed only after all chunks have been consumed as

        mean = sum / count.

    Thus memory use is bounded by the CSV chunk plus the per-station aggregate,
    rather than by the full input DataFrame.

    Args:
        file_path: Path to the semicolon-separated 1BRC input file.
        output_data: Whether to serialise and return the final station results.
        chunksize: Number of CSV rows Pandas should parse into each chunk.

    Returns:
        The 1BRC result string when ``output_data`` is true, otherwise an empty
        string.
    """
    aggregate = None

    for chunk in pd.read_csv(
        file_path,
        sep=";",
        names=["station", "measurement"],
        chunksize=chunksize,
    ):
        chunk_aggregate = chunk.groupby("station")["measurement"].agg(
            ["min", "max", "sum", "count"]
        )

        if aggregate is None:
            aggregate = chunk_aggregate
            continue

        aggregate = (
            pd.concat([aggregate, chunk_aggregate])
            .groupby(level=0)
            .agg(
                {
                    "min": "min",
                    "max": "max",
                    "sum": "sum",
                    "count": "sum",
                }
            )
        )

    if aggregate is None:
        return "{}" if output_data else ""

    aggregate["mean"] = aggregate["sum"] / aggregate["count"]
    df = aggregate[["min", "mean", "max"]].round(2)

    if output_data:
        result = []
        for station, min_val, mean_val, max_val in df.to_records():
            result.append(f"{station}={min_val}/{mean_val}/{max_val}")
        return "{" + ", ".join(result) + "}"
    return ""
