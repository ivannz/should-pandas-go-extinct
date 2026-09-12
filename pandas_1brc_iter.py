from pandas_should_go_extinct.constants import DATA_PATH
from pandas_should_go_extinct.pandas_1brc_iter import do_1brc_pandas_iter


def main() -> None:
    """Execute the chunked Pandas 1BRC benchmark."""
    do_1brc_pandas_iter(DATA_PATH, output_data=False)


if __name__ == "__main__":
    main()
