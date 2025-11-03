"""Utilities for loading n-gram frequency distributions from the corpus assets."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterator
from enum import Enum

class NgramType(Enum):
    MONOGRAM = 1
    BIGRAM = 2
    TRIGRAM = 3
    SKIPGRAM = 4

    @property
    def order(self) -> int:
        if self.value == 4:
            return 2
        return self.value

class FreqDist:
    def __init__(self, corpus_name: str, freqdist: dict[str | NgramType, dict[str, float]] | None = None):
        self.corpus_name = corpus_name
        self.freqdist = {t : {} for t in NgramType}

        if freqdist is None:
            return

        for ngram_name_or_type in freqdist:
            if isinstance(ngram_name_or_type, str):
                ngram_name = ngram_name_or_type
                if ngram_name not in [t.name for t in NgramType]:
                    print(f"Warning: '{ngram_name}' in {corpus_name} is not a valid ngram type. Ignoring.")
                    continue
                ngram_type = NgramType[ngram_name]
            else:
                ngram_type = ngram_name_or_type
            self.freqdist[ngram_type] = freqdist[ngram_name_or_type]

    @classmethod
    def from_ngram_file(cls, ngram_file_name: str) -> FreqDist:
        """Load the n-gram frequency distribution for the requested ngram file.

        Parameters
        ----------
        ngram_file_name:
            Name of the corpus file (without extension) to load from ``./corpus/ngrams``.

        Returns
        -------
        dict[str, Any]
            Parsed JSON payload representing the n-gram frequency distribution.

        Raises
        ------
        FileNotFoundError
            If the named ngram file does not exist.
        json.JSONDecodeError
            If the ngram file is malformed.
        """

        corpus_path = Path(__file__).resolve().parent / "corpus" / "ngrams" / f"{ngram_file_name}.json"
        with corpus_path.open("r", encoding="utf-8") as fp:
            return cls(ngram_file_name, freqdist=json.load(fp))


# add a main function to test the freqdist
if __name__ == "__main__":
    freqdist = FreqDist.from_ngram_file("en")

    # print each ngram type and if the dictionary is longer than 40 entries, print just 40 entries and ... with a count
    for ngram_type in freqdist.freqdist:
        if len(freqdist.freqdist[ngram_type]) > 40:
            print(f"{ngram_type.name}: {len(freqdist.freqdist[ngram_type])} entries")
            print(list(freqdist.freqdist[ngram_type].keys())[:40])
            print("...")
        else:
            print(f"{ngram_type.name}: {len(freqdist.freqdist[ngram_type])} entries")
            print(freqdist.freqdist[ngram_type])
        print()
