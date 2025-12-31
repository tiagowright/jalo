"""Utilities for loading n-gram frequency distributions from the corpus assets."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterator
from enum import Enum
import numpy as np
from functools import cache

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

NGRAM_FILENAMES: dict[NgramType, str] = {
    NgramType.MONOGRAM: "monograms.json",
    NgramType.BIGRAM: "bigrams.json",
    NgramType.TRIGRAM: "trigrams.json",
    NgramType.SKIPGRAM: "skipgrams.json",
}

class FreqDist:
    # sentinel value for out of distribution characters
    out_of_distribution = '__other__'

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
        
        # sort monograms by frequency descending
        self.char_seq = sorted(self.freqdist[NgramType.MONOGRAM].keys(), key=lambda x: self.freqdist[NgramType.MONOGRAM][x], reverse=True)
        self.char_seq.append(FreqDist.out_of_distribution)


    @classmethod
    def from_name(cls, name: str) -> FreqDist:
        """Load the n-gram frequency distribution for the requested corpus name.

        Parameters
        ----------
        name:
            Name of the corpus directory to load from ``./corpus/<name>``.

        Returns
        -------
        dict[str, Any]
            Parsed JSON payload representing the n-gram frequency distribution.

        Raises
        ------
        FileNotFoundError
            If the named corpus directory or legacy JSON file does not exist.
        json.JSONDecodeError
            If the ngram file is malformed.
        """

        corpus_root = Path(__file__).resolve().parent / "corpus"
        corpus_dir = corpus_root / name

        if corpus_dir.is_dir():
            freqdist_payload: dict[NgramType, dict[str, float]] = {}
            for ngram_type, filename in NGRAM_FILENAMES.items():
                ngram_path = corpus_dir / filename
                if not ngram_path.exists():
                    freqdist_payload[ngram_type] = {}
                    continue
                with ngram_path.open("r", encoding="utf-8") as fp:
                    freqdist_payload[ngram_type] = json.load(fp)
            return cls(name, freqdist=freqdist_payload)  # pyright: ignore[reportArgumentType]

        # Fall back to the legacy flat JSON file structure.
        corpus_path = corpus_root / "ngrams" / f"{name}.json"
        with corpus_path.open("r", encoding="utf-8") as fp:
            return cls(name, freqdist=json.load(fp))

    def select(self, char_set: set[str]) -> FreqDist:
        """Select a subset of the frequency distribution for the given list of characters."""
        return FreqDist(
            self.corpus_name, 
            freqdist={
                t : {
                    ngram : v 
                    for ngram, v in self.freqdist[t].items() 
                    if all(c in char_set for c in ngram)
                } for t in self.freqdist
            }
        )
    
    def top(self, n: int) -> FreqDist:
        """Select the top n n-grams for each ngram type."""
        top_monograms = sorted(
            self.freqdist[NgramType.MONOGRAM], 
            key=lambda x: self.freqdist[NgramType.MONOGRAM][x], 
            reverse=True)[:n]
        return self.select(set(top_monograms))
    
    def normalized(self) -> FreqDist:
        """Normalize the frequency distribution so that the sum of the frequencies is 1."""

        sum_monograms = sum(self.freqdist[NgramType.MONOGRAM].values())

        if not sum_monograms:
            return self
        
        return FreqDist(
            self.corpus_name, 
            freqdist={
                t : {ngram : v / sum_monograms for ngram, v in self.freqdist[t].items()} for t in self.freqdist
            }
        )
    
    # cache the result of this method
    @cache
    def to_numpy(self) -> dict[NgramType, np.ndarray]:
        """
        Convert each ngramtype in the frequency distribution to a numpy array.
        The numpy arrays are sorted by frequency descending of the monograms, 
        so an order 1 array at 0 will have the most frequent monogram and so on

        Example: in English, frequencies corresponding to monograms ['e', 't', 'a', ...]

        An order 2 ngram (bigram, skipgram) array at 0,0 will have the frequency of
        the bigram of the most frequent monogram followed by the same monogram.
        At 0,1 will have the frequency of the bigram of the most frequent monogram followed by the 
        second most frequent monogram.

        Example: in English, frequencies corresponding to bigrams:
        [['ee', 'et', 'ea', ...], ['te', 'tt', 'ta', ...], ...]

        The same pattern to order 3 ngrams.

        Example: in English, frequencies corresponding to trigrams:
        [
            [
                ['eee', 'eet', 'eea', ...], 
                ['ete', 'ett', 'eta', ...], 
                ...
            ], [
                ['tee', 'tet', 'tea', ...], 
                ['tte', 'ttt', 'tta', ...], 
                ...
            ], 
            ...
        ]
        
        """
 
        F = {}
        for ngramtype in self.freqdist:
            if ngramtype.order == 1:
                # F[ngramtype] = np_array(shape=(N,)) where F[ngramtype][i] = frequency of the i-th monogram in self.char_seq
                F[ngramtype] = np.array(
                    [
                        self.freqdist[ngramtype].get(char, 0.0) for char in self.char_seq
                    ], dtype=np.float64
                )
            elif ngramtype.order == 2:
                # F[ngramtype] = np_array(shape=(N,N)) where F[ngramtype][i,j] = frequency of the i-th and j-th self.char_seq in self.char_seq
                F[ngramtype] = np.array(
                    [
                        [
                            self.freqdist[ngramtype].get(char1 + char2, 0.0) for char2 in self.char_seq
                        ] for char1 in self.char_seq
                    ], dtype=np.float64
                )
            elif ngramtype.order == 3:
                # F[ngramtype] = np_array(shape=(N,N,N)) where F[ngramtype][i,j,k] = frequency of the i-th, j-th, and k-th self.char_seq in self.char_seq
                F[ngramtype] = np.array(
                    [
                        [
                            [
                                self.freqdist[ngramtype].get(char1 + char2 + char3, 0.0) for char3 in self.char_seq
                            ] for char2 in self.char_seq
                        ] for char1 in self.char_seq
                    ], dtype=np.float64
                )
            else:
                raise ValueError(f"Unsupported ngram type: {ngramtype} of order {ngramtype.order}. Only order 1, 2, and 3 are supported.")
        
        return F

# add a main function to test the freqdist
if __name__ == "__main__":
    freqdist = FreqDist.from_name("en")

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

    print("Top 3:")
    print(freqdist.top(3).normalized().freqdist)

    print()
    print("Numpy arrays for top 3:")
    print(freqdist.top(3).normalized().to_numpy())
