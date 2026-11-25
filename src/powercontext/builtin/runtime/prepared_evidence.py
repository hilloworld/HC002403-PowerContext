# Copyright (c) 2026 OceanBase.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Rank literal Memory spans by lexical relevance to a query."""

from __future__ import annotations

import re
import unicodedata
from collections import Counter
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from math import log, sqrt

_SENTENCE_END = re.compile(r"""[.!?]["'\u201d\u2019)\]]*(?=\s|$)|[\u3002\uff01\uff1f]+|\r\n?|[\n\u0085\u2028\u2029]""")
_SHORT_FORMS = re.compile(r"(?:\b(?i:e\.g|i\.e|fig|ref|dr|vs|et al)|\b[A-Z])\.$")
_WORD = re.compile(r"[\u3040-\u30ff\u3400-\u9fff]+|[^\W_\u3040-\u30ff\u3400-\u9fff]+")
_CJK = re.compile(r"[\u3040-\u30ff\u3400-\u9fff]")
_STOP_WORDS = frozenset(
    "a an and are as at be been by did do does for from had has have how in is it of on or that the "  # noqa: SIM905
    "their these they this those to was were what when where which who will with would".split()
)


@dataclass(frozen=True)
class EvidenceSpan:
    entry_index: int
    start: int
    end: int
    anchor: tuple[int, int] | None


@dataclass(frozen=True)
class _Sentence:
    start: int
    end: int
    word_count: int
    matches: dict[str, tuple[int, int]]


def _sentences(text: str) -> Iterator[tuple[int, int]]:
    start = 0
    for boundary in _SENTENCE_END.finditer(text):
        end = boundary.end()
        if boundary[0].startswith(".") and _SHORT_FORMS.search(text, max(start, end - 12), end):
            continue
        yield start, end
        start = end
    if start < len(text):
        yield start, len(text)


def _english_stem(word: str) -> str:
    # Cover common inflections and verb/nominalization pairs such as activate/activation.
    if len(word) > 4 and word.endswith(("ies", "ied")):
        return word[:-3] + "y"
    for suffix in ("ing", "ed", "ion", "es", "s"):
        if len(word) - len(suffix) < 3 or not word.endswith(suffix):
            continue
        if suffix == "es" and not word[:-2].endswith(("ch", "sh", "ss", "x", "z")):
            continue
        if suffix == "s" and word.endswith(("ss", "us", "is")):
            continue
        word = word[: -len(suffix)]
        if suffix in ("ing", "ed") and word[-1] == word[-2] and word[-1] in "bdgmnprt":
            word = word[:-1]
        break
    return word[:-1] if len(word) > 4 and word.endswith("e") else word


def _terms(text: str, start: int = 0, end: int | None = None) -> Iterator[tuple[str, int, int]]:
    for match in _WORD.finditer(text, start, len(text) if end is None else end):
        word = unicodedata.normalize("NFKC", match[0]).casefold()
        if word in _STOP_WORDS:
            continue
        if _CJK.match(match[0]) and len(match[0]) > 1:
            for index in range(match.start(), match.end() - 1):
                yield unicodedata.normalize("NFKC", text[index : index + 2]), index, index + 2
        else:
            yield _english_stem(word) if word.isascii() and word.isalpha() else word, match.start(), match.end()


def rank_evidence_spans(query: str, texts: Sequence[str]) -> list[EvidenceSpan]:
    """Prefer rare query terms and nearby supporting sentences, keeping source offsets."""
    query_terms = dict.fromkeys(term for term, _, _ in _terms(query))
    if not query_terms:
        return []

    sentences: list[list[_Sentence]] = []
    frequencies: Counter[str] = Counter()
    for text in texts:
        document: list[_Sentence] = []
        for start, end in _sentences(text):
            raw = text[start:end]
            stripped = raw.strip()
            if not stripped:
                continue
            start += len(raw) - len(raw.lstrip())
            end = start + len(stripped)
            matches: dict[str, tuple[int, int]] = {}
            word_count = 0
            for term, left, right in _terms(text, start, end):
                word_count += 1
                if term in query_terms:
                    matches.setdefault(term, (left, right))
            frequencies.update(matches.keys())
            document.append(_Sentence(start, end, word_count, matches))
        sentences.append(document)

    total_sentences = sum(map(len, sentences))
    weights = {term: 1.0 + log((total_sentences + 1) / (count + 1)) for term, count in frequencies.items()}
    ranked: list[tuple[float, EvidenceSpan]] = []
    for entry_index, document in enumerate(sentences):
        direct = [
            sum(weights[term] for term in sentence.matches) / (1.0 + 0.08 * sqrt(max(0, sentence.word_count - 20)))
            for sentence in document
        ]
        for index, sentence in enumerate(document):
            matches = sentence.matches
            nearby = max((score for score in direct[max(0, index - 1) : index + 2]), default=0.0)
            score = direct[index] + 0.18 * nearby
            if score:
                anchor = matches[max(matches, key=weights.__getitem__)] if matches else None
                ranked.append((score, EvidenceSpan(entry_index, sentence.start, sentence.end, anchor)))

    ranked.sort(key=lambda pair: pair[0], reverse=True)
    return [span for _, span in ranked]
