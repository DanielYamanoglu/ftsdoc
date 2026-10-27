"""Shared tokenisation so training data and inference see identical tokens."""
import re

_WORD = re.compile(r"\S+")
_TRAIL = ".,;:)("


def tokenize(text: str) -> list[tuple[str, int, int]]:
    """Whitespace tokens with trailing/leading punctuation split off.

    Returns (token, start, end) character offsets into ``text``.
    """
    out: list[tuple[str, int, int]] = []
    for m in _WORD.finditer(text):
        s, e = m.start(), m.end()
        word = m.group()
        # peel leading "(" and trailing ".,;:)" so entities stay clean
        while len(word) > 1 and word[0] in "(#":
            out.append((word[0], s, s + 1))
            s += 1
            word = word[1:]
        trail: list[tuple[str, int, int]] = []
        while len(word) > 1 and word[-1] in _TRAIL:
            trail.append((word[-1], e - 1, e))
            e -= 1
            word = word[:-1]
        out.append((word, s, e))
        out.extend(reversed(trail))
    return out
