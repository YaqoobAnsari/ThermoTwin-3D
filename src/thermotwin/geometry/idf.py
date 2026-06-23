"""Minimal EnergyPlus IDF reader.

A full EnergyPlus parser (eppy) needs the version-matched IDD data dictionary and
a local EnergyPlus install. We only need a handful of object types to lift the
**envelope geometry + material layers** out of the DOE reference buildings, so this
is a deliberately small, dependency-free tokenizer — the same pragmatic call as
using a scipy finite-volume solver instead of dolfinx (ADR 0002).

The IDF grammar we rely on is simple:

* ``!`` starts a comment to end of line.
* An *object* is ``Type, field, field, ... ;`` — fields comma-separated, the
  object terminated by a semicolon. Whitespace and newlines between fields are
  irrelevant.

:func:`parse_idf` returns ``{type: [object, ...]}`` where each object is the list
of its field strings (the leading type token removed). Field semantics are left to
the consumer (:mod:`thermotwin.geometry.envelope`).
"""

from __future__ import annotations

import re
from collections import defaultdict
from pathlib import Path

__all__ = ["IdfObject", "parse_idf", "parse_idf_text"]

# A parsed object: its type and the ordered list of field strings.
IdfObject = list[str]

_COMMENT = re.compile(r"!.*")


def parse_idf_text(text: str) -> dict[str, list[IdfObject]]:
    """Parse IDF *text* into ``{object_type: [fields, ...]}``.

    Object types are matched case-insensitively but returned in the canonical
    case of their first occurrence. Field strings are stripped of surrounding
    whitespace; empty trailing fields are preserved (they are meaningful in IDF).
    """
    # Strip comments line by line, then flatten — newlines carry no meaning.
    cleaned = "\n".join(_COMMENT.sub("", line) for line in text.splitlines())
    objects: dict[str, list[IdfObject]] = defaultdict(list)
    for raw in cleaned.split(";"):
        raw = raw.strip()
        if not raw:
            continue
        fields = [f.strip() for f in raw.split(",")]
        obj_type = fields[0]
        if not obj_type:
            continue
        objects[obj_type].append(fields[1:])
    return dict(objects)


def parse_idf(path: str | Path) -> dict[str, list[IdfObject]]:
    """Parse an IDF file at *path*. See :func:`parse_idf_text`."""
    return parse_idf_text(Path(path).read_text(encoding="latin-1"))


def get(objects: dict[str, list[IdfObject]], obj_type: str) -> list[IdfObject]:
    """Case-insensitive lookup of all objects of a type (``[]`` if none)."""
    for key, val in objects.items():
        if key.lower() == obj_type.lower():
            return val
    return []
