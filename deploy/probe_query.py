"""The vector probe's query: the one fixed question every build asks the
corpus and every verifier asks it again.

Its own module, beside manifest_contract.py rather than inside it, because
it is not a contract about the package's SHAPE at all -- it is an input to
a measurement, chosen against the live corpus and re-validated against it
on every build. What the two have in common is only that both the producer
(manifest_probe.py, build-time) and the verifiers (drift_probe.py, bundled)
must read the SAME one: the query was once a literal in each of them, kept
in sync by a comment.
"""
from __future__ import annotations

# A paraphrase of "an algebraic polynomial bounded from its values on a
# uniform grid" (the recurring theme of 1997_sm280 and related papers) with
# genuinely ZERO shared stemmed lexeme against its own nearest page's
# wording -- not merely zero shared surface forms, and not excusing the
# domain noun either: earlier wordings that kept "полином"/"величина"/
# "оценить" etc. kept landing on pages that use the exact same word, which
# the stemmed check (probe_overlap.stemmed_token_overlap) correctly
# rejected. This wording was accepted only after gather_manifest ran clean
# against the live corpus (verified: phraseto_tsquery also finds "no
# matches" for it). Nearest page can legitimately drift as the corpus/model
# change -- gather_manifest() re-checks the invariant against whatever page
# is actually nearest on every build and refuses to record a pair that
# overlaps, rather than trusting this comment to still hold.
VECTOR_PROBE_QUERY = (
    "какое предельное значение по абсолютной величине допускает "
    "рациональная форма, заданная своими данными на равноотстоящих "
    "узлах отрезка"
)
