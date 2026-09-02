"""WHOSE decision the artifact's citation mode was -- the one artifact-side
citation check that reads no dump byte, split out of
citation_content_checks.py for that reason and for module size
(kb/CLAUDE.md FILE_SIZE).

Everything in the sibling module holds the dump to the manifest. This one
holds the MANIFEST to itself: an artifact built with --policy-override is
byte-for-byte indistinguishable from an owner-classified one -- same
profile, same schemas, same counts, a perfectly consistent dump -- and the
filename is not part of the package. manifest.citation.policy_source is the
only field that answers it, so it is read here strictly, with no default
anywhere in the path.

Strictly means: the block must be there, and both its mode and its source
must be values this reader knows. Written defensively (`manifest.get(...,
{}).get(...)`), the whole citation half of the certification degraded to
"nothing to check" the moment the block was missing -- a manifest that says
nothing about the citation policy would then certify clean, which is
precisely the outcome PUBLIC_APPROVED_BY_OWNER and CITATION_POLICY_IS_DATA
forbid. profile_checks.py's version gate keeps the honest predecessor of
such a manifest (an artifact from before the field existed) from ever
reaching this check at all; what reaches it is a CURRENT manifest with a
hole in it.
"""
from __future__ import annotations

from manifest_contract import CitationMode, Key, PolicySource, Profile


def check_policy_is_the_owners(manifest: dict) -> tuple[bool, str]:
    """The citation mode this artifact applied must be the owner's decision.

    --policy-override forces a mode without reading citation.public_policy,
    which is legitimate for exercising the pipeline and never legitimate for
    an artifact anybody publishes. The build records which it was; this
    refuses the one it must.

    An override is refused BEFORE the mode is looked at: `--policy-override
    none` produces an artifact carrying no citation schema at all, and the
    refusal is about the provenance of the decision, not about how much it
    let through. For the same reason mode 'none' is certified rather than
    excused: "the graph does not travel" is itself a decision the owner
    records (CITATION_POLICY_IS_DATA), and an artifact asserting it must
    name who made it.

    WHICH source is required depends on the profile, because only the
    public one applies a policy: public must name the owner, and anything
    else must say "not-applicable" -- a full artifact claiming an owner
    decision names one nobody made, since the packager never reads
    citation.public_policy for that profile. The two are refused in each
    other's place, not merely accepted loosely.
    """
    citation = manifest.get(Key.CITATION)
    if not isinstance(citation, dict):
        return False, (
            f"манифест не несёт блока {Key.CITATION} — чьё решение применено "
            "к графу цитирований, из пакета не узнать; пересоберите артефакт "
            "текущим сборщиком"
        )
    mode = citation.get(Key.CITATION_MODE)
    source = citation.get(Key.CITATION_POLICY_SOURCE)
    if mode not in CitationMode.ALL:
        return False, (
            f"citation.mode={mode!r} — не из словаря {CitationMode.ALL}; "
            "режим, которого этот читатель не знает, не сертифицируется"
        )
    if source == PolicySource.OVERRIDE:
        return False, (
            "артефакт собран с --policy-override, не по решению владельца; "
            f"публиковать нельзя (mode={mode!r} задан командной строкой, а не "
            "citation.public_policy)"
        )
    wanted = (PolicySource.OWNER if manifest.get(Key.PROFILE) == Profile.PUBLIC
              else PolicySource.NOT_APPLICABLE)
    if source != wanted:
        return False, (
            f"citation.policy_source={source!r} при профиле "
            f"{manifest.get(Key.PROFILE)!r} — ожидалось {wanted!r} "
            f"(значения: {PolicySource.ALL}); пересоберите артефакт "
            "текущим сборщиком"
        )
    if wanted == PolicySource.NOT_APPLICABLE:
        return True, (f"policy_source={source!r}, mode={mode!r} — профиль "
                      "политики не применяет, решать было нечего")
    return True, f"policy_source={source!r}, mode={mode!r} — решение владельца"
