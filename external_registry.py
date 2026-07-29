"""theory/external/EXTERNAL_INDEX.md -- the registry of sources by OTHER authors.

The corpus proper is one author's work (theory/iis/). External literature --
somebody else's paper, preprint or handbook -- is a different kind of holding
and gets its own directory, its own loader (pg_load_external.py) and its own
registry, which this module reads. The registry is DATA in the data tree, not
code: the file lives beside the sources it describes and never enters git.

Two rules the format exists to enforce:

- **A source without a stated reason is not stored.** The knowledge base is
  not a bookmark folder; every row must say which task/issue the work serves,
  and the reason travels into corpus.documents.note where a reader who found
  the page by search will actually see it.
- **The regime is the strictest one, mechanically.** Every external document
  is loaded as LEGAL_CLASS / PUBLIC_DISTRIBUTION below, with no per-row say in
  the matter: this is somebody else's copyright, so not one byte of it may
  leave in a public artifact -- not the text, not the source blob, not the
  bibliography row, not a page vector. The per-row `правовой режим` column
  records the BASIS (which licence, checked where, on what date) and becomes
  legal_note; it does not decide the distribution.

That the packager itself knows nothing of this class is the point: it reads
corpus.documents.public_distribution and nothing else (deploy/legal_profile.py,
invariant LEGAL_IS_DATA). This module writes that data; corpus_completeness.py
re-reads it from the database and fails if a document under theory/external/
ever carries anything weaker.

Format: a markdown table, exactly COLUMNS in that order, `|` forbidden inside a
cell. Parsed strictly -- a row with the wrong number of cells raises rather
than being skipped, because a silently ignored row is a source that vanishes
from the registry while its file stays on disk.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from paths import EXTERNAL_SOURCE_DIR, default_external_dir

REGISTRY_FILENAME = "EXTERNAL_INDEX.md"
REGISTRY_DOCUMENT_ID = Path(REGISTRY_FILENAME).stem

# Written by pg_load_external.py into every row it loads. Not configurable per
# source: see the module docstring.
LEGAL_CLASS = "external-literature"
PUBLIC_DISTRIBUTION = "excluded"

# The registry file itself is the one document under theory/external/ that is
# OURS: a hand-curated list of what the project consulted and why, in our own
# words, exactly like theory/iis/INDEX.md. It is indexed on the same terms and
# carries the same class those two do -- a bibliography of public works is
# nobody else's copyright, and a knowledge base that cannot answer "which
# outside literature do we hold, and why" is missing the point of holding it.
REGISTRY_LEGAL_CLASS = "internal-metadata"
REGISTRY_DISTRIBUTION = "internal"
REGISTRY_NOTE = ("реестр внешних источников: что взято, откуда, зачем и на "
                 "каком правовом основании; ведётся вручную, читается "
                 "kb/external_registry.py")

COLUMNS = ("файл", "source_tier", "канонический URL", "правовой режим",
           "библиография", "зачем взяли")

# Suffixes a registry row may name. .pdf is a source document (loaded page by
# page); .md is a bibliography-only record we wrote ourselves for a work whose
# text we do not hold -- the honest form of "this work exists, here is why it
# matters, and we have not read it".
SOURCE_SUFFIXES = (".pdf", ".md")


class RegistryError(RuntimeError):
    """The registry cannot be read as written -- never guessed around."""


@dataclass(frozen=True)
class ExternalSource:
    filename: str
    source_tier: str
    source_url: str
    legal_note: str
    bibliography: str
    reason: str

    @property
    def document_id(self) -> str:
        return Path(self.filename).stem

    @property
    def is_pdf(self) -> bool:
        return Path(self.filename).suffix == ".pdf"

    @property
    def note(self) -> str:
        """corpus.documents.note: the bibliography and the reason, together.

        Both in one field on purpose -- a reader who lands on a page through
        search sees the note, not this registry.
        """
        return f"{self.bibliography} || зачем взято: {self.reason}"


def _cells(line: str) -> list[str]:
    stripped = line.strip()
    if not stripped.startswith("|"):
        return []
    return [cell.strip() for cell in stripped.strip("|").split("|")]


def parse_registry(text: str) -> list[ExternalSource]:
    """Every data row of the registry table, in file order.

    Rows before the header (prose, the format description) are ignored; from
    the header on, a row that does not have exactly len(COLUMNS) cells is an
    error naming the offending line.
    """
    sources: list[ExternalSource] = []
    seen: set[str] = set()
    header_seen = False
    for number, line in enumerate(text.splitlines(), start=1):
        cells = _cells(line)
        if not cells:
            continue
        if not header_seen:
            header_seen = tuple(cells) == COLUMNS
            continue
        if all(set(cell) <= set("-: ") for cell in cells):
            continue  # the markdown separator row
        if len(cells) != len(COLUMNS):
            raise RegistryError(
                f"{REGISTRY_FILENAME}:{number}: {len(cells)} ячеек вместо "
                f"{len(COLUMNS)} ({COLUMNS}); символ | внутри ячейки запрещён"
            )
        source = ExternalSource(*cells)
        _validate(source, number)
        if source.filename in seen:
            raise RegistryError(f"{REGISTRY_FILENAME}:{number}: файл "
                                f"{source.filename} перечислен дважды")
        seen.add(source.filename)
        sources.append(source)
    if not header_seen:
        raise RegistryError(
            f"{REGISTRY_FILENAME}: таблицы с колонками {COLUMNS} в файле нет")
    return sources


def _validate(source: ExternalSource, number: int) -> None:
    where = f"{REGISTRY_FILENAME}:{number} ({source.filename or '<без файла>'})"
    if Path(source.filename).suffix not in SOURCE_SUFFIXES:
        raise RegistryError(f"{where}: расширение не из {SOURCE_SUFFIXES}")
    if "/" in source.filename:
        raise RegistryError(f"{where}: каталог плоский, подкаталогов нет")
    if not source.source_url.startswith(("http://", "https://")):
        raise RegistryError(f"{where}: канонический URL обязателен")
    for field, value in (("source_tier", source.source_tier),
                         ("правовой режим", source.legal_note),
                         ("библиография", source.bibliography),
                         ("зачем взяли", source.reason)):
        if not value:
            raise RegistryError(f"{where}: пустая колонка «{field}»")


def load_registry(directory: Path | None = None) -> list[ExternalSource]:
    """The registry as written on disk; empty list when theory/external/ does
    not exist yet (a data tree with no external sources is legitimate, and a
    missing directory must not break the completeness predicate).
    """
    directory = directory or default_external_dir()
    path = directory / REGISTRY_FILENAME
    if not path.is_file():
        if directory.is_dir():
            raise RegistryError(
                f"{directory} есть, а {REGISTRY_FILENAME} в нём нет: источник "
                "без реестра — закладка, а не запись базы")
        return []
    return parse_registry(path.read_text(encoding="utf-8"))


def registry_problems(directory: Path, sources: list[ExternalSource]) -> list[str]:
    """Registry against the disk, both directions.

    A file with no row would be a source nobody can say why we hold; a row
    with no file would be a document the loader silently skips.
    """
    if not directory.is_dir():
        return [f"НЕТ КАТАЛОГА: {EXTERNAL_SOURCE_DIR} перечислен в реестре, "
                f"каталога нет"] if sources else []
    on_disk = {
        path.name for path in directory.iterdir()
        if path.is_file() and path.name != REGISTRY_FILENAME
        and path.suffix in SOURCE_SUFFIXES
    }
    listed = {source.filename for source in sources}
    problems = [f"НЕТ В РЕЕСТРЕ: {EXTERNAL_SOURCE_DIR}/{name} — дописать строку "
                f"в {REGISTRY_FILENAME} (зачем взято, URL, режим)"
                for name in sorted(on_disk - listed)]
    problems += [f"НЕТ ФАЙЛА: {REGISTRY_FILENAME} называет {name}, "
                 f"на диске его нет" for name in sorted(listed - on_disk)]
    return problems
