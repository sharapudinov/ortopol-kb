"""Is the dump on disk the one the manifest describes -- and the streaming
sha256 that answers it.

The dump embeds every source PDF/djvu blob in the corpus, so it is expected
to be hundreds of MB to several GB. Every side of the pipeline hashes it:
build_package.py after writing it, and every reader that certifies the
extracted package before trusting a byte of it. Neither side may read the
whole file into memory to do so.

The COMPARISON lives here beside the hash, and one implementation serves
both readers. It was smoke_test.py's alone -- three inline conditions in
the Docker path -- so profile_checks.py, the certifier that travels inside
the artifact and is documented (AGENT_GUIDE.md) as runnable on its own,
inspected a dump it never checked was the declared one. That is the one
question an unsigned manifest still lets a recipient answer about the
package as a whole: every other check reads the dump's contents, and
contents nobody tied to the manifest's own numbers are contents of some
file that happens to sit at that path.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

from manifest_keys import Key

CHUNK_SIZE = 1 << 20  # 1 MiB


def sha256_file(path: Path, chunk_size: int = CHUNK_SIZE) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def check_dump_matches_manifest(manifest: dict, artifact_dir: Path) -> tuple[bool, str]:
    """The named dump exists, is exactly as long as the manifest says, and
    hashes to the digest it declares.

    Both numbers, not just the digest: the size is what makes a truncated
    file say so cheaply, and it is the first thing to name in the verdict.

    Defensive about the block's shape rather than trusting it, because this
    is a gate and the shapes it would raise on -- a `dump` that is not a
    mapping, a missing name -- are exactly the hand-edited and
    partially-extracted cases it exists to report. (profile_checks.py has a
    shape gate one step earlier; smoke_test.py calls this before anything
    else touches the manifest.)
    """
    dump = manifest.get(Key.DUMP)
    if not isinstance(dump, dict) or not isinstance(dump.get(Key.FILE), str):
        return False, (f"manifest {Key.DUMP} не называет файла ({dump!r}): сверять "
                       "нечего -- пересоберите артефакт текущим сборщиком")
    path = artifact_dir / dump[Key.FILE]
    if not path.is_file():
        return False, f"{dump[Key.FILE]}: файла нет в {artifact_dir}"
    size = path.stat().st_size
    declared_size, declared_sha = dump.get(Key.BYTES), dump.get(Key.SHA256)
    if size != declared_size:
        return False, (f"{path.name}: {size} bytes, манифест обещает {declared_size!r} "
                       "-- дамп обрезан или подменён")
    digest = sha256_file(path)
    if digest != declared_sha:
        return False, (f"{path.name}: sha256 {digest}, манифест обещает {declared_sha!r} "
                       "-- содержимое не то, что описано")
    return True, f"{path.name}: {size} bytes, sha256 сошёлся"
