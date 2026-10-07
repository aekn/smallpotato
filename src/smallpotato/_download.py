__all__ = ()

import hashlib
from collections.abc import Sequence
from pathlib import Path
from tempfile import TemporaryDirectory
from urllib.request import Request, urlopen

_BLOCK_SIZE = 1024 * 1024
_TIMEOUT = 60.0
_USER_AGENT = "smallpotato"


def sha256_file(path: Path, /) -> str:
    if not path.exists():
        raise FileNotFoundError(path)
    if not path.is_file():
        raise RuntimeError(f"not a regular file: {path}")

    with path.open("rb") as file:
        return hashlib.file_digest(file, "sha256").hexdigest()


def verify_file(
    path: Path, /, *, sha256: str, size: int | None = None
) -> None:
    if not path.exists():
        raise FileNotFoundError(path)
    if not path.is_file():
        raise RuntimeError(f"not a regular file: {path}")

    actual_size = path.stat().st_size
    if size is not None and actual_size != size:
        raise RuntimeError(
            f"size mismatch for {path}: expected {size}, got {actual_size}"
        )

    digest = sha256_file(path)
    if digest != sha256:
        raise RuntimeError(
            f"SHA-256 mismatch for {path}: expected {sha256}, got {digest}"
        )


def acquire_file(
    path: Path,
    /,
    *,
    urls: Sequence[str],
    sha256: str,
    size: int | None = None,
) -> str | None:
    """Ensure that *path* contains the expected file.

    Return the source URL used to create the file, or None if the
    existing file already matches.
    """
    if path.exists():
        verify_file(path, sha256=sha256, size=size)
        return None

    if not urls:
        raise ValueError("urls must not be empty")

    path.parent.mkdir(parents=True, exist_ok=True)
    failures: list[tuple[str, Exception]] = []

    with TemporaryDirectory(
        dir=path.parent,
        prefix=f".{path.name}.",
    ) as directory:
        temporary = Path(directory) / path.name

        for url in urls:
            try:
                _download_file(
                    url,
                    temporary,
                    sha256=sha256,
                    size=size,
                )
            except (OSError, RuntimeError, ValueError) as exc:
                failures.append((url, exc))
                temporary.unlink(missing_ok=True)
                continue

            temporary.replace(path)
            return url

    error = RuntimeError(f"failed to acquire {path}")
    for url, exc in failures:
        error.add_note(f"{url}: {type(exc).__name__}: {exc}")
    raise error


def _download_file(
    url: str,
    path: Path,
    /,
    *,
    sha256: str,
    size: int | None,
) -> None:
    request = Request(url, headers={"User-Agent": _USER_AGENT})
    digest = hashlib.sha256()
    actual_size = 0

    with (
        urlopen(request, timeout=_TIMEOUT) as response,
        path.open("xb") as file,
    ):
        while chunk := response.read(_BLOCK_SIZE):
            file.write(chunk)
            digest.update(chunk)
            actual_size += len(chunk)

    if size is not None and actual_size != size:
        raise RuntimeError(
            f"size mismatch for {url}: expected {size}, got {actual_size}"
        )

    actual_sha256 = digest.hexdigest()
    if actual_sha256 != sha256:
        raise RuntimeError(
            f"SHA-256 mismatch for {url}: expected {sha256}, "
            f"got {actual_sha256}"
        )
