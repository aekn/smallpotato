import hashlib
from collections.abc import Sequence
from dataclasses import dataclass
from http.client import HTTPException
from pathlib import Path
from tempfile import TemporaryDirectory
from urllib.request import Request, urlopen

_BLOCK_SIZE = 1024 * 1024
_TIMEOUT = 60.0
_USER_AGENT = "smallpotato"


@dataclass(frozen=True, slots=True)
class Artifact:
    filename: str
    urls: tuple[str, ...]
    sha256: str
    size: int | None = None


def sha256_file(path: Path, /) -> str:
    with path.open("rb") as file:
        return hashlib.file_digest(file, "sha256").hexdigest()


def verify_file(
    path: Path,
    /,
    *,
    sha256: str,
    size: int | None = None,
) -> None:
    if size is not None:
        actual_size = path.stat().st_size
        if actual_size != size:
            raise RuntimeError(
                f"size mismatch for {path}: expected {size}, got {actual_size}"
            )

    actual_sha256 = sha256_file(path)
    if actual_sha256 != sha256:
        raise RuntimeError(
            f"SHA-256 mismatch for {path}: expected {sha256}, "
            f"got {actual_sha256}"
        )


def acquire_artifact(directory: Path, artifact: Artifact, /) -> Path:
    path = directory / artifact.filename
    acquire_file(
        path,
        urls=artifact.urls,
        sha256=artifact.sha256,
        size=artifact.size,
    )
    return path


def acquire_file(
    path: Path,
    /,
    *,
    urls: Sequence[str],
    sha256: str,
    size: int | None = None,
) -> None:
    if path.exists():
        verify_file(path, sha256=sha256, size=size)
        return
    if not urls:
        raise ValueError("urls must not be empty")

    path.parent.mkdir(parents=True, exist_ok=True)
    failures: list[str] = []

    with TemporaryDirectory(
        dir=path.parent,
        prefix=f".{path.name}.",
    ) as directory:
        temporary = Path(directory) / path.name

        for url in urls:
            try:
                _download_file(url, temporary, sha256=sha256, size=size)
            except (OSError, HTTPException, RuntimeError, ValueError) as exc:
                failures.append(f"{url}: {exc}")
                continue

            temporary.replace(path)
            return

    detail = "; ".join(failures)
    raise RuntimeError(f"failed to acquire {path}: {detail}")


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
        path.open("wb") as file,
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
