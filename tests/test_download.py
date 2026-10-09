import hashlib
from pathlib import Path

import pytest

from smallpotato._download import acquire_file, verify_file


def _identity(data: bytes) -> tuple[str, int]:
    return hashlib.sha256(data).hexdigest(), len(data)


def test_verify_file_rejects_wrong_digest(tmp_path: Path) -> None:
    path = tmp_path / "artifact.bin"
    path.write_bytes(b"smallpotato\n")

    with pytest.raises(RuntimeError, match="SHA-256 mismatch"):
        verify_file(path, sha256="0" * 64)


def test_acquire_file_reuses_valid_cached_file(tmp_path: Path) -> None:
    path = tmp_path / "artifact.bin"
    data = b"smallpotato\n"
    path.write_bytes(data)
    sha256, size = _identity(data)

    acquire_file(
        path,
        urls=("file:///does/not/exist",),
        sha256=sha256,
        size=size,
    )

    assert path.read_bytes() == data


def test_acquire_file_falls_back_to_valid_source(tmp_path: Path) -> None:
    bad = tmp_path / "bad.bin"
    good = tmp_path / "good.bin"
    path = tmp_path / "cache" / "artifact.bin"
    data = b"smallpotato\n"

    bad.write_bytes(b"wrong\n")
    good.write_bytes(data)
    sha256, size = _identity(data)

    acquire_file(
        path,
        urls=(bad.as_uri(), good.as_uri()),
        sha256=sha256,
        size=size,
    )

    assert path.read_bytes() == data


def test_acquire_file_preserves_invalid_cached_file(tmp_path: Path) -> None:
    source = tmp_path / "source.bin"
    path = tmp_path / "artifact.bin"
    data = b"smallpotato\n"

    source.write_bytes(data)
    path.write_bytes(b"wrong\n")
    sha256, size = _identity(data)

    with pytest.raises(RuntimeError, match="size mismatch"):
        acquire_file(
            path,
            urls=(source.as_uri(),),
            sha256=sha256,
            size=size,
        )

    assert path.read_bytes() == b"wrong\n"
