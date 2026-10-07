import hashlib
from pathlib import Path

import pytest

from smallpotato._download import acquire_file, sha256_file, verify_file


def _identity(data: bytes) -> tuple[str, int]:
    return hashlib.sha256(data).hexdigest(), len(data)


def test_sha256_file(tmp_path: Path) -> None:
    path = tmp_path / "file.bin"
    path.write_bytes(b"data")

    assert sha256_file(path) == hashlib.sha256(b"data").hexdigest()


def test_verify_file(tmp_path: Path) -> None:
    path = tmp_path / "artifact.bin"
    data = b"smallpotato\n"
    path.write_bytes(data)
    sha256, size = _identity(data)

    verify_file(path, sha256=sha256, size=size)


def test_verify_file_rejects_wrong_digest(tmp_path: Path) -> None:
    path = tmp_path / "artifact.bin"
    data = b"smallpotato\n"
    path.write_bytes(data)
    _, size = _identity(data)

    with pytest.raises(RuntimeError, match="SHA-256 mismatch"):
        verify_file(path, sha256="0" * 64, size=size)


def test_acquire_file_uses_valid_existing_file(tmp_path: Path) -> None:
    path = tmp_path / "artifact.bin"
    data = b"smallpotato\n"
    path.write_bytes(data)
    sha256, size = _identity(data)

    source = acquire_file(
        path,
        urls=("file:///does/not/exist",),
        sha256=sha256,
        size=size,
    )

    assert source is None
    assert path.read_bytes() == data


def test_acquire_file_falls_back_to_next_source(tmp_path: Path) -> None:
    bad = tmp_path / "bad.bin"
    good = tmp_path / "good.bin"
    path = tmp_path / "cache" / "artifact.bin"
    data = b"smallpotato\n"

    bad.write_bytes(b"wrong\n")
    good.write_bytes(data)
    sha256, _ = _identity(data)

    source = acquire_file(
        path,
        urls=(bad.as_uri(), good.as_uri()),
        sha256=sha256,
    )

    assert source == good.as_uri()
    assert path.read_bytes() == data


def test_acquire_file_does_not_replace_invalid_existing_file(
    tmp_path: Path,
) -> None:
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


def test_acquire_file_requires_source(tmp_path: Path) -> None:
    path = tmp_path / "artifact.bin"

    with pytest.raises(ValueError, match="urls must not be empty"):
        acquire_file(
            path,
            urls=(),
            sha256="0" * 64,
        )


def test_acquire_file_reports_source_failures(tmp_path: Path) -> None:
    first = tmp_path / "first.bin"
    second = tmp_path / "second.bin"
    path = tmp_path / "cache" / "artifact.bin"

    first.write_bytes(b"first\n")
    second.write_bytes(b"second\n")

    with pytest.raises(RuntimeError, match="failed to acquire") as exc_info:
        acquire_file(
            path,
            urls=(first.as_uri(), second.as_uri()),
            sha256="0" * 64,
        )

    notes = exc_info.value.__notes__
    assert notes is not None
    assert len(notes) == 2
    assert first.as_uri() in notes[0]
    assert "SHA-256 mismatch" in notes[0]
    assert second.as_uri() in notes[1]
    assert "SHA-256 mismatch" in notes[1]
    assert not path.exists()
