import hashlib
from pathlib import Path
from zipfile import ZipFile

import pytest

from smallpotato import _coco
from smallpotato._download import Artifact


def _write_zip(path: Path, root: str, files: dict[str, bytes]) -> None:
    with ZipFile(path, "w") as archive:
        for name, data in files.items():
            archive.writestr(f"{root}/{name}", data)


def _artifact(path: Path, filename: str | None = None) -> Artifact:
    data = path.read_bytes()
    return Artifact(
        filename=filename or path.name,
        urls=(path.as_uri(),),
        sha256=hashlib.sha256(data).hexdigest(),
        size=len(data),
    )


def _fake_coco(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[Path, bytes]:
    sources = tmp_path / "sources"
    sources.mkdir()

    val_archive = sources / "val2017.zip"
    _write_zip(
        val_archive,
        "val2017",
        {"0001.jpg": b"image\n"},
    )

    instances = b'{"images": []}\n'
    annotations_archive = sources / "annotations.zip"
    _write_zip(
        annotations_archive,
        "annotations",
        {"instances_val2017.json": instances},
    )

    monkeypatch.setattr(
        _coco,
        "VAL2017",
        _artifact(val_archive),
    )
    monkeypatch.setattr(
        _coco,
        "ANNOTATIONS",
        _artifact(
            annotations_archive,
            "annotations_trainval2017.zip",
        ),
    )
    monkeypatch.setattr(
        _coco,
        "INSTANCES_VAL2017_SHA256",
        hashlib.sha256(instances).hexdigest(),
    )
    monkeypatch.setattr(
        _coco,
        "INSTANCES_VAL2017_SIZE",
        len(instances),
    )

    return tmp_path / "cache", instances


def test_prepare_val2017_acquires_and_reuses_dataset(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cache, instances = _fake_coco(tmp_path, monkeypatch)

    prepared = _coco.prepare_val2017(cache)
    root = cache / "datasets" / "coco" / "2017"

    assert prepared.root == root
    assert prepared.images == root / "val2017"
    assert prepared.annotations == (
        root / "annotations" / "instances_val2017.json"
    )
    assert (prepared.images / "0001.jpg").read_bytes() == b"image\n"
    assert prepared.annotations.read_bytes() == instances

    assert _coco.prepare_val2017(cache) == prepared


def test_prepare_val2017_rejects_modified_extraction(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cache, _ = _fake_coco(tmp_path, monkeypatch)
    prepared = _coco.prepare_val2017(cache)

    (prepared.images / "0001.jpg").write_bytes(b"longer image\n")

    with pytest.raises(RuntimeError, match="size mismatch"):
        _coco.prepare_val2017(cache)


def test_prepare_val2017_rejects_invalid_archive(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cache, _ = _fake_coco(tmp_path, monkeypatch)

    archive = cache.parent / "sources" / "val2017.zip"
    with ZipFile(archive, "w") as file:
        file.writestr("unexpected/0001.jpg", b"image\n")

    monkeypatch.setattr(
        _coco,
        "VAL2017",
        _artifact(archive),
    )

    with pytest.raises(RuntimeError, match="unexpected member"):
        _coco.prepare_val2017(cache)

    assert not (cache / "datasets" / "coco" / "2017" / "val2017").exists()


def test_prepare_val2017_rejects_modified_annotations(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cache, instances = _fake_coco(tmp_path, monkeypatch)
    prepared = _coco.prepare_val2017(cache)

    modified = b'{"images": {}}\n'
    assert len(modified) == len(instances)

    prepared.annotations.write_bytes(modified)

    with pytest.raises(RuntimeError, match="SHA-256 mismatch"):
        _coco.prepare_val2017(cache)
