import hashlib
import json
import subprocess
from pathlib import Path
from types import SimpleNamespace
from zipfile import ZipFile

import pytest

from reproductions.rtmdet import _prepare

_CONFIG = Path("configs/rtmdet/rtmdet_tiny_8xb32-300e_coco.py")


def _write_zip(path: Path, root: str, files: dict[str, bytes]) -> None:
    with ZipFile(path, "w") as archive:
        for name, data in files.items():
            archive.writestr(f"{root}/{name}", data)


def _init_repo(path: Path) -> str:
    subprocess.run(("git", "init", "-q", str(path)), check=True)
    subprocess.run(
        ("git", "-C", str(path), "config", "user.email", "test@example.com"),
        check=True,
    )
    subprocess.run(
        ("git", "-C", str(path), "config", "user.name", "Test"),
        check=True,
    )

    config = path / _CONFIG
    config.parent.mkdir(parents=True)
    config.write_text("model = dict()\n", encoding="utf-8")

    subprocess.run(("git", "-C", str(path), "add", "."), check=True)
    subprocess.run(
        ("git", "-C", str(path), "commit", "-q", "-m", "test"),
        check=True,
    )
    result = subprocess.run(
        ("git", "-C", str(path), "rev-parse", "HEAD"),
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _acquired(path: Path) -> _prepare.AcquiredFile:
    return _prepare.AcquiredFile(path=path, source=None)


def _artifact(path: Path, filename: str | None = None) -> SimpleNamespace:
    data = path.read_bytes()
    return SimpleNamespace(
        filename=filename or path.name,
        urls=(path.as_uri(),),
        sha256=hashlib.sha256(data).hexdigest(),
        size=len(data),
    )


def _fake_inputs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[Path, bytes]:
    sources = tmp_path / "sources"
    sources.mkdir()

    upstream = sources / "mmdetection"
    revision = _init_repo(upstream)

    checkpoint = sources / "checkpoint.pth"
    checkpoint.write_bytes(b"checkpoint\n")

    val_archive = sources / "val2017.zip"
    _write_zip(val_archive, "val2017", {"0001.jpg": b"image\n"})

    instances = b'{"images": []}\n'
    ann_archive = sources / "annotations.zip"
    _write_zip(
        ann_archive,
        "annotations",
        {"instances_val2017.json": instances},
    )

    monkeypatch.setattr(
        _prepare,
        "MMDET_REPOSITORY",
        upstream.as_uri(),
    )
    monkeypatch.setattr(_prepare, "MMDET_REVISION", revision)
    monkeypatch.setattr(
        _prepare,
        "CHECKPOINT",
        _artifact(checkpoint, "checkpoint.pth"),
    )
    monkeypatch.setattr(
        _prepare,
        "VAL2017",
        _artifact(val_archive, "val2017.zip"),
    )
    monkeypatch.setattr(
        _prepare,
        "ANNOTATIONS",
        _artifact(ann_archive, "annotations_trainval2017.zip"),
    )
    monkeypatch.setattr(
        _prepare,
        "INSTANCES_VAL_SHA256",
        hashlib.sha256(instances).hexdigest(),
    )
    monkeypatch.setattr(_prepare, "INSTANCES_VAL_SIZE", len(instances))

    return tmp_path / "cache", instances


def test_prepare_inputs_acquires_verified_inputs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cache, instances = _fake_inputs(tmp_path, monkeypatch)

    prepared = _prepare.prepare_inputs(cache)

    assert (
        prepared.source.name
        == subprocess.run(
            ("git", "-C", str(prepared.source), "rev-parse", "HEAD"),
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    )
    assert prepared.config == prepared.source / _CONFIG
    assert prepared.source_fetched
    assert prepared.checkpoint.source is not None
    assert prepared.val_archive.source is not None
    assert prepared.annotations_archive.source is not None
    assert prepared.checkpoint.path.read_bytes() == b"checkpoint\n"
    assert (prepared.coco / "val2017" / "0001.jpg").read_bytes() == b"image\n"
    assert prepared.annotations.read_bytes() == instances

    cached = _prepare.prepare_inputs(cache)
    assert not cached.source_fetched
    assert cached.checkpoint.source is None
    assert cached.val_archive.source is None
    assert cached.annotations_archive.source is None


def test_prepare_inputs_rejects_modified_source(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cache, _ = _fake_inputs(tmp_path, monkeypatch)
    prepared = _prepare.prepare_inputs(cache)
    prepared.config.write_text(
        "model = dict(type='changed')\n",
        encoding="utf-8",
    )

    with pytest.raises(RuntimeError, match="source tree is not clean"):
        _prepare.prepare_inputs(cache)


def test_prepare_inputs_rejects_modified_extraction(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cache, _ = _fake_inputs(tmp_path, monkeypatch)
    prepared = _prepare.prepare_inputs(cache)
    image = prepared.coco / "val2017" / "0001.jpg"
    image.write_bytes(b"longer image\n")

    with pytest.raises(RuntimeError, match="size mismatch"):
        _prepare.prepare_inputs(cache)


def test_write_runtime_config_records_only_runtime_overrides(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source with spaces"
    config = source / _CONFIG
    config.parent.mkdir(parents=True)
    config.write_text("model = dict()\n", encoding="utf-8")

    coco = tmp_path / "coco data"
    annotations = coco / "annotations" / "instances_val2017.json"
    inputs = _prepare.PreparedInputs(
        source=source,
        source_fetched=False,
        config=config,
        checkpoint=_acquired(tmp_path / "checkpoint.pth"),
        coco=coco,
        annotations=annotations,
        val_archive=_acquired(tmp_path / "val2017.zip"),
        annotations_archive=_acquired(tmp_path / "annotations.zip"),
    )
    runtime = tmp_path / "run" / "runtime.py"
    prediction_prefix = tmp_path / "run" / "pred"

    _prepare.write_runtime_config(
        runtime,
        inputs,
        prediction_prefix=prediction_prefix,
        num_workers=2,
    )

    text = runtime.read_text(encoding="utf-8")
    compile(text, str(runtime), "exec")
    assert json.dumps(str(config.resolve())) in text
    assert "num_workers=2" in text
    assert "persistent_workers=True" in text
    assert "format_only=True" in text
    assert json.dumps(str(prediction_prefix.resolve())) in text
    assert "batch_size" not in text
    assert "score_thr" not in text
    assert "max_per_img" not in text


def test_write_runtime_config_disables_persistent_workers_at_zero(
    tmp_path: Path,
) -> None:
    inputs = _prepare.PreparedInputs(
        source=tmp_path,
        source_fetched=False,
        config=tmp_path / "config.py",
        checkpoint=_acquired(tmp_path / "checkpoint.pth"),
        coco=tmp_path / "coco",
        annotations=tmp_path / "instances.json",
        val_archive=_acquired(tmp_path / "val2017.zip"),
        annotations_archive=_acquired(tmp_path / "annotations.zip"),
    )
    runtime = tmp_path / "runtime.py"

    _prepare.write_runtime_config(
        runtime,
        inputs,
        prediction_prefix=tmp_path / "pred",
        num_workers=0,
    )

    text = runtime.read_text(encoding="utf-8")
    assert "num_workers=0" in text
    assert "persistent_workers=False" in text


def test_write_runtime_config_rejects_negative_workers(tmp_path: Path) -> None:
    inputs = _prepare.PreparedInputs(
        source=tmp_path,
        source_fetched=False,
        config=tmp_path / "config.py",
        checkpoint=_acquired(tmp_path / "checkpoint.pth"),
        coco=tmp_path / "coco",
        annotations=tmp_path / "instances.json",
        val_archive=_acquired(tmp_path / "val2017.zip"),
        annotations_archive=_acquired(tmp_path / "annotations.zip"),
    )

    with pytest.raises(ValueError, match="num_workers must not be negative"):
        _prepare.write_runtime_config(
            tmp_path / "runtime.py",
            inputs,
            prediction_prefix=tmp_path / "pred",
            num_workers=-1,
        )
