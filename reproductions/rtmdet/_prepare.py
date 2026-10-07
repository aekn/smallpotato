__all__ = ()

import json
import os
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from subprocess import CalledProcessError
from tempfile import TemporaryDirectory
from typing import final
from zipfile import ZipFile

from smallpotato._download import acquire_file, verify_file
from smallpotato._process import capture_output

MMDET_REPOSITORY = "https://github.com/open-mmlab/mmdetection.git"
MMDET_REVISION = "44ebd17b145c2372c4b700bfb9cb20dbd28ab64a"
MMDET_CONFIG = Path("configs/rtmdet/rtmdet_tiny_8xb32-300e_coco.py")

_COCO_MIRROR_REVISION = "5200d2cffae8121713bec767b4693bdafc0eeb0b"

INSTANCES_VAL_SHA256 = (
    "e8c7f7908f1d7278341fae127d0da654f102f11bd7b21d8aeefa635b8c810b6f"
)
INSTANCES_VAL_SIZE = 19_987_840


@final
@dataclass(frozen=True, slots=True, match_args=False)
class Artifact:
    filename: str
    urls: tuple[str, ...]
    sha256: str
    size: int | None = None


@final
@dataclass(frozen=True, slots=True, match_args=False)
class AcquiredFile:
    """A verified file and its transport source for this run."""

    path: Path
    source: str | None


@final
@dataclass(frozen=True, slots=True, match_args=False)
class PreparedInputs:
    """Inputs for the RTMDet reproduction."""

    source: Path
    source_fetched: bool
    config: Path
    checkpoint: AcquiredFile
    coco: Path
    annotations: Path
    val_archive: AcquiredFile
    annotations_archive: AcquiredFile


VAL2017 = Artifact(
    filename="val2017.zip",
    urls=(
        "http://images.cocodataset.org/zips/val2017.zip",
        "https://huggingface.co/datasets/pcuenq/coco-2017-mirror/resolve/"
        f"{_COCO_MIRROR_REVISION}/val2017.zip?download=true",
    ),
    sha256=(
        "4f7e2ccb2866ec5041993c9cf2a952bbed69647b115d0f74da7ce8f4bef82f05"
    ),
    size=815_585_330,
)

ANNOTATIONS = Artifact(
    filename="annotations_trainval2017.zip",
    urls=(
        "http://images.cocodataset.org/annotations/"
        "annotations_trainval2017.zip",
        "https://huggingface.co/datasets/pcuenq/coco-2017-mirror/resolve/"
        f"{_COCO_MIRROR_REVISION}/annotations_trainval2017.zip?download=true",
    ),
    sha256=(
        "113a836d90195ee1f884e704da6304dfaaecff1f023f49b6ca93c4aaae470268"
    ),
    size=252_907_541,
)

CHECKPOINT = Artifact(
    filename="rtmdet_tiny_8xb32-300e_coco_20220902_112414-78e30dcc.pth",
    urls=(
        "https://download.openmmlab.com/mmdetection/v3.0/rtmdet/"
        "rtmdet_tiny_8xb32-300e_coco/"
        "rtmdet_tiny_8xb32-300e_coco_20220902_112414-78e30dcc.pth",
    ),
    sha256=(
        "78e30dcce0c6f594eaff0d6977b84b4103688b4aff0ad1aa16008a8cc854a7fb"
    ),
)


def prepare_inputs(cache: Path, /) -> PreparedInputs:
    cache = cache.expanduser().resolve()
    source, source_fetched = _prepare_source(cache)
    checkpoint = _acquire(
        cache / "checkpoints" / "rtmdet" / CHECKPOINT.filename,
        CHECKPOINT,
    )
    coco, annotations, val_archive, annotations_archive = _prepare_coco(cache)

    return PreparedInputs(
        source=source,
        source_fetched=source_fetched,
        config=source / MMDET_CONFIG,
        checkpoint=checkpoint,
        coco=coco,
        annotations=annotations,
        val_archive=val_archive,
        annotations_archive=annotations_archive,
    )


def write_runtime_config(
    path: Path,
    inputs: PreparedInputs,
    /,
    *,
    prediction_prefix: Path,
    num_workers: int = 2,
) -> None:
    if num_workers < 0:
        raise ValueError("num_workers must not be negative")

    path.parent.mkdir(parents=True, exist_ok=True)
    text = (
        f"_base_ = {_python_string(inputs.config)}\n\n"
        "test_dataloader = dict(\n"
        f"    num_workers={num_workers},\n"
        f"    persistent_workers={num_workers > 0},\n"
        "    dataset=dict(\n"
        f"        data_root={_python_string(inputs.coco)},\n"
        "    ),\n"
        ")\n\n"
        "test_evaluator = dict(\n"
        f"    ann_file={_python_string(inputs.annotations)},\n"
        "    format_only=True,\n"
        f"    outfile_prefix={_python_string(prediction_prefix)},\n"
        ")\n"
    )
    path.write_text(text, encoding="utf-8")


def _prepare_source(cache: Path) -> tuple[Path, bool]:
    parent = cache / "sources" / "mmdetection"
    destination = parent / MMDET_REVISION

    if destination.exists():
        verify_source(destination)
        return destination, False

    parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=parent, prefix=".mmdetection.") as directory:
        checkout = Path(directory) / "checkout"
        _run_git(None, "init", "-q", str(checkout))
        _run_git(checkout, "remote", "add", "origin", MMDET_REPOSITORY)
        _run_git(
            checkout,
            "fetch",
            "-q",
            "--depth=1",
            "origin",
            MMDET_REVISION,
        )
        _run_git(checkout, "checkout", "-q", "--detach", "FETCH_HEAD")
        verify_source(checkout)
        checkout.replace(destination)

    return destination, True


def verify_source(path: Path, /) -> None:
    if not path.is_dir():
        raise RuntimeError(f"not a source directory: {path}")

    revision = _run_git(path, "rev-parse", "HEAD")
    if revision != MMDET_REVISION:
        raise RuntimeError(
            f"MMDetection revision mismatch: expected {MMDET_REVISION}, "
            f"got {revision}"
        )

    status = _run_git(path, "status", "--porcelain", "--untracked-files=all")
    if status:
        raise RuntimeError(f"source tree is not clean: {path}")

    config = path / MMDET_CONFIG
    if not config.is_file():
        raise RuntimeError(f"missing RTMDet config: {config}")


def _prepare_coco(
    cache: Path,
) -> tuple[Path, Path, AcquiredFile, AcquiredFile]:
    root = cache / "datasets" / "coco" / "2017"
    archives = root / "archives"

    val_archive = _acquire(archives / VAL2017.filename, VAL2017)
    ann_archive = _acquire(archives / ANNOTATIONS.filename, ANNOTATIONS)

    images = root / "val2017"
    annotations = root / "annotations"
    _extract_directory(val_archive.path, "val2017", images)
    _extract_directory(ann_archive.path, "annotations", annotations)

    _verify_extracted_tree(val_archive.path, "val2017", images)
    _verify_extracted_tree(ann_archive.path, "annotations", annotations)

    instances = annotations / "instances_val2017.json"
    verify_file(
        instances,
        sha256=INSTANCES_VAL_SHA256,
        size=INSTANCES_VAL_SIZE,
    )
    return root, instances, val_archive, ann_archive


def _acquire(path: Path, artifact: Artifact) -> AcquiredFile:
    source = acquire_file(
        path,
        urls=artifact.urls,
        sha256=artifact.sha256,
        size=artifact.size,
    )
    return AcquiredFile(path=path, source=source)


def _extract_directory(archive: Path, root: str, destination: Path) -> None:
    if destination.exists():
        return

    destination.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(
        dir=destination.parent,
        prefix=f".{destination.name}.",
    ) as directory:
        temporary = Path(directory)
        with ZipFile(archive) as file:
            file.extractall(temporary)

        extracted = temporary / root
        if not extracted.is_dir():
            raise RuntimeError(f"missing {root!r} directory in {archive}")
        extracted.replace(destination)


def _verify_extracted_tree(
    archive: Path,
    root: str,
    destination: Path,
) -> None:
    if not destination.is_dir():
        raise RuntimeError(f"missing extracted directory: {destination}")

    expected: dict[Path, int] = {}
    with ZipFile(archive) as file:
        for info in file.infolist():
            if info.is_dir():
                continue

            member = PurePosixPath(info.filename)
            if not member.parts or member.parts[0] != root:
                raise RuntimeError(
                    f"unexpected member in {archive}: {info.filename}"
                )
            relative = Path(*member.parts[1:])
            expected[relative] = info.file_size

    actual = {
        path.relative_to(destination): path.stat().st_size
        for path in destination.rglob("*")
        if path.is_file()
    }
    if actual.keys() != expected.keys():
        raise RuntimeError(f"contents mismatch for {destination}")

    for relative, size in expected.items():
        if actual[relative] != size:
            path = destination / relative
            raise RuntimeError(
                f"size mismatch for {path}: expected {size}, "
                f"got {actual[relative]}"
            )


def _run_git(cwd: Path | None, *args: str) -> str:
    command = ["git"]
    if cwd is not None:
        command.extend(("-C", str(cwd)))
    command.extend(args)

    env = os.environ.copy()
    env["GIT_TERMINAL_PROMPT"] = "0"
    try:
        return capture_output(command, env=env)
    except CalledProcessError as exc:
        name = args[0] if args else "command"
        error = RuntimeError(
            f"git {name} failed with exit code {exc.returncode}"
        )
        if stderr := exc.stderr.strip() if exc.stderr else "":
            error.add_note(stderr)
        raise error from exc


def _python_string(path: Path) -> str:
    return json.dumps(str(path.resolve()))
