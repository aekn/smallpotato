from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from tempfile import TemporaryDirectory
from zipfile import ZipFile

from ._download import Artifact, acquire_artifact, verify_file

_COCO_MIRROR_REVISION = "5200d2cffae8121713bec767b4693bdafc0eeb0b"

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

INSTANCES_VAL2017_SHA256 = (
    "e8c7f7908f1d7278341fae127d0da654f102f11bd7b21d8aeefa635b8c810b6f"
)
INSTANCES_VAL2017_SIZE = 19_987_840


@dataclass(frozen=True, slots=True)
class PreparedCOCO:
    root: Path
    images: Path
    annotations: Path


def prepare_val2017(cache: Path, /) -> PreparedCOCO:
    root = cache.expanduser().resolve() / "datasets" / "coco" / "2017"
    archives = root / "archives"

    val_archive = acquire_artifact(archives, VAL2017)
    annotations_archive = acquire_artifact(archives, ANNOTATIONS)

    images = root / "val2017"
    annotations_dir = root / "annotations"
    _prepare_directory(val_archive, "val2017", images)
    _prepare_directory(annotations_archive, "annotations", annotations_dir)

    annotations = annotations_dir / "instances_val2017.json"
    verify_file(
        annotations,
        sha256=INSTANCES_VAL2017_SHA256,
        size=INSTANCES_VAL2017_SIZE,
    )

    return PreparedCOCO(root=root, images=images, annotations=annotations)


def _prepare_directory(
    archive: Path,
    root: str,
    destination: Path,
    /,
) -> None:
    with ZipFile(archive) as file:
        expected = _archive_tree(file, archive, root)

        if destination.exists():
            _validate_extracted_tree(destination, expected)
            return

        destination.parent.mkdir(parents=True, exist_ok=True)
        with TemporaryDirectory(
            dir=destination.parent,
            prefix=f".{destination.name}.",
        ) as directory:
            temporary = Path(directory)
            file.extractall(temporary)

            extracted = temporary / root
            _validate_extracted_tree(extracted, expected)
            extracted.replace(destination)


def _archive_tree(
    file: ZipFile,
    archive: Path,
    root: str,
    /,
) -> dict[Path, int]:
    expected: dict[Path, int] = {}

    for info in file.infolist():
        member = PurePosixPath(info.filename)
        if (
            not member.parts
            or member.is_absolute()
            or member.parts[0] != root
            or ".." in member.parts
        ):
            raise RuntimeError(
                f"unexpected member in {archive}: {info.filename}"
            )
        if info.is_dir():
            continue
        if len(member.parts) == 1:
            raise RuntimeError(
                f"unexpected member in {archive}: {info.filename}"
            )

        relative = Path(*member.parts[1:])
        if relative in expected:
            raise RuntimeError(
                f"duplicate member in {archive}: {info.filename}"
            )
        expected[relative] = info.file_size

    return expected


def _validate_extracted_tree(
    destination: Path,
    expected: dict[Path, int],
    /,
) -> None:
    if not destination.is_dir():
        raise RuntimeError(f"missing extracted directory: {destination}")

    actual = {
        path.relative_to(destination): path.stat().st_size
        for path in destination.rglob("*")
        if path.is_file()
    }
    if actual.keys() != expected.keys():
        raise RuntimeError(f"contents mismatch for {destination}")

    for relative, expected_size in expected.items():
        actual_size = actual[relative]
        if actual_size != expected_size:
            path = destination / relative
            raise RuntimeError(
                f"size mismatch for {path}: expected {expected_size}, "
                f"got {actual_size}"
            )
