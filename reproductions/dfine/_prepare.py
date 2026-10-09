import os
from dataclasses import dataclass
from pathlib import Path
from subprocess import CalledProcessError
from tempfile import TemporaryDirectory

from smallpotato import _coco
from smallpotato._download import Artifact, acquire_artifact
from smallpotato._process import run_captured

DFINE_REPOSITORY = "https://github.com/Peterande/D-FINE.git"
DFINE_REVISION = "a15be7038b2149e787613505cda6199f366a7f59"
DFINE_CONFIG = Path("configs/dfine/dfine_hgnetv2_n_coco.yml")

CHECKPOINT = Artifact(
    filename="dfine_n_coco.pth",
    urls=(
        "https://github.com/Peterande/storage/releases/download/"
        "dfinev1.0/dfine_n_coco.pth",
    ),
    sha256=(
        "41973938d2784d38a9836990d805b8392855ebf611aba55f0f7add90e110744c"
    ),
    size=15_489_558,
)


@dataclass(frozen=True, slots=True)
class PreparedInputs:
    source: Path
    config: Path
    checkpoint: Path
    coco: Path
    images: Path
    annotations: Path


def prepare_inputs(cache: Path, /) -> PreparedInputs:
    cache = cache.expanduser().resolve()
    source = _prepare_source(cache)
    checkpoint = acquire_artifact(cache / "checkpoints" / "dfine", CHECKPOINT)
    coco = _coco.prepare_val2017(cache)

    return PreparedInputs(
        source=source,
        config=source / DFINE_CONFIG,
        checkpoint=checkpoint,
        coco=coco.root,
        images=coco.images,
        annotations=coco.annotations,
    )


def verify_source(path: Path, /) -> None:
    if not path.is_dir():
        raise RuntimeError(f"not a source directory: {path}")

    revision = _run_git(path, "rev-parse", "HEAD")
    if revision != DFINE_REVISION:
        raise RuntimeError(
            f"D-FINE revision mismatch: expected {DFINE_REVISION}, "
            f"got {revision}"
        )

    status = _run_git(path, "status", "--porcelain", "--untracked-files=all")
    if status:
        raise RuntimeError(f"source tree is not clean: {path}")

    config = path / DFINE_CONFIG
    if not config.is_file():
        raise RuntimeError(f"missing D-FINE-N config: {config}")


def _prepare_source(cache: Path, /) -> Path:
    parent = cache / "sources" / "dfine"
    destination = parent / DFINE_REVISION

    if destination.exists():
        verify_source(destination)
        return destination

    parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=parent, prefix=".dfine.") as directory:
        checkout = Path(directory) / "checkout"
        _run_git(None, "init", "-q", str(checkout))
        _run_git(checkout, "remote", "add", "origin", DFINE_REPOSITORY)
        _run_git(
            checkout,
            "fetch",
            "-q",
            "--depth=1",
            "--no-tags",
            "origin",
            DFINE_REVISION,
        )
        _run_git(checkout, "checkout", "-q", "--detach", "FETCH_HEAD")
        verify_source(checkout)
        checkout.replace(destination)

    return destination


def _run_git(cwd: Path | None, command: str, *args: str) -> str:
    argv = ["git"]
    if cwd is not None:
        argv.extend(("-C", str(cwd)))
    argv.extend((command, *args))

    env = os.environ.copy()
    env["GIT_TERMINAL_PROMPT"] = "0"
    try:
        return run_captured(argv, env=env).strip()
    except CalledProcessError as exc:
        raise RuntimeError(
            f"git {command} failed with exit code {exc.returncode}"
        ) from exc
