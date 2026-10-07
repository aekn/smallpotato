# Small Potato

Research on high-quality streaming monocular driving perception under severe
edge-compute constraints.

## Development

Python 3.12 and [uv](https://docs.astral.sh/uv/) are required.

```bash
uv sync --locked
uv run ruff format --check .
uv run ruff check .
uv run pyright
uv run pytest
uv build
```
