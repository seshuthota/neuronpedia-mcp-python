# AGENTS.md

## Entry points

- **Real entry point**: `src/neuronpedia_mcp/server.py` — thin runner. Module split: `models.py`, `client.py`, `tools.py`, `server.py`.
- `main.py` at the repo root is dead placeholder code. Ignore it.

## Commands

```bash
uv sync                          # install dependencies (editable install)
uv run python src/neuronpedia_mcp/server.py   # run (direct script)
uv run python -m neuronpedia_mcp.server       # run (module)
uv run neuronpedia-mcp                        # run (CLI entrypoint)
uv run python test_mcp.py api   # API smoke test only
uv run python test_mcp.py       # full MCP integration test
```

- Use `uv` for everything.
- Python >=3.12 required.

## Required env

`NEURONPEDIA_API_KEY` — server raises `ValueError` if unset at first tool invocation (lazy init in `client.get_client()`).

## Architecture

- `models.py` — Pydantic schemas (`AttributionGraphResponse`).
- `client.py` — `NeuronpediaClient` wraps the Neuronpedia REST API with `httpx.AsyncClient`. Module-level `get_client()` provides lazy singleton.
- `tools.py` — `@mcp.tool()` decorated functions, imported by `server.py` after `mcp` is created (deferred import pattern to avoid circular imports).
- `server.py` — `FastMCP` instance + `main()` + `if __name__ == "__main__": main()`.
- Client methods that accept JSON arrays from MCP deserialize inline with `import json; json.loads(...)` in the tool wrapper.

## Testing gotchas

- `test_mcp.py` used to have a hardcoded API key and Windows path — both fixed to read from env / use script directory.
- No CI, no pre-commit hooks, no lint/typecheck config exists in this repo.
