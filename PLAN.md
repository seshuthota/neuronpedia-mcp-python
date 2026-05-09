# Plan: Neuronpedia MCP Update & Refactor — Done

## Phase 1: Codebase cleanup ✅

- [x] 1.1 Delete `main.py`
- [x] 1.2 Fix `pyproject.toml` + add `main()` to server.py
- [x] 1.3 Fix `test_mcp.py` hardcoded API key + Windows path
- [x] 1.4 Populate `__init__.py`

## Phase 2: Refactor into modules ✅

- [x] 2.1 Create `models.py`
- [x] 2.2 Create `client.py`
- [x] 2.3 Create `tools.py`
- [x] 2.4 Update `server.py` — thin entry point
- [x] 2.5 Verify imports

## Phase 3: Fix schema mismatches (10 methods) ✅

- [x] 3.1 `generate_attribution_graph` — slug, new params, renames
- [x] 3.2 `get_activations` — feature object schema
- [x] 3.3 `search_all_features` — sourceSet, numResults
- [x] 3.4 `search_topk_by_token` — source, numResults
- [x] 3.5 `generate_explanation` — explanationType, explanationModelName
- [x] 3.6 `steer_generation` — features array + steer_method
- [x] 3.7 `steer_chat` — features array + steer_method
- [x] 3.8 `search_explanations` — layers array
- [x] 3.9 `steer_chat_advanced` — steer_method
- [x] 3.10 `steer_text_advanced` — steer_method

## Phase 4: Add missing endpoints (6 new) ✅

- [x] 4.1 `POST /api/activation/get`
- [x] 4.2 `POST /api/activation/source`
- [x] 4.3 `POST /api/list/update`
- [x] 4.4 `POST /api/list/edit-feature`
- [x] 4.5 `POST /api/explanation/search-release`
- [x] 4.6 `GET /api/sparsity/connected-neurons`

## Phase 5: Final remaining items ✅

- [x] `POST /api/graph/signed-put` + `/api/graph/save-to-db` (2-step graph upload)
- [x] `POST /api/graph/subgraph/list` / `save` / `delete` (subgraph CRUD)
- [x] `POST /api/explanation/score/{id}/delete` (score delete)
- [x] 36 MCP tools registered, server starts cleanly
