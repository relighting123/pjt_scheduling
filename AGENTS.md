# AGENTS.md

## Cursor Cloud specific instructions

This repo is a semiconductor scheduling system: a Python RL/heuristic backend (FastAPI) plus a
React + Vite frontend. An Oracle DB is optional and can be bypassed with `--nodb` + locally
generated sample data. See `README.md` for the full architecture and CLI reference.

### Environment layout
- Python deps live in a virtualenv at `.venv/` (Python 3.12). Run Python via `.venv/bin/python`
  (e.g. `.venv/bin/python main.py ...`, `.venv/bin/python -m pytest`). The startup update script
  recreates/refreshes this venv.
- `pytest` and `httpx` are required for the test suite but are **not** in `requirements.txt`; the
  update script installs them into the venv.
- Frontend deps are in `frontend/node_modules` (`npm install --prefix frontend`).

### Services (see README "UI" section)
| Service | Command | Port |
|---------|---------|------|
| FastAPI backend | `.venv/bin/python -m uvicorn api.server:app --host 127.0.0.1 --port 8000 --reload` | 8000 |
| Vite frontend | `npm run dev` (in `frontend/`) | 5173 |

- `python main.py ui` starts both together, but it calls `webbrowser.open(...)` and blocks; for a
  headless VM prefer starting the two services separately (as above). The Vite dev server proxies
  `/api` → `http://127.0.0.1:8000`.

### Sample data is required before inference / UI runs
- The checked-in dataset input contains only SQL query placeholders, so a fresh `infer`/UI run
  fails with `KeyError: 'PLAN_PROD_KEY'` unless real data is generated first. Generate local
  sample data once per VM: `.venv/bin/python main.py sample --facid FAC001 --bootstrap`.
- Without an Oracle DB you must run inference in "no DB" mode: pass `--nodb` on the CLI
  (`.venv/bin/python main.py infer --facid FAC001 --nodb`) or tick the
  "기존 JSON 사용 (--nodb …)" checkbox in the UI before clicking "추론 실행" (Run Inference).

### Tests / build / lint
- Tests: `.venv/bin/python -m pytest -q` (see README "테스트"). Note the full suite has
  pre-existing issues on this branch: `tests/test_conversion_scenario.py` and
  `tests/test_pacing_takt_suite.py` import modules that don't exist here, and several
  `test_scheduling_env.py` tests fail only when the full suite runs together due to shared global
  `CONFIG` state (they pass when that file is run on its own). Prefer running specific files, e.g.
  `.venv/bin/python -m pytest tests/test_scheduling_env.py -q`.
- Frontend build (also the closest thing to a lint/typecheck): `cd frontend && npm run build`
  (`tsc -b && vite build`).
