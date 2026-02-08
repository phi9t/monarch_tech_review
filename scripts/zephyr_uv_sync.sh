#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

SPACK_PYTHON="${ZEPHYR_SPACK_PYTHON:-/opt/spack_store/view/bin/python}"
if [[ ! -x "$SPACK_PYTHON" ]]; then
  SPACK_PYTHON="$(command -v python || true)"
fi
if [[ ! -x "$SPACK_PYTHON" ]]; then
  echo "error: expected Zephyr Spack Python at $SPACK_PYTHON" >&2
  echo "hint: run this inside Zephyr container infra or set ZEPHYR_SPACK_PYTHON." >&2
  exit 1
fi

need_rebuild=0
if [[ ! -f .venv/pyvenv.cfg ]]; then
  need_rebuild=1
elif ! rg -q "include-system-site-packages\\s*=\\s*true" .venv/pyvenv.cfg; then
  need_rebuild=1
elif [[ ! -x .venv/bin/python ]]; then
  need_rebuild=1
else
  venv_py_mm="$(.venv/bin/python -c 'import sys; print(f"{sys.version_info[0]}.{sys.version_info[1]}")' 2>/dev/null || true)"
  spack_py_mm="$("$SPACK_PYTHON" -c 'import sys; print(f"{sys.version_info[0]}.{sys.version_info[1]}")')"
  if [[ "$venv_py_mm" != "$spack_py_mm" ]]; then
    need_rebuild=1
  else
    torch_origin="$(.venv/bin/python -c 'import importlib.util as u; s=u.find_spec("torch"); print(s.origin if s else "")' 2>/dev/null || true)"
    if [[ "$torch_origin" == *"/.venv/"* ]]; then
      need_rebuild=1
    fi
  fi
fi

if [[ "$need_rebuild" -eq 1 ]]; then
  uv venv --clear --python "$SPACK_PYTHON" --system-site-packages .venv
fi

source .venv/bin/activate

# Never reinstall packages that Zephyr already provides in system site-packages.
# We compute the intersection with uv.lock package names so uv won't fetch wheels
# for anything already available from the container infra.
mapfile -t no_install_pkgs < <("$SPACK_PYTHON" - <<'PY'
import importlib.metadata as md
import pathlib
import tomllib

lock_path = pathlib.Path("uv.lock")
if lock_path.exists():
    lock_data = tomllib.loads(lock_path.read_text())
    lock_names = {
        (pkg.get("name") or "").lower().replace("_", "-")
        for pkg in lock_data.get("package", [])
    }
else:
    lock_names = set()

sys_names = {
    (dist.metadata.get("Name") or "").lower().replace("_", "-")
    for dist in md.distributions()
}

for name in sorted((lock_names & sys_names) - {"monarch-gpu-mode"}):
    if name:
        print(name)
PY
)

sync_args=(--active)
for pkg in "${no_install_pkgs[@]}"; do
  sync_args+=(--no-install-package "$pkg")
done

# Force Zephyr-provided CUDA/Torch runtime stack; never install wheel variants.
for pkg in torch triton cuda-bindings cuda-pathfinder; do
  sync_args+=(--no-install-package "$pkg")
done

# Block any CUDA wheel package families that can conflict with Zephyr runtime libs.
if [[ -f uv.lock ]]; then
  while IFS= read -r pkg; do
    [[ -n "$pkg" ]] && sync_args+=(--no-install-package "$pkg")
  done < <("$SPACK_PYTHON" - <<'PY'
import pathlib
import tomllib

data = tomllib.loads(pathlib.Path("uv.lock").read_text())
for pkg in data.get("package", []):
    name = (pkg.get("name") or "").lower().replace("_", "-")
    if name.startswith("nvidia-") or name.startswith("cuda-"):
        print(name)
PY
)
fi

uv sync "${sync_args[@]}" "$@"
