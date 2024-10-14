## to use uv instead of conda
- Install uv from the [installation](https://docs.astral.sh/uv/getting-started/installation/) page
- run `uv sync` , there should now be a .venv folder
- activate in terminal `source /.venv/bin/activate`
- or choose via your IDE

To start from scratch in linux:
```
curl -LsSf https://astral.sh/uv/install.sh | sh
uv init .
uv add -r requirements.txt
```

To use existing uv.lock:
```
uv sync
```

## to add new dependencies
``` 
uv add torch
uv add -r requirements.txt
uv remove tensosrflow
```
for more check the [dependencies](https://docs.astral.sh/uv/concepts/dependencies/) page

## update .bashrc
```
. "$HOME/.cargo/env"

export PIP_CACHE_DIR=/local/s3777103/.cache/.pip
export UV_CACHE_DIR=/local/s3777103/.cache/.uv
```
