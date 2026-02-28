# Espresso Latte Art Ranker (Web App)

This project is now a deployable **web app** where you enter:
- location
- number of businesses
- score threshold

Then it fetches businesses from Yelp or Google Places, scores photos with your TensorFlow model, shows ranked results, and lets you download CSV.

## Features

- Web form UI for ranking parameters
- Results table with:
  - business name
  - address
  - Yelp URL
  - aggregate score
  - images downloaded
  - images above threshold
- CSV download from browser
- Supports data source selection via web form, CLI flag, or env vars:
  - `yelp_scrape` (default, no key)
  - `yelp_api` (requires `YELP_API_KEY`)
  - `google` (requires `GOOGLE_PLACES_API_KEY`)

## Requirements

- Python 3.10 or 3.11 recommended (TensorFlow wheel compatibility)
- TensorFlow model file/folder path available on disk
- Model path can be a `.keras`/`.h5` file, or a TensorFlow SavedModel directory (Keras 3 fallback uses signature endpoint).

## Setup

### Standard setup (Linux/Windows/macOS Intel)

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
pip install -r requirements-tensorflow.txt
```

### Apple Silicon (M1/M2/M3) setup

If you see AVX/jaxlib errors or TensorFlow install failures, use an ARM64 Python + Apple TensorFlow packages:

```bash
# Verify you're on arm64 Python (should print arm64)
python -c "import platform; print(platform.machine())"

python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel

# Install app deps first
pip install -r requirements.txt

# Remove incompatible x86 wheels if present
pip uninstall -y tensorflow tensorflow-cpu tensorflow-intel jax jaxlib

# Install Apple Silicon TensorFlow stack
pip install tensorflow-macos tensorflow-metal
```

### If `tensorflow` says "No matching distribution found"

This usually means one of these:
- you are on Python 3.12+/3.13 where your TensorFlow wheel is unavailable,
- or you're on macOS with an x86/ARM mismatch.

Use Python 3.10 or 3.11, then reinstall:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
pip install -r requirements-tensorflow.txt
```

On macOS, also verify architecture:

```bash
python -c "import platform; print(platform.machine())"  # should be arm64 on Apple Silicon
```

## Train a new latte-art model (good vs bad)

You can train a binary scoring model and plug it directly into this app.

Dataset layout:

```text
/path/to/dataset/
  good/
    *.jpg|png
  bad/
    *.jpg|png
```

Training command:

```bash
python train_latte_art_model.py \
  --dataset-dir /path/to/dataset \
  --output-model models/latte_art_model.keras \
  --img-size 224 \
  --batch-size 32 \
  --epochs 12 \
  --fine-tune
```

Then point the app/CLI at the trained model:

```bash
export LATTE_ART_MODEL_PATH=$PWD/models/latte_art_model.keras
python app.py
```

The trained model outputs a sigmoid score in `[0, 1]` where higher means more likely to be good latte art.

## Environment variables

Required:

```bash
export LATTE_ART_MODEL_PATH=/path/to/your/model
export LATTE_ART_CALL_ENDPOINT=serving_default   # optional; for SavedModel endpoint selection
# If you get endpoint errors, try: export LATTE_ART_CALL_ENDPOINT=serve
export LATTE_ART_INPUT_HEIGHT=224             # optional; use if model input shape cannot be inferred
export LATTE_ART_INPUT_WIDTH=224
export LATTE_ART_INPUT_CHANNELS=3
```

Optional:

```bash
export BUSINESS_SOURCE=yelp_scrape   # yelp_scrape | yelp_api | google
export YELP_SOURCE=scrape          # legacy alias supported
export YELP_API_KEY=...            # required for yelp_api
export GOOGLE_PLACES_API_KEY=...   # required for google
export PORT=8080
export DEFAULT_BUSINESS_LIMIT=20
export DEFAULT_SCORE_THRESHOLD=0.7
export REQUEST_TIMEOUT_S=20
export REQUEST_SLEEP_S=0.2
export LOG_LEVEL=INFO
export YELP_SCRAPE_USER_AGENT="Mozilla/5.0 ... Chrome/... Safari/537.36"
```

## Web usage

```bash
python app.py
```

Open: `http://localhost:8080`

## CLI usage

You can also run the ranking flow directly from the command line:

```bash
python latte_art_ranker.py \
  --location "San Francisco, CA" \
  --model-path /path/to/your/model \
  --call-endpoint serving_default \
  --input-height 224 --input-width 224 --input-channels 3 \
  --business-limit 20 \
  --score-threshold 0.70 \
  --source yelp_scrape \
  --output-csv latte_art_results.csv
```

Notes:
- For SavedModel directories, Keras 3 may require endpoint selection; if `serving_default` is missing, try `serve` via `LATTE_ART_CALL_ENDPOINT` (or `--call-endpoint` in CLI).
- If your SavedModel has no signatures/endpoints, set manual shape overrides (`LATTE_ART_INPUT_HEIGHT`, `LATTE_ART_INPUT_WIDTH`, `LATTE_ART_INPUT_CHANNELS` or CLI equivalents).
- If you still see `Available endpoints: []`, your SavedModel may expose only callable attributes. The loader now probes common callables automatically; if shape inference still fails, set the three input-shape overrides.
- Use `--source yelp_api` with `--yelp-api-key` (or `YELP_API_KEY`) for Yelp Fusion API mode.
- Use `--source google` with `--google-api-key` (or `GOOGLE_PLACES_API_KEY`) for Google Places mode.
- Use `--include-non-drink-photos` to score all discovered photos.


To inspect available SavedModel signatures locally:

```bash
python - <<'PY'
import tensorflow as tf
model = tf.saved_model.load('/path/to/your/model')
print('signatures:', list(model.signatures.keys()))
PY
```

## Deploy notes

Any platform that supports Python web apps works (Render, Railway, Fly.io, Heroku-like platforms, etc.).
Use `python app.py` as start command and set env vars in your deployment settings.

## API/route summary

- `GET /` - web form + results page
- `POST /run` - executes ranking
- `POST /download` - downloads CSV of current results
- `GET /healthz` - health check endpoint

## Caveats

- On Apple Silicon, use ARM64 Python plus `tensorflow-macos`/`tensorflow-metal` to avoid AVX/jaxlib x86 wheel crashes.
- Console logging is enabled for both CLI and web app; adjust verbosity with `LOG_LEVEL` (e.g., `DEBUG`, `INFO`).
- Scraping mode is best-effort and may break if Yelp markup changes.
- If Yelp scrape requests return 403, set `YELP_SCRAPE_USER_AGENT` to your local Chrome user agent string.
- Confirm Yelp Terms of Service compliance for your usage.
- Google Places mode requires billing-enabled Places API credentials.
- Latte/drink photo filtering remains heuristic unless your model handles non-drink photos robustly.
