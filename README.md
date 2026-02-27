# Yelp Espresso Latte Art Ranker (Web App)

This project is now a deployable **web app** where you enter:
- location
- number of businesses
- score threshold

Then it fetches Yelp businesses, scores photos with your TensorFlow model, shows ranked results, and lets you download CSV.

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
- Supports data source via env var:
  - `YELP_SOURCE=scrape` (default, no API key)
  - `YELP_SOURCE=api` (uses Yelp Fusion API + `YELP_API_KEY`)

## Requirements

- Python 3.10+
- TensorFlow model file/folder path available on disk

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Environment variables

Required:

```bash
export LATTE_ART_MODEL_PATH=/path/to/your/model
```

Optional:

```bash
export YELP_SOURCE=scrape          # or api
export YELP_API_KEY=...            # required if YELP_SOURCE=api
export PORT=8080
export DEFAULT_BUSINESS_LIMIT=20
export DEFAULT_SCORE_THRESHOLD=0.7
export REQUEST_TIMEOUT_S=20
export REQUEST_SLEEP_S=0.2
export LOG_LEVEL=INFO
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
  --business-limit 20 \
  --score-threshold 0.70 \
  --source scrape \
  --output-csv latte_art_results.csv
```

Notes:
- Use `--source api` with `--yelp-api-key` (or `YELP_API_KEY`) if you want Yelp Fusion API mode.
- Use `--include-non-drink-photos` to score all discovered photos.

## Deploy notes

Any platform that supports Python web apps works (Render, Railway, Fly.io, Heroku-like platforms, etc.).
Use `python app.py` as start command and set env vars in your deployment settings.

## API/route summary

- `GET /` - web form + results page
- `POST /run` - executes ranking
- `POST /download` - downloads CSV of current results
- `GET /healthz` - health check endpoint

## Caveats

- Console logging is enabled for both CLI and web app; adjust verbosity with `LOG_LEVEL` (e.g., `DEBUG`, `INFO`).
- Scraping mode is best-effort and may break if Yelp markup changes.
- Confirm Yelp Terms of Service compliance for your usage.
- Latte/drink photo filtering remains heuristic unless your model handles non-drink photos robustly.
