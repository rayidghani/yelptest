#!/usr/bin/env python3
"""Flask web app for ranking Yelp espresso businesses by latte art quality."""

from __future__ import annotations

import csv
import io
import logging
import os
from functools import lru_cache
from typing import List

from flask import Flask, Response, redirect, render_template, request, url_for

from latte_art_ranker import (
    BusinessResult,
    LatteArtModel,
    YelpApiProvider,
    YelpScrapeProvider,
    build_results,
    configure_logging,
)

app = Flask(__name__)
LOGGER = logging.getLogger(__name__)
configure_logging(os.getenv("LOG_LEVEL", "INFO"))


def _int_env(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except ValueError:
        return default


def _float_env(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except ValueError:
        return default


@lru_cache(maxsize=1)
def get_model() -> LatteArtModel:
    model_path = os.getenv("LATTE_ART_MODEL_PATH", "").strip()
    if not model_path:
        raise RuntimeError("LATTE_ART_MODEL_PATH is not set.")
    LOGGER.info("Loading TensorFlow model from %s", model_path)
    return LatteArtModel(model_path=model_path)


def get_provider():
    source = os.getenv("YELP_SOURCE", "scrape").strip().lower()
    timeout = _int_env("REQUEST_TIMEOUT_S", 20)
    LOGGER.info("Selecting provider source=%s timeout=%ss", source, timeout)
    if source == "api":
        api_key = os.getenv("YELP_API_KEY", "").strip()
        if not api_key:
            raise RuntimeError("YELP_SOURCE=api requires YELP_API_KEY.")
        return YelpApiProvider(api_key=api_key, timeout_s=timeout)
    return YelpScrapeProvider(timeout_s=timeout)


def results_to_csv(results: List[BusinessResult]) -> str:
    buffer = io.StringIO()
    writer = csv.writer(buffer)
    writer.writerow(
        [
            "business_name",
            "address",
            "business_url",
            "yelp_url",
            "aggregate_score",
            "images_downloaded",
            "images_above_threshold",
        ]
    )
    for row in results:
        writer.writerow(
            [
                row.name,
                row.address,
                row.business_url,
                row.yelp_url,
                f"{row.aggregate_score:.6f}",
                row.images_downloaded,
                row.images_above_threshold,
            ]
        )
    return buffer.getvalue()


@app.route("/", methods=["GET"])
def index():
    return render_template(
        "index.html",
        defaults={
            "location": "San Francisco, CA",
            "business_limit": _int_env("DEFAULT_BUSINESS_LIMIT", 20),
            "score_threshold": _float_env("DEFAULT_SCORE_THRESHOLD", 0.7),
        },
    )


@app.route("/run", methods=["POST"])
def run():
    location = request.form.get("location", "").strip()
    if not location:
        return render_template("index.html", error="Location is required.", defaults=request.form)

    try:
        business_limit = int(request.form.get("business_limit", "20"))
        score_threshold = float(request.form.get("score_threshold", "0.7"))
    except ValueError:
        return render_template(
            "index.html",
            error="Business limit must be an integer and threshold must be a number.",
            defaults=request.form,
        )

    if business_limit < 1 or business_limit > 50:
        return render_template(
            "index.html",
            error="Business limit must be between 1 and 50.",
            defaults=request.form,
        )

    if score_threshold < 0 or score_threshold > 1:
        return render_template(
            "index.html",
            error="Score threshold must be between 0 and 1.",
            defaults=request.form,
        )

    try:
        LOGGER.info(
            "Web run requested: location='%s', business_limit=%d, threshold=%.3f",
            location,
            business_limit,
            score_threshold,
        )
        model = get_model()
        provider = get_provider()
        results = build_results(
            provider=provider,
            model=model,
            location=location,
            business_limit=business_limit,
            score_threshold=score_threshold,
            include_non_drink_photos=False,
            sleep_s=_float_env("REQUEST_SLEEP_S", 0.2),
        )
    except Exception as exc:
        LOGGER.exception("Web run failed")
        return render_template("index.html", error=str(exc), defaults=request.form)

    LOGGER.info("Web run complete: %d result rows", len(results))
    csv_text = results_to_csv(results)
    return render_template(
        "index.html",
        defaults=request.form,
        results=results,
        csv_text=csv_text,
    )


@app.route("/healthz", methods=["GET"])
def healthz():
    return {"status": "ok"}


@app.route("/download", methods=["POST"])
def download():
    csv_text = request.form.get("csv_text", "")
    if not csv_text:
        LOGGER.warning("Download requested without CSV payload")
        return redirect(url_for("index"))
    LOGGER.info("CSV download requested")
    return Response(
        csv_text,
        mimetype="text/csv",
        headers={"Content-Disposition": "attachment; filename=latte_art_results.csv"},
    )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", "8080")), debug=False)
