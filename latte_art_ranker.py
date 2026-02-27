#!/usr/bin/env python3
"""Rank Yelp espresso businesses by latte art quality from their photos.

Supports two data sources:
- `api`: official Yelp Fusion API (requires API key)
- `scrape`: parse public Yelp HTML pages (no API key)
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import logging
import os
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Protocol, Sequence
from urllib.parse import quote_plus, urljoin

import requests
from bs4 import BeautifulSoup
import numpy as np
from PIL import Image

YELP_API_BASE = "https://api.yelp.com/v3"
YELP_WEB_BASE = "https://www.yelp.com"
LOGGER = logging.getLogger(__name__)


def configure_logging(level: str = "INFO") -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    )


@dataclass
class BusinessSummary:
    id: str


@dataclass
class BusinessResult:
    name: str
    address: str
    business_url: str
    yelp_url: str
    aggregate_score: float
    images_downloaded: int
    images_above_threshold: int


class BusinessProvider(Protocol):
    timeout_s: int

    def search_espresso(self, location: str, limit: int) -> List[BusinessSummary]: ...

    def business_details(self, business_id: str) -> dict: ...


class YelpApiProvider:
    def __init__(self, api_key: str, timeout_s: int = 20) -> None:
        self.session = requests.Session()
        self.session.headers.update({"Authorization": f"Bearer {api_key}"})
        self.timeout_s = timeout_s

    def search_espresso(self, location: str, limit: int) -> List[BusinessSummary]:
        LOGGER.info("[1/6] API search: term='espresso', location='%s', limit=%d", location, limit)
        params = {
            "term": "espresso",
            "location": location,
            "limit": limit,
            "sort_by": "best_match",
        }
        resp = self.session.get(
            f"{YELP_API_BASE}/businesses/search", params=params, timeout=self.timeout_s
        )
        resp.raise_for_status()
        businesses = resp.json().get("businesses", [])
        LOGGER.info("API search returned %d businesses", len(businesses))
        return [BusinessSummary(id=b["id"]) for b in businesses if b.get("id")]

    def business_details(self, business_id: str) -> dict:
        LOGGER.info("Fetching business details via API: %s", business_id)
        resp = self.session.get(
            f"{YELP_API_BASE}/businesses/{business_id}", timeout=self.timeout_s
        )
        resp.raise_for_status()
        payload = resp.json()
        return {
            "name": payload.get("name", ""),
            "address": normalize_address(payload.get("location", {})),
            "business_url": payload.get("url", ""),
            "yelp_url": payload.get("url", ""),
            "photos": payload.get("photos") or [],
        }


class YelpScrapeProvider:
    """Best-effort Yelp web scraping provider (no API key required)."""

    def __init__(self, timeout_s: int = 20) -> None:
        self.session = requests.Session()
        self.session.headers.update(
            {
                "User-Agent": (
                    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
                    "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
                )
            }
        )
        self.timeout_s = timeout_s

    def search_espresso(self, location: str, limit: int) -> List[BusinessSummary]:
        LOGGER.info("[1/6] Scrape search: term='espresso', location='%s', limit=%d", location, limit)
        ids: List[str] = []
        start = 0
        while len(ids) < limit:
            search_url = (
                f"{YELP_WEB_BASE}/search?find_desc={quote_plus('espresso')}&"
                f"find_loc={quote_plus(location)}&start={start}"
            )
            resp = self.session.get(search_url, timeout=self.timeout_s)
            resp.raise_for_status()

            # Extract /biz/<slug> links from search cards.
            slugs = set(re.findall(r'"(/biz/[^"?]+)', resp.text))
            added = 0
            for slug in slugs:
                if slug.startswith("/biz/"):
                    bid = slug[len("/biz/") :]
                    if bid and bid not in ids:
                        ids.append(bid)
                        added += 1
                        if len(ids) >= limit:
                            break

            if added == 0:
                break
            start += 10
            if start > 90:  # avoid crawling too deeply
                break
            time.sleep(0.25)

        LOGGER.info("Scrape search discovered %d businesses", len(ids[:limit]))
        return [BusinessSummary(id=i) for i in ids[:limit]]

    def business_details(self, business_id: str) -> dict:
        LOGGER.info("Fetching business details via scrape: %s", business_id)
        biz_url = f"{YELP_WEB_BASE}/biz/{business_id}"
        resp = self.session.get(biz_url, timeout=self.timeout_s)
        resp.raise_for_status()

        soup = BeautifulSoup(resp.text, "html.parser")

        name = ""
        address = ""
        photos: List[str] = []

        # Preferred: JSON-LD block.
        for tag in soup.find_all("script", attrs={"type": "application/ld+json"}):
            txt = (tag.string or "").strip()
            if not txt:
                continue
            try:
                data = json.loads(txt)
            except json.JSONDecodeError:
                continue
            blobs = data if isinstance(data, list) else [data]
            for blob in blobs:
                if not isinstance(blob, dict):
                    continue
                if blob.get("@type") in {"LocalBusiness", "Restaurant", "CafeOrCoffeeShop"}:
                    name = blob.get("name", name)
                    img = blob.get("image")
                    if isinstance(img, str):
                        photos.append(img)
                    elif isinstance(img, list):
                        photos.extend([i for i in img if isinstance(i, str)])
                    addr = blob.get("address")
                    if isinstance(addr, dict):
                        parts = [
                            addr.get("streetAddress", ""),
                            addr.get("addressLocality", ""),
                            addr.get("addressRegion", ""),
                            addr.get("postalCode", ""),
                        ]
                        address = ", ".join([p for p in parts if p])

        # Fallbacks from html meta tags.
        if not name:
            name = (soup.find("h1") or {}).get_text(strip=True) if soup.find("h1") else ""
        if not address:
            addr_meta = soup.find("meta", attrs={"property": "business:contact_data:street_address"})
            if addr_meta and addr_meta.get("content"):
                address = addr_meta["content"]

        # Best-effort photo extraction from image urls in html.
        html_photo_urls = re.findall(r'https://s3-media\d+\.fl\.yelpcdn\.com/bphoto/[^"\']+', resp.text)
        photos.extend(html_photo_urls)

        # de-duplicate while preserving order
        deduped: List[str] = []
        seen = set()
        for p in photos:
            if p not in seen:
                deduped.append(p)
                seen.add(p)

        return {
            "name": name,
            "address": address,
            "business_url": "",
            "yelp_url": biz_url,
            "photos": deduped,
        }


class LatteArtModel:
    def __init__(self, model_path: str, call_endpoint: str = "serving_default"):
        try:
            import tensorflow as tf  # noqa: PLC0415
        except ImportError as exc:
            raise RuntimeError(
                "TensorFlow is not installed/compatible in this environment. "
                "Install from requirements-tensorflow.txt or use tensorflow-macos on Apple Silicon."
            ) from exc

        self.tf = tf
        self.call_endpoint = call_endpoint
        self.input_key: str | None = None

        try:
            self.model = tf.keras.models.load_model(model_path)
            LOGGER.info("Loaded model with keras.models.load_model: %s", model_path)
            shape = self.model.input_shape
            if isinstance(shape, list):
                shape = shape[0]
            self.predict_fn = lambda batch: self.model(batch, training=False)
        except (ValueError, OSError) as exc:
            if "File format not supported" not in str(exc):
                raise
            LOGGER.warning(
                "Keras load_model could not load '%s'. Falling back to SavedModel/TFSMLayer.",
                model_path,
            )
            shape = self._configure_saved_model_predictor(model_path, call_endpoint)

        if len(shape) != 4:
            raise ValueError(f"Expected NHWC model input, got {shape}")
        self.input_height = int(shape[1])
        self.input_width = int(shape[2])
        self.channels = int(shape[3])

    def _shape_from_signature(self, signature: object) -> tuple[object, str | None]:
        _, kw = signature.structured_input_signature
        if not kw:
            raise RuntimeError("SavedModel signature has no keyword input tensors.")
        input_key = next(iter(kw.keys()))
        return kw[input_key].shape, input_key

    def _configure_saved_model_predictor(self, model_path: str, call_endpoint: str):
        saved = self.tf.saved_model.load(model_path)
        signatures = getattr(saved, "signatures", {})

        endpoint_to_use = None
        if call_endpoint in signatures:
            endpoint_to_use = call_endpoint
        elif signatures:
            endpoint_to_use = next(iter(signatures.keys()))
            LOGGER.warning(
                "SavedModel endpoint '%s' not found. Falling back to first available endpoint '%s'.",
                call_endpoint,
                endpoint_to_use,
            )

        if endpoint_to_use is not None:
            signature = signatures[endpoint_to_use]
            shape, self.input_key = self._shape_from_signature(signature)
            self.predict_fn = lambda batch: signature(**{self.input_key: batch})
            return shape

        LOGGER.warning(
            "SavedModel signatures are empty. Attempting keras.layers.TFSMLayer endpoint resolution."
        )
        endpoint_candidates = [call_endpoint, "serve", "serving_default"]
        last_error = None
        for candidate in endpoint_candidates:
            try:
                layer = self.tf.keras.layers.TFSMLayer(model_path, call_endpoint=candidate)
                fn = getattr(layer, "_call_endpoint_fn", None)
                if fn is None:
                    continue
                shape, self.input_key = self._shape_from_signature(fn)
                self.predict_fn = lambda batch, lyr=layer: lyr(batch)
                LOGGER.info("Using TFSMLayer endpoint '%s'", candidate)
                return shape
            except Exception as exc:  # best-effort endpoint probing
                last_error = exc
                continue

        raise RuntimeError(
            "Could not resolve a callable SavedModel endpoint. "
            "Try setting --call-endpoint / LATTE_ART_CALL_ENDPOINT to your exported endpoint "
            "(commonly 'serve' or 'serving_default')."
        ) from last_error

    def _extract_score(self, pred: object) -> float:
        if isinstance(pred, dict):
            if not pred:
                raise ValueError("Model prediction dictionary was empty")
            pred = next(iter(pred.values()))
        arr = np.array(pred)
        return float(arr.squeeze())

    def score_image_bytes(self, image_bytes: bytes) -> float:
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        image = image.resize((self.input_width, self.input_height))

        arr = self.tf.keras.utils.img_to_array(image)
        if self.channels == 1:
            arr = self.tf.image.rgb_to_grayscale(arr)

        arr = arr / 255.0
        batch = self.tf.expand_dims(arr, axis=0)
        pred = self.predict_fn(batch)
        return self._extract_score(pred)


def normalize_address(location: dict) -> str:
    display = location.get("display_address") or []
    if display:
        return ", ".join(display)
    return ""


def is_likely_drink_photo(url: str) -> bool:
    lowered = url.lower()
    keywords = ["drink", "latte", "espresso", "coffee", "cappuccino", "mocha"]
    return any(k in lowered for k in keywords)


def download_image(session: requests.Session, image_url: str, timeout_s: int = 20) -> bytes | None:
    try:
        resp = session.get(image_url, timeout=timeout_s)
        resp.raise_for_status()
        return resp.content
    except requests.RequestException:
        return None


def score_business_photos(
    details: dict,
    model: LatteArtModel,
    score_threshold: float,
    include_non_drink_photos: bool,
    timeout_s: int,
) -> tuple[float, int, int]:
    photos = details.get("photos") or []
    LOGGER.info("[3/6] Retrieved %d photo candidates for '%s'", len(photos), details.get("name", ""))
    if not include_non_drink_photos:
        filtered = [p for p in photos if is_likely_drink_photo(p)]
        if filtered:
            photos = filtered
    LOGGER.info("[3/6] %d photos selected for scoring", len(photos))

    session = requests.Session()
    scores: List[float] = []
    above = 0
    for photo_url in photos:
        image_bytes = download_image(session, photo_url, timeout_s=timeout_s)
        if image_bytes is None:
            LOGGER.debug("Skipping photo download failure: %s", photo_url)
            continue
        try:
            score = model.score_image_bytes(image_bytes)
        except Exception:
            LOGGER.exception("Model scoring failed for photo: %s", photo_url)
            continue
        scores.append(score)
        if score >= score_threshold:
            above += 1

    if not scores:
        LOGGER.info("No scorable photos for '%s'", details.get("name", ""))
        return 0.0, 0, 0

    LOGGER.info("[4/6] Scored %d photos, %d above threshold", len(scores), above)
    return sum(scores) / len(scores), len(scores), above


def build_results(
    provider: BusinessProvider,
    model: LatteArtModel,
    location: str,
    business_limit: int,
    score_threshold: float,
    include_non_drink_photos: bool,
    sleep_s: float,
) -> List[BusinessResult]:
    LOGGER.info("Starting ranking run for location='%s'", location)
    businesses = provider.search_espresso(location=location, limit=business_limit)
    LOGGER.info("[2/6] Processing %d businesses", len(businesses))
    out: List[BusinessResult] = []

    for idx, b in enumerate(businesses, start=1):
        LOGGER.info("Processing business %d/%d (%s)", idx, len(businesses), b.id)
        try:
            details = provider.business_details(b.id)
        except requests.RequestException:
            LOGGER.exception("Failed to fetch details for business id=%s", b.id)
            continue

        agg, downloaded, above = score_business_photos(
            details=details,
            model=model,
            score_threshold=score_threshold,
            include_non_drink_photos=include_non_drink_photos,
            timeout_s=provider.timeout_s,
        )

        out.append(
            BusinessResult(
                name=details.get("name", ""),
                address=details.get("address", ""),
                business_url=details.get("business_url", ""),
                yelp_url=details.get("yelp_url", ""),
                aggregate_score=agg,
                images_downloaded=downloaded,
                images_above_threshold=above,
            )
        )
        LOGGER.info(
            "[5/6] Aggregated '%s': score=%.4f, downloaded=%d, above_threshold=%d",
            details.get("name", ""),
            agg,
            downloaded,
            above,
        )
        if sleep_s > 0:
            time.sleep(sleep_s)

    out.sort(key=lambda x: x.aggregate_score, reverse=True)
    LOGGER.info("[6/6] Ranking complete: %d businesses in output", len(out))
    return out


def write_csv(path: Path, results: Sequence[BusinessResult]) -> None:
    LOGGER.info("Writing CSV to %s", path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
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
        for r in results:
            writer.writerow(
                [
                    r.name,
                    r.address,
                    r.business_url,
                    r.yelp_url,
                    f"{r.aggregate_score:.6f}",
                    r.images_downloaded,
                    r.images_above_threshold,
                ]
            )


def print_results(results: Sequence[BusinessResult], top_n: int | None = None) -> None:
    shown = results if top_n is None else results[:top_n]
    if not shown:
        print("No results to display.")
        return

    header = f"{'Rank':<4} {'Score':<8} {'Imgs':<4} {'Good':<4} {'Business':<35} Address"
    print(header)
    print("-" * len(header))
    for idx, r in enumerate(shown, start=1):
        print(
            f"{idx:<4} {r.aggregate_score:<8.4f} {r.images_downloaded:<4} "
            f"{r.images_above_threshold:<4} {r.name[:35]:<35} {r.address}"
        )


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Search Yelp espresso businesses and rank by latte art quality."
    )
    parser.add_argument("--location", required=True, help="Location string for Yelp search")
    parser.add_argument("--model-path", required=True, help="Path to TensorFlow model")
    parser.add_argument("--call-endpoint", default=os.getenv("LATTE_ART_CALL_ENDPOINT", "serving_default"), help="SavedModel call endpoint for Keras 3 fallback (default: serving_default)")
    parser.add_argument(
        "--source",
        choices=["api", "scrape"],
        default="scrape",
        help="Business data source. Use 'scrape' to avoid Yelp API key.",
    )
    parser.add_argument("--yelp-api-key", default=os.getenv("YELP_API_KEY"))
    parser.add_argument("--business-limit", type=int, default=20)
    parser.add_argument("--score-threshold", type=float, default=0.7)
    parser.add_argument(
        "--include-non-drink-photos",
        action="store_true",
        help="If set, score all discovered photos instead of trying to filter to drinks.",
    )
    parser.add_argument("--sleep-s", type=float, default=0.2)
    parser.add_argument("--output-csv", type=Path, default=Path("latte_art_results.csv"))
    parser.add_argument("--top-n", type=int, default=20, help="Rows to print in terminal")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    configure_logging(os.getenv("LOG_LEVEL", "INFO"))
    args = parse_args(argv or sys.argv[1:])
    LOGGER.info("CLI started with source=%s, limit=%d, threshold=%.3f", args.source, args.business_limit, args.score_threshold)

    if args.source == "api":
        if not args.yelp_api_key:
            print("Missing Yelp API key. Pass --yelp-api-key or set YELP_API_KEY.", file=sys.stderr)
            return 2
        provider: BusinessProvider = YelpApiProvider(api_key=args.yelp_api_key)
    else:
        provider = YelpScrapeProvider()

    model = LatteArtModel(model_path=args.model_path, call_endpoint=args.call_endpoint)

    results = build_results(
        provider=provider,
        model=model,
        location=args.location,
        business_limit=args.business_limit,
        score_threshold=args.score_threshold,
        include_non_drink_photos=args.include_non_drink_photos,
        sleep_s=args.sleep_s,
    )

    write_csv(args.output_csv, results)
    print_results(results, top_n=args.top_n)
    print(f"\nSaved CSV: {args.output_csv}")
    LOGGER.info("CLI run finished successfully")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
