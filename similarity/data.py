"""Streaming loader that turns the raw LDJSON dump into clean, typed products.

The raw fields are strings with sentinels and mixed units ("999999999" weights,
"50%" discounts, nested category dicts), so all parsing lives here and the rest
of the code only sees typed values (``None`` = unknown).
"""

from __future__ import annotations

import io
import json
import re
import zipfile
from collections import Counter, defaultdict
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import unquote, urlparse

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
LDJSON_NAME = (
    "marketing_sample_for_amazon_com-amazon_fashion_products__20200201_20200430__30k_data.ldjson"
)
MISSING_WEIGHT = "999999999"


@dataclass(slots=True)
class Product:
    uniq_id: str
    asin: str
    name: str
    brand: str | None
    sales_price: float | None  # INR
    list_price: float | None  # INR (MRP), derived from the discount
    weight_g: float | None
    rating: float | None
    review_count: int | None
    category: str | None  # most specific category, e.g. "WomensKurtasKurtis"
    browsenode: str | None
    segment: str | None  # women / men / girls / boys / baby / kids / unisex
    colors: tuple[str, ...]  # colour families, see COLOR_FAMILIES
    image_urls: tuple[str, ...]  # main image first; later ones are fallbacks for dead links
    co_bought: tuple[str, ...] = ()  # normalised names; used only for offline evaluation


# --------------------------------------------------------------------------- parsing


def norm_text(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip().lower()


def parse_price(value: str | None) -> float | None:
    if not value:
        return None
    try:
        price = float(value.replace(",", "").replace("₹", "").strip())
    except ValueError:
        return None
    return price if price > 0 else None


def parse_discount(value: str | None) -> float | None:
    if not value:
        return None
    try:
        discount = float(value.strip().rstrip("%")) / 100
    except ValueError:
        return None
    return discount if 0 <= discount < 1 else None


_WEIGHT_RE = re.compile(r"(\d+(?:\.\d+)?)\s*(kilograms?|kgs?|grams?|g)\b", re.IGNORECASE)


def parse_weight_grams(value: str | None) -> float | None:
    if not value or value.strip() == MISSING_WEIGHT:
        return None
    match = _WEIGHT_RE.search(value.replace(",", ""))
    if not match:
        return None
    grams = float(match.group(1)) * (1000 if match.group(2).lower().startswith("k") else 1)
    return grams if 0 < grams < 50_000 else None


def parse_rating(value: str | None) -> float | None:
    try:
        rating = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    return rating if 1.0 <= rating <= 5.0 else None


_REVIEWS_RE = re.compile(r"([\d,]+)\s+customer reviews?", re.IGNORECASE)


def parse_review_count(no_of_reviews: str | None, details: dict | None) -> int | None:
    if no_of_reviews:
        try:
            return int(no_of_reviews.replace(",", ""))
        except ValueError:
            pass
    text = (details or {}).get("Customer_Reviews") or ""
    if "first to review" in text:
        return 0
    match = _REVIEWS_RE.search(text)
    return int(match.group(1).replace(",", "")) if match else None


_PLACEHOLDER_BRANDS = {"generic", "unbranded", "unknown", "na", "n/a", "none", "other", "others", "no brand"}


def normalize_brand(value: str | None) -> str | None:
    if not value:
        return None
    brand = norm_text(value).strip(" .,-_'\"")
    return None if not brand or brand in _PLACEHOLDER_BRANDS else brand


def parse_category(categories: dict | None) -> str | None:
    """Most specific category; the dict lists the parent first, then the child."""
    if not categories:
        return None
    return list(categories)[-1]


_SEGMENT_FROM_CATEGORY = (("Womens", "women"), ("Mens", "men"), ("Girls", "girls"), ("Boys", "boys"), ("Baby", "baby"))
_SEGMENT_PATTERNS = {
    "baby": re.compile(r"\b(baby|babies|infants?|newborns?|toddlers?)\b"),
    "girls": re.compile(r"\bgirls?\b"),
    "boys": re.compile(r"\bboys?\b"),
    "kids": re.compile(r"\b(kids?|children|childrens|child)\b"),
    "women": re.compile(r"\b(women|womens|woman|womans|ladies|lady)\b"),
    "men": re.compile(r"\b(men|mens|gents)\b"),
    "unisex": re.compile(r"\bunisex\b"),
}


def parse_segment(category: str | None, name: str) -> str | None:
    """Target shopper; the structured category wins over keywords in the name."""
    for prefix, segment in _SEGMENT_FROM_CATEGORY:
        if category and category.startswith(prefix):
            return segment
    text = re.sub(r"['’`]", "", name.lower())
    found = {segment for segment, pattern in _SEGMENT_PATTERNS.items() if pattern.search(text)}
    if "baby" in found:
        return "baby"
    if {"girls", "boys"} <= found:
        return "kids"
    for segment in ("girls", "boys", "kids"):
        if segment in found:
            return segment
    if {"women", "men"} <= found or "unisex" in found:
        return "unisex"
    return "women" if "women" in found else "men" if "men" in found else None


# Shade -> family. Families keep "navy" and "royal blue" comparable; the image
# embedding is responsible for finer visual differences.
COLOR_FAMILIES: dict[str, tuple[str, ...]] = {
    "black": ("black", "jet black"),
    "white": ("white", "off white", "offwhite", "ivory", "milky white"),
    "grey": ("grey", "gray", "charcoal", "slate"),
    "blue": ("blue", "navy", "navy blue", "royal blue", "sky blue", "dark blue", "light blue", "indigo",
             "turquoise", "aqua", "cyan", "denim blue", "airforce blue", "firozi", "ink blue"),
    "red": ("red", "maroon", "wine", "burgundy", "crimson"),
    "pink": ("pink", "baby pink", "light pink", "peach", "coral", "magenta", "fuchsia", "rani pink", "onion pink"),
    "green": ("green", "olive", "mint", "sea green", "bottle green", "dark green", "rama green", "teal", "lime",
              "pista"),
    "yellow": ("yellow", "mustard", "lemon"),
    "orange": ("orange", "rust"),
    "purple": ("purple", "violet", "lavender", "lilac", "mauve"),
    "brown": ("brown", "tan", "coffee", "camel", "chocolate"),
    "beige": ("beige", "cream", "khaki", "nude", "fawn"),
    "gold": ("gold", "golden", "copper", "bronze"),
    "silver": ("silver",),
    "multi": ("multicolor", "multicolour", "multicolored", "multicoloured", "multi color", "multi colour",
              "multi coloured", "multi colored"),
}
_SHADE_TO_FAMILY = {shade: family for family, shades in COLOR_FAMILIES.items() for shade in shades}
_COLOR_RE = re.compile(
    r"\b(" + "|".join(re.escape(s) for s in sorted(_SHADE_TO_FAMILY, key=len, reverse=True)) + r")\b"
)


def extract_colors(text: str, exclude: str | None = None) -> tuple[str, ...]:
    """Colour families mentioned in free text, ignoring the brand (e.g. "Red Tape")."""
    text = norm_text(re.sub(r"[-_/|,()]+", " ", unquote(text).lower()))
    if exclude:
        text = text.replace(exclude, " ")
    return tuple(dict.fromkeys(_SHADE_TO_FAMILY[m] for m in _COLOR_RE.findall(text)))


def parse_colors(record: dict, brand: str | None) -> tuple[str, ...]:
    """Own colour from the URL slug / name / tech specs, else the listed variant colours."""
    slug = urlparse(record.get("product_url") or "").path.split("/")[1:2]
    tech = (record.get("technical_details__k_v_pairs") or {}).get("Color") or ""
    own = extract_colors(" ".join([*slug, record.get("product_name") or "", tech]), exclude=brand)
    return own or extract_colors(record.get("colour") or "", exclude=brand)


_IMAGE_SUFFIX_RE = re.compile(r"(\._[^/]*_)?\.(jpe?g|png|gif)$", re.IGNORECASE)


def parse_image_urls(record: dict, size: int = 224, limit: int = 3) -> tuple[str, ...]:
    """Product images, rewritten to ask Amazon's CDN for a ``size`` px version."""
    for field in ("medium", "large", "image_urls__small"):
        urls = [u.strip() for u in (record.get(field) or "").split("|") if _IMAGE_SUFFIX_RE.search(u.strip())]
        if urls:
            return tuple(_IMAGE_SUFFIX_RE.sub(rf"._SL{size}_.\2", u) for u in urls[:limit])
    return ()


def parse_record(record: dict) -> Product:
    details = record.get("product_details__k_v_pairs")
    brand = normalize_brand(record.get("brand"))
    sales_price = parse_price(record.get("sales_price"))
    discount = parse_discount(record.get("discount_percentage"))
    category = parse_category(record.get("parent___child_category__all"))
    name = re.sub(r"\s+", " ", record.get("product_name") or "").strip()
    return Product(
        uniq_id=record["uniq_id"],
        asin=record.get("asin") or record["uniq_id"],
        name=name,
        brand=brand,
        sales_price=sales_price,
        # No discount shown means the product sells at list price.
        list_price=sales_price / (1 - discount) if sales_price and discount else sales_price,
        weight_g=parse_weight_grams(record.get("weight")),
        rating=parse_rating(record.get("rating")),
        review_count=parse_review_count(record.get("no__of_reviews"), details),
        category=category,
        browsenode=record.get("browsenode") or None,
        segment=parse_segment(category, name),
        colors=parse_colors(record, brand),
        image_urls=parse_image_urls(record),
        co_bought=tuple(
            norm_text(n) for n in (record.get("other_items_customers_buy") or "").split("|") if n.strip()
        ),
    )


# --------------------------------------------------------------------------- loading


def default_data_path() -> Path:
    extracted = DATA_DIR / LDJSON_NAME
    return extracted if extracted.exists() else DATA_DIR / "archive.zip"


def iter_records(path: Path | None = None) -> Iterator[dict]:
    """Yield raw JSON records from the ``.ldjson`` file or straight from the zip."""
    path = Path(path) if path else default_data_path()
    if path.suffix == ".zip":
        with zipfile.ZipFile(path) as archive:
            member = next(n for n in archive.namelist() if n.endswith((".ldjson", ".jsonl")))
            with archive.open(member) as raw:
                yield from (json.loads(line) for line in io.TextIOWrapper(raw, encoding="utf-8") if line.strip())
    else:
        with path.open(encoding="utf-8") as handle:
            yield from (json.loads(line) for line in handle if line.strip())


def _infer_missing_brands(products: list[Product]) -> None:
    """Fill a missing brand when the name starts with a brand seen on other products."""
    counts = Counter(p.brand for p in products if p.brand)
    by_first_word: dict[str, list[str]] = defaultdict(list)
    for brand, count in counts.items():
        if count >= 2 and len(brand) >= 3:
            by_first_word[brand.split(" ", 1)[0]].append(brand)
    for brands in by_first_word.values():
        brands.sort(key=len, reverse=True)
    for product in products:
        if product.brand:
            continue
        name = norm_text(product.name)
        for brand in by_first_word.get(name.split(" ", 1)[0], ()):
            if name == brand or name.startswith(brand + " "):
                product.brand = brand
                break


def _impute_categories(products: list[Product]) -> None:
    """Products without a category inherit the majority category of their browse node."""
    votes: dict[str, Counter] = defaultdict(Counter)
    for p in products:
        if p.category and p.browsenode:
            votes[p.browsenode][p.category] += 1
    for p in products:
        if not p.category and p.browsenode in votes:
            p.category = votes[p.browsenode].most_common(1)[0][0]
            p.segment = parse_segment(p.category, p.name)


def load_products(path: Path | None = None) -> list[Product]:
    seen: set[str] = set()
    products = []
    for record in iter_records(path):
        if record.get("uniq_id") and record["uniq_id"] not in seen:
            seen.add(record["uniq_id"])
            products.append(parse_record(record))
    _infer_missing_brands(products)
    _impute_categories(products)
    return products
