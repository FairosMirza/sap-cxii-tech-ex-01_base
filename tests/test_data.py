import json
import zipfile

import pytest

from similarity.build import load_aligned
from similarity.data import (
    extract_colors,
    load_products,
    normalize_brand,
    parse_colors,
    parse_discount,
    parse_image_urls,
    parse_price,
    parse_rating,
    parse_record,
    parse_review_count,
    parse_segment,
    parse_weight_grams,
)


@pytest.mark.parametrize("raw, expected", [("931.00", 931.0), ("1,299.00", 1299.0), ("", None), (None, None),
                                           ("n/a", None), ("0", None)])
def test_parse_price(raw, expected):
    assert parse_price(raw) == expected


@pytest.mark.parametrize("raw, expected", [("50%", 0.5), ("0%", 0.0), ("100%", None), ("abc", None), (None, None)])
def test_parse_discount(raw, expected):
    assert parse_discount(raw) == expected


@pytest.mark.parametrize("raw, expected", [("999999999", None), ("200 g", 200.0), ("68 Grams", 68.0),
                                           ("1.2 Kilograms", 1200.0), ("1 Kg", 1000.0), ("look slimmer", None)])
def test_parse_weight_grams(raw, expected):
    assert parse_weight_grams(raw) == pytest.approx(expected) if expected else parse_weight_grams(raw) is None


@pytest.mark.parametrize("raw, expected", [("4.3", 4.3), ("5.0", 5.0), ("0", None), ("abc", None), (None, None)])
def test_parse_rating(raw, expected):
    assert parse_rating(raw) == expected


def test_parse_review_count():
    assert parse_review_count("39", None) == 39
    assert parse_review_count(None, {"Customer_Reviews": "4.3 out of 5 stars 1,049 customer reviews"}) == 1049
    assert parse_review_count(None, {"Customer_Reviews": "Be the first to review this item"}) == 0
    assert parse_review_count(None, None) is None


def test_normalize_brand():
    assert normalize_brand("  BIBA ") == "biba"
    assert normalize_brand("Generic") is None
    assert normalize_brand(None) is None


@pytest.mark.parametrize("category, name, expected", [
    ("WomensKurtasKurtis", "Anything", "women"),
    ("MensT_Shirts", "Anything", "men"),
    (None, "Puma Unisex Regular T-Shirt", "unisex"),
    (None, "Mothercare Girls' A-Line Dress", "girls"),
    (None, "Boys and Girls Night Suit", "kids"),
    (None, "Men's Formal Shirt", "men"),
    (None, "Women's Top", "women"),
    (None, "Plain Socks", None),
])
def test_parse_segment(category, name, expected):
    assert parse_segment(category, name) == expected


def test_extract_colors_maps_shades_and_ignores_brand():
    assert extract_colors("Navy Blue Kurta") == ("blue",)
    assert extract_colors("Red Tape Men's Black Shoes", exclude="red tape") == ("black",)
    assert extract_colors("Multi-Coloured Print Saree") == ("multi",)


def test_parse_colors_prefers_own_colour_over_variant_list():
    record = {"product_url": "https://www.amazon.in/Chhabra-Lehenga-DPNQ1376_Sea-Green_X-Large/dp/B076HR4FVZ/",
              "product_name": "Chhabra Lehenga", "colour": "Red|Blue"}
    assert parse_colors(record, brand=None) == ("green",)
    assert parse_colors({"product_name": "Lehenga", "colour": "Red|Navy"}, brand=None) == ("red", "blue")


def test_parse_image_urls_requests_small_versions():
    record = {"medium": "https://x/images/I/51mw03cLo6L.jpg|https://x/images/I/41KCH3%2B1BSL._SS40_.jpg"}
    assert parse_image_urls(record) == ("https://x/images/I/51mw03cLo6L._SL224_.jpg",
                                        "https://x/images/I/41KCH3%2B1BSL._SL224_.jpg")
    assert parse_image_urls({}) == ()


def test_parse_record_derives_list_price_from_discount():
    product = parse_record({"uniq_id": "u1", "asin": "A", "product_name": "Kurta", "sales_price": "500.00",
                            "discount_percentage": "50%", "rating": "4.0", "weight": "999999999"})
    assert (product.sales_price, product.list_price, product.weight_g) == (500.0, 1000.0, None)


def test_build_rejects_git_lfs_pointer_files(tmp_path):
    pointer = tmp_path / "text_minilm.npz"
    pointer.write_text("version https://git-lfs.github.com/spec/v1\noid sha256:0\nsize 1\n")
    with pytest.raises(RuntimeError, match="git lfs install && git lfs pull"):
        load_aligned(pointer, ["a"])


def test_load_products_from_zip_infers_brand_and_category(tmp_path):
    records = [
        {"uniq_id": "1", "asin": "A1", "product_name": "Puma Men's T-Shirt", "brand": "Puma", "browsenode": "7",
         "parent___child_category__all": {"ClothingAccessories": "#1", "MensT_Shirts": "#2"}},
        {"uniq_id": "2", "asin": "A2", "product_name": "Puma Men's Polo", "brand": "PUMA", "browsenode": "7",
         "parent___child_category__all": {"ClothingAccessories": "#1", "MensT_Shirts": "#3"}},
        {"uniq_id": "3", "asin": "A3", "product_name": "Puma Sports Tee", "brand": None, "browsenode": "7"},
        {"uniq_id": "3", "asin": "A3", "product_name": "duplicate row", "browsenode": "7"},
    ]
    archive = tmp_path / "archive.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr("data.ldjson", "\n".join(json.dumps(r) for r in records))

    products = load_products(archive)
    assert [p.uniq_id for p in products] == ["1", "2", "3"]
    assert products[2].brand == "puma"  # name starts with a brand seen on 2+ products
    assert products[2].category == "MensT_Shirts"  # majority category of its browse node
    assert products[2].segment == "men"
