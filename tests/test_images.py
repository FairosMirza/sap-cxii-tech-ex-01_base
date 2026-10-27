import os
import threading
from concurrent.futures import ThreadPoolExecutor

import httpx
import pytest

from similarity.images import _download_one


class _ImageClient:
    def get(self, url):
        return httpx.Response(200, headers={"content-type": "image/jpeg"}, content=b"jpeg-bytes")


def test_concurrent_downloads_of_one_url_do_not_collide(tmp_path, monkeypatch):
    both_written = threading.Barrier(2, timeout=5)
    real_replace = os.replace

    def replace_after_both_wrote(src, dst):
        both_written.wait()  # forces the race: both workers have written before either renames
        real_replace(src, dst)

    monkeypatch.setattr(os, "replace", replace_after_both_wrote)
    dest = tmp_path / "image.jpg"
    with ThreadPoolExecutor(2) as pool:
        results = list(pool.map(lambda _: _download_one(_ImageClient(), "https://img/x.jpg", dest), range(2)))

    assert results == [True, True]
    assert dest.read_bytes() == b"jpeg-bytes"
    assert not list(tmp_path.glob("*.part"))


def test_failed_write_leaves_no_temp_file(tmp_path, monkeypatch):
    def fail(src, dst):
        raise OSError("disk full")

    monkeypatch.setattr(os, "replace", fail)
    with pytest.raises(OSError):
        _download_one(_ImageClient(), "https://img/x.jpg", tmp_path / "image.jpg")
    assert not list(tmp_path.iterdir())
