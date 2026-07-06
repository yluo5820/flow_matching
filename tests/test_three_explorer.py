from fm_lab.image_diagnostics import three_explorer


def test_load_three_source_reuses_geometry_cache_before_network(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    filename = f"three-{three_explorer.THREE_VERSION}.min.js"
    cached_path = tmp_path / "outputs" / "geometry_explorer" / "assets" / "vendor" / filename
    cached_path.parent.mkdir(parents=True)
    cached_path.write_text("cached-three-source", encoding="utf-8")
    requested_dir = tmp_path / "runs" / "example" / "plots" / "assets" / "vendor"

    def fail_download(*args, **kwargs) -> None:
        raise AssertionError("download should not be attempted when a local cache exists")

    monkeypatch.setattr(three_explorer.urllib.request, "urlretrieve", fail_download)

    assert three_explorer._load_three_source(requested_dir) == "cached-three-source"
    assert (requested_dir / filename).read_text(encoding="utf-8") == "cached-three-source"
