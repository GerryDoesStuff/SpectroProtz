from spectro_app.plugins.uvvis.plugin import UvVisPlugin


def _write_simple_csv(path):
    path.write_text("Wavelength,SampleA\n200,0.10\n210,0.11\n", encoding="utf-8")


def test_queue_override_marks_blank(tmp_path):
    csv_path = tmp_path / "sample.csv"
    _write_simple_csv(csv_path)
    plugin = UvVisPlugin(enable_manifest=False)

    specs = plugin.load([str(csv_path)])
    assert specs
    assert specs[0].meta.get("role") == "sample"

    plugin.set_queue_overrides({str(csv_path): {"role": "blank", "blank_id": "C1"}})
    specs = plugin.load([str(csv_path)])
    assert specs[0].meta.get("role") == "blank"
    assert specs[0].meta.get("blank_id") == "C1"


def test_queue_override_blank_id_defaults(tmp_path):
    csv_path = tmp_path / "blank.csv"
    _write_simple_csv(csv_path)
    plugin = UvVisPlugin(enable_manifest=False)

    plugin.set_queue_overrides({str(csv_path): {"role": "blank"}})
    specs = plugin.load([str(csv_path)])
    assert specs[0].meta.get("role") == "blank"
    assert specs[0].meta.get("blank_id") in {"SampleA", "blank"}
