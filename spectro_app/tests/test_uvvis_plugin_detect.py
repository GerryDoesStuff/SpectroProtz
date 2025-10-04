from spectro_app.plugins.uvvis.plugin import UvVisPlugin


def test_uvvis_detect_accepts_xls_extension():
    plugin = UvVisPlugin()
    assert plugin.detect(["/path/to/spectrum.xls"]) is True
