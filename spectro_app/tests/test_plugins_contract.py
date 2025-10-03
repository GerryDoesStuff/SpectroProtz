from spectro_app.engine.plugin_api import SpectroscopyPlugin

def test_plugin_base_methods():
    p = SpectroscopyPlugin()
    assert p.detect([]) is False
