from spectro_app.engine.recipe_model import Recipe


def test_recipe_validation_passes_for_reasonable_recipe():
    params = {
        "smoothing": {"enabled": True, "window": 15, "polyorder": 3},
        "join": {"enabled": True, "window": 5, "threshold": 0.5},
        "blank": {"subtract": True, "require": True},
    }
    assert Recipe(params=params).validate() == []


def test_recipe_validation_flags_join_and_smoothing_issues():
    params = {
        "smoothing": {"enabled": True, "window": 4, "polyorder": 3},
        "join": {"enabled": True, "window": 0, "threshold": -1},
    }
    errs = Recipe(params=params).validate()
    assert "Savitzky–Golay window must be odd" in errs
    assert "Join correction window must be positive" in errs
    assert "Join detection threshold must be positive" in errs


def test_recipe_validation_allows_optional_blank_without_fallback():
    params = {
        "blank": {"subtract": True, "require": False},
    }
    assert Recipe(params=params).validate() == []


def test_recipe_validation_flags_invalid_drift_limits():
    params = {
        "qc": {
            "drift": {
                "enabled": True,
                "window": {"min": 300, "max": 200},
                "max_slope_per_hour": -1,
                "max_delta": "not-a-number",
            }
        }
    }
    errs = Recipe(params=params).validate()
    assert any("Drift window" in err for err in errs)
    assert any("slope limit" in err for err in errs)
    assert any("delta limit" in err for err in errs)
