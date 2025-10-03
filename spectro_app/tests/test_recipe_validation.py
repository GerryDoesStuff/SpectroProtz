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


def test_recipe_validation_requires_blank_fallback_when_optional():
    params = {
        "blank": {"subtract": True, "require": False},
    }
    errs = Recipe(params=params).validate()
    assert "fallback" in errs[0].lower()
