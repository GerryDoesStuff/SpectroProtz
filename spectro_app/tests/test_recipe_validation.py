from spectro_app.engine.recipe_model import Recipe

def test_recipe_validation():
    r = Recipe(params={"smoothing": {"window": 15, "polyorder": 3}})
    assert r.validate() == []
