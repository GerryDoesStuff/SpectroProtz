from spectro_app.engine.ftir_lookup import parse_lookup_text


def test_formula_token_routes_to_molform():
    criteria = parse_lookup_text("C6H6")

    assert criteria.filters == {"molform": ["C6H6"]}
    assert criteria.peaks == []
    assert criteria.errors == []


def test_formula_and_title_tokens_split_filters():
    criteria = parse_lookup_text("C6H6 acetone")

    assert criteria.filters == {"molform": ["C6H6"], "title": ["acetone"]}
