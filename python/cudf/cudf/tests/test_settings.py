from cudf.settings import set_options, settings


def test_set_options():
    def get_setting():
        return settings.formatting.get("nrows")

    assert get_setting() is None
    val = 13
    with set_options(formatting={"nrows": val}):
        assert get_setting() == val
    assert get_setting() is None
