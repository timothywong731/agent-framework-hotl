from hotl_demo.main import model_present

TAGS = {"models": [{"name": "gemma4:31b"}, {"name": "qwen3.6:latest"}]}


def test_model_present_exact_tag():
    assert model_present(TAGS, "gemma4:31b") is True


def test_model_present_base_name_matches_any_tag():
    assert model_present(TAGS, "qwen3.6") is True


def test_model_absent():
    assert model_present(TAGS, "gemma4:9b") is False
    assert model_present({}, "gemma4:31b") is False
