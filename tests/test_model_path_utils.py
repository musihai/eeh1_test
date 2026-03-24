from recipe.time_series_forecast.model_path_utils import (
    has_loadable_transformers_weights,
    resolve_transformers_model_dir,
)


def test_has_loadable_transformers_weights_recognizes_canonical_files(tmp_path):
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    (model_dir / "model.safetensors").write_text("stub")

    assert has_loadable_transformers_weights(model_dir)


def test_resolve_transformers_model_dir_falls_back_to_sibling_hf_merged(tmp_path):
    step_dir = tmp_path / "global_step_50"
    huggingface_dir = step_dir / "huggingface"
    hf_merged_dir = step_dir / "hf_merged"
    huggingface_dir.mkdir(parents=True)
    hf_merged_dir.mkdir(parents=True)
    (huggingface_dir / "config.json").write_text("{}")
    (huggingface_dir / "model-00001-of-00002.safetensors").write_text("shard")
    (hf_merged_dir / "model.safetensors").write_text("merged")

    assert resolve_transformers_model_dir(huggingface_dir) == hf_merged_dir


def test_resolve_transformers_model_dir_raises_when_no_loadable_weights_exist(tmp_path):
    model_dir = tmp_path / "broken"
    model_dir.mkdir()
    (model_dir / "config.json").write_text("{}")

    try:
        resolve_transformers_model_dir(model_dir)
    except FileNotFoundError as exc:
        assert "No loadable Transformers weights found" in str(exc)
    else:
        raise AssertionError("Expected resolve_transformers_model_dir to fail")
