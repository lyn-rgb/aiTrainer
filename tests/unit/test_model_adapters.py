from aitrainer.plugins.models import BERTAdapter, GPTAdapter, LlamaAdapter, TransformerModelConfig


def test_model_adapter_contracts_validate_without_torch():
    for adapter in (LlamaAdapter(), GPTAdapter(), BERTAdapter()):
        adapter.config.validate()
        assert adapter.family


def test_model_config_rejects_non_divisible_heads():
    try:
        TransformerModelConfig(hidden_size=10, heads=3).validate()
    except ValueError:
        return
    raise AssertionError("hidden size/head mismatch must fail")
