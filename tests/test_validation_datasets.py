from ml_platform.data.validation_datasets import _sanitize_label_token


def test_sanitize_label_token_handles_spaces_and_symbols() -> None:
    assert _sanitize_label_token("Ankle boot") == "Ankle_boot"
    assert _sanitize_label_token("T-shirt/top") == "T_shirt_top"
