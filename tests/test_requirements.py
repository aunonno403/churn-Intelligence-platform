"""The deployed app must install the exact library versions that saved its model."""

import importlib.metadata
from pathlib import Path

REQUIREMENTS = Path(__file__).resolve().parents[1] / "app" / "requirements.txt"


def test_app_pins_match_the_training_environment():
    pins = {}
    for line in REQUIREMENTS.read_text().splitlines():
        spec = line.split("#")[0].strip()
        if "==" in spec:
            name, version = spec.split("==")
            pins[name.strip()] = version.strip()
    assert pins, "app/requirements.txt should pin the libraries that pickle the model"
    installed = {name: importlib.metadata.version(name) for name in pins}
    assert pins == installed, (
        "Retrained with different versions? Update the pins in app/requirements.txt "
        "to match the environment that saved the model."
    )
