"""Run the Streamlit app headlessly with every model and every sample leaf.

  python tests/test_app.py
"""
from pathlib import Path

from streamlit.testing.v1 import AppTest

APP = Path(__file__).resolve().parents[1] / "app" / "app.py"


def test_every_model_and_sample():
    at = AppTest.from_file(str(APP), default_timeout=300).run()
    assert not at.exception
    at.main.radio[0].set_value("Use a sample leaf").run()
    options = [o for o in at.main.selectbox[0].options if o != "(none)"]
    assert len(options) == 10
    for model in at.sidebar.radio[0].options:
        at.sidebar.radio[0].set_value(model).run()
        hits = 0
        for o in options:
            at.main.selectbox[0].set_value(o).run()
            assert not at.exception, (model, o, at.exception)
            hits += at.main.subheader[0].value.replace(" - ", " ").lower() == o.lower()
        print(f"{model}: {hits}/{len(options)} samples correct", flush=True)
        assert hits >= 6


if __name__ == "__main__":
    test_every_model_and_sample()
    print("app test passed")
