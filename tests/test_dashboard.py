from pathlib import Path

from streamlit.testing.v1 import AppTest


def test_dashboard_empty_dataset_and_pages(monkeypatch):
    # AppTest has no real session manager; exercise component rendering without one.
    monkeypatch.setattr("streamlit_webrtc.component.get_this_session_info", lambda: None)
    app = AppTest.from_file(str(Path(__file__).parents[1] / "app.py"), default_timeout=30).run()
    assert not app.exception
    assert app.title[0].value == "SignVerse AI"
    for page in ["Train & evaluate", "Recognize", "Collect", "Dataset"]:
        app.sidebar.radio[0].set_value(page).run()
        assert not app.exception
