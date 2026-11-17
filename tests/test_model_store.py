from possession import model_store


def test_get_trains_once_and_then_loads(tmp_path, monkeypatch):
    calls = []

    def train():
        calls.append(1)
        return {"weights": [1, 2, 3]}

    monkeypatch.setattr(model_store, "DIRECTORY", None)
    assert model_store.get("m", train) == {"weights": [1, 2, 3]} and len(calls) == 1
    monkeypatch.setattr(model_store, "DIRECTORY", str(tmp_path))
    assert model_store.get("m", train) == {"weights": [1, 2, 3]} and len(calls) == 2
    assert model_store.get("m", train) == {"weights": [1, 2, 3]} and len(calls) == 2   # loaded
    assert (tmp_path / "m.pkl").is_file()
