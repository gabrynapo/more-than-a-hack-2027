import numpy as np
import pandas as pd

from possession import extra_features as ef
from possession.tracking import AWAY, HOME


def toy(n=200):
    """Two teams of three; home keeper at x = 5, away keeper at x = 100; home attacks +x."""
    team = np.array([HOME] * 3 + [AWAY] * 3, dtype=np.int8)
    pos = np.zeros((n, 6, 2))
    pos[:, :, 1] = 34.0
    pos[:, 0, 0], pos[:, 1, 0], pos[:, 2, 0] = 5.0, 40.0, 50.0
    pos[:, 3, 0], pos[:, 4, 0], pos[:, 5, 0] = 100.0, 60.0, 70.0
    ball = np.full((n, 2), np.nan)
    ball[:50] = [52.0, 2.0]                       # seen near the touchline, then lost
    tracking = {"pos": pos.astype(np.float32), "team": team, "ball": ball.astype(np.float32),
                "pitch": np.array([105.0, 68.0], dtype=np.float32), "speed": np.zeros((n, 6), np.float32)}
    half = np.ones(n, dtype=np.int8)
    return tracking, half, pd.DataFrame({"ball_still_s": np.r_[np.full(50, 2.0), np.zeros(n - 50)]})


def test_prev_touchline_and_keepers():
    tracking, half, features = toy()
    out = ef.extra_columns(tracking, half, features)
    assert np.allclose(out["ball_prev_d_touchline"].values[60:], 2.0)
    assert out["gk_h_in_box"].values[0] == 1 and out["gk_a_in_box"].values[0] == 1


def test_restart_taker_is_the_nearest_side_and_is_carried_on():
    tracking, half, features = toy()
    out = ef.extra_columns(tracking, half, features)
    assert out["rs_taker"].values[10] == 1.0          # home player at x = 50 is nearest to (52, 2)
    assert out["rs_last_taker"].values[120] == 1.0 and out["rs_since_s"].values[120] > 2


def test_meeting_point_of_players_running_to_one_spot():
    target = np.array([30.0, 20.0])
    starts = np.array([[10.0, 10.0], [50.0, 5.0], [40.0, 60.0], [0.0, 40.0]])
    velocity = (target - starts) / np.linalg.norm(target - starts, axis=1, keepdims=True) * 5
    point = ef.meeting_point(starts[None], velocity[None])
    assert np.allclose(point[0], target, atol=1e-6)


def test_with_extras_appends_and_caches(tmp_path, monkeypatch):
    from possession import dfl, proxy
    monkeypatch.setattr(dfl, "CACHE_DIR", str(tmp_path))
    tracking, half, features = toy()
    calls = []

    def noisy():
        calls.append(1)
        return tracking

    first = proxy.with_extras("M", "n3", "default", features, noisy, half)
    again = proxy.with_extras("M", "n3", "default", features, noisy, half)
    assert len(calls) == 1                                     # second call read the cache
    assert list(first.columns[:1]) == ["ball_still_s"] and "gk_h_d_ball" in first.columns
    pd.testing.assert_frame_equal(first, again)
    assert proxy.extras_cache("M", "n3", "default").endswith("M_n3_default_extra.parquet")


def test_model_names_without_extras_are_unchanged():
    import pytest
    from possession.pipeline import check_extras, model_name
    assert model_name("sup_tau1.5_cx0_n3_ev", False, False) == "sup_tau1.5_cx0_n3_ev_m0"
    assert model_name("sup_tau1.5_cx0_n3_ev", False, True) == "sup_tau1.5_cx0_n3_ev_m0_xf"
    with pytest.raises(ValueError):
        check_extras(True, True)
