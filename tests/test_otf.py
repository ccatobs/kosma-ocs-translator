"""
Tests of the OTF mode C (obs_otf_mode C): dispatch of PRETP / OTFTP, the OTF
path, the PRETP move-to and the status of an OTF scan cookie.
"""
import json
import time
from unittest.mock import MagicMock

import numpy as np
import pytest
from astropy import units as u
from astropy.coordinates import AltAz, SkyCoord
from astropy.time import Time

import kosma_ocs_translator as kot
from kosma_ocs_translator import _get_time_from_acu_format, _otf_scan_time
from helpers import BASE_OBS2TEL, CCAT_LOCATION, write_obs2tel

# a line in J2000 offsets with the true angle, scanning in +lam
OTF_LINE = {
    "obs_otf_mode": "C",
    "obs_lam_del": -300.0, "obs_bet_del": 60.0,      # line start S [arcsec]
    "obs_lam_vel": 100.0, "obs_bet_vel": 0.0,        # [arcsec/s]
}


def _ok():
    return MagicMock(status_code=200, json=lambda: {"status": "ok"})


def _rejected():
    return MagicMock(status_code=200, json=lambda: {"status": "error", "message": "bad path"})


def _set_obs2tel(t, tmp_path, cookie, **values):
    """Write a new KOSMA_obs2tel.set (BASE_OBS2TEL + values) for process_obs2tel."""
    write_obs2tel(tmp_path, {**BASE_OBS2TEL, "obs_cookie": cookie, **values})


def _use_obs2tel(t, cookie, **values):
    """Set the obs2tel of the next command directly (for the send methods)."""
    t.obs2tel = {**BASE_OBS2TEL, "obs_cookie": cookie, **values}
    t.next_tel_return_cookie = cookie


def _sky_altaz(lam_del, bet_del, t_unix):
    """Independent reference: az/el [deg] of the BASE source + J2000 true-angle offset."""
    src = SkyCoord(BASE_OBS2TEL["obs_lam_on"] * u.deg, BASE_OBS2TEL["obs_bet_on"] * u.deg,
                   frame="icrs")
    pos = src.spherical_offsets_by(lam_del * u.arcsec, bet_del * u.arcsec)
    altaz = pos.transform_to(AltAz(obstime=Time(t_unix, format="unix"), location=CCAT_LOCATION))
    return altaz.az.deg, altaz.alt.deg


# ---------------------------------------------------------------------------
# effective scan time f(tau)
# ---------------------------------------------------------------------------

def test_otf_scan_time_profile():
    ramp, stop, d = 3.0, 2.0, 6.0
    tau = np.array([-10.0, -ramp, -ramp / 2, 0.0, 3.0, d, d + stop, d + 10.0])
    f = _otf_scan_time(tau, d, ramp, stop)
    assert f == pytest.approx([-1.5, -1.5, -1.5 + 3.0 * (1 / 8 - 1 / 32), 0.0, 3.0, 6.0, 7.0, 7.0])


def test_otf_scan_time_smooth():
    """Speed and acceleration have no jumps at the ends of ramp and stop."""
    ramp, stop, d, h = 3.0, 2.0, 6.0, 1e-4
    for tau0, speed in [(-ramp, 0.0), (0.0, 1.0), (d, 1.0), (d + stop, 0.0)]:
        f = _otf_scan_time([tau0 - 2 * h, tau0 - h, tau0, tau0 + h, tau0 + 2 * h], d, ramp, stop)
        v = np.diff(f) / h
        assert v == pytest.approx(speed, abs=1e-3)
        assert np.diff(v) / h == pytest.approx(0.0, abs=1e-2)


def test_otf_scan_time_without_ramp_and_stop():
    f = _otf_scan_time([-1.0, 0.5, 7.0], 6.0, 0.0, 0.0)
    assert f == pytest.approx([0.0, 0.5, 6.0])


# ---------------------------------------------------------------------------
# dispatch
# ---------------------------------------------------------------------------

def test_otftp_starts_otf_scan_even_for_same_target(translator_factory, tmp_path):
    t = translator_factory(obs2tel_overrides=OTF_LINE)
    t.last_payload["type"] = "move_to"        # e.g. the PRETP move-to of the line
    t.last_payload["start_time"] = time.time()
    t.tel_return_cookie = 1
    t.start_command = MagicMock()
    _set_obs2tel(t, tmp_path, 2, **OTF_LINE, obs_track_duration=6.0,
                 obs_start_time=time.time() + 14)

    assert t.process_obs2tel() is True

    t.start_command.assert_called_once_with(t._send_otf_scan)
    assert t.tel_return_cookie == 1           # returned by the command thread


@pytest.mark.parametrize("changes,kept", [
    pytest.param({},                            True,  id="same_position"),
    pytest.param({"obs_lam_del": 0.0},          False, id="other_offset"),
    pytest.param({"obs_coord_sys_del": "GALACTIC"}, False, id="other_frame"),
])
def test_pretp_keeps_running_path_at_same_position(translator_factory, tmp_path, changes, kept):
    # ON preset at the line start: otf_mode N, no velocity
    on_preset = {**OTF_LINE, "obs_otf_mode": "N", "obs_lam_vel": 0.0, "obs_bet_vel": 0.0}
    t = translator_factory(obs2tel_overrides=on_preset)
    t.last_payload["start_time"] = time.time() - 5.0     # running, 595 s left
    t.tel_return_cookie = 1
    t.tel_lost_track = "Y"
    t.start_command = MagicMock()
    _set_obs2tel(t, tmp_path, 2, **{**OTF_LINE, **changes}, obs_track_duration=0)

    assert t.process_obs2tel() is True

    if kept:
        assert not t.start_command.called
        assert t.tel_return_cookie == 2
        assert t.tel_lost_track == "N"
        assert t.last_obs2tel["obs_otf_mode"] == "C"
        assert t.otf_scan_window is None
    else:
        t.start_command.assert_called_once_with(t._send_pre_otf)
        assert t.tel_return_cookie == 1


@pytest.mark.parametrize("setup", ["ended_soon", "move_to", "otf", "failed", "aborted"])
def test_pretp_does_not_keep(translator_factory, tmp_path, setup):
    on_preset = {**OTF_LINE, "obs_otf_mode": "N", "obs_lam_vel": 0.0, "obs_bet_vel": 0.0}
    t = translator_factory(obs2tel_overrides=on_preset)
    t.last_payload["start_time"] = time.time() - 5.0
    if setup == "ended_soon":
        t.last_payload["start_time"] = time.time() - 600 + kot.OTF_PRE_KEEP_MIN_REMAINING / 2
    elif setup == "failed":
        t.tel_path_failed = True
    elif setup == "aborted":
        t.last_aborted = True
    else:
        t.last_payload["type"] = setup
    t.start_command = MagicMock()
    _set_obs2tel(t, tmp_path, 2, **OTF_LINE, obs_track_duration=0)

    t.process_obs2tel()

    t.start_command.assert_called_once_with(t._send_pre_otf)


def test_pattern_mode_not_implemented(translator_factory, tmp_path):
    t = translator_factory()
    t.tel_return_cookie = 1
    t.start_command = MagicMock()
    _set_obs2tel(t, tmp_path, 2, obs_otf_mode="P", obs_track_duration=6.0)

    assert t.process_obs2tel() is False
    assert not t.start_command.called


def test_no_reuse_of_otf_path(translator_factory):
    t = translator_factory(obs2tel_overrides=OTF_LINE)
    t.last_payload["type"] = "otf"
    t.last_payload["start_time"] = time.time()
    _use_obs2tel(t, 3, **OTF_LINE)

    assert t.reuse_last_command() is False


# ---------------------------------------------------------------------------
# OTF path
# ---------------------------------------------------------------------------

def test_otf_payload_follows_backend_offsets(translator_factory, monkeypatch):
    monkeypatch.setattr(kot, "OTF_RAMP_TIME", 3.0)
    monkeypatch.setattr(kot, "OTF_SETTLE_TIME", 0.0)
    monkeypatch.setattr(kot, "OTF_STOP_TIME", 0.0)
    t = translator_factory()
    obs2tel = {**t.obs2tel, **OTF_LINE}
    t0, d = 1.8e9 + 0.37, 6.0

    payload = t._build_otf_payload(obs2tel, t0, d)

    pts = np.array(payload["points"])
    assert payload["start_time"] == pytest.approx(t0 - 3.0)
    assert pts[:, 0] == pytest.approx(np.arange(91) * 0.1)
    # run-up point, line start at T0, line end at T0 + D (offset = del + vel * (t - T0))
    for i, lam in [(0, -300.0 - 150.0), (30, -300.0), (90, -300.0 + 600.0)]:
        az, el = _sky_altaz(lam, 60.0, payload["start_time"] + pts[i, 0])
        assert pts[i, 1] == pytest.approx(az, abs=1e-5)
        assert pts[i, 2] == pytest.approx(el, abs=1e-5)


@pytest.mark.parametrize("true_angle", ["Y", "N"])
def test_otf_payload_horizon_map_offsets(translator_factory, monkeypatch, true_angle):
    """Map offset in HORIZON: del + vel * f applied in az/el after the transform."""
    monkeypatch.setattr(kot, "OTF_RAMP_TIME", 3.0)
    monkeypatch.setattr(kot, "OTF_SETTLE_TIME", 0.0)
    monkeypatch.setattr(kot, "OTF_STOP_TIME", 0.0)
    t = translator_factory()
    obs2tel = {**t.obs2tel, **OTF_LINE,
               "obs_coord_sys_del": "HORIZON", "obs_true_angle_del": true_angle}
    t0, d = 1.8e9 + 0.37, 6.0

    payload = t._build_otf_payload(obs2tel, t0, d)

    pts = np.array(payload["points"])
    src = SkyCoord(BASE_OBS2TEL["obs_lam_on"] * u.deg, BASE_OBS2TEL["obs_bet_on"] * u.deg,
                   frame="icrs")
    # run-up point, line start at T0, line end at T0 + D
    for i, lam in [(0, -300.0 - 150.0), (30, -300.0), (90, -300.0 + 600.0)]:
        t_unix = payload["start_time"] + pts[i, 0]
        altaz = src.transform_to(
            AltAz(obstime=Time(t_unix, format="unix"), location=CCAT_LOCATION))
        if true_angle == "Y":
            ref = altaz.spherical_offsets_by(lam * u.arcsec, 60.0 * u.arcsec)
            az, el = ref.az.deg, ref.alt.deg
        else:
            az, el = altaz.az.deg + lam / 3600.0, altaz.alt.deg + 60.0 / 3600.0
        assert pts[i, 1] == pytest.approx(az % 360.0, abs=1e-5)
        assert pts[i, 2] == pytest.approx(el, abs=1e-5)


# HORIZON source: obs_lam/bet_on are az/el
HORIZON_SOURCE = {"obs_coord_sys_on": "HORIZON", "obs_lam_on": 180.0, "obs_bet_on": 45.0}


@pytest.mark.parametrize("true_angle", ["Y", "N"])
def test_otf_payload_horizon_source_horizon_map(translator_factory, monkeypatch, true_angle):
    """HORIZON source and map: az/el on + del + vel * f, independent of time."""
    monkeypatch.setattr(kot, "OTF_RAMP_TIME", 3.0)
    monkeypatch.setattr(kot, "OTF_SETTLE_TIME", 0.0)
    monkeypatch.setattr(kot, "OTF_STOP_TIME", 0.0)
    t = translator_factory()
    obs2tel = {**t.obs2tel, **OTF_LINE, **HORIZON_SOURCE,
               "obs_coord_sys_del": "HORIZON", "obs_true_angle_del": true_angle}
    t0, d = 1.8e9 + 0.37, 6.0

    payload = t._build_otf_payload(obs2tel, t0, d)

    pts = np.array(payload["points"])
    assert payload["start_time"] == pytest.approx(t0 - 3.0)
    on = SkyCoord(180.0 * u.deg, 45.0 * u.deg, frame="altaz")
    for i, lam in [(0, -300.0 - 150.0), (30, -300.0), (90, -300.0 + 600.0)]:
        if true_angle == "Y":
            ref = on.spherical_offsets_by(lam * u.arcsec, 60.0 * u.arcsec)
            az, el = ref.az.deg, ref.alt.deg
        else:
            az, el = 180.0 + lam / 3600.0, 45.0 + 60.0 / 3600.0
        assert pts[i, 1] == pytest.approx(az, abs=1e-6)
        assert pts[i, 2] == pytest.approx(el, abs=1e-6)
    # a constant velocity in az during the scan, zero in el
    assert pts[31:90, 3] == pytest.approx(
        np.diff(pts[[30, 90], 1])[0] / 6.0, rel=1e-2 if true_angle == "Y" else 1e-6)
    assert pts[31:90, 4] == pytest.approx(0.0, abs=1e-3)


def test_otf_payload_horizon_source_sky_map(translator_factory, monkeypatch):
    """HORIZON source, J2000 map: offset applied in J2000 at the time of each point."""
    monkeypatch.setattr(kot, "OTF_RAMP_TIME", 3.0)
    monkeypatch.setattr(kot, "OTF_SETTLE_TIME", 0.0)
    monkeypatch.setattr(kot, "OTF_STOP_TIME", 0.0)
    t = translator_factory()
    obs2tel = {**t.obs2tel, **OTF_LINE, **HORIZON_SOURCE}     # obs_coord_sys_del J2000
    t0, d = 1.8e9 + 0.37, 6.0

    payload = t._build_otf_payload(obs2tel, t0, d)

    pts = np.array(payload["points"])
    for i, lam in [(0, -300.0 - 150.0), (30, -300.0), (90, -300.0 + 600.0)]:
        frame = AltAz(obstime=Time(payload["start_time"] + pts[i, 0], format="unix"),
                      location=CCAT_LOCATION)
        ref = (SkyCoord(180.0 * u.deg, 45.0 * u.deg, frame=frame)
               .transform_to("icrs")
               .spherical_offsets_by(lam * u.arcsec, 60.0 * u.arcsec)
               .transform_to(frame))
        assert pts[i, 1] == pytest.approx(ref.az.deg, abs=1e-5)
        assert pts[i, 2] == pytest.approx(ref.alt.deg, abs=1e-5)


def test_pretp_keeps_move_to_of_horizon_source(translator_factory, tmp_path):
    # ON preset of a HORIZON source: a move-to at the line start
    on_preset = {**OTF_LINE, **HORIZON_SOURCE,
                 "obs_otf_mode": "N", "obs_lam_vel": 0.0, "obs_bet_vel": 0.0}
    t = translator_factory(obs2tel_overrides=on_preset)
    t.last_payload["type"] = "move_to"
    t.last_payload["start_time"] = time.time() - 3600.0     # a move-to does not end
    t.tel_return_cookie = 1
    t.start_command = MagicMock()
    _set_obs2tel(t, tmp_path, 2, **OTF_LINE, **HORIZON_SOURCE, obs_track_duration=0)

    assert t.process_obs2tel() is True

    assert not t.start_command.called
    assert t.tel_return_cookie == 2


def test_otf_payload_velocities_match_positions(translator_factory):
    t = translator_factory()
    obs2tel = {**t.obs2tel, **OTF_LINE}

    payload = t._build_otf_payload(obs2tel, 1.8e9, 6.0)

    pts = np.array(payload["points"])
    az = np.unwrap(pts[:, 1], period=360.0)
    mid = slice(1, -1)
    assert pts[mid, 3] == pytest.approx(np.gradient(az, pts[:, 0])[mid], abs=1e-6)
    assert pts[mid, 4] == pytest.approx(np.gradient(pts[:, 2], pts[:, 0])[mid], abs=1e-6)


def test_otf_payload_stop_segment(translator_factory, monkeypatch):
    monkeypatch.setattr(kot, "OTF_STOP_TIME", 2.0)
    monkeypatch.setattr(kot, "OTF_SETTLE_TIME", 1.0)
    t = translator_factory()
    obs2tel = {**t.obs2tel, **OTF_LINE}
    t0 = 1.8e9

    payload = t._build_otf_payload(obs2tel, t0, 6.0)

    pts = np.array(payload["points"])
    assert payload["start_time"] == pytest.approx(t0 - kot.OTF_RAMP_TIME - 1.0)
    assert payload["start_time"] + pts[-1, 0] == pytest.approx(t0 + 8.0)


# ---------------------------------------------------------------------------
# OTFTP: send the OTF path
# ---------------------------------------------------------------------------

@pytest.fixture
def otf_translator(translator_factory, mock_ocs):
    """Translator with a running tracking path; next command: OTFTP of OTF_LINE."""
    def _make(t0_in, duration=6.0):
        t = translator_factory()
        t.last_payload["start_time"] = time.time() - 5.0     # running
        t.tel_return_cookie = 1
        t0 = time.time() + t0_in
        _use_obs2tel(t, 2, **OTF_LINE, obs_track_duration=duration, obs_start_time=t0)
        mock_ocs.scan_pattern.return_value = _ok()
        return t, t0
    return _make


def test_otftp_sends_path_and_returns_cookie(otf_translator, mock_ocs):
    t, t0 = otf_translator(16.0)
    order = MagicMock()
    order.attach_mock(mock_ocs.abort, "abort")
    order.attach_mock(mock_ocs.scan_pattern, "scan_pattern")

    t._command_loop(t._send_otf_scan)

    assert [c[0] for c in order.mock_calls] == ["abort", "scan_pattern"]
    sent = mock_ocs.scan_pattern.call_args.kwargs["data"]
    assert sent["start_time"] == pytest.approx(t0 - kot.OTF_RAMP_TIME - kot.OTF_SETTLE_TIME)
    assert "type" not in sent
    assert t.last_payload["type"] == "otf"
    assert t.tel_return_cookie == 2
    assert t.tel_error == 0
    assert t.on_track_not_before == pytest.approx(t0)
    assert t.otf_scan_window == pytest.approx((t0, t0 + 6.0))


def test_otftp_payload_saved(otf_translator, mock_ocs, tmp_path):
    t, _ = otf_translator(16.0)
    t.payload_dir = str(tmp_path / "payloads")
    (tmp_path / "payloads").mkdir()

    t._command_loop(t._send_otf_scan)

    sent = mock_ocs.scan_pattern.call_args.kwargs["data"]
    files = list((tmp_path / "payloads").glob("*.json"))
    assert [f.name for f in files] == [
        kot.PAYLOAD_FILE_FORMAT.format(start_time=sent["start_time"], kind="OTF")]
    saved = json.loads(files[0].read_text())
    assert saved["kind"] == "OTF"
    assert saved["start_time"] == sent["start_time"]
    assert saved["coordsys"] == sent["coordsys"]
    assert np.allclose(saved["points"], sent["points"], rtol=0, atol=1e-12)


def test_payload_not_saved_by_default(otf_translator, mock_ocs, tmp_path):
    t, _ = otf_translator(16.0)
    assert t.payload_dir is None

    t._command_loop(t._send_otf_scan)

    assert mock_ocs.scan_pattern.called
    assert not list(tmp_path.rglob("path_*.json"))


def test_otftp_too_late_not_sent(otf_translator, mock_ocs):
    t, _ = otf_translator(5.0)     # path start 2 s ahead < OTF_MIN_SEND_LEAD

    t._command_loop(t._send_otf_scan)

    assert not mock_ocs.scan_pattern.called
    assert not mock_ocs.abort.called            # the running path is left alone
    assert t.tel_return_cookie == 2
    assert t.tel_error == kot.TEL_ERROR_OTF_TOO_LATE
    assert t.tel_lost_track == "Y"              # flags the data of the scan
    assert t.otf_scan_window is None


def test_otftp_rejected_flags_lost_track(otf_translator, mock_ocs):
    t, _ = otf_translator(16.0)
    mock_ocs.scan_pattern.return_value = _rejected()

    t._command_loop(t._send_otf_scan)

    assert t.tel_return_cookie == 2
    assert t.tel_error == 1
    assert t.tel_lost_track == "Y"
    assert t.tel_path_failed is True


def test_otftp_exception_flags_lost_track(otf_translator):
    t, t0 = otf_translator(16.0)
    t.obs2tel = {**t.obs2tel, "obs_coord_sys_del": "NOFRAME"}   # ValueError

    t._command_loop(t._send_otf_scan)

    assert t.tel_return_cookie == 2
    assert t.tel_path_failed is True
    assert t.tel_lost_track == "Y"


# ---------------------------------------------------------------------------
# PRETP without a running path at its position: move-to the run-up point
# ---------------------------------------------------------------------------

def test_pretp_move_to_run_up_point(translator_factory, mock_ocs, monkeypatch):
    monkeypatch.setattr(kot, "OTF_SETTLE_TIME", 0.0)
    mock_ocs.move_to.return_value = _ok()
    t = translator_factory()
    t.tel_return_cookie = 1
    _use_obs2tel(t, 2, **OTF_LINE, obs_track_duration=0)
    now = time.time()

    t._command_loop(t._send_pre_otf)

    az, el = mock_ocs.move_to.call_args.args
    # run-up point = line start - vel * ramp / 2, at the expected path start
    t_path = now + kot.OTF_EXPECTED_PRE_OTF_TIME - kot.OTF_RAMP_TIME
    ref_az, ref_el = _sky_altaz(-300.0 - 100.0 * kot.OTF_RAMP_TIME / 2, 60.0, t_path)
    # the expected path start is only approximate: a second is a few arcsec here
    assert az == pytest.approx(ref_az, abs=0.01)
    assert el == pytest.approx(ref_el, abs=0.01)
    assert t.last_payload["type"] == "move_to"
    assert t.last_obs2tel["obs_otf_mode"] == "C"
    assert t.tel_return_cookie == 2
    assert t.otf_scan_window is None


# ---------------------------------------------------------------------------
# status of an OTF scan cookie
# ---------------------------------------------------------------------------

def _status_time(mock_ocs):
    status = mock_ocs.get_status.return_value
    return _get_time_from_acu_format(status["Time"], status["Year"]).unix


@pytest.mark.parametrize("since_t0,on_track", [
    pytest.param(-1.0, "N", id="before_scan"),
    pytest.param(0.5,  "Y", id="in_scan"),
    pytest.param(7.0,  "N", id="after_scan"),
])
def test_on_track_only_within_scan(translator_factory, mock_ocs, since_t0, on_track):
    t = translator_factory()    # payload static at the mock position: on the path
    t0 = _status_time(mock_ocs) - since_t0
    t.tel_return_cookie = 1
    t._return_cookie(2, t0, otf_scan_window=(t0, t0 + 6.0))

    result, _ = t.set_tel2obs_dict()

    assert result["tel_on_track"] == on_track
    assert result["tel_lost_track"] == "N"
    assert result["tel_time_del"] == pytest.approx(since_t0, abs=1e-3)


def test_lost_track_within_scan_without_settling(translator_factory, mock_ocs):
    t = translator_factory()
    status = mock_ocs.get_status.return_value
    t0 = _status_time(mock_ocs) - 0.5
    t._return_cookie(2, t0, otf_scan_window=(t0, t0 + 6.0))

    # off the path right at the start: lost track, although never on track before
    status["Azimuth current position"] = 182.0
    result, _ = t.set_tel2obs_dict()
    assert (result["tel_on_track"], result["tel_lost_track"]) == ("N", "Y")

    # stays lost when back on the path
    status["Azimuth current position"] = 180.0
    result, _ = t.set_tel2obs_dict()
    assert (result["tel_on_track"], result["tel_lost_track"]) == ("Y", "Y")


def test_off_path_outside_scan_is_not_lost(translator_factory, mock_ocs):
    t = translator_factory()
    status = mock_ocs.get_status.return_value
    t0 = _status_time(mock_ocs) + 2.0           # ramp: scan not started yet
    t._return_cookie(2, t0, otf_scan_window=(t0, t0 + 6.0))
    status["Azimuth current position"] = 182.0

    result, _ = t.set_tel2obs_dict()

    assert (result["tel_on_track"], result["tel_lost_track"]) == ("N", "N")


def test_new_tracking_cookie_clears_scan_window(translator_factory):
    t = translator_factory()
    t._return_cookie(2, 0.0, otf_scan_window=(1.0, 7.0))
    t._return_cookie(3, 0.0)
    assert t.otf_scan_window is None
