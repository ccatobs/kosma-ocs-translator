"""
Integration tests for KOSMA_translator.set_tel2obs_dict.

Adding a new scenario is a one-liner: append a Case to CASES with only the
fields that differ from the defaults in helpers.py.  The parametrised test
function handles the rest.

"""
from pathlib import Path
from unittest.mock import MagicMock
import pytest
import numpy as np
from dataclasses import dataclass, field
from astropy.coordinates import SkyCoord, AltAz
from astropy import units as u
from astropy.time import Time

from kosma_ocs_translator import _get_time_from_acu_format
from helpers import CCAT_LOCATION


# ---------------------------------------------------------------------------
# Test-case definition
# ---------------------------------------------------------------------------

@dataclass
class Case:
    """
    One scenario for set_tel2obs_dict.

    obs2tel and status are shallow-merged on top of helpers.BASE_OBS2TEL /
    helpers.DEFAULT_STATUS.  checks maps output-dict keys to expected values
    or single-argument predicates (must return True).
    """
    id: str
    obs2tel: dict = field(default_factory=dict)
    status: dict = field(default_factory=dict)
    checks: dict = field(default_factory=dict)


CASES = [
    # ---- coordinate systems ------------------------------------------------
    Case(
        id="horizon_tel_angle_focal_plane_is_zero",
        obs2tel={"obs_coord_sys_del": "HORIZON", "obs_lam_on": 180.0, "obs_bet_on": 45.0},
        status={"Azimuth commanded position": 180.0, "Elevation commanded position": 45.0},
        checks={"tel_angle_focal_plane": 0.0},
    ),
    Case(
        id="galactic_coord_sys_on_smoke",
        # Orion A in Galactic coordinates; no crash expected
        obs2tel={
            "obs_coord_sys_on": "GALACTIC",
            "obs_lam_on": 209.01,
            "obs_bet_on": -19.38,
            "obs_coord_sys_del": "GALACTIC",
            "obs_coord_sys_off": "GALACTIC",
        },
        checks={"tel_telescope": "CCAT"},
    ),
    Case(
        id="j2000_on_with_galactic_del_smoke",
        # cross-system offset: source in J2000, map offset in Galactic
        obs2tel={
            "obs_coord_sys_on": "J2000",
            "obs_coord_sys_del": "GALACTIC",
            "obs_lam_del": 100.0,
            "obs_bet_del": 0.0,
        },
        checks={"tel_telescope": "CCAT"},
    ),

    # ---- on/off track ------------------------------------------------------
    Case(
        id="on_track_exact",
        # actual == commanded
        checks={"tel_on_track": "Y"},
    ),
    Case(
        id="on_track_small_error_below_tolerance",
        # 0.1 deg in az at el=45: actdiff ≈ 254 arcsec < default tolerance 3600 arcsec
        status={
            "Azimuth current position": 180.1,
            "Elevation current position": 45.0,
        },
        checks={"tel_on_track": "Y"},
    ),
    Case(
        id="off_track_large_azimuth_error",
        # 2 deg in az at el=45: actdiff ≈ 5091 arcsec > 3600 arcsec
        status={
            "Azimuth current position": 182.0,
            "Elevation current position": 45.0,
        },
        checks={"tel_on_track": "N"},
    ),
    Case(
        id="off_track_sets_tel_lost_track",
        status={
            "Azimuth current position": 182.0,
            "Elevation current position": 45.0,
        },
        checks={"tel_lost_track": "Y"},
    ),
    Case(
        id="tight_tolerance_small_error_off_track",
        # tolerance 10 arcsec; 0.01 deg az at el=45 ≈ 25 arcsec → off-track
        obs2tel={"obs_tolerance": 10},
        status={
            "Azimuth current position": 180.01,
            "Elevation current position": 45.0,
        },
        checks={"tel_on_track": "N"},
    ),

    # ---- commanded position interpolation ----------------------------------
    Case(
        id="cmd_position_interpolated_from_static_payload",
        # payload is static at az=180, el=45; interpolated cmd must match
        checks={
            "tel_azm_cmd": lambda v: abs(v - 180.0) < 0.01,
            "tel_elv_cmd": lambda v: abs(v - 45.0) < 0.01,
        },
    ),
    Case(
        id="actual_position_is_cmd_plus_acu_error",
        status={
            "Azimuth current position": 180.5,
            "Azimuth commanded position": 180.0,
            "Elevation current position": 45.2,
            "Elevation commanded position": 45.0,
        },
        checks={
            "tel_azm_act": lambda v: abs(v - (180.0 + 0.5)) < 0.01,
            "tel_elv_act": lambda v: abs(v - (45.0 + 0.2)) < 0.01,
        },
    ),

    # ---- fixed / constant output values ------------------------------------
    Case(
        id="telescope_identifier_is_CCAT",
        checks={"tel_telescope": "CCAT"},
    ),
    Case(
        id="tel_pos_in_range_always_Y",
        checks={"tel_pos_in_range": "Y"},
    ),
    Case(
        id="tel_supports_ephemeris_is_N",
        checks={"tel_supports_ephemeris": "N"},
    ),

    # ---- focal plane offsets -----------------------------------------------
    Case(
        id="nonzero_focal_plane_offset_no_crash",
        obs2tel={"obs_x_focal_plane": 30.0, "obs_y_focal_plane": -20.0},
        checks={"tel_telescope": "CCAT"},
    ),

    # ---- zero vs nonzero offsets -------------------------------------------
    Case(
        id="zero_del_and_off_offsets",
        obs2tel={"obs_lam_del": 0.0, "obs_bet_del": 0.0,
                 "obs_lam_off": 0.0, "obs_bet_off": 0.0},
        checks={"tel_on_track": "Y"},
    ),
    Case(
        id="nonzero_del_offset_in_j2000",
        obs2tel={"obs_lam_del": 3600.0, "obs_bet_del": 0.0,
                 "obs_coord_sys_del": "J2000"},
        checks={"tel_telescope": "CCAT"},  # smoke: no crash
    ),
    Case(
        id="nonzero_off_offset_in_j2000",
        obs2tel={"obs_lam_off": 0.0, "obs_bet_off": 3600.0,
                 "obs_coord_sys_off": "J2000"},
        checks={"tel_telescope": "CCAT"},
    ),
    Case(
        id="horizon_del_offset_nonzero",
        obs2tel={"obs_coord_sys_on": "HORIZON", "obs_lam_on": 180.0, "obs_bet_on": 45.0,
                 "obs_lam_del": 600.0, "obs_bet_del": 0.0,
                 "obs_coord_sys_del": "HORIZON"},
        status={"Azimuth commanded position": 180.0, "Elevation commanded position": 45.0},
        checks={"tel_angle_focal_plane": 0.0},
    ),
]


# ---------------------------------------------------------------------------
# Parametrised test
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("case", CASES, ids=[c.id for c in CASES])
def test_set_tel2obs_dict(translator_factory, case):
    t = translator_factory(obs2tel_overrides=case.obs2tel, status_overrides=case.status)
    result, _ = t.set_tel2obs_dict()

    for key, exp in case.checks.items():
        val = result[key]
        if callable(exp):
            assert exp(val), f"{key}={val!r} failed predicate"
        else:
            assert val == exp, f"{key}: expected {exp!r}, got {val!r}"


# ---------------------------------------------------------------------------
# Dedicated test: tel_angle_focal_plane vs. reference for J2000 source
# ---------------------------------------------------------------------------

def test_tel_angle_focal_plane_j2000_matches_reference(translator_factory, mock_ocs):
    """
    tel_angle_focal_plane for a J2000 source should agree with an independent
    reference calculation (same geometry, implemented directly in astropy)
    to within 0.1 degrees.

    """
    az_cmd, el_cmd = 180.0, 45.0

    t = translator_factory()  # default: J2000, Orion A
    result, _ = t.set_tel2obs_dict()
    computed = result["tel_angle_focal_plane"]

    # Reference: same geometry as the code, built directly with astropy
    status = mock_ocs.get_status.return_value
    obstime = _get_time_from_acu_format(status["Time"], status["Year"])
    altaz_frame = AltAz(obstime=obstime, location=CCAT_LOCATION)

    pos_altaz = SkyCoord(az=az_cmd * u.deg, alt=el_cmd * u.deg, frame=altaz_frame)
    # Transform the commanded position to ICRS, then offset slightly north
    pos_icrs = pos_altaz.transform_to("icrs")
    epsilon = 1e-3  # arcsec
    north_icrs = SkyCoord(
        ra=pos_icrs.ra, dec=pos_icrs.dec + epsilon * u.arcsec, frame="icrs"
    )
    north_altaz = north_icrs.transform_to(altaz_frame)

    dalt = north_altaz.alt.deg - pos_altaz.alt.deg
    daz = (north_altaz.az.deg - pos_altaz.az.deg) * np.cos(pos_altaz.alt.rad)
    ref_angle = np.rad2deg(np.arctan2(dalt, daz)) - 90.0
    if ref_angle < -180.0:
        ref_angle += 360.0
    ref_angle *= -1  # KOSMA convention: ccw from the reference system's second axis

    assert abs(computed - ref_angle) < 0.1, (
        f"tel_angle_focal_plane={computed:.3f} deg, "
        f"reference={ref_angle:.3f} deg, diff={abs(computed - ref_angle):.3f} deg"
    )


# ---------------------------------------------------------------------------
# Dedicated test: tel_angle_focal_plane equals parallactic angle for J2000
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("az_cmd,el_cmd", [
    pytest.param(180.0, 45.0, id="south_midsky"),
    pytest.param( 90.0, 30.0, id="east_low"),    # rising:  HA < 0 → q < 0
    pytest.param(270.0, 30.0, id="west_low"),    # setting: HA > 0 → q > 0
    pytest.param(  0.0, 60.0, id="north_high"),  # north of zenith at CCAT lat=-23°
    pytest.param(180.0, 20.0, id="south_low"),
])
def test_tel_angle_focal_plane_equals_parallactic_angle_j2000(
    translator_factory, mock_ocs, az_cmd, el_cmd
):
    """
    For J2000 coord_sys_del, tel_angle_focal_plane equals the parallactic angle
    (angle between the zenith direction and celestial north at each commanded
    az/el position), verifying magnitude and sign. Here "north" is the ICRS
    pole, which is what J2000 means; it differs from the north pole of date
    (the textbook parallactic angle) by up to ~0.4 deg at the current epoch.

    The reference is the spherical bearing from the commanded az/el to the
    ICRS pole, so the comparison is independent of obs_lam_on/obs_bet_on.
    """
    t = translator_factory(
        obs2tel_overrides={"obs_coord_sys_del": "J2000"},
        status_overrides={
            "Azimuth commanded position": az_cmd,
            "Elevation commanded position": el_cmd,
            "Azimuth current position": az_cmd,
            "Elevation current position": el_cmd,
        },
    )
    result, _ = t.set_tel2obs_dict()
    computed = result["tel_angle_focal_plane"]

    status = mock_ocs.get_status.return_value
    obstime = _get_time_from_acu_format(status["Time"], status["Year"])
    altaz_frame = AltAz(obstime=obstime, location=CCAT_LOCATION)

    # Direction from the commanded position to the ICRS north pole, as a
    # spherical bearing in (az, alt) measured from the zenith direction towards
    # increasing az. The ICRS pole is used (not the pole of date), because
    # J2000 axes differ from the equinox of date by ~0.3-0.4 deg now.
    pole = SkyCoord(ra=0 * u.deg, dec=90 * u.deg, frame="icrs").transform_to(altaz_frame)
    az, alt = np.deg2rad(az_cmd), np.deg2rad(el_cmd)
    daz, alt_p = pole.az.rad - az, pole.alt.rad

    para = np.rad2deg(np.arctan2(
        np.sin(daz) * np.cos(alt_p),
        np.cos(alt) * np.sin(alt_p) - np.sin(alt) * np.cos(alt_p) * np.cos(daz),
    ))

    diff = (computed - para + 180.0) % 360.0 - 180.0
    assert abs(diff) < 0.1, (
        f"az={az_cmd}, el={el_cmd}: "
        f"tel_angle_focal_plane={computed:.3f} deg, "
        f"parallactic_angle={para:.3f} deg, diff={abs(diff):.3f} deg"
    )


# ---------------------------------------------------------------------------
# Dedicated test: tel_vel_azm / tel_vel_elv converted from deg/s to arcsec/s
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("vel_azi,vel_ele", [
    pytest.param( 0.0,   0.0,   id="stationary"),
    pytest.param( 0.01,  0.0,   id="azimuth_only"),
    pytest.param( 0.0,  -0.005, id="elevation_only_negative"),
    pytest.param(-0.25,  0.1,   id="both"),
])
def test_tel_vel_converted_to_arcsec_per_sec(translator_factory, vel_azi, vel_ele):
    t = translator_factory(
        status_overrides={
            "Azimuth current velocity": vel_azi,
            "Elevation current velocity": vel_ele,
        },
    )
    result, _ = t.set_tel2obs_dict()

    assert result["tel_vel_azm"] == pytest.approx(vel_azi * 3600.0)
    assert result["tel_vel_elv"] == pytest.approx(vel_ele * 3600.0)


# ---------------------------------------------------------------------------
# Dedicated test: cookie of an accepted path is returned only after it started
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("t_rel,returned", [
    pytest.param(-10.0, False, id="before_start"),
    pytest.param(  0.5, False, id="within_margin"),
    pytest.param( 30.0, True,  id="after_start"),
])
def test_pending_cookie_returned_after_path_start(translator_factory, mock_ocs, t_rel, returned):
    t = translator_factory()
    status = mock_ocs.get_status.return_value
    status_time = _get_time_from_acu_format(status["Time"], status["Year"])
    t.last_payload["start_time"] = status_time.unix - t_rel
    t.tel_return_cookie = 1
    t.pending_tel_return_cookie = 2
    t.tel_lost_track = "Y"

    result, _ = t.set_tel2obs_dict()

    if returned:
        assert result["tel_return_cookie"] == 2
        assert t.pending_tel_return_cookie is None
        assert t.old_tel_return_cookie == 1
        assert result["tel_lost_track"] == "N"
    else:
        assert result["tel_return_cookie"] == 1
        assert t.pending_tel_return_cookie == 2


def test_failed_path_returns_cookie_out_of_range(translator_factory):
    t = translator_factory()
    t.tel_return_cookie = 1
    t._fail_path(2, 503)

    result, _ = t.set_tel2obs_dict()

    assert result["tel_return_cookie"] == 2
    assert result["tel_error"] == 503
    assert result["tel_pos_in_range"] == "N"
    # actual == commanded in the mock status, but the failed path is not tracked
    assert result["tel_on_track"] == "N"


def test_started_path_clears_previous_failure(translator_factory):
    t = translator_factory()  # status time is 30 s after the payload start
    t._fail_path(2, 1)
    t.pending_tel_return_cookie = 3

    result, _ = t.set_tel2obs_dict()

    assert result["tel_return_cookie"] == 3
    assert result["tel_error"] == 0
    assert result["tel_pos_in_range"] == "Y"
    assert result["tel_on_track"] == "Y"


def test_superseded_failure_not_reported(translator_factory):
    t = translator_factory()
    t.tel_return_cookie = 1
    t._track_stop_event.set()
    t._fail_path(2, 1)

    assert t.tel_return_cookie == 1
    assert t.tel_path_failed is False


def test_first_path_failed_writes_status(translator_factory, mock_ocs):
    t = translator_factory()
    t.last_payload = None
    t.last_obs2tel = None
    t._fail_path(2, 1)

    result, _ = t.set_tel2obs_dict()

    status = mock_ocs.get_status.return_value
    assert result["tel_return_cookie"] == 2
    assert result["tel_pos_in_range"] == "N"
    assert result["tel_on_track"] == "N"
    assert result["tel_azm_cmd"] == status["Azimuth commanded position"]
    assert result["tel_angle_focal_plane"] == 0.0


@pytest.mark.parametrize("last_start,aborted", [
    pytest.param(None,     False, id="no_previous_path"),
    pytest.param(-10000.0, False, id="previous_path_ended"),
    pytest.param(-10.0,    True,  id="previous_path_running"),
    pytest.param(5.0,      True,  id="previous_path_not_started"),
])
def test_abort_only_if_previous_path_not_ended(translator_factory, mock_ocs, last_start, aborted):
    import time
    t = translator_factory()   # payload is 600 s long
    if last_start is None:
        t.last_payload = None
    else:
        t.last_payload["start_time"] = time.time() + last_start

    t._track_loop()

    assert mock_ocs.abort.called is aborted


def test_no_abort_after_move_to(translator_factory, mock_ocs):
    import time
    t = translator_factory()
    t.last_payload["type"] = "move_to"
    t.last_payload["start_time"] = time.time()

    t._track_loop()

    assert not mock_ocs.abort.called


HORIZON_OBS2TEL = {
    "obs_coord_sys_on": "HORIZON", "obs_lam_on": 170.0, "obs_bet_on": 50.0,
    "obs_coord_sys_del": "HORIZON", "obs_coord_sys_off": "HORIZON",
}


@pytest.mark.parametrize("status_code,body,accepted", [
    pytest.param(200, {"status": "ok"},    True,  id="accepted"),
    pytest.param(503, {"status": "busy"},  False, id="rejected"),
])
def test_move_to_response_check(translator_factory, mock_ocs, status_code, body, accepted):
    mock_ocs.move_to.return_value = MagicMock(status_code=status_code, json=lambda: body)
    t = translator_factory(obs2tel_overrides=HORIZON_OBS2TEL)
    t.tel_return_cookie = 1
    t.next_tel_return_cookie = 2

    t._track_loop()

    assert mock_ocs.move_to.called
    if accepted:
        assert t.last_payload["type"] == "move_to"
        assert t.pending_tel_return_cookie == 2
        assert t.tel_return_cookie == 1
        assert t.tel_path_failed is False
    else:
        assert t.last_payload["type"] == "path"   # previous one kept
        assert t.pending_tel_return_cookie is None
        assert t.tel_return_cookie == 2
        assert t.tel_path_failed is True
        assert t.tel_error == 503


def _new_obs2tel(t, cookie, **changes):
    """Simulate reading a new obs2tel: same as the last one except for changes."""
    t.obs2tel = {**t.last_obs2tel, "obs_cookie": cookie, **changes}
    t.next_tel_return_cookie = cookie


@pytest.mark.parametrize("changes,start_offset,pending,reused", [
    pytest.param({},                         -30.0,   False, True,  id="same_target_running"),
    pytest.param({},                         -30.0,   True,  True,  id="same_target_pending"),
    pytest.param({"obs_lam_del": 30.0},      -30.0,   False, False, id="different_offset"),
    pytest.param({},                         -350.0,  False, False, id="path_ends_too_early"),
    pytest.param({},                         -250.0,  False, True,  id="path_long_enough"),
])
def test_reuse_last_track(translator_factory, changes, start_offset, pending, reused):
    import time
    t = translator_factory()   # payload is 600 s long, reuse needs >= 300 s left
    t.last_payload["start_time"] = time.time() + start_offset
    t.tel_return_cookie = 1
    if pending:
        t.pending_tel_return_cookie = 2
    t.tel_lost_track = "Y"
    _new_obs2tel(t, 3, **changes)

    assert t.reuse_last_track() is reused

    if not reused:
        assert t.tel_return_cookie == 1
    elif pending:
        assert t.pending_tel_return_cookie == 3
        assert t.tel_return_cookie == 1
    else:
        assert t.tel_return_cookie == 3
        assert t.old_tel_return_cookie == 1
        assert t.tel_lost_track == "N"


def test_reuse_last_track_not_after_failure(translator_factory):
    import time
    t = translator_factory()
    t.last_payload["start_time"] = time.time() - 30
    t._fail_path(2, 1)
    _new_obs2tel(t, 3)

    assert t.reuse_last_track() is False


def test_reuse_last_move_to_without_end(translator_factory):
    t = translator_factory()
    t.last_payload["type"] = "move_to"
    t.last_payload["start_time"] = 0.0   # long ago; a move-to holds its position
    _new_obs2tel(t, 3)

    assert t.reuse_last_track() is True
    assert t.tel_return_cookie == 3


def test_unexpected_exception_in_track_fails_path(translator_factory):
    t = translator_factory(obs2tel_overrides={"obs_coord_sys_on": "NOT_A_FRAME"})
    t.tel_return_cookie = 1
    t.next_tel_return_cookie = 2

    t._track_loop()   # ValueError for the unknown frame must not escape

    assert t.tel_return_cookie == 2
    assert t.tel_path_failed is True


# ---------------------------------------------------------------------------
# Output file format inspection
# ---------------------------------------------------------------------------

def test_write_tel2obs_file_produces_output(translator_factory, tmp_path):
    """
    Write KOSMA_tel2obs.set and KOSMA_ccdpos.sim to tests/output/ for format inspection.
    Files are also written under pytest's tmp_path (the translator's working directory).
    """
    output_dir = Path(__file__).parent / "output"
    output_dir.mkdir(exist_ok=True)

    t = translator_factory()
    t.write_tel2obs_file()

    tel2obs_path = tmp_path / "KOSMA_tel2obs.set"
    ccdpos_path = tmp_path / "KOSMA_ccdpos.sim"

    assert tel2obs_path.exists(), "KOSMA_tel2obs.set was not written"
    assert ccdpos_path.exists(), "KOSMA_ccdpos.sim was not written"

    (output_dir / "KOSMA_tel2obs.set").write_text(tel2obs_path.read_text())
    (output_dir / "KOSMA_ccdpos.sim").write_text(ccdpos_path.read_text())
