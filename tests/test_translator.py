"""
Integration tests for KOSMA_translator.set_tel2obs_dict.

Adding a new scenario is a one-liner: append a Case to CASES with only the
fields that differ from the defaults in helpers.py.  The parametrised test
function handles the rest.

"""
import logging
import time
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
        # never on track for this cookie yet: settling, not lost
        # (lost after being on track: test_lost_track_only_after_on_track)
        id="off_track_before_on_track_not_lost",
        status={
            "Azimuth current position": 182.0,
            "Elevation current position": 45.0,
        },
        checks={"tel_lost_track": "N"},
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

@pytest.mark.parametrize("az_nom,el_nom", [
    pytest.param(180.0, 45.0, id="south_midsky"),   # over-the-top az 0
    pytest.param( 30.0, 49.0, id="northeast"),      # over-the-top az 210, el 131
    pytest.param(300.0, 80.0, id="near_zenith"),
])
def test_tel_angle_focal_plane_extended_range(translator_factory, az_nom, el_nom):
    """
    In the extended range, tel_*_cmd is the over-the-top position (el > 90).
    tel_angle_focal_plane is computed at the nominal sky position, as astra does.
    """
    def angle(az, el):
        t = translator_factory(
            obs2tel_overrides={"obs_coord_sys_del": "J2000"},
            status_overrides={
                "Azimuth commanded position": az,
                "Elevation commanded position": el,
                "Azimuth current position": az,
                "Elevation current position": el,
            },
        )
        result, _ = t.set_tel2obs_dict()
        return result["tel_angle_focal_plane"]

    nominal = angle(az_nom, el_nom)
    overtop = angle((az_nom + 180.0) % 360.0, 180.0 - el_nom)
    assert overtop == pytest.approx(nominal, abs=1e-6)


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
# Dedicated test: site coordinates in KOSMA convention (longitude +west)
# ---------------------------------------------------------------------------

def test_tel_site_kosma_convention(translator_factory):
    t = translator_factory()
    result, _ = t.set_tel2obs_dict()

    # CCAT is west of Greenwich: KOSMA (+west) longitude is positive
    assert result["tel_longitude"] == pytest.approx(-CCAT_LOCATION.lon.deg)
    assert result["tel_longitude"] > 0
    assert result["tel_latitude"] == pytest.approx(CCAT_LOCATION.lat.deg)
    assert result["tel_altitude"] == pytest.approx(CCAT_LOCATION.height.to(u.m).value)


# ---------------------------------------------------------------------------
# Dedicated test: cookie of an accepted path is returned at once, but
# tel_on_track stays "N" until the path has started
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("t_rel,on_track", [
    pytest.param(-10.0, "N", id="before_start"),
    pytest.param( -0.5, "N", id="just_before_start"),
    pytest.param(  0.5, "Y", id="just_after_start"),
    pytest.param( 30.0, "Y", id="after_start"),
])
def test_on_track_held_until_path_start(translator_factory, mock_ocs, t_rel, on_track):
    t = translator_factory()   # actual == commanded: on track once started
    status = mock_ocs.get_status.return_value
    status_time = _get_time_from_acu_format(status["Time"], status["Year"])
    t.last_payload["start_time"] = status_time.unix - t_rel
    t.tel_return_cookie = 1
    t.tel_lost_track = "Y"
    t._return_cookie(2, t.last_payload["start_time"])

    result, _ = t.set_tel2obs_dict()

    assert result["tel_return_cookie"] == 2
    assert t.old_tel_return_cookie == 1
    assert result["tel_on_track"] == on_track
    # not started yet is not lost track
    assert result["tel_lost_track"] == "N"


def test_lost_track_only_after_on_track(translator_factory, mock_ocs):
    t = translator_factory()   # path started 30 s ago, static at az=180, el=45
    status = mock_ocs.get_status.return_value
    t.tel_return_cookie = 1
    t._return_cookie(2, t.last_payload["start_time"])

    # settling: off track, but never on track yet -> not lost
    status["Azimuth current position"] = 182.0
    result, _ = t.set_tel2obs_dict()
    assert (result["tel_on_track"], result["tel_lost_track"]) == ("N", "N")

    # arrived
    status["Azimuth current position"] = 180.0
    result, _ = t.set_tel2obs_dict()
    assert (result["tel_on_track"], result["tel_lost_track"]) == ("Y", "N")

    # left the track: lost, and it stays lost when back on track
    status["Azimuth current position"] = 182.0
    result, _ = t.set_tel2obs_dict()
    assert (result["tel_on_track"], result["tel_lost_track"]) == ("N", "Y")
    status["Azimuth current position"] = 180.0
    result, _ = t.set_tel2obs_dict()
    assert (result["tel_on_track"], result["tel_lost_track"]) == ("Y", "Y")

    # a new cookie starts settling afresh
    t._return_cookie(3, t.last_payload["start_time"])
    status["Azimuth current position"] = 182.0
    result, _ = t.set_tel2obs_dict()
    assert (result["tel_on_track"], result["tel_lost_track"]) == ("N", "N")


def test_on_track_held_while_slewing(translator_factory):
    t = translator_factory()
    t.tel_return_cookie = 1
    t._return_cookie(2, np.inf)   # slew accepted, path not sent yet

    result, _ = t.set_tel2obs_dict()

    assert result["tel_return_cookie"] == 2
    assert result["tel_on_track"] == "N"
    assert result["tel_lost_track"] == "N"


def test_failure_after_returned_cookie_sets_error(translator_factory):
    t = translator_factory()
    t.tel_return_cookie = 1
    t._return_cookie(2, np.inf)   # slew accepted
    t._fail_path(2, 503)          # path rejected after the slew

    result, _ = t.set_tel2obs_dict()

    assert result["tel_return_cookie"] == 2
    assert t.old_tel_return_cookie == 1
    assert result["tel_error"] == 503
    assert result["tel_on_track"] == "N"


def test_failed_path_returns_cookie_with_error(translator_factory):
    t = translator_factory()
    t.tel_return_cookie = 1
    t._fail_path(2, 503)

    result, _ = t.set_tel2obs_dict()

    assert result["tel_return_cookie"] == 2
    assert result["tel_error"] == 503
    assert result["tel_pos_in_range"] == "Y"
    # actual == commanded in the mock status, but the failed path is not tracked
    assert result["tel_on_track"] == "N"


def test_started_path_clears_previous_failure(translator_factory):
    t = translator_factory()  # status time is 30 s after the payload start
    t._fail_path(2, 1)
    t._return_cookie(3, t.last_payload["start_time"])

    result, _ = t.set_tel2obs_dict()

    assert result["tel_return_cookie"] == 3
    assert result["tel_error"] == 0
    assert result["tel_pos_in_range"] == "Y"
    assert result["tel_on_track"] == "Y"


def test_superseded_failure_not_reported(translator_factory):
    t = translator_factory()
    t.tel_return_cookie = 1
    t._command_stop_event.set()
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
    assert result["tel_error"] == 1
    assert result["tel_on_track"] == "N"
    # no payload: cmd is the current position minus the (zero) pointing correction
    assert result["tel_azm_cmd"] == status["Azimuth current position"]
    assert result["tel_angle_focal_plane"] == 0.0


# ---------------------------------------------------------------------------
# Status while slewing to the start of a path (slew_target)
# ---------------------------------------------------------------------------

def _status_unix(mock_ocs):
    status = mock_ocs.get_status.return_value
    return _get_time_from_acu_format(status["Time"], status["Year"]).unix


def test_no_status_before_any_cookie(translator_factory):
    # start-up: no command recorded, no cookie returned yet
    t = translator_factory()
    t.last_payload = None
    t.last_obs2tel = None

    assert t.set_tel2obs_dict() == (None, None)


def test_first_slew_writes_status(translator_factory, mock_ocs):
    # slew before the first path: no command recorded in last_payload yet
    t = translator_factory()
    t.last_payload = None
    t.last_obs2tel = None
    t.tel_return_cookie = 1
    t._return_cookie(2, np.inf, slew_target=(200.0, 30.0, _status_unix(mock_ocs) - 5.0))

    result, _ = t.set_tel2obs_dict()

    assert result["tel_return_cookie"] == 2
    assert result["tel_error"] == 0
    assert result["tel_on_track"] == "N"
    assert result["tel_lost_track"] == "N"
    assert (result["tel_azm_cmd"], result["tel_elv_cmd"]) == (200.0, 30.0)
    assert result["tel_time_del"] == pytest.approx(5.0, abs=1e-3)
    assert result["tel_angle_focal_plane"] == 0.0


def test_slew_target_replaces_previous_path(translator_factory, mock_ocs):
    # the previous path (static at az=180, el=45) and its HORIZON map offset
    # must not be reported for the new cookie while slewing
    t = translator_factory(obs2tel_overrides={
        "obs_wavelength": 0.6, "obs_coord_sys_del": "HORIZON",
        "obs_lam_del": 60.0, "obs_bet_del": 30.0})
    t._return_cookie(2, np.inf, slew_target=(200.0, 30.0, _status_unix(mock_ocs) - 5.0))

    result, ccd = t.set_tel2obs_dict()

    assert (result["tel_azm_cmd"], result["tel_elv_cmd"]) == (200.0, 30.0)
    assert result["tel_time_del"] == pytest.approx(5.0, abs=1e-3)
    assert result["tel_angle_focal_plane"] == 0.0
    assert ccd["op_azm"] == pytest.approx(np.deg2rad(200.0))
    assert ccd["op_elv"] == pytest.approx(np.deg2rad(30.0))
    assert ccd["op_vel_elv"] == 0.0
    assert (ccd["op_oazm"], ccd["op_oelv"]) == (0, 0)


def test_accepted_path_clears_slew_target(translator_factory):
    t = translator_factory()   # status time is 30 s after the payload start
    t._return_cookie(2, np.inf, slew_target=(200.0, 30.0, 0.0))
    t._return_cookie(2, t.last_payload["start_time"])   # path after the slew

    result, _ = t.set_tel2obs_dict()

    assert t.slew_target is None
    assert (result["tel_azm_cmd"], result["tel_elv_cmd"]) == (180.0, 45.0)
    assert result["tel_time_del"] == pytest.approx(30.0, abs=1e-3)
    assert result["tel_on_track"] == "Y"


def test_failure_clears_slew_target(translator_factory):
    t = translator_factory()
    t._return_cookie(2, np.inf, slew_target=(200.0, 30.0, 0.0))
    t._fail_path(2, 1)   # e.g. slew timed out

    assert t.slew_target is None
    assert t.cookie_reported is True


def test_superseded_slew_keeps_previous_slew_target(translator_factory):
    t = translator_factory()
    t._return_cookie(2, np.inf, slew_target=(200.0, 30.0, 0.0))
    t._command_stop_event.set()
    t._return_cookie(3, np.inf, slew_target=(210.0, 35.0, 1.0))

    # cookie 3 not returned: the status of cookie 2 is unchanged
    assert t.tel_return_cookie == 2
    assert t.slew_target == (200.0, 30.0, 0.0)


@pytest.mark.parametrize("current,correction,on_track,act_minus_cmd", [
    # payload is static at az=180, el=45; correction is SPEM + tiltmeters [deg]
    pytest.param((180.0, 45.0), (0.0, 0.0),  "Y", (0.0, 0.0),   id="no_correction"),
    pytest.param((182.0, 46.5), (2.0, 1.5),  "Y", (0.0, 0.0),   id="current_follows_correction"),
    pytest.param((180.0, 45.0), (2.0, 1.5),  "N", (-2.0, -1.5), id="correction_not_applied"),
])
def test_on_track_uses_payload_plus_pointing_correction(
    translator_factory, mock_ocs, current, correction, on_track, act_minus_cmd
):
    """
    tel_on_track compares the current position with the payload position plus
    the total pointing correction; the ACU commanded position is ignored.
    """
    pointing = {
        "SPEM correction AZ": correction[0] / 2,
        "Tiltmeter Az correction AZ": correction[0] / 4,
        "Tiltmeter El correction AZ": correction[0] / 4,
        "SPEM correction EL": correction[1] / 2,
        "Tiltmeter Az correction EL": correction[1] / 4,
        "Tiltmeter El correction EL": correction[1] / 4,
    }
    mock_ocs.get_dataset.side_effect = lambda name: (
        pointing if name == "CmdPointingCorrection" else {}
    )
    t = translator_factory(obs2tel_overrides={"obs_wavelength": 0.6}, status_overrides={
        "Azimuth current position": current[0],
        "Elevation current position": current[1],
    })
    # ACU commanded position deliberately inconsistent: must not be used
    mock_ocs.get_status.return_value["Azimuth commanded position"] = 0.0
    mock_ocs.get_status.return_value["Elevation commanded position"] = 0.0

    result, ccd = t.set_tel2obs_dict()

    assert result["tel_on_track"] == on_track
    assert result["tel_azm_cmd"] == pytest.approx(180.0)
    assert result["tel_elv_cmd"] == pytest.approx(45.0)
    assert result["tel_azm_act"] - result["tel_azm_cmd"] == pytest.approx(act_minus_cmd[0])
    assert result["tel_elv_act"] - result["tel_elv_cmd"] == pytest.approx(act_minus_cmd[1])
    # azimuth: ACU axis difference -> on sky, at the commanded elevation of 45 deg
    cos_elv = np.cos(np.deg2rad(45.0))
    assert ccd["del_tel_tot_azm"] == pytest.approx(np.deg2rad(correction[0]) * cos_elv)
    assert ccd["del_tel_tot_elv"] == pytest.approx(np.deg2rad(correction[1]))
    # repeatable part: SPEM only, which is half of the correction here
    assert ccd["del_tel_rep_azm"] == pytest.approx(np.deg2rad(correction[0] / 2) * cos_elv)
    assert ccd["del_tel_rep_elv"] == pytest.approx(np.deg2rad(correction[1] / 2))


@pytest.mark.parametrize("coord_sys_on,coord_sys_del,expected_deg", [
    pytest.param("J2000",   "HORIZON", (1.0, -0.5), id="horizon_offset_on_j2000_source"),
    pytest.param("HORIZON", "HORIZON", (1.0, -0.5), id="horizon_offset_on_horizon_source"),
    pytest.param("HORIZON", "J2000",   (0.0, 0.0),  id="j2000_offset_not_reported"),
])
def test_ccdpos_offsets(translator_factory, coord_sys_on, coord_sys_del, expected_deg):
    t = translator_factory(obs2tel_overrides={
        "obs_coord_sys_on": coord_sys_on, "obs_lam_on": 170.0, "obs_bet_on": 50.0,
        "obs_coord_sys_del": coord_sys_del,
        "obs_lam_del": 3600.0, "obs_bet_del": -1800.0,   # arcsec
        "obs_wavelength": 0.6,
    })

    _, ccd = t.set_tel2obs_dict()

    assert ccd["op_oazm"] == pytest.approx(np.deg2rad(expected_deg[0]))
    assert ccd["op_oelv"] == pytest.approx(np.deg2rad(expected_deg[1]))


@pytest.mark.parametrize("last_start,abort_if_unknown,aborted", [
    pytest.param(None,     True,  True,  id="no_previous_path"),
    pytest.param(None,     False, False, id="no_previous_path_no_abort_if_unknown"),
    pytest.param(-10000.0, True,  False, id="previous_path_ended"),
    pytest.param(-10.0,    True,  True,  id="previous_path_running"),
    pytest.param(5.0,      True,  True,  id="previous_path_not_started"),
])
def test_abort_only_if_previous_path_not_ended(
    translator_factory, mock_ocs, last_start, abort_if_unknown, aborted
):
    import time
    t = translator_factory()   # payload is 600 s long
    t.abort_if_unknown = abort_if_unknown
    if last_start is None:
        t.last_payload = None
    else:
        t.last_payload["start_time"] = time.time() + last_start

    t._command_loop()

    assert mock_ocs.abort.called is aborted


def test_abort_after_move_to(translator_factory, mock_ocs, monkeypatch):
    """
    A path after a move-to is preceded by an abort, even long after the
    move-to: it may still be moving, and TCS would reject the path as busy.
    """
    import kosma_ocs_translator as kot
    monkeypatch.setattr(kot, "SLEW_BEFORE_PATH_MIN_DIST", 1e6)   # no slew
    mock_ocs.scan_pattern.return_value = _ok()
    t = translator_factory()
    t.last_payload["type"] = "move_to"
    t.last_payload["start_time"] = time.time() - 10000.0
    order = MagicMock()
    order.attach_mock(mock_ocs.abort, "abort")
    order.attach_mock(mock_ocs.scan_pattern, "scan_pattern")

    t._command_loop()

    assert mock_ocs.abort.call_count == 1
    assert not mock_ocs.move_to.called
    assert [c[0] for c in order.mock_calls] == ["abort", "scan_pattern"]
    assert t.last_aborted is False     # the new path is running


HORIZON_OBS2TEL = {
    "obs_coord_sys_on": "HORIZON", "obs_lam_on": 170.0, "obs_bet_on": 50.0,
    "obs_coord_sys_del": "HORIZON", "obs_coord_sys_off": "HORIZON",
}


@pytest.mark.parametrize("status_code,body,accepted", [
    pytest.param(200, {"status": "ok"},    True,  id="accepted"),
    pytest.param(503, {"status": "busy"},  False, id="rejected"),
])
def test_move_to_response_check(translator_factory, mock_ocs, monkeypatch,
                                status_code, body, accepted):
    import kosma_ocs_translator as kot
    monkeypatch.setattr(kot, "MOVE_TO_BUSY_TIMEOUT", 0.0)
    mock_ocs.move_to.return_value = MagicMock(status_code=status_code, json=lambda: body)
    t = translator_factory(obs2tel_overrides=HORIZON_OBS2TEL)
    t.tel_return_cookie = 1
    t.next_tel_return_cookie = 2

    t._command_loop()

    assert mock_ocs.move_to.called
    if accepted:
        assert t.last_payload["type"] == "move_to"
        assert t.tel_return_cookie == 2
        assert t.on_track_not_before == pytest.approx(t.last_payload["start_time"])
        assert t.tel_path_failed is False
    else:
        assert t.last_payload["type"] == "path"   # previous one kept
        assert t.tel_return_cookie == 2
        assert t.tel_path_failed is True
        assert t.tel_error == 503


@pytest.mark.parametrize("last,abort_if_unknown,aborted", [
    pytest.param("path_running", True,  True,  id="path_running"),
    pytest.param("path_ended",   True,  False, id="path_ended"),
    pytest.param("move_to",      True,  True,  id="move_to_may_still_move"),
    pytest.param("aborted",      True,  False, id="already_aborted"),
    pytest.param(None,           True,  True,  id="unknown"),
    pytest.param(None,           False, False, id="unknown_no_abort_if_unknown"),
])
def test_horizon_move_to_aborts_previous(
    translator_factory, mock_ocs, last, abort_if_unknown, aborted
):
    """ocs.move_to does not abort by itself: the translator does it when needed."""
    import time
    mock_ocs.move_to.return_value = _ok()
    t = translator_factory(obs2tel_overrides=HORIZON_OBS2TEL)
    t.abort_if_unknown = abort_if_unknown
    # the fixture path starts at the mock ACU status time (in the past)
    if t.last_payload is not None:
        t.last_payload["start_time"] = time.time() - 10.0
    if last is None:
        t.last_payload = None
    elif last == "path_ended":
        t.last_payload["start_time"] = time.time() - 10000.0
    elif last == "move_to":
        t.last_payload["type"] = "move_to"
    elif last == "aborted":
        t.last_aborted = True
    order = MagicMock()
    order.attach_mock(mock_ocs.abort, "abort")
    order.attach_mock(mock_ocs.move_to, "move_to")

    t._command_loop()

    assert mock_ocs.abort.called is aborted
    assert mock_ocs.move_to.call_count == 1
    if aborted:
        assert [c[0] for c in order.mock_calls][:2] == ["abort", "move_to"]
    assert t.last_aborted is False     # the move-to is running now


def test_move_to_retried_while_busy_after_abort(translator_factory, mock_ocs, monkeypatch):
    import kosma_ocs_translator as kot
    monkeypatch.setattr(kot, "MOVE_TO_BUSY_RETRY_INTERVAL", 0.01)
    mock_ocs.move_to.side_effect = [_ocs_busy(), *_responses(503, 200)]
    t = translator_factory(obs2tel_overrides=HORIZON_OBS2TEL)
    t.last_payload["start_time"] = time.time() - 10.0   # previous path still running

    t._command_loop()

    assert mock_ocs.abort.call_count == 1
    assert mock_ocs.move_to.call_count == 3
    assert t.last_payload["type"] == "move_to"
    assert t.tel_path_failed is False


def test_abort_failure_fails_move_to(translator_factory, mock_ocs):
    mock_ocs.abort.side_effect = RuntimeError("no connection")
    t = translator_factory(obs2tel_overrides=HORIZON_OBS2TEL)
    t.last_payload["start_time"] = time.time() - 10.0   # previous path still running
    t.next_tel_return_cookie = 2

    t._command_loop()

    assert not mock_ocs.move_to.called
    assert t.tel_path_failed is True
    assert t.tel_error == 1


def test_no_reuse_of_aborted_path(translator_factory):
    t = translator_factory()
    t.last_aborted = True
    assert t.reuse_last_command() is False


def _responses(*codes):
    return [MagicMock(status_code=c, json=lambda c=c: {"status": "ok" if c == 200 else "busy"})
            for c in codes]


@pytest.mark.parametrize("codes,busy_timeout,expected_code,expected_calls", [
    pytest.param([200],                1.0,  200, 1, id="accepted_first_try"),
    pytest.param([503, 503, 200],      1.0,  200, 3, id="accepted_after_busy"),
    pytest.param([503] * 50,           0.05, 503, None, id="busy_until_timeout"),
    pytest.param([400],                1.0,  400, 1, id="other_error_not_retried"),
])
def test_scan_pattern_with_retry(translator_factory, mock_ocs, codes, busy_timeout,
                                 expected_code, expected_calls):
    mock_ocs.scan_pattern.side_effect = _responses(*codes)
    t = translator_factory()

    response = t._scan_pattern_with_retry({"points": []}, busy_timeout=busy_timeout,
                                          retry_interval=0.01)

    assert response.status_code == expected_code
    if expected_calls is not None:
        assert mock_ocs.scan_pattern.call_count == expected_calls
    else:
        assert 1 < mock_ocs.scan_pattern.call_count < len(codes)
    for call in mock_ocs.scan_pattern.call_args_list:
        assert call.kwargs == {"data": {"points": []}}


def _ocs_busy():
    """TCS busy as passed on by the OCS API: HTTP 200, error in the body."""
    return MagicMock(status_code=200, json=lambda: {"status": "error", "message": "busy"})


@pytest.mark.parametrize("status_code,body,busy,error", [
    pytest.param(503, {"status": "error", "message": "busy"}, True,  503, id="tcs_503"),
    pytest.param(200, {"status": "error", "message": "busy"}, True,  503, id="ocs_200_busy"),
    pytest.param(200, {"status": "error", "message": "bad az"}, False, 1, id="ocs_200_other_error"),
    pytest.param(400, {"status": "error", "message": "bad az"}, False, 400, id="http_400"),
    pytest.param(200, {"status": "ok"},                       False, 1,   id="ok"),
])
def test_is_busy_and_error_code(status_code, body, busy, error):
    import kosma_ocs_translator as kot
    response = MagicMock(status_code=status_code, json=lambda: body)
    assert kot._is_busy(response) is busy
    assert kot._error_code(response) == error


def test_scan_pattern_retry_on_ocs_busy(translator_factory, mock_ocs):
    mock_ocs.scan_pattern.side_effect = [_ocs_busy(), _ocs_busy(), *_responses(200)]
    t = translator_factory()

    response = t._scan_pattern_with_retry({"points": []}, busy_timeout=1.0,
                                          retry_interval=0.01)

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}
    assert mock_ocs.scan_pattern.call_count == 3


def test_scan_pattern_retry_stops_when_superseded(translator_factory, mock_ocs):
    mock_ocs.scan_pattern.side_effect = _responses(503, 200)
    t = translator_factory()
    t._command_stop_event.set()

    response = t._scan_pattern_with_retry({"points": []}, busy_timeout=1.0,
                                          retry_interval=0.01)

    assert response is None
    assert mock_ocs.scan_pattern.call_count == 1


def _new_obs2tel(t, cookie, **changes):
    """Simulate reading a new obs2tel: same as the last one except for changes."""
    t.obs2tel = {**t.last_obs2tel, "obs_cookie": cookie, **changes}
    t.next_tel_return_cookie = cookie


@pytest.mark.parametrize("changes,start_offset,pending,reused", [
    pytest.param({},                         -30.0,   False, True,  id="same_target_running"),
    pytest.param({},                         -30.0,   True,  True,  id="same_target_pending"),
    pytest.param({"obs_lam_del": 30.0},      -30.0,   False, False, id="different_offset"),
    pytest.param({"obs_wavelength": 350.0},  -30.0,   False, False, id="different_wavelength"),
    pytest.param({},                         -350.0,  False, False, id="path_ends_too_early"),
    pytest.param({},                         -250.0,  False, True,  id="path_long_enough"),
])
def test_reuse_last_command(translator_factory, changes, start_offset, pending, reused):
    import time
    t = translator_factory()   # payload is 600 s long, reuse needs >= 300 s left
    t.last_payload["start_time"] = time.time() + start_offset
    t.tel_return_cookie = 1
    hold = time.time() + 5.0 if pending else -np.inf   # path not started yet
    t.on_track_not_before = hold
    t.tel_lost_track = "Y"
    _new_obs2tel(t, 3, **changes)

    assert t.reuse_last_command() is reused

    if not reused:
        assert t.tel_return_cookie == 1
    else:
        assert t.tel_return_cookie == 3
        assert t.old_tel_return_cookie == 1
        assert t.tel_lost_track == "N"
    # a not yet started path keeps holding tel_on_track at "N"
    assert t.on_track_not_before == hold


def test_reuse_last_command_logs_changed_keys(translator_factory, caplog):
    import time
    t = translator_factory()
    t.last_payload["start_time"] = time.time() - 30
    old_lam_del = t.last_obs2tel["obs_lam_del"]
    _new_obs2tel(t, 3, obs_lam_del=old_lam_del + 30.0, obs_wavelength=350.0)

    with caplog.at_level(logging.DEBUG, logger=t.log.name):
        assert t.reuse_last_command() is False

    messages = [r.getMessage() for r in caplog.records if r.levelno == logging.DEBUG]
    assert len(messages) == 1
    assert messages[0].startswith("last path not reused: target changed: ")
    assert f"obs_lam_del {old_lam_del!r} -> {old_lam_del + 30.0!r}" in messages[0]
    assert "obs_wavelength" in messages[0]
    assert "obs_bet_del" not in messages[0]


def test_reuse_last_command_logs_path_ending(translator_factory, caplog):
    import time
    t = translator_factory()
    t.last_payload["start_time"] = time.time() - 350   # 250 s left < 300 s
    _new_obs2tel(t, 3)

    with caplog.at_level(logging.DEBUG, logger=t.log.name):
        assert t.reuse_last_command() is False

    assert any(r.getMessage().startswith("last path not reused: the last path ends in ")
               for r in caplog.records)


def test_reuse_last_command_not_after_failure(translator_factory):
    import time
    t = translator_factory()
    t.last_payload["start_time"] = time.time() - 30
    t._fail_path(2, 1)
    _new_obs2tel(t, 3)

    assert t.reuse_last_command() is False


def test_reuse_last_move_to_without_end(translator_factory):
    t = translator_factory()
    t.last_payload["type"] = "move_to"
    t.last_payload["start_time"] = 0.0   # long ago; a move-to holds its position
    _new_obs2tel(t, 3)

    assert t.reuse_last_command() is True
    assert t.tel_return_cookie == 3


def test_unexpected_exception_in_track_fails_path(translator_factory):
    t = translator_factory(obs2tel_overrides={"obs_coord_sys_on": "NOT_A_FRAME"})
    t.tel_return_cookie = 1
    t.next_tel_return_cookie = 2

    t._command_loop()   # ValueError for the unknown frame must not escape

    assert t.tel_return_cookie == 2
    assert t.tel_path_failed is True


# ---------------------------------------------------------------------------
# Slew (move-to) before a path whose start is far from the telescope
# ---------------------------------------------------------------------------

def _ok():
    return MagicMock(status_code=200, json=lambda: {"status": "ok"})


@pytest.fixture
def slew_translator(translator_factory, mock_ocs, monkeypatch):
    """
    Translator for a J2000 path, with no previous command (no abort sent) and
    a fake ACU position: far from any path start until a move-to was sent, then
    at the move-to target. arrive=False keeps it far forever.
    """
    import kosma_ocs_translator as kot
    monkeypatch.setattr(kot, "SLEW_POLL_INTERVAL", 0.01)

    def _make(arrive=True, on_poll=None):
        t = translator_factory()
        t.last_payload = None
        t.abort_if_unknown = False
        t.tel_return_cookie = 1
        t.next_tel_return_cookie = 2
        target = {}

        def move_to(az, el):
            target["azel"] = (az, el)
            return _ok()

        def acu_position():
            if on_poll and "azel" in target:
                on_poll(t)
            if arrive and "azel" in target:
                return (*target["azel"], 0.0, 0.0)
            return (-100.0, 5.0, 0.0, 0.0)   # path az is in [0, 360): always far

        mock_ocs.move_to.side_effect = move_to
        mock_ocs.scan_pattern.return_value = _ok()
        t._acu_position = acu_position
        return t, target

    return _make


def test_no_slew_if_close_to_path_start(slew_translator, mock_ocs, monkeypatch):
    import kosma_ocs_translator as kot
    monkeypatch.setattr(kot, "SLEW_BEFORE_PATH_MIN_DIST", 1e6)   # everything is close
    t, _ = slew_translator()

    t._command_loop()

    assert not mock_ocs.move_to.called
    assert mock_ocs.scan_pattern.call_count == 1
    assert t.tel_return_cookie == 2


def test_slew_before_path_if_far(slew_translator, mock_ocs):
    t, target = slew_translator()

    t._command_loop()

    assert mock_ocs.move_to.call_count == 1
    assert mock_ocs.scan_pattern.call_count == 1
    sent = mock_ocs.scan_pattern.call_args.kwargs["data"]
    # slewed to the start of the path (recomputed after the slew, source moved little)
    assert sent["points"][0][1] == pytest.approx(target["azel"][0], abs=0.1)
    assert sent["points"][0][2] == pytest.approx(target["azel"][1], abs=0.1)
    assert t.slew_in_progress is False
    assert t.last_payload["type"] == "path"
    assert t.tel_return_cookie == 2
    # held "N" until the path sent after the slew starts
    assert t.on_track_not_before == pytest.approx(sent["start_time"])
    assert t.tel_path_failed is False
    assert t.slew_target is None


def test_status_written_during_first_slew(slew_translator, mock_ocs):
    # first command after start-up (no last_payload): the returned cookie and
    # the slew target are reported while slewing
    during_slew = []
    t, target = slew_translator(on_poll=lambda t: during_slew.append(t.set_tel2obs_dict()[0]))
    t.last_obs2tel = None

    t._command_loop()

    result = during_slew[0]
    assert result is not None
    assert result["tel_return_cookie"] == 2
    assert result["tel_on_track"] == "N"
    assert (result["tel_azm_cmd"], result["tel_elv_cmd"]) == target["azel"]


def _send_once(t, mock_ocs):
    """Replace the short busy retry by a single send (fast tests)."""
    t._scan_pattern_with_retry = (
        lambda payload, busy_timeout, retry_interval: mock_ocs.scan_pattern(data=payload))


def test_busy_after_slew_resends_recomputed_path(slew_translator, mock_ocs):
    t, _ = slew_translator()
    _send_once(t, mock_ocs)
    mock_ocs.scan_pattern.side_effect = _responses(503, 503, 200)

    t._command_loop()

    assert mock_ocs.scan_pattern.call_count == 3
    starts = [c.kwargs["data"]["start_time"] for c in mock_ocs.scan_pattern.call_args_list]
    assert starts[0] < starts[1] < starts[2]     # recomputed with a fresh lead
    assert t.last_payload["start_time"] == starts[2]
    assert t.tel_path_failed is False
    assert t.tel_return_cookie == 2


def test_ocs_busy_after_slew_resends(slew_translator, mock_ocs):
    t, _ = slew_translator()
    _send_once(t, mock_ocs)
    mock_ocs.scan_pattern.side_effect = [_ocs_busy(), *_responses(200)]

    t._command_loop()

    assert mock_ocs.scan_pattern.call_count == 2
    assert t.tel_path_failed is False
    assert t.tel_return_cookie == 2


def test_ocs_busy_after_slew_timeout_reports_503(slew_translator, mock_ocs, monkeypatch):
    import kosma_ocs_translator as kot
    monkeypatch.setattr(kot, "SLEW_BUSY_RESEND_TIMEOUT", 0.0)
    t, _ = slew_translator()
    _send_once(t, mock_ocs)
    mock_ocs.scan_pattern.side_effect = [_ocs_busy()]

    t._command_loop()

    assert t.tel_path_failed is True
    assert t.tel_error == 503     # not the HTTP 200 of the OCS


def test_busy_after_slew_fails_after_timeout(slew_translator, mock_ocs, monkeypatch):
    import kosma_ocs_translator as kot
    monkeypatch.setattr(kot, "SLEW_BUSY_RESEND_TIMEOUT", 0.0)
    t, _ = slew_translator()
    _send_once(t, mock_ocs)
    mock_ocs.scan_pattern.side_effect = _responses(503, 200)

    t._command_loop()

    assert mock_ocs.scan_pattern.call_count == 1
    assert t.tel_path_failed is True
    assert t.tel_error == 503
    assert t.tel_return_cookie == 2


def test_busy_without_slew_not_resent(slew_translator, mock_ocs, monkeypatch):
    import kosma_ocs_translator as kot
    monkeypatch.setattr(kot, "SLEW_BEFORE_PATH_MIN_DIST", 1e6)   # no slew
    t, _ = slew_translator()
    _send_once(t, mock_ocs)
    mock_ocs.scan_pattern.side_effect = _responses(503, 200)

    t._command_loop()

    assert not mock_ocs.move_to.called
    assert mock_ocs.scan_pattern.call_count == 1
    assert t.tel_path_failed is True
    assert t.tel_error == 503


def test_slew_arrival_uses_own_tolerance(slew_translator, mock_ocs, monkeypatch):
    import kosma_ocs_translator as kot
    monkeypatch.setattr(kot, "SLEW_TIMEOUT", 0.05)
    monkeypatch.setattr(kot, "SLEW_POLL_INTERVAL", 0.01)
    t, target = slew_translator()
    # stopped 0.5 deg from the target: within SLEW_BEFORE_PATH_MIN_DIST, but not arrived
    t._acu_position = lambda: (
        (target["azel"][0] + 0.5, target["azel"][1], 0.0, 0.0) if "azel" in target
        else (-100.0, 5.0, 0.0, 0.0))

    t._command_loop()

    assert mock_ocs.move_to.call_count == 1
    assert not mock_ocs.scan_pattern.called
    assert t.tel_path_failed is True     # slew timed out


def test_slew_timeout_fails_path(slew_translator, mock_ocs, monkeypatch):
    import kosma_ocs_translator as kot
    monkeypatch.setattr(kot, "SLEW_TIMEOUT", 0.05)
    t, _ = slew_translator(arrive=False)

    t._command_loop()

    assert mock_ocs.move_to.call_count == 1
    assert not mock_ocs.scan_pattern.called
    assert t.tel_path_failed is True
    assert t.tel_return_cookie == 2
    assert t.tel_error == 1


def test_slew_rejected_fails_path(slew_translator, mock_ocs, monkeypatch):
    import kosma_ocs_translator as kot
    monkeypatch.setattr(kot, "MOVE_TO_BUSY_TIMEOUT", 0.0)
    t, _ = slew_translator()
    mock_ocs.move_to.side_effect = None
    mock_ocs.move_to.return_value = MagicMock(status_code=503, json=lambda: {"status": "busy"})

    t._command_loop()

    assert not mock_ocs.scan_pattern.called
    assert t.slew_in_progress is False
    assert t.tel_path_failed is True
    assert t.tel_error == 503


def test_superseded_slew_is_aborted_by_next_command(slew_translator, mock_ocs, monkeypatch):
    import kosma_ocs_translator as kot
    # superseded while slewing
    t, _ = slew_translator(arrive=False, on_poll=lambda t: t._command_stop_event.set())
    t._command_loop()
    assert not mock_ocs.scan_pattern.called
    assert t.slew_in_progress is True
    assert t.tel_path_failed is False      # superseded, not failed

    # the move-to may still be running: no reuse of the last path ...
    t.last_payload = {"type": "move_to", "start_time": 0.0,
                      "points": [[0.0, 0.0, 0.0, 0.0, 0.0]]}
    t.last_obs2tel = dict(t.obs2tel)
    assert t.reuse_last_command() is False

    # ... and the next path aborts it, even if the last accepted command was
    # a path that has ended (no abort then without the slew)
    t.last_payload = {"type": "path", "start_time": 0.0,
                      "points": [[0.0, 0.0, 0.0, 0.0, 0.0]]}
    t._command_stop_event.clear()
    monkeypatch.setattr(kot, "SLEW_BEFORE_PATH_MIN_DIST", 1e6)
    t._acu_position = lambda: (0.0, 0.0, 0.0, 0.0)
    t._command_loop()
    assert mock_ocs.abort.called
    assert t.slew_in_progress is False
    assert mock_ocs.scan_pattern.call_count == 1


@pytest.mark.parametrize("last_type,n_aborts", [
    pytest.param("path",    1, id="running_path_aborted_once"),
    pytest.param("move_to", 1, id="previous_move_to_aborted_once"),
])
def test_slew_abort(slew_translator, mock_ocs, last_type, n_aborts):
    """
    A running path or a previous move-to is aborted once by _send_track,
    not again by _slew_to before its own move-to.
    """
    import time
    t, _ = slew_translator()
    t.last_payload = {"type": last_type, "start_time": time.time() - 10.0,
                      "points": [[0.0, 0.0, 0.0, 0.0, 0.0],
                                 [600.0, 0.0, 0.0, 0.0, 0.0]]}
    order = MagicMock()
    order.attach_mock(mock_ocs.abort, "abort")
    order.attach_mock(mock_ocs.move_to, "move_to")

    t._command_loop()

    assert mock_ocs.abort.call_count == n_aborts
    assert [c[0] for c in order.mock_calls][:2] == ["abort", "move_to"]
    assert mock_ocs.scan_pattern.call_count == 1
    assert t.last_aborted is False


# ---------------------------------------------------------------------------
# Refraction correction in the AltAz transforms
# ---------------------------------------------------------------------------

def _stub_weather(t, pressure=428.199, temperature=2.72, humidity=0.39):
    """Replace the weather source of t, independent of what get_weather reads."""
    t.get_weather = MagicMock(return_value={
        "pressure": pressure * u.Torr,
        "temperature": temperature * u.deg_C,
        "relative_humidity": humidity,
    })


def _assert_default_weather(params):
    assert params["pressure"].to_value(u.bar) == pytest.approx(0.518)
    assert params["temperature"].to_value(u.deg_C) == pytest.approx(-5.9)
    assert params["relative_humidity"] == pytest.approx(0.24)


def test_refraction_off_does_not_read_weather(translator_factory):
    t = translator_factory()   # default: OFF
    t.get_weather = MagicMock()

    t._update_refraction_params(850.0)

    t.get_weather.assert_not_called()
    assert t._altaz_frame(Time("2026-10-19T12:00:00")).pressure == 0 * u.hPa


def test_refraction_on_raises_elevation(translator_factory):
    t = translator_factory()
    _stub_weather(t)
    obstime = Time("2026-10-19T12:00:00")
    # south celestial pole: at elevation of about 23 deg for any time
    pole = SkyCoord(ra=0 * u.deg, dec=-90 * u.deg, frame="icrs")
    el_off = pole.transform_to(t._altaz_frame(obstime)).alt.deg

    t.refraction_correction = True
    t._update_refraction_params(850.0)
    frame = t._altaz_frame(obstime)
    el_on = pole.transform_to(frame).alt.deg

    assert frame.obswl.to_value(u.micron) == pytest.approx(850.0)
    assert 10.0 < (el_on - el_off) * 3600.0 < 300.0


def test_refraction_not_applied_without_wavelength(translator_factory):
    t = translator_factory()
    t.refraction_correction = True
    t._refraction_params = {"pressure": 500 * u.hPa}
    t.get_weather = MagicMock()

    t._update_refraction_params(0.0)

    t.get_weather.assert_not_called()
    assert t._refraction_params == {}


@pytest.mark.parametrize("weather", [
    pytest.param({"humidity": 1.2},             id="humidity_above_1"),
    pytest.param({"humidity": -0.1},            id="negative_humidity"),
    pytest.param({"humidity": "N/A"},           id="humidity_not_a_number"),
    pytest.param({"pressure": 0.0},             id="zero_pressure"),
    pytest.param({"temperature": float("nan")}, id="temperature_nan"),
])
def test_refraction_bad_weather_uses_defaults(translator_factory, weather):
    t = translator_factory()
    _stub_weather(t, **weather)
    t.refraction_correction = True
    t.log = MagicMock()

    t._update_refraction_params(850.0)

    t.log.warning.assert_called_once()
    _assert_default_weather(t._refraction_params)
    assert t._refraction_params["obswl"].to_value(u.micron) == pytest.approx(850.0)


@pytest.mark.parametrize("side_effect", [
    pytest.param(TypeError("missing arguments"), id="ocs_call_fails"),
    pytest.param(None,                           id="no_weather_returned"),
])
def test_refraction_no_weather_source_uses_defaults(translator_factory, mock_ocs, side_effect):
    mock_ocs.get_measurement_data.side_effect = side_effect
    t = translator_factory()
    t.refraction_correction = True
    t.log = MagicMock()

    t._update_refraction_params(850.0)

    t.log.warning.assert_called_once()
    _assert_default_weather(t._refraction_params)


def test_path_uses_refraction_frame(slew_translator, mock_ocs, monkeypatch):
    import kosma_ocs_translator as kot
    monkeypatch.setattr(kot, "SLEW_BEFORE_PATH_MIN_DIST", 1e6)
    altaz = MagicMock(side_effect=AltAz)
    monkeypatch.setattr(kot, "AltAz", altaz)
    t, _ = slew_translator()
    _stub_weather(t)
    t.refraction_correction = True

    t._command_loop()

    assert mock_ocs.scan_pattern.call_count == 1
    assert altaz.call_count == 1
    assert altaz.call_args.kwargs["pressure"].to_value(u.Torr) == pytest.approx(428.199)
    assert altaz.call_args.kwargs["obswl"].to_value(u.micron) == pytest.approx(850.0)


def test_path_uses_default_weather_if_missing(slew_translator, mock_ocs, monkeypatch):
    import kosma_ocs_translator as kot
    monkeypatch.setattr(kot, "SLEW_BEFORE_PATH_MIN_DIST", 1e6)
    mock_ocs.get_measurement_data.side_effect = TypeError("missing arguments")
    t, _ = slew_translator()
    t.refraction_correction = True
    t.log = MagicMock()

    t._command_loop()

    t.log.warning.assert_called_once()
    _assert_default_weather(t._refraction_params)
    assert mock_ocs.scan_pattern.call_count == 1
    assert t.tel_path_failed is False


@pytest.mark.parametrize("refraction,acu_on,warnings", [
    pytest.param(True,  True,  1, id="both_on_warns_once"),
    pytest.param(True,  False, 0, id="acu_off"),
    pytest.param(False, True,  0, id="translator_off"),
])
def test_acu_refraction_check(translator_factory, mock_ocs, refraction, acu_on, warnings):
    from helpers import DEFAULT_POINTING
    mock_ocs.get_dataset.side_effect = lambda name: {
        **DEFAULT_POINTING, "RF refraction correction on": acu_on,
    }
    t = translator_factory()
    t.refraction_correction = refraction
    t.log = MagicMock()

    t.set_tel2obs_dict()
    t.set_tel2obs_dict()

    messages = [c.args[0] for c in t.log.warning.call_args_list]
    assert len([m for m in messages if "RF refraction" in m]) == warnings


# ---------------------------------------------------------------------------
# Start-up dump of the ACU SPEM parameters to KOSMA_file_io
# ---------------------------------------------------------------------------

ACU_SPEM_RAW = {"Time": 1.0, "Year": 2026, "Parameter IA": 1.5, "Parameter CA": -2.0}


@pytest.mark.parametrize("wavelength", [
    pytest.param(0.6,   id="optical"),
    pytest.param(850.0, id="submm"),
])
def test_dump_spem_to_kosma(translator_factory, mock_ocs, monkeypatch, wavelength):
    import kosma_ocs_translator as kot
    put = MagicMock(return_value={"IA": 1.5, "CA": -2.0})
    monkeypatch.setattr(kot, "put_kosma_spem", put)
    mock_ocs.get_dataset.side_effect = lambda name: (
        dict(ACU_SPEM_RAW) if name == "CmdSPEMParameter" else {}
    )
    t = translator_factory(obs2tel_overrides={"obs_wavelength": wavelength})

    result = t.dump_spem_to_kosma()

    put.assert_called_once()
    assert put.call_args.args[0] == {"Time": 1.0, "Year": 2026, "IA": 1.5, "CA": -2.0}
    assert result == {"IA": 1.5, "CA": -2.0}


def test_dump_spem_to_kosma_failure_does_not_raise(translator_factory, mock_ocs, monkeypatch):
    import kosma_ocs_translator as kot
    monkeypatch.setattr(kot, "put_kosma_spem",
                        MagicMock(side_effect=RuntimeError("Kset_point failed")))
    mock_ocs.get_dataset.side_effect = lambda name: dict(ACU_SPEM_RAW)
    t = translator_factory(obs2tel_overrides={"obs_wavelength": 0.6})

    assert t.dump_spem_to_kosma() is None


# ---------------------------------------------------------------------------
# op_tel_model_type / op_tel_model_date copied from KOSMA_ccatacu_point.in
# ---------------------------------------------------------------------------

def _write_ccatacu_model(directory, model_type, model_date):
    (Path(directory) / "KOSMA_ccatacu_point.in").write_text(
        "30-Oct-2026  14:46:42  1793368002.442420   File update time stamp   ! Kset_point\n"
        "1.5   ccatacu_point[0]   ! CCAT-ACU pointing correction coefficient [mdeg]\n"
        "IA   string_ccatacu_point[0]   ! description text\n"
        f"{model_type}   ccatacu_model_type   ! identifier string for CCAT-ACU pointing model type\n"
        f"{model_date}   ccatacu_model_date   ! creation date of CCAT_ACU pointing model values\n"
    )


def test_ccdpos_model_type_and_date_from_ccatacu_point(translator_factory, tmp_path):
    _write_ccatacu_model(tmp_path, "CCAT-ACU_VA_1.3", "2026-10-30:14:44:00Z")
    t = translator_factory(obs2tel_overrides={"obs_wavelength": 0.6})

    _, ccd = t.set_tel2obs_dict()

    assert ccd["op_tel_model_type"] == "CCAT-ACU_VA_1.3"
    assert ccd["op_tel_model_date"] == "2026-10-30:14:44:00Z"


def test_ccdpos_model_type_with_blanks_takes_first_column(translator_factory, tmp_path):
    _write_ccatacu_model(tmp_path, "CCAT-ACU VA Error Compensations-document Release 1.3",
                         "2026-10-30:14:44:00Z")
    t = translator_factory(obs2tel_overrides={"obs_wavelength": 0.6})

    _, ccd = t.set_tel2obs_dict()

    assert ccd["op_tel_model_type"] == "CCAT-ACU"
    assert ccd["op_tel_model_date"] == "2026-10-30:14:44:00Z"


def test_ccdpos_model_marker_without_ccatacu_point(translator_factory):
    from kosma_ocs_translator import POINTING_MODEL_MISSING
    # tmp_path has no KOSMA_ccatacu_point.in
    t = translator_factory(obs2tel_overrides={"obs_wavelength": 0.6})

    _, ccd = t.set_tel2obs_dict()

    assert (ccd["op_tel_model_type"], ccd["op_tel_model_date"]) == POINTING_MODEL_MISSING


@pytest.mark.parametrize("drop,expected", [
    ("ccatacu_model_type", ("CCAT-ACU-MISSING", "2026-10-30:14:44:00Z")),
    ("ccatacu_model_date", ("CCAT-ACU_VA_1.3", "MISSING")),
])
def test_ccdpos_model_marker_for_missing_entry(translator_factory, tmp_path, drop, expected):
    _write_ccatacu_model(tmp_path, "CCAT-ACU_VA_1.3", "2026-10-30:14:44:00Z")
    path = tmp_path / "KOSMA_ccatacu_point.in"
    path.write_text("".join(
        line for line in path.read_text().splitlines(keepends=True) if drop not in line
    ))
    t = translator_factory(obs2tel_overrides={"obs_wavelength": 0.6})

    _, ccd = t.set_tel2obs_dict()

    assert (ccd["op_tel_model_type"], ccd["op_tel_model_date"]) == expected


def test_ccdpos_model_marker_differs_from_acfg_defaults():
    # starpoint holds the ACFG defaults if the entries are missing; the markers must
    # differ from them (strcmp fails) but keep the "CCAT-ACU" prefix (strncmp passes)
    from kosma_ocs_translator import POINTING_MODEL_MISSING
    model_type, model_date = POINTING_MODEL_MISSING
    assert model_type.startswith("CCAT-ACU") and model_type != "CCAT-ACU"
    assert model_date != "YYYY-MM-DDTHH:MM:SSZ"
    assert " " not in model_type and " " not in model_date   # read with %s


def test_ccdpos_model_date_follows_file_update(translator_factory, tmp_path):
    import os
    _write_ccatacu_model(tmp_path, "CCAT-ACU", "2026-10-30:14:44:00Z")
    t = translator_factory(obs2tel_overrides={"obs_wavelength": 0.6})
    _, ccd = t.set_tel2obs_dict()
    assert ccd["op_tel_model_date"] == "2026-10-30:14:44:00Z"

    # e.g. update_spem_acu.py wrote a new SPEM set; force a different mtime
    _write_ccatacu_model(tmp_path, "CCAT-ACU", "2026-11-05:10:00:00Z")
    path = tmp_path / "KOSMA_ccatacu_point.in"
    st = path.stat()
    os.utime(path, ns=(st.st_atime_ns, st.st_mtime_ns + 10**9))

    _, ccd = t.set_tel2obs_dict()
    assert ccd["op_tel_model_date"] == "2026-11-05:10:00:00Z"


def test_ccdpos_model_not_reread_if_file_unchanged(translator_factory, tmp_path, monkeypatch):
    import kosma_ocs_translator as kot
    _write_ccatacu_model(tmp_path, "CCAT-ACU", "2026-10-30:14:44:00Z")
    t = translator_factory(obs2tel_overrides={"obs_wavelength": 0.6})
    reads = []
    original = kot.read_first_column
    monkeypatch.setattr(kot, "read_first_column",
                        lambda *a, **kw: reads.append(a) or original(*a, **kw))

    for _ in range(3):
        t.set_tel2obs_dict()

    assert len(reads) == 1


def test_ccdpos_file_contains_model_type_and_date(translator_factory, tmp_path):
    _write_ccatacu_model(tmp_path, "CCAT-ACU_VA_1.3", "2026-10-30:14:44:00Z")
    t = translator_factory(obs2tel_overrides={"obs_wavelength": 0.6})

    t.write_tel2obs_file()

    lines = (tmp_path / "KOSMA_ccdpos.sim").read_text().splitlines()
    fields = {l.partition("!")[0].split()[-1]: l.partition("!")[0].split()[0]
              for l in lines if "!" in l and "File update time stamp" not in l}
    assert fields["op_tel_model_type"] == "CCAT-ACU_VA_1.3"
    assert fields["op_tel_model_date"] == "2026-10-30:14:44:00Z"


@pytest.mark.parametrize("wavelength,written", [
    pytest.param(0.6,   True,  id="optical"),
    pytest.param(99.9,  True,  id="just_below_100um"),
    pytest.param(100.0, False, id="100um"),
    pytest.param(850.0, False, id="submm"),
])
def test_ccdpos_file_written_only_below_submm(translator_factory, tmp_path, wavelength, written):
    t = translator_factory(obs2tel_overrides={"obs_wavelength": wavelength})

    t.write_tel2obs_file()

    assert (tmp_path / "KOSMA_tel2obs.set").exists()
    assert (tmp_path / "KOSMA_ccdpos.sim").exists() is written


def test_ccdpos_params_not_built_for_submm(translator_factory, monkeypatch):
    t = translator_factory(obs2tel_overrides={"obs_wavelength": 850.0})
    build = MagicMock()
    monkeypatch.setattr(t, "_build_ccdpos_params", build)

    result, ccd = t.set_tel2obs_dict()

    assert result is not None
    assert ccd is None
    build.assert_not_called()


# ---------------------------------------------------------------------------
# op_vel_elv: commanded (path) elevation velocity in KOSMA_ccdpos.sim
# ---------------------------------------------------------------------------

def _set_payload_el_velocity(t, vel_of_time):
    """Replace the el velocity column of t.last_payload by vel_of_time(dt) [deg/s]."""
    for p in t.last_payload["points"]:
        p[4] = vel_of_time(p[0])


def test_op_vel_elv_zero_for_static_payload(translator_factory):
    # static payload: velocities are 0
    t = translator_factory(obs2tel_overrides={"obs_wavelength": 0.6})

    _, ccd = t.set_tel2obs_dict()

    assert ccd["op_vel_elv"] == 0.0


def test_op_vel_elv_interpolated_from_payload(translator_factory):
    # status time is 30 s after the payload start
    t = translator_factory(obs2tel_overrides={"obs_wavelength": 0.6})
    _set_payload_el_velocity(t, lambda dt: 0.001 + 1e-6 * dt)

    result, ccd = t.set_tel2obs_dict()

    t_rel = result["tel_time_del"]
    assert t_rel == pytest.approx(30.0, abs=1e-3)
    assert ccd["op_vel_elv"] == pytest.approx((0.001 + 1e-6 * t_rel) * 3600.0)


def test_op_vel_elv_zero_without_payload(translator_factory):
    t = translator_factory(obs2tel_overrides={"obs_wavelength": 0.6})
    t.last_payload = None
    t.last_obs2tel = None
    t._fail_path(2, 1)   # first path failed: status is still written

    _, ccd = t.set_tel2obs_dict()

    assert ccd["op_vel_elv"] == 0.0


def test_tel_vel_elv_stays_acu_measured_velocity(translator_factory):
    # tel2obs reports the ACU velocity, ccdpos the commanded one
    t = translator_factory(obs2tel_overrides={"obs_wavelength": 0.6},
                           status_overrides={"Elevation current velocity": 0.01})
    _set_payload_el_velocity(t, lambda dt: 0.002)

    result, ccd = t.set_tel2obs_dict()

    assert result["tel_vel_elv"] == pytest.approx(0.01 * 3600.0)
    assert ccd["op_vel_elv"] == pytest.approx(0.002 * 3600.0)


def test_op_vel_elv_written_only_to_ccdpos(translator_factory, tmp_path):
    t = translator_factory(obs2tel_overrides={"obs_wavelength": 0.6})
    _set_payload_el_velocity(t, lambda dt: 0.002)

    t.write_tel2obs_file()

    def values(path):
        out = {}
        for line in path.read_text().splitlines():
            data, sep, _ = line.partition("!")
            fields = data.split()
            if sep and fields and "File update time stamp" not in data:
                out[fields[-1]] = fields[0]
        return out

    ccdpos = values(tmp_path / "KOSMA_ccdpos.sim")
    tel2obs = values(tmp_path / "KOSMA_tel2obs.set")
    assert float(ccdpos["op_vel_elv"]) == pytest.approx(0.002 * 3600.0)
    # the helper key is not written to KOSMA_tel2obs.set
    assert "tel_vel_elv_cmd" not in tel2obs
    assert "op_vel_elv" not in tel2obs


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

    t = translator_factory(obs2tel_overrides={"obs_wavelength": 0.6})
    t.write_tel2obs_file()

    tel2obs_path = tmp_path / "KOSMA_tel2obs.set"
    ccdpos_path = tmp_path / "KOSMA_ccdpos.sim"

    assert tel2obs_path.exists(), "KOSMA_tel2obs.set was not written"
    assert ccdpos_path.exists(), "KOSMA_ccdpos.sim was not written"

    (output_dir / "KOSMA_tel2obs.set").write_text(tel2obs_path.read_text())
    (output_dir / "KOSMA_ccdpos.sim").write_text(ccdpos_path.read_text())
