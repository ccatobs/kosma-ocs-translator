"""
Integration tests for KOSMA_translator.set_tel2obs_dict.

Adding a new scenario is a one-liner: append a Case to CASES with only the
fields that differ from the defaults in helpers.py.  The parametrised test
function handles the rest.

NOTE: test_tel_angle_focal_plane_j2000_matches_reference will currently fail
because set_tel2obs_dict creates the SkyCoord for *pos* without passing
frame=altaz_frame.  Fix that line first, then this test should pass.
"""
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
        obs2tel={"obs_coord_sys_on": "HORIZON", "obs_lam_on": 180.0, "obs_bet_on": 45.0},
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

    NOTE: this test currently fails because set_tel2obs_dict creates pos =
    SkyCoord(az=..., alt=...) without passing frame=altaz_frame.  Add that
    argument and the test will pass.
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

    assert abs(computed - ref_angle) < 0.1, (
        f"tel_angle_focal_plane={computed:.3f} deg, "
        f"reference={ref_angle:.3f} deg, diff={abs(computed - ref_angle):.3f} deg"
    )
