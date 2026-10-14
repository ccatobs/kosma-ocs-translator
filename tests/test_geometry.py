"""Geometry golden tests: the target matches astra (astra -t -f 4 -n -r) within 0.1 arcsec."""

import json
import time
from pathlib import Path

import numpy as np
import ocs
import pytest

from kosma_ocs_translator import geometry, kio

FIXTURES = Path(__file__).parent / "fixtures"
GOLDEN = json.loads((FIXTURES / "golden.json").read_text())
_, TEMPLATE = kio.read_obs2tel(FIXTURES / "ccat_chai" / "KOSMA_obs2tel.set")
SITE = geometry.Site(latitude=-22.9856389, longitude=-67.7402778, height=5611.8)
EOP = geometry.read_bulletin_a(FIXTURES / "bulletinA")


def separation_arcsec(az1, el1, az2, el2):
    daz = (np.asarray(az1) - az2 + 180) % 360 - 180
    return np.hypot(daz * np.cos(np.radians(el2)), np.asarray(el1) - el2) * 3600


def track_of(fixture):
    command = {**TEMPLATE, **fixture["command"]}
    return geometry.Track.from_command(command, pointing_offsets=fixture["track_point"])


@pytest.mark.parametrize("fixture", GOLDEN["fixtures"], ids=lambda f: f["name"])
def test_target_matches_astra_within_a_tenth_of_an_arcsec(fixture):
    t = np.array([s["utc"] for s in fixture["samples"]])
    tau = np.array([s["tau"] for s in fixture["samples"]])

    az, el = geometry.target(track_of(fixture), t, tau, SITE, EOP)

    sep = separation_arcsec(
        az, el, [s["az"] for s in fixture["samples"]], [s["el"] for s in fixture["samples"]]
    )
    assert sep.max() < 0.1, sep


@pytest.mark.parametrize("fixture", GOLDEN["fixtures"], ids=lambda f: f["name"])
def test_focal_plane_angle_matches_astra(fixture):
    golden = fixture["tel_angle_focal_plane"]
    tau = fixture["samples"][-1]["tau"]

    angle = geometry.focal_plane_angle(track_of(fixture), golden["utc"], tau, SITE, EOP)

    assert angle == pytest.approx(golden["deg"], abs=1e-4)


def test_a_window_through_north_keeps_one_azimuth_branch_near_the_acu():
    fixture = next(f for f in GOLDEN["fixtures"] if f["name"] == "north_transit")
    t0 = fixture["samples"][0]["utc"] - 600
    t = t0 + np.arange(1500.0)

    for az_ref in (350.0, -10.0, 5.0):
        points = np.array(geometry.window(track_of(fixture), t, 0 * t, SITE, EOP, az_ref))
        az = points[:, 1]
        assert np.abs(np.diff(az)).max() < 0.1  # no jump of 360 deg inside the window
        assert -180 <= az.min() and az.max() <= 360
        # the other branches leave the TCS range or start farther from the ACU
        for other in (az - 360, az + 360):
            inside = -180 <= other.min() and other.max() <= 360
            assert not inside or abs(other[0] - az_ref) >= abs(az[0] - az_ref)


def test_a_1500_point_window_fits_the_ocs_body_limit():
    fixture = next(f for f in GOLDEN["fixtures"] if f["name"] == "otf_line_running")
    t = fixture["samples"][0]["utc"] + np.arange(1500.0)

    points = geometry.window(track_of(fixture), t, np.clip(t - t[0], -10, 1500), SITE, EOP)

    body = ocs.check_path(15.0, points, "Horizon")  # relative start: the lead is 15 s
    assert len(ocs._encode(body)) < ocs.MAX_BODY_BYTES


@pytest.mark.parametrize("name", ["STAR", "@MARS", "@MOON"])
def test_a_1500_point_window_is_cheap_and_agrees_with_exact_targets(name):
    command = {**TEMPLATE, "obs_source_name": name, "obs_lam_on": 132.3, "obs_bet_on": 19.1}
    track = geometry.Track.from_command(command)
    t = 1792062410.4 + np.arange(1500.0)
    geometry.window(track, t, 0 * t, SITE, EOP)  # warm caches

    start = time.perf_counter()
    points = np.array(geometry.window(track, t, 0 * t, SITE, EOP))
    elapsed = time.perf_counter() - start

    assert elapsed < 0.05
    for k in range(0, 1500, 97):  # one sample at a time is exact
        az, el = geometry.target(track, t[k : k + 1], 0.0, SITE, EOP)
        assert separation_arcsec(points[k, 1], points[k, 2], az, el)[0] < 0.001
