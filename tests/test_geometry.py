"""Geometry golden tests: the target matches astra (astra -t -f 4 -n -r) within 0.1 arcsec."""

import json
from pathlib import Path

import numpy as np
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
