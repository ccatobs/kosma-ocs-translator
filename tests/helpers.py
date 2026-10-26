"""
Shared constants and utility functions for the test suite.
Not a pytest module — imported directly by conftest.py and test files.
"""
import time
from pathlib import Path
import numpy as np
from astropy.coordinates import EarthLocation
from astropy import units as u

DATA_DIR = Path(__file__).parent / "data"

# CCAT site coordinates
CCAT_LOCATION = EarthLocation(
    lat=-22.985639 * u.deg,
    lon=-67.740278 * u.deg,
    height=5611.8 * u.m,
)

# Default obs2tel values — mirror of tests/data/KOSMA_obs2tel.set.
# Tests override only the keys they care about; add new defaults here
# when new obs2tel variables become required.
BASE_OBS2TEL = {
    "obs_tolerance": 3600,
    "obs_cookie": 1,
    "obs_tel_info_update_time": 1.0,
    "obs_coord_sys_on": "J2000",
    "obs_lam_on": 83.8221,
    "obs_bet_on": -5.3911,
    "obs_pm_lam": 0.0,
    "obs_pm_bet": 0.0,
    "obs_paralx": 0.0,
    "obs_extended_elevation_range": "N",
    "obs_lam_del": 0.0,
    "obs_bet_del": 0.0,
    "obs_coord_sys_del": "J2000",
    "obs_true_angle_del": "Y",
    "obs_lam_vel": 0.0,
    "obs_bet_vel": 0.0,
    "obs_otf_mode": "N",
    "obs_lam_off": 0.0,
    "obs_bet_off": 0.0,
    "obs_coord_sys_off": "J2000",
    "obs_true_angle_off": "Y",
    "obs_x_focal_plane": 0.0,
    "obs_y_focal_plane": 0.0,
    "obs_wavelength": 850.0,
    "obs_track_duration": 600,
}

# Default mock telescope status
DEFAULT_STATUS = {
    "Azimuth current position": 180.0,
    "Elevation current position": 45.0,
    "Azimuth commanded position": 180.0,
    "Elevation commanded position": 45.0,
    "Time": 292.5,   # day 292.5 of 2026 ≈ 2026-10-19 12:00 UTC
    "Year": 2026,
}

# Default mock pointing corrections (all zero)
DEFAULT_POINTING = {
    "SPEM correction AZ": 0.0,
    "SPEM correction EL": 0.0,
    "Tiltmeter Az correction AZ": 0.0,
    "Tiltmeter El correction AZ": 0.0,
    "Tiltmeter Az correction EL": 0.0,
    "Tiltmeter El correction EL": 0.0,
}


def write_obs2tel(directory, data):
    """Write a KOSMA_obs2tel.set file to *directory* from a variable→value dict."""
    path = Path(directory) / "KOSMA_obs2tel.set"
    ts = time.time()
    with open(path, "w") as f:
        f.write(
            f"19-Oct-2026  14:00:00   {ts:.3f}   File update time stamp   ! test\n"
        )
        for key, val in data.items():
            f.write(f"{val}   {key}   ! test\n")


def make_payload(az, el, t_start, duration=600):
    """Build a minimal static-pointing track payload dict."""
    n = int(duration) + 1
    times = np.linspace(0, duration, n)
    points = np.column_stack([
        times,
        np.full(n, az),
        np.full(n, el),
        np.zeros(n),   # vaz
        np.zeros(n),   # vel
    ]).tolist()
    return {
        "start_time": float(t_start),
        "coordsys": "Horizon",
        "points": points,
    }
