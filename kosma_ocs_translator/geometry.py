"""Geometry: the target, topocentric az/el without refraction (ADR 0001), stage by stage as astra.

Angles are degrees at the interface and radians inside. Times are unix seconds (UTC).
ERFA (IAU 2006/2000A) does the mean-to-apparent-to-topocentric step, with DUT1 and
polar motion; the OCS adds refraction and the pointing model to every path sample.
"""

import re
from dataclasses import dataclass

import erfa
import numpy as np

ARCSEC = np.pi / 180 / 3600
FRAMES = ("J2000", "B1950", "GALACTIC", "HIPPARCOS", "HORIZON")

# IAU 1958 galactic system: v_gal = GALACTIC_MATRIX @ v_J2000 (slaEqgal, slaGaleq)
GALACTIC_MATRIX = np.array(
    [
        [-0.054875539726, -0.873437108010, -0.483834985808],
        [+0.494109453312, -0.444829589425, +0.746982251810],
        [-0.867666135858, -0.198076386122, +0.455983795705],
    ]
)


@dataclass(frozen=True)
class Site:
    latitude: float  # deg
    longitude: float  # deg, east-positive
    height: float  # m


@dataclass(frozen=True)
class Eop:
    """Earth orientation from IERS Bulletin A: MJD, polar motion x/y (arcsec), UT1-UTC (s)."""

    mjd: np.ndarray
    x: np.ndarray
    y: np.ndarray
    dut1: np.ndarray

    def at(self, mjd):
        if np.min(mjd) < self.mjd[0] or np.max(mjd) > self.mjd[-1]:
            raise ValueError(
                f"IERS data cover MJD {self.mjd[0]:.0f}-{self.mjd[-1]:.0f}, not {np.min(mjd):.2f}"
            )
        return (
            np.interp(mjd, self.mjd, self.x) * ARCSEC,
            np.interp(mjd, self.mjd, self.y) * ARCSEC,
            np.interp(mjd, self.mjd, self.dut1),
        )


# Bulletin A rows: "26 10  2  61315 0.17334 .00009 0.32459 .00009 -0.023137 0.000021"
# (rapid service) and "2026 10  9  61322  0.1672  0.3223  -0.03029" (predictions).
_RAPID = re.compile(
    r"^\s*\d{2}\s+\d{1,2}\s+\d{1,2}\s+(\d{5})\s+(\S+)\s+\S+\s+(\S+)\s+\S+\s+(\S+)\s+\S+\s*$"
)
_PREDICTED = re.compile(r"^\s*\d{4}\s+\d{1,2}\s+\d{1,2}\s+(\d{5})\s+(\S+)\s+(\S+)\s+(\S+)\s*$")


def read_bulletin_a(path):
    """Read IERS Bulletin A (ser7.dat), the file astra reads as share/astra/bulletinA."""
    rows = {}
    with open(path) as f:
        for line in f:
            m = _RAPID.match(line) or _PREDICTED.match(line)
            if m:
                rows[int(m[1])] = [float(v) for v in m.groups()[1:]]
    mjd = np.array(sorted(rows), dtype=float)
    x, y, dut1 = np.array([rows[int(d)] for d in mjd]).T
    return Eop(mjd, x, y, dut1)


def _rotate(matrix, lam, bet):
    v = np.einsum("ij,...j->...i", matrix, erfa.s2c(lam, bet))
    return erfa.c2s(v)


@dataclass(frozen=True)
class Offset:
    """A map or OFF offset in its frame, radians; velocity in rad/s (map only)."""

    frame: str
    true_angle: bool
    lam: float
    bet: float
    vlam: float = 0.0
    vbet: float = 0.0


@dataclass(frozen=True)
class Track:
    """One telescope command as the geometry needs it. Angles in radians."""

    frame: str
    lam: float
    bet: float
    pm: tuple  # (dlam/dt, dbet/dt) rad/yr, dlam/dt without cos(bet)
    parallax: float  # arcsec
    map: Offset
    off: Offset
    focal_plane: tuple  # (x, y) az/el offset, true angle
    extended: bool
    pointing: tuple  # (az, el) observer pointing offset, true angle

    @classmethod
    def from_command(cls, command, pointing_offsets=(0.0, 0.0)):
        """Build the track of an obs2tel command (kio.read_obs2tel); raises ValueError."""
        c = command
        for key in ("obs_coord_sys_on", "obs_coord_sys_del", "obs_coord_sys_off"):
            if c[key] not in FRAMES:
                raise ValueError(f"{key} {c[key]} is not one of {', '.join(FRAMES)}")
        otf = c["obs_otf_mode"] == "C"
        return cls(
            frame=c["obs_coord_sys_on"],
            lam=np.radians(c["obs_lam_on"]),
            bet=np.radians(c["obs_bet_on"]),
            pm=(c["obs_pm_lam"] * ARCSEC, c["obs_pm_bet"] * ARCSEC),
            parallax=c["obs_paralx"],
            map=Offset(
                c["obs_coord_sys_del"],
                c["obs_true_angle_del"] == "Y",
                c["obs_lam_del"] * ARCSEC,
                c["obs_bet_del"] * ARCSEC,
                c["obs_lam_vel"] * ARCSEC if otf else 0.0,
                c["obs_bet_vel"] * ARCSEC if otf else 0.0,
            ),
            off=Offset(
                c["obs_coord_sys_off"],
                c["obs_true_angle_off"] == "Y",
                c["obs_lam_off"] * ARCSEC,
                c["obs_bet_off"] * ARCSEC,
            ),
            focal_plane=(c["obs_x_focal_plane"] * ARCSEC, c["obs_y_focal_plane"] * ARCSEC),
            extended=c["obs_extended_elevation_range"] == "Y",
            pointing=(pointing_offsets[0] * ARCSEC, pointing_offsets[1] * ARCSEC),
        )


def add_offset(lam, bet, dlam, dbet, true_angle=True):
    """astra_addoffset: MODE_TRUE divides dlam by cos of the new beta; MODE_NONE adds."""
    bet = bet + dbet
    return lam + (dlam / np.cos(bet) if true_angle else dlam), bet


def _apply(offset, lam, bet, tau, nudge=0.0):
    """Add an offset, then its velocity times the line time tau, as astra's ADD_OFFSET."""
    lam, bet = add_offset(lam, bet, offset.lam, offset.bet + nudge, offset.true_angle)
    if offset.vlam or offset.vbet:
        lam, bet = add_offset(lam, bet, offset.vlam * tau, offset.vbet * tau, offset.true_angle)
    return lam, bet


def _mean(track, tau, nudge):
    """Source frame to mean J2000, with the map and the OFF offset in their frames.

    Returns (ra, dec, pm_ra, pm_dec, parallax, rv) for erfa.atciq.
    """
    lam, bet = track.lam, track.bet + nudge
    star = (*track.pm, track.parallax, 0.0)
    if track.frame == "B1950":
        lam, bet, *star = erfa.fk425(lam, bet, *star)
    elif track.frame == "GALACTIC":
        lam, bet = _rotate(GALACTIC_MATRIX.T, lam, bet)
    for offset, n in ((track.map, nudge), (track.off, 0.0)):
        if offset.frame == "GALACTIC":
            gl, gb = _rotate(GALACTIC_MATRIX, lam, bet)
            lam, bet = _rotate(GALACTIC_MATRIX.T, *_apply(offset, gl, gb, tau, n))
        elif offset.frame == "B1950":
            bl, bb, *bstar = erfa.fk524(lam, bet, *star)
            lam, bet, *star = erfa.fk425(*_apply(offset, bl, bb, tau, n), *bstar)
        elif offset.frame != "HORIZON":
            lam, bet = _apply(offset, lam, bet, tau, n)
    return lam, bet, *star


def _observed(track, t, tau, site, eop, nudge=0.0):
    """Observed az/el (rad) without refraction, before the HORIZON-frame offsets.

    nudge shifts beta by that much in the source frame and again in the map frame,
    as astra does for the second point of its parallactic angle.
    """
    t = np.atleast_1d(np.asarray(t, dtype=float))
    if track.frame == "HORIZON":  # astra skips the sky stages and offsets in other frames
        return np.full(t.shape, track.lam), np.full(t.shape, track.bet)
    xp, yp, dut1 = eop.at(t / 86400 + 40587)
    astrom, _ = erfa.apco13(
        2440587.5,
        t / 86400,
        dut1,
        np.radians(site.longitude),
        np.radians(site.latitude),
        site.height,
        xp,
        yp,
        0.0,
        0.0,
        0.0,
        0.0,
    )
    ri, di = erfa.atciq(*_mean(track, tau, nudge), astrom)
    az, zd, *_ = erfa.atioq(ri, di, astrom)
    return az, np.pi / 2 - zd


def target(track, t, tau, site, eop):
    """Target az/el (deg) of track at unix times t; tau is the OTF line time (s):
    the map offset is del + v * tau."""
    tau = np.asarray(tau, dtype=float)
    az, el = _observed(track, t, tau, site, eop)
    for offset in (track.off, track.map):  # HORIZON-frame offsets, OFF first as astra
        if offset.frame == "HORIZON":
            az, el = _apply(offset, az, el, tau)
    az, el = add_offset(az, el, *track.focal_plane)
    if track.extended:  # beyond 90 deg elevation, as astra
        az, el = az + np.pi, np.pi - el
    az, el = add_offset(az, el, *track.pointing)
    return np.degrees(az) % 360, np.degrees(el)


def focal_plane_angle(track, t, tau, site, eop):
    """tel_angle_focal_plane (deg): astra's a_parang, the bearing in az/el of a point
    0.001 arcsec up in beta of the source frame and again of the map frame."""
    if track.frame == "HORIZON" or track.map.frame == "HORIZON":
        return 0.0
    az0, el0 = _observed(track, t, tau, site, eop)
    az1, el1 = _observed(track, t, tau, site, eop, nudge=0.001 * ARCSEC)
    return float(np.degrees(erfa.pas(az0, el0, az1, el1))[0])


# Path limits of the TCS, as the OCS client checks them (commands.go).
AZIMUTH_MIN, AZIMUTH_MAX = -180.0, 360.0
ELEVATION_MIN, ELEVATION_MAX = -90.0, 180.0
AZIMUTH_SPEED_MAX, ELEVATION_SPEED_MAX = 3.0, 1.5


def window(track, t, tau, site, eop, az_ref=180.0):
    """Path points [t - t[0], az, el, az_vel, el_vel] (s, deg, deg/s) of the target.

    The window keeps one azimuth branch inside AZIMUTH_MIN..AZIMUTH_MAX, the one whose
    first point lies nearest to az_ref. Values are rounded to 1e-7 deg (0.4 mas) and
    1e-7 deg/s so that 1500 points stay below the 100 KB body limit of the OCS.
    Raises ValueError when the target leaves the TCS limits.
    """
    t = np.asarray(t, dtype=float)
    az, el = target(track, t, tau, site, eop)
    az = np.degrees(np.unwrap(np.radians(az)))
    branches = sorted((az + 360 * k for k in range(-2, 3)), key=lambda a: abs(a[0] - az_ref))
    az = next((a for a in branches if a.min() >= AZIMUTH_MIN and a.max() <= AZIMUTH_MAX), None)
    if az is None:
        raise ValueError("the azimuth of the window spans more than the TCS range")
    if el.min() < ELEVATION_MIN or el.max() > ELEVATION_MAX:
        raise ValueError(f"elevation {el.min():.2f}..{el.max():.2f} deg outside the TCS range")
    vaz, vel = np.gradient(az, t), np.gradient(el, t)
    if np.abs(vaz).max() > AZIMUTH_SPEED_MAX or np.abs(vel).max() > ELEVATION_SPEED_MAX:
        raise ValueError("the target moves faster than the TCS speed limits")
    return np.column_stack([np.round(t - t[0], 3), az, el, vaz, vel]).round(7).tolist()
