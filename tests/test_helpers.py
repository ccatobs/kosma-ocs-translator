"""Unit tests for module-level helper functions."""
import pytest
import numpy as np
from astropy.coordinates import SkyCoord, AltAz
from astropy import units as u
from astropy.time import Time

from kosma_ocs_translator import (
    _apply_offset, _get_time_from_acu_format, _hermite_resample,
    TRACK_COARSE_STEP, TRACK_TIME_STEP,
)
from helpers import CCAT_LOCATION


class TestHermiteResample:

    def test_quadratic_is_exact(self):
        # second-order slopes are exact for a quadratic, so is the cubic Hermite
        t_c = np.arange(11) * 1.0
        t_f = np.arange(101) * 0.1
        y, dydt = _hermite_resample(t_c, 3.0 + 2.0 * t_c - 0.5 * t_c**2, t_f)
        np.testing.assert_allclose(y, 3.0 + 2.0 * t_f - 0.5 * t_f**2, atol=1e-12)
        np.testing.assert_allclose(dydt, 2.0 - t_f, atol=1e-12)

    def test_nodes_are_reproduced(self):
        t_c = np.arange(5) * 1.0
        y_c = np.array([0.0, 1.0, 4.0, 2.0, 7.0])
        y, _ = _hermite_resample(t_c, y_c, t_c)
        np.testing.assert_allclose(y, y_c, atol=1e-12)

    def test_sidereal_track_near_zenith(self):
        # source passing 3 deg from zenith (fast azimuth): interpolated track
        # matches the full transform at every path point
        t0 = Time("2026-10-19T03:00:00")
        source = SkyCoord(
            az=0.0 * u.deg, alt=87.0 * u.deg,
            frame=AltAz(obstime=t0, location=CCAT_LOCATION)).icrs
        t_f = np.arange(601) * TRACK_TIME_STEP
        t_c = np.arange(int(t_f[-1] / TRACK_COARSE_STEP) + 1) * TRACK_COARSE_STEP

        def altaz(t):
            return source.transform_to(
                AltAz(obstime=t0 - 30 * u.s + t * u.s, location=CCAT_LOCATION))

        coarse, full = altaz(t_c), altaz(t_f)
        az, vaz = _hermite_resample(t_c, np.unwrap(coarse.az.deg, period=360.0), t_f)
        el, vel = _hermite_resample(t_c, coarse.alt.deg, t_f)
        d_az = (np.mod(az, 360.0) - full.az.deg + 180.0) % 360.0 - 180.0
        assert np.max(np.abs(d_az)) * 3600 < 0.01
        assert np.max(np.abs(el - full.alt.deg)) * 3600 < 0.01
        # velocities against finite differences of the full transform
        full_vaz = np.gradient(np.unwrap(full.az.deg, period=360.0), t_f)
        full_vel = np.gradient(full.alt.deg, t_f)
        assert np.max(np.abs(vaz - full_vaz)[1:-1]) * 3600 < 0.1
        assert np.max(np.abs(vel - full_vel)[1:-1]) * 3600 < 0.1


class TestApplyOffset:

    def test_zero_offset_is_identity(self):
        coord = SkyCoord(ra=83.8 * u.deg, dec=-5.4 * u.deg, frame="icrs")
        assert _apply_offset(coord, 0.0, 0.0, "Y").separation(coord).arcsec < 1e-9

    @pytest.mark.parametrize("true_angle", ["Y", "N"])
    def test_zero_offset_both_modes(self, true_angle):
        coord = SkyCoord(ra=83.8 * u.deg, dec=-5.4 * u.deg, frame="icrs")
        assert _apply_offset(coord, 0.0, 0.0, true_angle).separation(coord).arcsec < 1e-9

    def test_true_angle_y_preserves_angular_separation(self):
        coord = SkyCoord(ra=83.8 * u.deg, dec=-5.4 * u.deg, frame="icrs")
        result = _apply_offset(coord, 3600.0, 0.0, "Y")
        assert abs(result.separation(coord).arcsec - 3600.0) < 1.0

    def test_true_angle_n_direct_addition_in_lon(self):
        # at dec=0, true_angle N adds arcsec directly to RA
        coord = SkyCoord(ra=83.8 * u.deg, dec=0.0 * u.deg, frame="icrs")
        result = _apply_offset(coord, 3600.0, 0.0, "N")
        assert abs(result.ra.deg - (coord.ra.deg + 1.0)) < 1e-9

    def test_bet_offset_moves_latitude(self):
        coord = SkyCoord(ra=83.8 * u.deg, dec=0.0 * u.deg, frame="icrs")
        result = _apply_offset(coord, 0.0, 3600.0, "Y")
        assert abs(result.dec.deg - 1.0) < 0.01

    def test_galactic_frame(self):
        coord = SkyCoord(l=294.0 * u.deg, b=12.2 * u.deg, frame="galactic")
        result = _apply_offset(coord, 0.0, 3600.0, "Y")
        assert abs(result.b.deg - (coord.b.deg + 1.0)) < 0.01

    def test_altaz_frame(self):
        frame = AltAz(obstime=Time("2026-10-19T12:00:00"), location=CCAT_LOCATION)
        coord = SkyCoord(az=180.0 * u.deg, alt=45.0 * u.deg, frame=frame)
        result = _apply_offset(coord, 0.0, 3600.0, "Y")
        assert abs(result.alt.deg - 46.0) < 0.01

    @pytest.mark.parametrize("d_lam,d_bet", [
        (100.0,   0.0),
        (  0.0, 100.0),
        (100.0, 100.0),
        (-60.0,  80.0),
    ])
    def test_true_angle_separation_equals_vector_magnitude(self, d_lam, d_bet):
        coord = SkyCoord(ra=83.8 * u.deg, dec=-5.4 * u.deg, frame="icrs")
        expected = (d_lam**2 + d_bet**2) ** 0.5
        result = _apply_offset(coord, d_lam, d_bet, "Y")
        assert abs(result.separation(coord).arcsec - expected) < 1.0


class TestGetTimeFromAcuFormat:

    def test_day_1_is_jan_1(self):
        t = _get_time_from_acu_format(1.0, 2026)
        assert t.iso[:10] == "2026-01-01"

    def test_noon_jan_1(self):
        t = _get_time_from_acu_format(1.5, 2026)
        ref = Time("2026-01-01T12:00:00", scale="utc")
        assert abs(t.unix - ref.unix) < 1.0

    def test_day_292_is_oct_19_in_2026(self):
        # 2026 is not a leap year; day 292 = Oct 19
        t = _get_time_from_acu_format(292.0, 2026)
        assert t.iso[:10] == "2026-10-19"

    def test_correct_year(self):
        assert _get_time_from_acu_format(1.0, 2025).iso[:4] == "2025"
        assert _get_time_from_acu_format(1.0, 2026).iso[:4] == "2026"

    def test_returns_astropy_time(self):
        assert isinstance(_get_time_from_acu_format(100.0, 2026), Time)

    def test_scale_is_utc(self):
        assert _get_time_from_acu_format(1.0, 2026).scale == "utc"
