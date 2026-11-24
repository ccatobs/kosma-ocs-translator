import time
import logging
import os
import socket
from astropy.coordinates import SkyCoord, AltAz, Distance
from astropy import units as u
from astropy.time import Time
import numpy as np
import argparse
import threading
import signal,sys
from dataclasses import dataclass
# sun avoidance
try:
    from sun_avoidance.telescope import get_overtop_altaz_from_nominal
except ImportError:
    print("sun-avoidance not found. Install from git@github.com:ccatobs/sun-avoidance.git")
    sys.exit(1)

from kosma_ocs_lib import (
    DEFAULT_LOGGER_NAME,
    KOSMA_SPEM_FILE,
    copy_point_in_to_status,
    setup_logging,
    setup_ocs,
    ImportKOSMAReadWriteIntoDictionary,
    read_first_column,
    write_atomic,
    strip_parameter_prefix,
    put_kosma_spem,
)


# log file, written into the current directory
LOG_FILE = "kosma-ocs-translator.log"

# ---------------------------------------------------------------------------
# Coordinate systems
# ---------------------------------------------------------------------------
# KOSMA obs_coord_sys_* -> astropy frame name
COORD_SYS_MAP = {
    "J2000": "icrs",
    "B1950": "fk4",
    "GALACTIC": "galactic",
    "HORIZON": "altaz",
}

# Proper-motion keyword names in astropy SkyCoord differ by frame
_FRAME_PM_KEYS = {
    "icrs":     ("pm_ra_cosdec", "pm_dec"),
    "fk4":      ("pm_ra_cosdec", "pm_dec"),
    "galactic": ("pm_l_cosb",    "pm_b"),
}

# Reference epoch for catalogue coordinates by frame
# only matters for proper motion correction
_FRAME_EPOCHS = {
    "icrs":     Time("J2000"),
    "fk4":      Time(1950.0, format="byear"),
    "galactic": Time("J2000"),
}

# ---------------------------------------------------------------------------
# Pointing model (KOSMA_SPEM_FILE and the other SPEM constants: kosma_ocs_lib)
# ---------------------------------------------------------------------------
# op_tel_model_type / op_tel_model_date written to KOSMA_ccdpos.sim if
# KOSMA_SPEM_FILE or its ccatacu_model_type / ccatacu_model_date is missing.
# Deliberately not the ACFG defaults ("CCAT-ACU", "YYYY-MM-DDTHH:MM:SSZ"): starpoint
# then also holds these defaults for ccatacu_model_type / ccatacu_model_date and
# its consistency check (strcmp, starpoint.c) would pass on placeholders. The type
# keeps the "CCAT-ACU" prefix, so that starpoint runs the check and aborts.
POINTING_MODEL_MISSING = ("CCAT-ACU-MISSING", "MISSING")

# ---------------------------------------------------------------------------
# Observing wavelength and refraction
# ---------------------------------------------------------------------------
# [micron] obs_wavelength separating submm observations (at or above this value)
# from optical / infrared camera observations (below this value).
# KOSMA_ccdpos.sim is written only for the latter.
SUBMM_MIN_WAVELENGTH = 100.0

# used for the refraction correction if the weather parameters are bad or missing
DEFAULT_WEATHER = {
    "pressure": 0.518 * u.bar,
    "temperature": -5.9 * u.deg_C,
    "relative_humidity": 0.24,
}

# ---------------------------------------------------------------------------
# Slew before a path (see KOSMA_translator._slew_to)
# ---------------------------------------------------------------------------
# [deg] If the telescope is at least this far from the first point of a new path
# (in azimuth or in elevation, not distance on sky), it is first slewed there with move-to
# and the path is sent only after it has arrived. Otherwise the path is sent directly
# and the ACU itself moves to the first point with its spline path definition.
SLEW_BEFORE_PATH_MIN_DIST = 1.0
# [sec] give up waiting for the slew to finish after this
SLEW_TIMEOUT = 300.0
# [deg] the slew has finished once both axes are this close to the target (in
# azimuth and in elevation, ACU position minus pointing correction) ...
SLEW_ARRIVAL_TOLERANCE = 0.01
# [deg/s] ... and slower than this.
# TCS finishes the move-to only once |current - commanded| < 1e-4 deg and the
# speed < 1e-4 deg/s on both axes (telescope-control-system commands.go); until
# then a path is rejected as busy, see SLEW_BUSY_RESEND_TIMEOUT.
SLEW_SPEED_TOLERANCE = 1e-4
# [sec] after a slew, a path rejected as busy (TCS still finishing the move-to)
# is recomputed with a fresh lead and resent until this time after the first send
SLEW_BUSY_RESEND_TIMEOUT = 10.0
# [sec] interval of reading the ACU status while waiting for the slew
SLEW_POLL_INTERVAL = 0.5
# [sec] a move-to rejected as busy (TCS not yet idle after the abort sent just
# before it) is resent for at most this long, MOVE_TO_BUSY_RETRY_INTERVAL apart
MOVE_TO_BUSY_TIMEOUT = 5.0
MOVE_TO_BUSY_RETRY_INTERVAL = 0.3

# ---------------------------------------------------------------------------
# Path timing (see KOSMA_translator._send_track)
# ---------------------------------------------------------------------------
# [sec] start_time of a path is this far after the time its computation starts.
# TCS requires start_time >= 9.8 sec ahead when it processes the path.
TRACK_OVERHEAD = 11.0
# [sec] minimum lead (start_time - now) right before sending the path. If the
# computation left less, the path is recomputed with a larger overhead. The margin
# above 9.8 sec covers transfer time and a clock offset between here and the TCS.
TRACK_MIN_SEND_LEAD = 10.3
# [sec] time step between path points sent to the TCS
TRACK_TIME_STEP = 0.1
# [sec] time step of the full (astropy) coordinate transform; the points in between
# are cubic Hermite interpolated (errors << 0.01 arcsec for sidereal motion)
TRACK_COARSE_STEP = 1.0

# ---------------------------------------------------------------------------
# OTF, obs_otf_mode C (see the section "OTF" of KOSMA_translator)
# ---------------------------------------------------------------------------
# KOSMA otf_server sends PRETP (obs_track_duration = 0) and then OTFTP
# (obs_track_duration = scan duration D > 0, obs_start_time = T0). The backend
# assigns the offset obs_lam/bet_del + obs_lam/bet_vel * (t - T0) to the data
# (writeheader.c): the scan passes obs_lam/bet_del at full speed at T0.
# [sec] before T0 the speed along the scan rises from 0 (relative to the sky) to
# the scan speed with a smoothstep profile (zero acceleration at both ends).
# Run-up length = scan speed * OTF_RAMP_TIME / 2.
OTF_RAMP_TIME = 3.0
# [sec] tracking of the run-up point before the ramp
OTF_SETTLE_TIME = 0.0
# [sec] smoothstep deceleration after T0 + D; 0: the path ends at T0 + D
OTF_STOP_TIME = 0.0
# [sec] the OTF path must be sent at least this long before its start. T0 is
# fixed by KOSMA, so a path that is too late is not sent (no shift of T0).
OTF_MIN_SEND_LEAD = TRACK_MIN_SEND_LEAD
# [sec] expected time from the translator reading PRETP to T0: pre_otf_time in
# KOSMA master_parameters plus the KOSMA overhead in between (cookie handshake,
# backend IDLE). Only used to aim the PRETP move-to; an error of 1 s moves the
# aim by ~15 arcsec at sidereal rate, which the ACU absorbs within the path lead.
OTF_EXPECTED_PRE_OTF_TIME = 15.5
# [sec] a running tracking path at the PRETP position is kept only if it runs
# at least this much longer
OTF_PRE_KEEP_MIN_REMAINING = 60.0
# tel_error if the OTF path could not be sent in time for T0
TEL_ERROR_OTF_TOO_LATE = 2

# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------
# [sec] interval of checking KOSMA_obs2tel.set for a new command. The delay (up
# to this interval) counts against pre_otf_time for OTF: pre_otf_time must be
# >= OTF_MIN_SEND_LEAD + OTF_RAMP_TIME + OTF_SETTLE_TIME + this interval
# + computation/abort (~0.5 s), i.e. ~15 s with the defaults.
OBS2TEL_POLL_INTERVAL = 0.5


def _apply_offset(coord, d_lam_arcsec, d_bet_arcsec, true_angle):
    """
    Apply a sky offset to a SkyCoord.

    Parameters
    ----------
    coord: astropy.SkyCoord
        Coordinates of the central position
    d_lam_arcsec, d_bet_arcsec: float
        offset to be added in the coordinate system given for coord in unit of arcsec.
    true_angle : str
        If 'Y', longitude requires cos(lat) correction. Use spherical_offsets_by
        If anything else, direct coordinate addition is applied.

    Returns
    -------
    The updated position : astropy.SkyCoord
    """

    if true_angle == "Y":
        # if required to be faster or completely consistent with KOSMA calculation,
        # use the following simple delta calculation
        # cos_lat = np.cos(coord.data.lat.rad)
        # return SkyCoord(
        #     coord.data.lon + (d_lam_arcsec / cos_lat) * u.arcsec,
        #     coord.data.lat + d_bet_arcsec * u.arcsec,
        #     frame=coord.frame,
        # )

        # spherical_offset_by is a full precision position angle
        return coord.spherical_offsets_by(
            d_lam_arcsec * u.arcsec, d_bet_arcsec * u.arcsec)

    return SkyCoord(
        coord.data.lon + d_lam_arcsec * u.arcsec,
        coord.data.lat + d_bet_arcsec * u.arcsec,
        frame=coord.frame,
    )


def _hermite_resample(t_coarse, y_coarse, t_fine):
    """
    Cubic Hermite interpolation of y_coarse(t_coarse), with the slopes at the
    nodes from second-order finite differences.

    Parameters
    ----------
    t_coarse : np.ndarray
        Increasing node times, at least 3 [sec]
    y_coarse : np.ndarray
        Values at t_coarse (must be continuous, e.g. unwrapped azimuth)
    t_fine : np.ndarray
        Times to interpolate at, within [t_coarse[0], t_coarse[-1]] [sec]

    Returns
    -------
    y, dydt : np.ndarray
        Interpolated values and their time derivatives at t_fine
    """
    slope = np.gradient(y_coarse, t_coarse, edge_order=2)
    i = np.clip(np.searchsorted(t_coarse, t_fine, side="right") - 1, 0, len(t_coarse) - 2)
    h = t_coarse[i + 1] - t_coarse[i]
    s = (t_fine - t_coarse[i]) / h
    y0, y1 = y_coarse[i], y_coarse[i + 1]
    m0, m1 = slope[i] * h, slope[i + 1] * h
    s2 = s * s
    s3 = s2 * s
    y = ((2 * s3 - 3 * s2 + 1) * y0 + (s3 - 2 * s2 + s) * m0
         + (-2 * s3 + 3 * s2) * y1 + (s3 - s2) * m1)
    dydt = ((6 * s2 - 6 * s) * y0 + (3 * s2 - 4 * s + 1) * m0
            + (-6 * s2 + 6 * s) * y1 + (3 * s2 - 2 * s) * m1) / h
    return y, dydt


def _otf_scan_time(tau, duration, ramp, stop):
    """
    Position along the OTF scan line at time tau = t - T0, expressed in
    seconds of scanning at the nominal velocity. It is evaluated for every
    path point: the map offset commanded at that point is

        obs_lam_del + obs_lam_vel * f(tau),  obs_bet_del + obs_bet_vel * f(tau)

    f replaces t - T0 so that this one scan formula also gives the ramp-up/down.

    During the scan (0 <= tau <= duration) f = tau, i.e. exactly the offset
    the backend assigns to the data taken at t. Outside the scan f is not
    tau, so that the telescope does not jump to full speed:

    - before T0: f is held at -ramp/2 (the run-up point R), then within the
      last `ramp` seconds the speed df/dtau rises smoothly from 0 to 1
      (smoothstep 3u^2 - 2u^3), reaching the scan start with full speed at
      tau = 0. Example with ramp = 3 s: f = -1.5 for tau <= -3,
      f(-1.5) = -1.219, f(0) = 0, with df/dtau = 0, 0.5, 1 at these times.
    - after T0 + D: the speed falls back to 0 the same way within `stop`
      seconds, and f is held at duration + stop/2.

    The acceleration is zero at both ends of the ramp and of the stop.

    Parameters
    ----------
    tau : array_like
        Time since T0 [sec]
    duration : float
        Scan duration D [sec]
    ramp, stop : float
        Duration of the acceleration before T0 and of the deceleration after
        T0 + D [sec]. 0: speed step (hold before T0, end at T0 + D).

    Returns
    -------
    f : np.ndarray
        Position along the scan line in seconds of nominal scan speed, one
        value per tau [sec]
    """
    tau = np.asarray(tau, dtype=float)
    f = tau.copy()
    if ramp > 0:
        u = np.clip((tau + ramp) / ramp, 0.0, 1.0)
        f = np.where(tau < 0, -ramp / 2 + ramp * (u**3 - u**4 / 2), f)
    else:
        f = np.where(tau < 0, 0.0, f)
    if stop > 0:
        u = np.clip((tau - duration) / stop, 0.0, 1.0)
        f = np.where(tau > duration, duration + stop * (u - u**3 + u**4 / 2), f)
    else:
        f = np.where(tau > duration, float(duration), f)
    return f


def _get_time_from_acu_format(day_of_year, year):
    """
    Convert the ACU provided time-related variables to datetime.
    Input is assumed to be in UTC.
    Used the same logic as housekeeping/monitors/acu/monitorAcu.py
    but with astropy.time instead of datetime

    Parameters
    ----------
    day_of_year : float
        Unit in sec
    year : int
        Year

    Returns
    -------
    t : astropy.time.Time
        Corresponding Time object of the given time
    """

    t = Time(f"{year}-01-01", format="iso", scale="utc") + (day_of_year - 1) * u.day

    return t


def _total_pointing_correction(response_pointing):
    """
    Total pointing correction (SPEM + tiltmeters) applied by the ACU.

    Parameters
    ----------
    response_pointing : dict
        ACU dataset "CmdPointingCorrection"

    Returns
    -------
    del_tot_azm, del_tot_elv : float
        Azimuth and elevation corrections in degrees
    """

    # This is the pointing model only and must not include refraction: it is
    # removed from the encoder positions to get the positions for KOSMA, which keep
    # the refraction, and it is written as del_tel_tot_* to KOSMA_ccdpos.sim.
    # TODO: when the refraction correction is done at OCS, add it to tel_*_cmd
    # in _build_tel2obs_params, not here

    del_tot_azm = (
        response_pointing["SPEM correction AZ"]
        + response_pointing["Tiltmeter Az correction AZ"]
        + response_pointing["Tiltmeter El correction AZ"]
    )
    del_tot_elv = (
        response_pointing["SPEM correction EL"]
        + response_pointing["Tiltmeter Az correction EL"]
        + response_pointing["Tiltmeter El correction EL"]
    )
    return del_tot_azm, del_tot_elv


def _response_body(response):
    """
    JSON body of an OCS response as a dict. Errors raised in front of the
    API (e.g. an nginx 413 for a too large path payload) come back as HTML,
    not JSON: then return status None and the start of the raw text as
    message, so that the caller logs the real HTTP error instead of a
    JSONDecodeError.
    """
    try:
        body = response.json()
    except ValueError:
        return {"status": None, "message": response.text[:200].strip()}
    if not isinstance(body, dict):
        return {"status": None, "message": str(body)[:200]}
    return body


def _is_busy(response):
    """
    True if TCS rejected the command because the previous one is still
    running. TCS answers HTTP 503 with message "busy", but the OCS API passes
    the TCS JSON body on with HTTP 200 (status "error", message "busy"),
    so both forms are accepted.
    """
    if response.status_code == 503:
        return True
    body = _response_body(response)
    return body.get("status") != "ok" and body.get("message") == "busy"


def _error_code(response):
    """
    tel_error for a rejected command. The OCS reports TCS errors with HTTP
    200 and status "error" in the body: busy is reported as 503, any other
    rejection with HTTP 200 as 1, never as 0 or 200.
    """
    if _is_busy(response):
        return 503
    if response.status_code == 200:
        return 1
    return response.status_code


@dataclass(frozen=True)
class PathTarget:
    """
    Target of a path (non-HORIZON source): everything needed to compute the
    path points except the time (see KOSMA_translator._build_path_payload).

    Attributes
    ----------
    coord : astropy.SkyCoord
        Source position in its catalogue frame, with proper motion and the
        non-HORIZON sky/map and off-position offsets already applied
    horizon_offsets : tuple of (d_lam_arcsec, d_bet_arcsec, true_angle)
        Offsets applied in AltAz after the transform, in this order:
        HORIZON sky/map offset, HORIZON off-position offset, focal plane offset.
        Zero offsets are left out.
    extended : bool
        Extended elevation range (over the top)
    """
    coord: SkyCoord
    horizon_offsets: tuple = ()
    extended: bool = False


class KOSMA_translator:
    """
    Translates KOSMA obs2tel commands into OCS/TCS commands and reports the
    telescope status back to KOSMA. Three threads use this object:

    - main thread: watches KOSMA_obs2tel.set and dispatches new commands
      (sections "obs2tel input" and "Command dispatch")
    - track thread: one per command, computes and sends it to TCS
      (sections "Tracking", "TCS communication", "Cookie handshake")
    - background thread: polls the ACU and writes KOSMA_tel2obs.set and
      KOSMA_ccdpos.sim (section "Status output")

    The track thread and the background thread share the cookie handshake
    state (tel_return_cookie, tel_error, on_track_not_before, ...) and
    last_payload / last_obs2tel.
    """

    # =======================================================================
    # Set-up (main thread, at start-up)
    # =======================================================================

    def __init__(self, ocs, abort_if_unknown=True, refraction_correction=False):
        # --- configuration --------------------------------------------------
        self.ocs = ocs
        self.log = logging.getLogger(DEFAULT_LOGGER_NAME)
        self._host_pid = f"({socket.gethostname()}:{os.getpid()})"
        self.read_write_dir = os.environ.get(
            "WRITE_DIR", "/net/KOSMA_file_io/ReadWrite/"
        )
        self.log.info(f"Using KOSMA ReadWrite directory: {self.read_write_dir}")
        # ocs.earth_location is set when the OCS client connects: it is fetched from
        # TCS /telescope-position (hard-coded FYST constants in TCS astro.go, same as
        # the TCS uses for its own az/el), with a built-in default if that fails.
        self.location = ocs.earth_location
        self.log.info(f"Telescope location: {self.location}")
        # abort before the first path if there is no record of the previous command
        self.abort_if_unknown = abort_if_unknown
        # [sec] length of a path sent to TCS (and of the move-to bookkeeping payload)
        self.track_duration = 600
        # [sec] minimum remaining time of the last path to keep it for a new obs2tel
        # commanding the same target (see reuse_last_track)
        self.reuse_min_remaining = self.track_duration / 2

        # --- status output templates ----------------------------------------
        self.load_tel2obs_template()
        self.load_ccdpossim_template()

        # --- cookie handshake (track thread -> background thread) ------------
        self.tel_return_cookie = 0
        self.old_tel_return_cookie = 0
        self.next_tel_return_cookie = 0
        # The cookie is returned as soon as TCS has accepted the command, but
        # tel_on_track is "N" until the ACU executes it: before the path starts, the
        # telescope is still on the previous target and would give a stale "Y".
        # [unix sec] tel_on_track is "N" for ACU status times before this, i.e. the
        # path start_time (inf while slewing to the start of a path not sent yet).
        # No margin: the ACU and this host are NTP synchronized, and settling onto
        # the path is handled by tel_was_on_track.
        self.on_track_not_before = -np.inf
        self._logged_on_track_not_before = -np.inf   # log "path started" once
        # True once tel_on_track was "Y" for the current cookie. tel_lost_track is
        # set only after that, so the time the telescope needs to settle onto a
        # new path is not reported as lost track.
        self.tel_was_on_track = False
        # True if the path for tel_return_cookie was rejected or failed to be sent
        self.tel_path_failed = False
        self.tel_error = 0
        self.tel_lost_track = "N"
        # (T0, T0 + D) [unix sec] while the returned cookie is an OTF scan (OTFTP),
        # else None. Then tel_on_track is "N" outside this window, tel_lost_track
        # is set by any status outside the tolerance within it (no settling), and
        # tel_time_del is the time since T0.
        self.otf_scan_window = None
        # (az, el, start) [deg, deg, unix sec] while the returned cookie is
        # slewing to the start of its path (path not accepted yet), else None.
        # Then tel_*_cmd is the slew target and tel_time_del the time since the
        # slew started.
        self.slew_target = None
        # True once a cookie was returned or failed: from then on
        # KOSMA_tel2obs.set is written, also before any command was recorded in
        # last_payload (slew before the first path, first path failed)
        self.cookie_reported = False

        # --- commands (main thread / track thread) ---------------------------
        self._track_thread = None
        self._track_stop_event = threading.Event()
        # record of the last accepted command and the obs2tel it was sent for
        self.last_payload = None
        self.last_obs2tel = None
        # True while a move-to sent by _slew_to may still be running on TCS
        # (e.g. its track was superseded before arrival): the next command aborts it
        self.slew_in_progress = False
        # True after the last command (path, move-to or slew) was aborted and
        # nothing was accepted by TCS since: nothing is running, so no further
        # abort is needed and the last path must not be reused
        self.last_aborted = False

        # --- coordinates and refraction -------------------------------------
        # apply the atmospheric refraction in the AltAz transforms
        self.refraction_correction = refraction_correction
        self.log.info(f"Refraction correction: {'ON' if refraction_correction else 'OFF'}")
        # AltAz keyword arguments for refraction, updated for each new track.
        # Empty: no refraction (AltAz default pressure is 0)
        self._refraction_params = {}
        # last known state of the ACU RF refraction correction
        self._acu_refraction_on = False

        # --- status output ---------------------------------------------------
        # pointing model identifier for KOSMA_ccdpos.sim, cached by the
        # modification time of KOSMA_ccatacu_point.in (see _pointing_model_info)
        self._model_info_key = None
        self._model_info = POINTING_MODEL_MISSING

        # --- obs2tel input (sets obs2tel, next_tel_return_cookie, ...) -------
        self.read_obs2tel_file()

    def load_tel2obs_template(self):
        self.tel2obs = """\
        {0[timestring]}  {0[timestamp]}   File update time stamp   ! ccat_translator {0[host_pid]}
        {0[tel_telescope]}   tel_telescope   ! Telescope Identifier (for display and backend_name in FITS header)
        {0[tel_on_track]}   tel_on_track   ! Y if tracking on commanded position/track within tolerance [Y/N]
        {0[tel_lost_track]}   tel_lost_track   ! Y if tracking got beyond tolerance since start of track [Y/N]
        {0[tel_pos_in_range]}   tel_pos_in_range   ! Y if commanded position is within telescope range [Y/N]
        {0[tel_error]}   tel_error   ! flags telescope command syntax/consistency and functional errors (=0: ok)
        NONE   tel_error_string   ! Error message associated with telescope command (=NONE: OK)
        5   tel_error_level   ! Error level for message in tel_error_string [1: trace, 2: debug, 4: info, 5: message, 7: warn, 8: error, 9: fatal]
        {0[tel_return_cookie]}  tel_return_cookie   ! return cookie to identify proper observation
        {0[tel_plate_scale]}  tel_plate_scale   ! plate scale of focal plane [arcsec/mm]
        {0[tel_latitude]}   tel_latitude   ! latitude of observatory (needed for Doppler correction) [degree]
        {0[tel_longitude]}  tel_longitude   ! longitude of observatory (needed for Doppler correction) [degree, +west (astronomical notation)]
        {0[tel_altitude]}  tel_altitude   ! height above sea level (needed for Doppler correction) [meter]
        {0[tel_angle_focal_plane]}   tel_angle_focal_plane   ! ccw angle from 2nd coordinate of obs_coord_sys_del to 2nd coordinate of telescope reference system (TARF,HORIZON) [degree] (NOTE: was formerly opposite, but wrong)
        0   tel_los_act   ! actual los-angle [degree]
        {0[tel_time_act]}   tel_time_act   ! time when status data were valid [seconds since 1970 UTC]
        {0[tel_time_del]}   tel_time_del   ! time since start of track (may be negative, if start in the future) [seconds]
        {0[tel_azm_cmd]:3.4f}   tel_azm_cmd   ! commanded azimuth (for tracking display) [degree,cw toward east]
        {0[tel_elv_cmd]:3.4f}   tel_elv_cmd   ! commanded elevation (for tracking display and for atmospheric calibration) [degree]
        {0[tel_azm_act]:3.4f}   tel_azm_act   ! actual azimuth (for tracking display) [degree,cw toward east]
        {0[tel_elv_act]:3.4f}   tel_elv_act   ! actual elevation (for tracking display) [degree]
        {0[tel_supports_ephemeris]}   tel_supports_ephemeris   ! Telescope supports ephemeris tracking [Y/N] !
        {0[tel_mjd1]}    tel_mjd1 ! modified Julian Date UT1 [days]
        {0[tel_vel_azm]} tel_vel_azm ! azimuth velocity [arcsec/sec]
        {0[tel_vel_elv]} tel_vel_elv ! elevation velocity [arcsec/sec]
        """

    def load_ccdpossim_template(self):
        self.ccdpossim = """\
        {0[timestring]}  {0[timestamp]}   File update time stamp   ! ccat_translator {0[host_pid]}
        {0[op_azm]}  op_azm   ! azimuth [radian] for optical pointing (commanded to telescope, without pointing model corrections)
        {0[op_elv]}  op_elv   ! elevatin [radian] for optical pointing (commanded to telescope, without pointing model corrections)
        {0[op_vel_elv]}  op_vel_elv   ! elevation velocity [arcsec/sec], recorded for optical pointing simulation
        {0[op_oazm]}  op_oazm   ! azimuth pointing offset [rad]; optical pointing (HORIZON system)
        {0[op_oelv]}  op_oelv   ! elevation pointing offset [rad]; optical pointing (HORIZON system)
        {0[op_tel_model_type]}   op_tel_model_type   ! identifier string for telescope task pointing model applied [KOSMA_3m,CCAT_*,CCAT-ACU*]
        {0[op_tel_model_date]}   op_tel_model_date   ! creation date for telescope task pointing model applied
        {0[del_tel_tot_azm]}   del_tel_tot_azm   ! azimuth  pointing offset [rad] from telescope pointing model (including non repeatable terms)
        {0[del_tel_tot_elv]}   del_tel_tot_elv   ! elevation pointing offset [rad] from telescope pointing model (including non repeatable terms)
        {0[del_tel_rep_azm]}   del_tel_rep_azm   ! azimuth  pointing offset [rad] from telescope pointing model (only repeatable terms)
        {0[del_tel_rep_elv]}   del_tel_rep_elv   ! elevation pointing offset [rad] from telescope pointing model (only repeatable terms)
        """

    def dump_spem_to_kosma(self):
        """
        Mirror the SPEM parameters currently used by the ACU into KOSMA_file_io,
        so that the pointing fit in KOSMA starts from the actual ACU model.
        Failures are logged but do not stop the translator.

        Returns
        -------
        spem : dict or None
            SPEM parameters read back from KOSMA_file_io, None if the dump failed
        """
        self.log.info(f"dumping ACU SPEM parameters to {KOSMA_SPEM_FILE}")
        try:
            spem_acu = strip_parameter_prefix(self.ocs.get_dataset("CmdSPEMParameter"))
            self.log.info(f"ACU SPEM parameters: {spem_acu}")
            spem_kosma = put_kosma_spem(spem_acu, logger=self.log)
        except Exception as e:
            self.log.error(f"Failed to dump ACU SPEM parameters to {KOSMA_SPEM_FILE}: {e}")
            return None
        self.log.info(f"SPEM parameters in {KOSMA_SPEM_FILE}: {spem_kosma}")
        return spem_kosma

    def copy_point_status(self):
        """
        Copy the astra pointing model KOSMA_point.in to KOSMA_point.status (as
        astra does at its start). Failures are logged but do not stop the translator.

        Returns
        -------
        bool
            True if KOSMA_point.status was written
        """
        try:
            copy_point_in_to_status(logger=self.log)
        except Exception as e:
            self.log.error(f"Failed to copy KOSMA_point.in to KOSMA_point.status: {e}")
            return False
        return True

    # =======================================================================
    # obs2tel input (main thread)
    # =======================================================================

    def check_for_obs2tel_update(self):
        # check modification time of obs2tel file and store

        current_mod_time = os.path.getmtime(f"{self.read_write_dir}/KOSMA_obs2tel.set")
        # on start up set mod time to 0 and return false
        if not hasattr(self, "obs2tel_mod_time"):
            self.obs2tel_mod_time = current_mod_time
            return False
        if current_mod_time > self.obs2tel_mod_time:
            self.obs2tel_mod_time = current_mod_time
            return True
        else:
            return False

    def read_obs2tel_file(self):
        #
        self.log.info("reading KOSMA_obs2tel.set file")
        self.kio_files = ImportKOSMAReadWriteIntoDictionary(
            files=["KOSMA_obs2tel.set"], variable=None, update_mod_time=True
        )
        self.obs_tolerance = self.kio_files["KOSMA_obs2tel.set"]["obs_tolerance"]
        self.log.info(f"obs_tolerance set to {self.obs_tolerance} arcseconds")
        self.next_tel_return_cookie = self.kio_files["KOSMA_obs2tel.set"]["obs_cookie"]
        self.obs_tel_info_update_time = self.kio_files["KOSMA_obs2tel.set"][
            "obs_tel_info_update_time"
        ]
        # read all variables from obs2tel file into the object
        self.obs2tel = self.kio_files["KOSMA_obs2tel.set"]

        # explicitly checks that the all required keys exist. Do not use a fallback to
        # an arbitrary default value
        required_keys = [
            "obs_cookie", "obs_coord_sys_on",
            "obs_lam_on", "obs_bet_on",
            "obs_pm_lam", "obs_pm_bet", "obs_paralx",
            "obs_extended_elevation_range",
            "obs_lam_del", "obs_bet_del", "obs_coord_sys_del", "obs_true_angle_del",
            "obs_lam_vel", "obs_bet_vel", "obs_otf_mode",
            "obs_lam_off", "obs_bet_off", "obs_coord_sys_off", "obs_true_angle_off",
            "obs_x_focal_plane", "obs_y_focal_plane",
            "obs_wavelength", "obs_tolerance",
            "obs_track_duration", "obs_start_time",
        ]
        missing = [k for k in required_keys if k not in self.obs2tel]
        if missing:
            self.log.error(f"Missing required obs2tel keys: {missing}")
            raise KeyError(f"Missing required obs2tel keys: {missing}")

    # =======================================================================
    # Command dispatch (main thread)
    # =======================================================================

    def process_obs2tel(self):
        """
        Read the changed KOSMA_obs2tel.set and act on it: ignore it if the
        cookie has not changed, keep the last path for the same target, or
        start a new command depending on obs_otf_mode.

        obs_otf_mode C (see section "OTF"): obs_track_duration > 0 is the OTF scan
        (OTFTP), never served by a kept path; obs_track_duration = 0 is the
        pre-OTF position (PRETP).

        Returns
        -------
        handled : bool
            True if a new cookie was handled (path reused or track started).
            False if nothing was done; the caller then waits before polling again.
        """
        self.read_obs2tel_file()
        if self.next_tel_return_cookie == self.tel_return_cookie:
            self.log.info(
                f"obs2tel cookie has not changed ({self.next_tel_return_cookie}), no action taken"
            )
            return False
        otf_mode = self.obs2tel["obs_otf_mode"]
        if otf_mode == "C" and self.obs2tel["obs_track_duration"] > 0:
            self.log.info("OTF scan command (OTFTP) received")
            self.track(self._send_otf_scan)
            return True
        if self.reuse_last_track():
            return True
        if otf_mode == "C":
            self.log.info("pre-OTF command (PRETP) received")
            if self.keep_path_for_pre_otf():
                return True
            self.track(self._send_pre_otf)
            return True
        if otf_mode == "P":
            self.log.info("OTF pattern mode (P) command received, not yet implemented")
            return False
        if otf_mode == "N":
            self.log.info("track mode command received")
            self.track()
            return True
        self.log.warning(f"unknown obs_otf_mode {otf_mode!r}, no action taken")
        return False

    # obs2tel variables that define the commanded path; obs2tel files that differ
    # only in other variables (e.g. the cookie) command the same path.
    # obs_otf_mode and obs_lam/bet_vel are needed although an OTF path is never
    # reused: the PRETP move-to goes to the run-up point R = S - vel * ramp/2,
    # not to the line start S its position keys describe, so a later track (N)
    # at S, or a PRETP with another velocity, must not reuse it.
    SAME_TARGET_KEYS = [
        "obs_lam_on", "obs_bet_on", "obs_coord_sys_on",
        "obs_lam_del", "obs_bet_del", "obs_coord_sys_del",
        "obs_lam_off", "obs_bet_off", "obs_coord_sys_off",
        "obs_extended_elevation_range",
        "obs_x_focal_plane", "obs_y_focal_plane",
        "obs_otf_mode", "obs_lam_vel", "obs_bet_vel",
        "obs_wavelength",   # the refraction depends on it
    ]
    # PRETP carries the same position (offset = line start) as the ON preset of
    # control_tel before it, but obs_otf_mode C and the scan velocity
    PRE_OTF_SAME_POSITION_KEYS = [
        k for k in SAME_TARGET_KEYS
        if k not in ("obs_otf_mode", "obs_lam_vel", "obs_bet_vel")
    ]

    def reuse_last_track(self):
        """
        If the new obs2tel commands the same target as the last accepted path
        and that path has at least reuse_min_remaining left, keep it instead of
        aborting and resending it (e.g. control_tel presets the next position,
        then commands the same position again with a new cookie).
        Returns True if the last path is reused.
        """
        if self.last_payload is None or self.last_obs2tel is None or self.tel_path_failed:
            return False
        if self.slew_in_progress:
            # the telescope has left the last path for a (superseded) slew
            return False
        if self.last_aborted:
            # the last path was aborted for a (superseded) newer command
            return False
        if self._track_thread and self._track_thread.is_alive():
            return False
        if self.last_payload["type"] == "otf":
            # an OTF scan path is only valid for its own start time
            return False
        if any(self.obs2tel[k] != self.last_obs2tel[k] for k in self.SAME_TARGET_KEYS):
            return False
        if self.last_payload["type"] != "move_to":
            last_end_time = self.last_payload["start_time"] + self.last_payload["points"][-1][0]
            if last_end_time - time.time() < self.reuse_min_remaining:
                return False

        self._keep_last_command("same target, last path kept")
        return True

    def keep_path_for_pre_otf(self):
        """
        PRETP: if the running tracking path is at the same position (the ON
        preset of control_tel at the line start, nextflg=Y), keep it and return
        the cookie at once. The OTF path (OTFTP) then starts from there.
        Returns True if the path is kept.
        """
        if self.last_payload is None or self.last_obs2tel is None or self.tel_path_failed:
            return False
        if self.slew_in_progress or self.last_aborted:
            return False
        if self._track_thread and self._track_thread.is_alive():
            return False
        # a HORIZON source is held by a move-to (fixed az/el, no end)
        horizon_source = COORD_SYS_MAP[self.obs2tel["obs_coord_sys_on"]] == "altaz"
        if self.last_payload["type"] != ("move_to" if horizon_source else "path"):
            return False
        if any(self.obs2tel[k] != self.last_obs2tel[k]
               for k in self.PRE_OTF_SAME_POSITION_KEYS):
            return False
        if not horizon_source:
            last_end_time = self.last_payload["start_time"] + self.last_payload["points"][-1][0]
            if last_end_time - time.time() < OTF_PRE_KEEP_MIN_REMAINING:
                return False

        self._keep_last_command("PRETP at the position of the running path, path kept")
        return True

    def _keep_last_command(self, reason):
        """
        Return the new cookie (next_tel_return_cookie) for the command that is
        already running on TCS.
        """
        cookie = self.next_tel_return_cookie
        self.last_obs2tel = self.obs2tel
        # on_track_not_before is kept: if the path has not started yet,
        # tel_on_track stays "N" until it has
        self.log.info(f"{reason}, returning cookie {cookie}")
        self.otf_scan_window = None
        self.slew_target = None
        self.cookie_reported = True
        self.old_tel_return_cookie = self.tel_return_cookie
        self.tel_return_cookie = cookie
        self.tel_lost_track = "N"    # initialize for the new cookie
        self.tel_was_on_track = False

    def track(self, send=None):
        """
        Start the track thread for the current obs2tel.

        Parameters
        ----------
        send : callable, optional
            send(cookie) computes and sends the command (default _send_track)
        """

        # 1. If a track is already running → stop it
        if self._track_thread and self._track_thread.is_alive():
            self.log.info("Stopping previous track before starting new one")
            self.stop_track()

        # 2. Reset stop event
        self._track_stop_event.clear()

        # 3. Start new thread
        self._track_thread = threading.Thread(
            target=self._track_loop,
            args=(send,),
            daemon=True
        )
        self._track_thread.start()

    def stop_track(self):
        if self._track_thread and self._track_thread.is_alive():
            self.log.info("Signaling track thread to stop")

            self._track_stop_event.set()   # signal thread

            self._track_thread.join()      # WAIT until thread exits

            self.log.info("Track thread stopped")

            self._track_thread = None

    # =======================================================================
    # Status output (background thread): KOSMA_tel2obs.set, KOSMA_ccdpos.sim
    # =======================================================================

    def write_tel2obs_file(self):
        input_dict, ccd_input_dict = self.set_tel2obs_dict()
        if input_dict is None:
            return

        self.log.debug(
            f"writing tel2obs file with obs cookie: {self.tel_return_cookie}"
        )
        write_atomic(
            f"{self.read_write_dir}/KOSMA_tel2obs.set", self.tel2obs.format(input_dict)
        )

        if ccd_input_dict:
            write_atomic(
                f"{self.read_write_dir}/KOSMA_ccdpos.sim", self.ccdpossim.format(ccd_input_dict)
            )

    def set_tel2obs_dict(self):
        """
        Set dictionary of KOSMA_tel2obs and KOSMA_ccdpos.sim. The latter is None
        for submm observations (obs_wavelength >= SUBMM_MIN_WAVELENGTH), for
        which KOSMA_ccdpos.sim is not written.
        """
        if (self.last_payload is None or self.last_obs2tel is None) and not self.cookie_reported:
            self.log.info('No record of successfully sent command to telescope. Wait...')
            return None, None

        response = self.ocs.get_status()
        response_pointing = self.ocs.get_dataset("CmdPointingCorrection")
        self._check_acu_refraction(response_pointing)

        input_dict = self._build_tel2obs_params(response, response_pointing)
        # KOSMA_ccdpos.sim is only for optical / infrared camera observations
        ccd_input_dict = None
        if self.obs2tel["obs_wavelength"] < SUBMM_MIN_WAVELENGTH:
            ccd_input_dict = self._build_ccdpos_params(input_dict, response_pointing)

        return input_dict, ccd_input_dict

    def _build_tel2obs_params(self, response: dict, response_pointing: dict) -> dict:
        """
        Derive the KOSMA_tel2obs parameter dictionary from a single ACU status
        response and the pointing corrections (dataset CmdPointingCorrection).
        """
        input_dict = {}

        # Fixed values
        input_dict["tel_latitude"] = self.location.lat.deg
        input_dict["tel_longitude"] = self.location.lon.deg
        input_dict["tel_altitude"] = self.location.height.to(u.m).value
        input_dict["tel_telescope"] = "CCAT"
        input_dict["tel_supports_ephemeris"] = "N"
        input_dict["tel_plate_scale"] = "13.89"
        # TODO: for the moment put the default constant value tel_plate_scale
        #       follow-up how this is used in combination of
        #       instrument plate scale factor

        # time stamp of the status and its relative time to the start of the last sent path
        status_time = _get_time_from_acu_format(response["Time"], response["Year"])

        # Only the track thread sets on_track_not_before, otf_scan_window and
        # slew_target; read them once here
        otf_scan_window = self.otf_scan_window
        on_track_not_before = self.on_track_not_before
        slew_target = self.slew_target

        if slew_target is not None:
            # slewing to the start of the path of the returned cookie: last_payload
            # is the previous command (or None), not related to this cookie
            t_rel = status_time.unix - slew_target[2]
        elif self.last_payload is None:
            # no path accepted yet; only reached when the first path failed
            t_rel = 0.0
        else:
            t_rel = status_time.unix - self.last_payload["start_time"]
        path_started = status_time.unix >= on_track_not_before
        if (path_started and np.isfinite(on_track_not_before)
                and on_track_not_before != self._logged_on_track_not_before):
            self._logged_on_track_not_before = on_track_not_before
            self.log.info(
                f"path started (t_rel={t_rel:.2f} s), "
                f"tel_on_track evaluated for cookie {self.tel_return_cookie}"
            )

        # values set elsewhere
        input_dict["tel_return_cookie"] = self.tel_return_cookie
        input_dict["tel_error"] = self.tel_error
        input_dict["tel_pos_in_range"] = "Y"     # practically not used

        current_azi = response["Azimuth current position"]
        current_ele = response["Elevation current position"]

        # The ACU "commanded position" is not used.
        # cmd_azi = response["Azimuth commanded position"]
        # cmd_ele = response["Elevation commanded position"]
        # acu_azm_del = current_azi - cmd_azi
        # acu_elv_del = current_ele - cmd_ele

        # Use delta reported by ACU instead
        # KOSMA expects the positions of a perfect telescope: with refraction,
        # without the pointing model. The ACU current position is the encoder position:
        #   current = path position + total pointing correction + tracking error
        #   tel_*_cmd = path position (interpolated from the last payload)
        #               including refraction correction
        #   tel_*_act = current - total pointing correction
        del_tot_azm, del_tot_elv = _total_pointing_correction(response_pointing)
        input_dict["tel_azm_act"] = current_azi - del_tot_azm
        input_dict["tel_elv_act"] = current_ele - del_tot_elv
        # not in KOSMA_tel2obs.set; for KOSMA_ccdpos.sim
        input_dict["slewing"] = slew_target is not None
        if slew_target is not None:
            # the slew target (start of the path), constant during the slew
            input_dict["tel_azm_cmd"] = slew_target[0]
            input_dict["tel_elv_cmd"] = slew_target[1]
            input_dict["tel_vel_elv_cmd"] = 0.0
        elif self.last_payload is None:
            # no path to interpolate (only when the first path failed):
            # report the current position; tel_on_track is "N" anyway
            input_dict["tel_azm_cmd"] = input_dict["tel_azm_act"]
            input_dict["tel_elv_cmd"] = input_dict["tel_elv_act"]
            # not in KOSMA_tel2obs.set; used for op_vel_elv in KOSMA_ccdpos.sim [deg/s]
            input_dict["tel_vel_elv_cmd"] = 0.0
        else:
            pts = np.array(self.last_payload["points"])   # shape (N, 5): dt, az, el, vaz, vel
            # TODO: if the refraction correction is done at OCS or ACU, corresponding
            # correction has to be added here. If it is done at translator,
            # the path already includes the refraction correction.
            input_dict["tel_azm_cmd"] = np.interp(t_rel, pts[:, 0], pts[:, 1])
            input_dict["tel_elv_cmd"] = np.interp(t_rel, pts[:, 0], pts[:, 2])
            # interpolated here, from the same payload as tel_elv_cmd
            input_dict["tel_vel_elv_cmd"] = np.interp(t_rel, pts[:, 0], pts[:, 4])
        # tracking error
        acu_azm_del = input_dict["tel_azm_act"] - input_dict["tel_azm_cmd"]
        acu_elv_del = input_dict["tel_elv_act"] - input_dict["tel_elv_cmd"]

        self.log.debug(
            f'expected az/el = {input_dict["tel_azm_cmd"]}, {input_dict["tel_elv_cmd"]} current az/el = {input_dict["tel_azm_act"]}, {input_dict["tel_elv_act"]}'
        )

        actdiff = 3600 * (
            (acu_azm_del*np.cos(np.deg2rad(input_dict["tel_elv_cmd"])))**2 + acu_elv_del**2
        ) ** 0.5

        if self.tel_path_failed:
            # the ACU is not executing the commanded path
            input_dict["tel_on_track"] = "N"
        elif otf_scan_window is not None:
            # OTF scan: "N" before and after the scan (obs2tel definition of
            # obs_start_time); within it any status off the path is lost track,
            # also before the first "Y" (the data are taken from T0 on)
            if not otf_scan_window[0] <= status_time.unix <= otf_scan_window[1]:
                input_dict["tel_on_track"] = "N"
            elif actdiff < self.obs_tolerance:
                input_dict["tel_on_track"] = "Y"
                self.tel_was_on_track = True
            else:
                input_dict["tel_on_track"] = "N"
                self.tel_lost_track = "Y"
        elif not path_started:
            # still on the previous target (or slewing): not on track, not lost
            input_dict["tel_on_track"] = "N"
        elif actdiff < self.obs_tolerance:
            input_dict["tel_on_track"] = "Y"
            self.tel_was_on_track = True
        else:
            input_dict["tel_on_track"] = "N"
            if self.tel_was_on_track:
                # was on track for this cookie and left it; not while settling
                self.tel_lost_track = "Y"
        input_dict["tel_lost_track"] = self.tel_lost_track

        # velocity
        vel_azi = response["Azimuth current velocity"]
        vel_ele = response["Elevation current velocity"]
        input_dict["tel_vel_azm"] = vel_azi * 3600.    # deg/s to arcsec/s
        input_dict["tel_vel_elv"] = vel_ele * 3600.    # deg/s to arcsec/s
        
        # commanded and actual positions and timestamp
        input_dict["tel_time_act"] = status_time.unix
        input_dict["tel_mjd1"] = status_time.ut1.mjd

        # time of the status since the start time of the table sent, for an OTF
        # scan since the scan start T0
        if otf_scan_window is not None:
            input_dict["tel_time_del"] = status_time.unix - otf_scan_window[0]
        else:
            input_dict["tel_time_del"] = t_rel

        # Calculate angle between coordinate systems. While slewing,
        # last_obs2tel is the previous command: no angle reported.
        if (slew_target is not None or self.last_obs2tel is None
                or self.last_obs2tel["obs_coord_sys_del"] == "HORIZON"):
            input_dict["tel_angle_focal_plane"] = 0.0
        else:
            cmd_coord_sys_del = self.last_obs2tel["obs_coord_sys_del"]
            # Calculation is needed:
            # Calculate north (+epsilon in the second axis cmd_coord_sys_del) in Horizon
            # then csys_del -> horizon (opposite sign)

            altaz_frame = self._altaz_frame(status_time)

            # In the extended range tel_*_cmd is the over-the-top position
            # (elevation > 90 deg). The angle is computed at the nominal sky
            # position, as astra computes a_parang before its extended-range
            # conversion (astra_track.c).
            azm_cmd, elv_cmd = input_dict["tel_azm_cmd"], input_dict["tel_elv_cmd"]
            if elv_cmd > 90.0:
                azm_cmd, elv_cmd = (azm_cmd + 180.0) % 360.0, 180.0 - elv_cmd

            pos = SkyCoord(az=azm_cmd, alt=elv_cmd, unit='deg', frame=altaz_frame)

            epsilon = 1e-3  # about 4 arcsec, small angle approximation is sufficient
            north = _apply_offset(
                pos.transform_to(COORD_SYS_MAP[cmd_coord_sys_del]), 0, epsilon, 'Y'
            ).transform_to(altaz_frame)

            angle = np.arctan2(
                north.alt.deg - pos.alt.deg,
                (north.az.deg - pos.az.deg) * np.cos(pos.alt.rad)
            )
            tel_angle_focal_plane = np.rad2deg(angle) - 90.   # from second axis in degrees
            if tel_angle_focal_plane < -180:
                tel_angle_focal_plane += 360.

            input_dict["tel_angle_focal_plane"] = tel_angle_focal_plane * (-1)

        #self.log.info(f"on track: {input_dict['tel_on_track']} az:{input_dict['tel_azm_act']:.4f} ele:{input_dict['tel_elv_act']:.4f} cmd_az:{input_dict['tel_azm_cmd']:.4f} cmd_ele:{input_dict['tel_elv_cmd']:.4f} obs_tolerance: {self.obs_tolerance/3600.0:.4f} deg ")

        fmt = "%d-%b-%Y  %H:%M:%S"
        current = time.localtime()
        input_dict["timestamp"] = time.time()
        input_dict["timestring"] = time.strftime(fmt, current)
        input_dict["host_pid"] = self._host_pid

        return input_dict

    def _build_ccdpos_params(self, input_dict: dict, response_pointing: dict) -> dict:
        """
        Derive the KOSMA_ccdpos.sim parameter dictionary.
        Reuses tel_azm_cmd / tel_elv_cmd from the already-computed input_dict,
        as well as timestamp and timestring
        """
        spem_azm = response_pointing["SPEM correction AZ"]
        spem_elv = response_pointing["SPEM correction EL"]
        del_tot_azm, del_tot_elv = _total_pointing_correction(response_pointing)

        ccd_input_dict = {}
        ccd_input_dict["op_azm"] = np.deg2rad(input_dict["tel_azm_cmd"])
        ccd_input_dict["op_elv"] = np.deg2rad(input_dict["tel_elv_cmd"])
        # commanded (path) elevation velocity, deg/s -> arcsec/s, as in astra_track
        ccd_input_dict["op_vel_elv"] = input_dict["tel_vel_elv_cmd"] * 3600.0

        # last_obs2tel is None if the very first path failed, and the previous
        # command while slewing
        if (not input_dict["slewing"] and self.last_obs2tel is not None
                and self.last_obs2tel["obs_coord_sys_del"] == "HORIZON"):
            # obs_lam_del, obs_bet_del are in arcsec
            ccd_input_dict["op_oazm"] = np.deg2rad(self.last_obs2tel["obs_lam_del"] / 3600.0)
            ccd_input_dict["op_oelv"] = np.deg2rad(self.last_obs2tel["obs_bet_del"] / 3600.0)
        else:
            # these values are irrelevant if not Horizon
            ccd_input_dict["op_oazm"] = 0
            ccd_input_dict["op_oelv"] = 0

        model_type, model_date = self._pointing_model_info()
        ccd_input_dict["op_tel_model_type"] = model_type
        ccd_input_dict["op_tel_model_date"] = model_date

        # The ACU azimuth corrections are axis (encoder) differences, while KOSMA
        # expects them on sky: astra applies del_azm / cos(el) to the axis
        cos_elv = np.cos(ccd_input_dict["op_elv"])
        ccd_input_dict["del_tel_tot_azm"] = np.deg2rad(del_tot_azm) * cos_elv
        ccd_input_dict["del_tel_tot_elv"] = np.deg2rad(del_tot_elv)
        ccd_input_dict["del_tel_rep_azm"] = np.deg2rad(spem_azm) * cos_elv
        ccd_input_dict["del_tel_rep_elv"] = np.deg2rad(spem_elv)

        ccd_input_dict["timestamp"] = input_dict["timestamp"]
        ccd_input_dict["timestring"] = input_dict["timestring"]
        ccd_input_dict["host_pid"] = input_dict["host_pid"]

        return ccd_input_dict

    def _pointing_model_info(self):
        """
        Pointing model type and date applied by the ACU, for op_tel_model_type /
        op_tel_model_date in KOSMA_ccdpos.sim. Copied from ccatacu_model_type /
        ccatacu_model_date in KOSMA_ccatacu_point.in. The file is re-read only when its
        modification time changes. A missing file or entry is an error: the
        POINTING_MODEL_MISSING marker is used for it, which makes starpoint abort.

        Returns
        -------
        model_type, model_date : str
        """
        path = f"{self.read_write_dir}/{KOSMA_SPEM_FILE}"
        try:
            st = os.stat(path)
            key = (st.st_mtime_ns, st.st_size)
        except FileNotFoundError:
            key = "missing"
        if key == self._model_info_key:
            return self._model_info

        self._model_info_key = key
        if key == "missing":
            self.log.error(
                f"{path} not found, writing pointing model type/date "
                f"{POINTING_MODEL_MISSING} to KOSMA_ccdpos.sim (starpoint will abort)"
            )
            self._model_info = POINTING_MODEL_MISSING
            return self._model_info

        found = read_first_column(path, ["ccatacu_model_type", "ccatacu_model_date"])
        missing = [k for k in ("ccatacu_model_type", "ccatacu_model_date") if k not in found]
        if missing:
            self.log.error(
                f"{', '.join(missing)} not found in {path}, writing "
                f"{POINTING_MODEL_MISSING} for it to KOSMA_ccdpos.sim (starpoint will abort)"
            )
        self._model_info = (
            found.get("ccatacu_model_type", POINTING_MODEL_MISSING[0]),
            found.get("ccatacu_model_date", POINTING_MODEL_MISSING[1]),
        )
        self.log.info(
            f"pointing model for KOSMA_ccdpos.sim: type={self._model_info[0]} "
            f"date={self._model_info[1]}"
        )
        return self._model_info

    # =======================================================================
    # Tracking (track thread): compute the command for obs2tel and send it
    # =======================================================================

    def _track_loop(self, send=None):
        """
        Thread target: send(cookie), default _send_track. Any unexpected
        exception fails the path, so that the cookie is still returned to the
        observer.
        """
        if send is None:
            send = self._send_track
        # the cookie this thread works for; next_tel_return_cookie may already be
        # overwritten by a newer obs2tel while this thread is being stopped
        cookie = self.next_tel_return_cookie
        try:
            send(cookie)
        except Exception as e:
            self.log.exception(f"Track error: {e}")
            # a failed OTF scan must flag the data taken by the backend
            self._fail_path(cookie, 1, lost_track="Y" if send == self._send_otf_scan else "N")

    def _validate_coord_sys(self, obs2tel):
        """Raise ValueError if a coordinate system in obs2tel is unknown."""
        for label in ("obs_coord_sys_on", "obs_coord_sys_del", "obs_coord_sys_off"):
            csys = obs2tel[label]
            if csys not in COORD_SYS_MAP:
                msg = f"Unknown coordinate system {label}={csys!r}, must be one of {list(COORD_SYS_MAP)}"
                self.log.error(msg)
                raise ValueError(msg)

    def _send_track(self, cookie):
        """
        Send the command for the current obs2tel to TCS: a move-to for a HORIZON
        source (_send_move_to), otherwise a path (_send_path).

        This routine is called every time when KOSMA_obs2tel.set is updated. If a
        track is already running, the previous one is stopped and the new one starts.
        Therefore practically the duration of the track in this routine is overwritten
        by a new track triggered by the KOSMA_obs2tel update.
        This justifies to use the fixed obs_x/y_focal_plane, which is a function of
        elevation and changes over time: once the value changes more than a tolerance,
        a new track is triggered with updated obs_x/y_focal_plane values.

        obs2tel variables used:
        - obs_lam_on, obs_bet_on [deg], obs_coord_sys_on: commanded position
        - obs_lam_del, obs_bet_del [arcsec], obs_coord_sys_del, obs_true_angle_del (Y/N):
          sky/map offsets, applied before the AltAz transform unless HORIZON frame
        - obs_lam_off, obs_bet_off [arcsec], obs_coord_sys_off, obs_true_angle_off (Y/N):
          off-position offsets, applied before the AltAz transform unless HORIZON frame
        - obs_x_focal_plane, obs_y_focal_plane [arcsec]: focal plane offsets, always
          applied after the AltAz transform, always with cos-correction (true_angle=Y).
          For CCAT obs_coord_sys_focal_plane is always set to HORIZON.
        - obs_pm_lam, obs_pm_bet [arcsec/yr], obs_paralx [arcsec]: proper motion and
          parallax of the source (path only). obs_pm_lam/obs_pm_bet are raw coordinate
          rates dlam/dt, not cos(bet)*dlam/dt; slalib h2fk5.c note 1: "proper motions
          in RA are dRA/dt rather than cos(Dec)*dRA/dt"
        - obs_extended_elevation_range (Y/N)
        - obs_wavelength [micron]: used for the refraction correction
        """
        self._validate_coord_sys(self.obs2tel)

        # get weather
        self._update_refraction_params(self.obs2tel["obs_wavelength"])

        # If coordinate system of ON is Horizon, use move-to.
        # For this case, it does not make much sense if map or off offset is given
        # by other coordinate system but such a case is also covered in _send_move_to.
        if COORD_SYS_MAP[self.obs2tel["obs_coord_sys_on"]] == "altaz":
            self._send_move_to(cookie)
        else:
            self._send_path(cookie)

    def _send_move_to(self, cookie):
        """
        HORIZON source: apply the offsets to the AltAz position now and command
        it with a move-to (constant position, no path). See _send_track for the
        obs2tel variables.
        """
        obs2tel = self.obs2tel
        cmd_coord_sys_on = obs2tel["obs_coord_sys_on"]
        focal_plane_az, focal_plane_el = obs2tel["obs_x_focal_plane"], obs2tel["obs_y_focal_plane"]
        if_extended = obs2tel["obs_extended_elevation_range"] == "Y"
        track_duration = self.track_duration  # seconds

        coord = self._offset_horizon_source(
            obs2tel, self._altaz_frame(Time.now()),
            obs2tel["obs_lam_del"], obs2tel["obs_bet_del"])

        self.log.info(
            f"tracking to {coord.az.deg} {coord.alt.deg} in {cmd_coord_sys_on} frame"
        )

        # Apply focal plane offsets
        if focal_plane_az != 0.0 or focal_plane_el != 0.0:
            coord = _apply_offset(coord, focal_plane_az, focal_plane_el, "Y")

        # command move-to
        if if_extended:
            cmd_el, cmd_az = get_overtop_altaz_from_nominal(
                alt_nominal=coord.alt.deg, az_nominal=coord.az.deg)
        else:
            cmd_az = coord.az.deg
            cmd_el = coord.alt.deg
        self._send_move_to_azel(cookie, cmd_az, cmd_el, obs2tel, track_duration)

    def _offset_horizon_source(self, obs2tel, altaz_frame, lam_del, bet_del):
        """
        HORIZON source of obs2tel (obs_lam/bet_on as az/el) in altaz_frame,
        with the sky/map and off-position offsets applied (focal plane offset
        not). A non-HORIZON offset is applied in its frame at the obstime of
        altaz_frame. See _send_track for the obs2tel variables.

        Parameters
        ----------
        obs2tel : dict
        altaz_frame : astropy.coordinates.AltAz
            Its obstime may be an array (OTF): the result has its shape
        lam_del, bet_del : float or np.ndarray
            Sky/map offset [arcsec] in obs_coord_sys_del, used instead of
            obs_lam_del / obs_bet_del (arrays for OTF)

        Returns
        -------
        coord : astropy.SkyCoord
            in altaz_frame
        """
        shape = np.shape(altaz_frame.obstime)
        coord_sys_del, true_angle_del = obs2tel["obs_coord_sys_del"], obs2tel["obs_true_angle_del"]
        lam_off, bet_off = obs2tel["obs_lam_off"], obs2tel["obs_bet_off"]
        coord_sys_off, true_angle_off = obs2tel["obs_coord_sys_off"], obs2tel["obs_true_angle_off"]

        az_on, el_on = float(obs2tel["obs_lam_on"]), float(obs2tel["obs_bet_on"])
        if shape:   # one position per obstime
            az_on, el_on = np.full(shape, az_on), np.full(shape, el_on)
        coord = SkyCoord(az_on * u.deg, el_on * u.deg, frame=altaz_frame)
        if np.any(lam_del != 0.0) or np.any(bet_del != 0.0):
            if coord_sys_del == "HORIZON":
                coord = _apply_offset(coord, lam_del, bet_del, true_angle_del)
            else:
                coord = _apply_offset(
                    coord.transform_to(COORD_SYS_MAP[coord_sys_del]),
                    lam_del, bet_del, true_angle_del,
                ).transform_to(altaz_frame)
        if lam_off != 0.0 or bet_off != 0.0:
            if coord_sys_off == "HORIZON":
                coord = _apply_offset(coord, lam_off, bet_off, true_angle_off)
            else:
                coord = _apply_offset(
                    coord.transform_to(COORD_SYS_MAP[coord_sys_off]),
                    lam_off, bet_off, true_angle_off,
                ).transform_to(altaz_frame)
        return coord

    def _send_move_to_azel(self, cookie, cmd_az, cmd_el, obs2tel, track_duration):
        """
        Abort the previous command if needed and command a move-to to (cmd_az,
        cmd_el) [deg, Horizon, as sent to TCS]. On acceptance, record a constant
        bookkeeping payload of track_duration seconds and return the cookie,
        with tel_on_track evaluated against this position from now on.

        Parameters
        ----------
        obs2tel : dict
            obs2tel the move-to is sent for (recorded as last_obs2tel)
        """
        if self._track_stop_event.is_set():
            self.log.info("Track superseded, move-to not sent")
            return
        # ocs.move_to does not abort the previous command: do it here
        if not self._abort_previous(cookie):
            return
        response = self._move_to_with_retry(cmd_az, cmd_el)
        if response is None:
            self.log.info("Track superseded while TCS was busy, move-to not sent")
            return
        body = _response_body(response)
        if response.status_code != 200 or body.get("status") != "ok":
            self.log.error(
                f"move_to rejected: HTTP {response.status_code}, "
                f"status={body.get('status')}, message={body.get('message')}"
            )
            self._fail_path(cookie, _error_code(response))
        else:
            self.log.info(f"move_to accepted: {body}")
            self.last_aborted = False
            # Bookkeeping only (not sent): the ACU starts moving on receipt
            # and holds the position, so the commanded position is constant.
            self.last_payload = {
                "type": "move_to",
                "start_time": time.time(),
                "coordsys": "Horizon",
                "points": [[0.0, cmd_az, cmd_el, 0.0, 0.0],
                           [float(track_duration), cmd_az, cmd_el, 0.0, 0.0]],
            }
            self.last_obs2tel = obs2tel
            self._return_cookie(cookie, self.last_payload["start_time"])

    def _send_path(self, cookie):
        """
        Non-HORIZON source: compute a path (scan_pattern) of track_duration
        seconds, slew to its start first if the telescope is far from it, and
        send it. See _send_track for the obs2tel variables.
        """
        obs2tel = self.obs2tel
        track_duration = self.track_duration  # seconds
        coord, horizon_offsets = self._offset_source(
            obs2tel, obs2tel["obs_lam_del"], obs2tel["obs_bet_del"])
        self.log.info(
            f"tracking to {coord.data.lon.deg} {coord.data.lat.deg} "
            f"in {obs2tel['obs_coord_sys_on']} frame"
        )
        target = PathTarget(
            coord=coord, horizon_offsets=horizon_offsets,
            extended=obs2tel["obs_extended_elevation_range"] == "Y")
        self._send_path_target(cookie, target, track_duration, obs2tel)

    def _offset_source(self, obs2tel, lam_del, bet_del):
        """
        Non-HORIZON source of obs2tel with proper motion and parallax, and
        the non-HORIZON sky/map and off-position offsets applied (before the
        AltAz transform). See _send_track for the obs2tel variables.

        Parameters
        ----------
        obs2tel : dict
        lam_del, bet_del : float or np.ndarray
            Sky/map offset [arcsec] in obs_coord_sys_del, used instead of
            obs_lam_del / obs_bet_del (arrays for OTF give an array coordinate)

        Returns
        -------
        coord : astropy.SkyCoord
            in the frame of obs_coord_sys_on
        horizon_offsets : tuple of (d_lam_arcsec, d_bet_arcsec, true_angle)
            Offsets to apply in AltAz after the transform, in this order:
            HORIZON sky/map offset, HORIZON off-position offset, focal plane
            offset (always true angle). Zero offsets are left out.
        """
        cmd_lam, cmd_bet = obs2tel["obs_lam_on"], obs2tel["obs_bet_on"]
        cmd_coord_sys_on = obs2tel["obs_coord_sys_on"]
        frame = COORD_SYS_MAP[cmd_coord_sys_on]
        coord_sys_del, true_angle_del = obs2tel["obs_coord_sys_del"], obs2tel["obs_true_angle_del"]
        lam_off, bet_off = obs2tel["obs_lam_off"], obs2tel["obs_bet_off"]
        coord_sys_off, true_angle_off = obs2tel["obs_coord_sys_off"], obs2tel["obs_true_angle_off"]
        focal_plane_az, focal_plane_el = obs2tel["obs_x_focal_plane"], obs2tel["obs_y_focal_plane"]
        pm_lam, pm_bet, paralx = obs2tel["obs_pm_lam"], obs2tel["obs_pm_bet"], obs2tel["obs_paralx"]
        del_nonzero = bool(np.any(lam_del != 0.0) or np.any(bet_del != 0.0))

        # non-HORIZON source: make into an astropy coordinate object,
        # applying proper motion and parallax propagation when non-zero.
        if pm_lam != 0.0 or pm_bet != 0.0 or paralx != 0.0:
            # Convert raw coordinate rate to true angular rate for astropy 
            # (multiply by cos(beta))
            cos_bet = np.cos(np.deg2rad(cmd_bet))
            pm_lon_key, pm_lat_key = _FRAME_PM_KEYS[frame]
            pm_kwargs = {
                pm_lon_key: pm_lam * cos_bet * u.arcsec/u.yr,
                pm_lat_key: pm_bet * u.arcsec/u.yr,
                "obstime": _FRAME_EPOCHS[frame],
            }
            if paralx != 0.0:
                pm_kwargs["distance"] = Distance(parallax=paralx * u.arcsec)
            coord_with_pm = SkyCoord(
                cmd_lam * u.deg, cmd_bet * u.deg, frame=frame, **pm_kwargs)
            # Propagate to current epoch; proper motion over the 600 s track window is
            # negligible so one propagation at the start of the window suffices
            propagated = coord_with_pm.apply_space_motion(new_obstime=Time.now())
            coord = SkyCoord(propagated.data.lon, propagated.data.lat, frame=frame)
        else:
            coord = SkyCoord(cmd_lam * u.deg, cmd_bet * u.deg, frame=frame)

        # Apply (non-Horizon) sky/map offsets in obs_coord_sys_del frame 
        # before AltAz transform
        if coord_sys_del != "HORIZON" and del_nonzero:
            if coord_sys_del == cmd_coord_sys_on:
                coord = _apply_offset(coord, lam_del, bet_del, true_angle_del)
            else:
                coord = _apply_offset(
                    coord.transform_to(COORD_SYS_MAP[coord_sys_del]),
                    lam_del, bet_del, true_angle_del,
                ).transform_to(frame)

        # Apply (non-Horizon) off-position offsets in obs_coord_sys_off frame 
        # before AltAz transform
        if coord_sys_off != "HORIZON" and (lam_off != 0.0 or bet_off != 0.0):
            if coord_sys_off == cmd_coord_sys_on:
                coord = _apply_offset(coord, lam_off, bet_off, true_angle_off)
            else:
                coord = _apply_offset(
                    coord.transform_to(COORD_SYS_MAP[coord_sys_off]),
                    lam_off, bet_off, true_angle_off,
                ).transform_to(frame)

        # Offsets applied in AltAz after the transform: HORIZON sky/map offset,
        # HORIZON off-position offset, focal plane offset (always true angle)
        horizon_offsets = []
        if coord_sys_del == "HORIZON" and del_nonzero:
            horizon_offsets.append((lam_del, bet_del, true_angle_del))
        if coord_sys_off == "HORIZON" and (lam_off != 0.0 or bet_off != 0.0):
            horizon_offsets.append((lam_off, bet_off, true_angle_off))
        if focal_plane_az != 0.0 or focal_plane_el != 0.0:
            horizon_offsets.append((focal_plane_az, focal_plane_el, "Y"))
        return coord, tuple(horizon_offsets)

    def _send_path_target(self, cookie, target, track_duration, obs2tel):
        """
        Compute the tracking path for target, slew to its start first if the
        telescope is far from it, and send it (see _send_path). obs2tel is the
        one the path is sent for (recorded as last_obs2tel).
        """
        # Abort the previous path first, if it has not ended yet.
        # This must come before time_array is built: TCS rejects a path starting
        # less than 9.8 sec from its arrival, so the abort must not eat into the
        # overhead below. The telescope is not tracking from here until the new
        # path starts.
        if self._track_stop_event.is_set():
            self.log.info("Track superseded, previous path not aborted")
            return
        if not self._abort_previous(cookie):
            return

        payload = self._build_path_payload(target, track_duration)

        # If the telescope is far from the start of the path, slew there first,
        # then recompute the path so that it starts track_overhead after arrival.
        first_az, first_el = payload["points"][0][1], payload["points"][0][2]
        current_az, current_el, _, _ = self._acu_position()
        distance = max(abs(first_az - current_az), abs(first_el - current_el))
        if distance >= SLEW_BEFORE_PATH_MIN_DIST:
            self.log.info(
                f"telescope at az={current_az:.4f} el={current_el:.4f} is {distance:.3f} deg "
                f"(>= {SLEW_BEFORE_PATH_MIN_DIST} deg) from the path start "
                f"az={first_az:.4f} el={first_el:.4f}: slewing there first"
            )
            if not self._slew_to(first_az, first_el, cookie):
                return
            payload = self._build_path_payload(target, track_duration)
            # TCS may still be finishing the move-to: resend busy-rejected paths
            resend_deadline = time.monotonic() + SLEW_BUSY_RESEND_TIMEOUT
        else:
            resend_deadline = time.monotonic()   # no resending

        try:
            while True:
                # The computation (or anything else since) may have used up the margin:
                # recompute once, with the overhead increased by the shortfall.
                lead = payload["start_time"] - time.time()
                if lead < TRACK_MIN_SEND_LEAD:
                    overhead = TRACK_OVERHEAD + (TRACK_MIN_SEND_LEAD - lead)
                    self.log.warning(
                        f"start_time lead {lead:.3f} s < {TRACK_MIN_SEND_LEAD} s: "
                        f"recomputing path with track_overhead {overhead:.3f} s"
                    )
                    payload = self._build_path_payload(target, track_duration, overhead)

                # TODO: find out a way to pass obs_wavelength together with payload
                # A newer track has superseded this one while the payload was computed
                if self._track_stop_event.is_set():
                    self.log.info("Track superseded, payload not sent")
                    return
                # TCS answers 503 (busy) until it is idle after the abort. Retry only
                # while the lead stays >= TRACK_MIN_SEND_LEAD; a later retry would be
                # rejected as starting too soon anyway.
                lead_send = payload["start_time"] - time.time()
                response = self._scan_pattern_with_retry(
                    payload, busy_timeout=max(lead_send - TRACK_MIN_SEND_LEAD, 0.0),
                    retry_interval=0.3
                )
                if response is None:
                    self.log.info("Track superseded while TCS was busy, payload not sent")
                    return
                lead_done = payload["start_time"] - time.time()
                self.log.info(
                    f"start_time lead: {lead_send:.3f} s before send, "
                    f"{lead_done:.3f} s after response (TCS needs >= 9.8 s)"
                )
                if _is_busy(response) and time.monotonic() < resend_deadline:
                    # after a slew: TCS still finishing the move-to
                    self.log.warning(
                        "TCS still busy after the slew: recomputing path and resending"
                    )
                    payload = self._build_path_payload(target, track_duration)
                    continue
                break

            body = _response_body(response)
            if response.status_code != 200 or body.get("status") != "ok":
                self.log.error(
                    f"scan_pattern rejected: HTTP {response.status_code}, "
                    f"status={body.get('status')}, message={body.get('message')}"
                )
                self._fail_path(cookie, _error_code(response))
            else:
                self.log.info(f"scan_pattern accepted: {body}")
                self.last_aborted = False
                # "type" only for bookkeeping, not part of the payload sent to TCS
                self.last_payload = {**payload, "type": "path"}
                self.last_obs2tel = obs2tel
                # Until start_time the ACU still holds the previous target:
                # tel_on_track stays "N" until then.
                self._return_cookie(cookie, payload["start_time"])
        except Exception as e:
            self.log.error(f"Track error: {e}")
            self._fail_path(cookie, 1)

    def _build_path_payload(self, target, track_duration, track_overhead=None):
        """
        Path for scan_pattern of track_duration seconds, starting track_overhead
        seconds from now.

        Parameters
        ----------
        target : PathTarget
            Source position and the offsets applied after the AltAz transform
        track_duration : float
            Length of the path [sec]
        track_overhead : float, optional
            start_time of the path is this far after now [sec].
            Default TRACK_OVERHEAD (read at call time).

        Returns
        -------
        payload : dict
            start_time (unix), coordsys and points [[t, az, el, v_az, v_el], ...]
        """
        if track_overhead is None:
            track_overhead = TRACK_OVERHEAD
        # Points every TRACK_TIME_STEP from now + overhead to now + overhead + track_duration.
        # If track_duration is not a multiple of the step, the path ends at the
        # last full step before track_duration. The small tolerance avoids
        # dropping a step due to floating-point error (e.g. 600 / 0.1).
        n_steps = int(track_duration / TRACK_TIME_STEP + 1e-9) + 1
        t_fine = np.arange(n_steps) * TRACK_TIME_STEP
        # The full transform runs only on a coarse grid covering the path
        # (at least 3 nodes for the second-order slopes)
        n_coarse = max(int(np.ceil(t_fine[-1] / TRACK_COARSE_STEP - 1e-9)) + 1, 3)
        t_coarse = np.arange(n_coarse) * TRACK_COARSE_STEP
        t_build = time.perf_counter()
        t_start = Time.now() + track_overhead * u.second
        time_array = t_start + t_coarse * u.second

        # calculate the altaz coordinates for each time step
        altaz_frames = self._altaz_frame(time_array)
        altaz = target.coord.transform_to(altaz_frames)

        # Apply HORIZON-frame sky/map, off-position and focal plane offsets
        for d_lam, d_bet, true_angle in target.horizon_offsets:
            altaz = _apply_offset(altaz, d_lam, d_bet, true_angle)

        if target.extended:
            el_coarse, az_coarse = get_overtop_altaz_from_nominal(
                alt_nominal=altaz.alt.deg, az_nominal=altaz.az.deg)
        else:
            az_coarse = altaz.az.deg
            el_coarse = altaz.alt.deg

        # Interpolate to the path points; positions and velocities from the
        # same cubic. Azimuth is unwrapped so a 0/360 crossing is interpolated
        # (and its velocity computed) correctly, then wrapped back to [0, 360).
        # NaN (unreachable overtop) would spread through the unwrap: skip it.
        if np.all(np.isfinite(az_coarse)):
            az_coarse = np.unwrap(az_coarse, period=360.0)
        az_array, az_velocities = _hermite_resample(t_coarse, az_coarse, t_fine)
        az_array = np.mod(az_array, 360.0)
        el_array, el_velocities = _hermite_resample(t_coarse, el_coarse, t_fine)

        # program track mode, see definition here ICD-1000000-32000-02-00 VA Webserver
        # - Remote Protocol
        # Assuming all arrays are 1D and of the same length
        points = np.column_stack(
            [
                t_fine,  # time offset in seconds since start of track
                az_array,
                el_array,
                az_velocities,
                el_velocities,
            ]
        )
        payload = {
            "start_time": t_start.unix,
            "coordsys": "Horizon",
            "points": points.tolist(),
        }
        self.log.info(f"payload computed in {time.perf_counter() - t_build:.3f} s")
        return payload

    # =======================================================================
    # TCS communication (track thread): slew, abort, send with busy retry
    # =======================================================================

    def _acu_position(self):
        """
        ACU current position minus the total pointing correction, i.e. comparable
        with path / move-to positions, and the current axis velocities.

        Returns
        -------
        az, el : float
            [deg]
        vel_az, vel_el : float
            [deg/s]
        """
        status = self.ocs.get_status()
        del_tot_azm, del_tot_elv = _total_pointing_correction(
            self.ocs.get_dataset("CmdPointingCorrection"))
        return (
            status["Azimuth current position"] - del_tot_azm,
            status["Elevation current position"] - del_tot_elv,
            status["Azimuth current velocity"],
            status["Elevation current velocity"],
        )

    def _slew_to(self, az, el, cookie):
        """
        Slew the telescope to (az, el) [deg, Horizon], where a path is going to
        start, and wait until it has arrived.

        For now this is a plain move-to. The slew is isolated here so that it can
        later be replaced by a slew path avoiding the sun
        (sun_avoidance/slew.py) without touching _send_track.

        Returns True once the telescope has arrived. Returns False if the track
        was superseded, or if the slew failed (then the path for cookie is
        failed with _fail_path).
        """
        if self._track_stop_event.is_set():
            self.log.info("Track superseded, slew not sent")
            return False
        # ocs.move_to does not abort the previous command: do it here. Usually
        # _send_track has aborted it already (then nothing is sent again).
        if not self._abort_previous(cookie):
            return False
        response = self._move_to_with_retry(az, el)
        if response is None:
            self.log.info("Track superseded while TCS was busy, slew not sent")
            return False
        body = _response_body(response)
        if response.status_code != 200 or body.get("status") != "ok":
            self.log.error(
                f"slew move_to rejected: HTTP {response.status_code}, "
                f"status={body.get('status')}, message={body.get('message')}"
            )
            self._fail_path(cookie, _error_code(response))
            return False
        self.slew_in_progress = True
        self.last_aborted = False
        self.log.info(f"slew move_to accepted: {body}")
        # The command is accepted: return the cookie now, so that the observer is
        # not blocked for the slew. tel_on_track stays "N" until the path, sent
        # after the slew, has started (a failure later still sets tel_error).
        # Until the path is accepted, the slew target is reported as commanded.
        self._return_cookie(cookie, np.inf, slew_target=(az, el, time.time()))

        # Arrived = stopped close to the target. TCS uses a tighter position
        # tolerance (against the ACU commanded position), so it may still be
        # finishing the move-to: _send_track then resends the path.
        t0 = time.monotonic()
        while True:
            # interruptible: stop waiting as soon as a newer track arrives
            if self._track_stop_event.wait(SLEW_POLL_INTERVAL):
                # slew_in_progress stays True: the next track aborts the move-to
                self.log.info("Track superseded during slew")
                return False
            cur_az, cur_el, vel_az, vel_el = self._acu_position()
            if (max(abs(az - cur_az), abs(el - cur_el)) < SLEW_ARRIVAL_TOLERANCE
                    and abs(vel_az) < SLEW_SPEED_TOLERANCE
                    and abs(vel_el) < SLEW_SPEED_TOLERANCE):
                self.slew_in_progress = False
                self.log.info(
                    f"slew finished in {time.monotonic() - t0:.1f} s "
                    f"at az={cur_az:.4f} el={cur_el:.4f}"
                )
                return True
            if time.monotonic() - t0 > SLEW_TIMEOUT:
                self.log.error(
                    f"slew not finished within {SLEW_TIMEOUT} s, "
                    f"at az={cur_az:.4f} el={cur_el:.4f}"
                )
                self._fail_path(cookie, 1)
                return False

    def _abort_previous(self, cookie):
        """
        Abort the previous command if it may still be running on TCS. The OCS
        client does not abort by itself before scan_pattern or move_to.

        A finished path needs no abort. A previous move-to is always aborted:
        it may still be moving (TCS would answer busy), and its end is not
        known without polling the ACU.

        Returns True if the new command can be sent. Returns False if the abort
        failed (then the path for cookie is failed with _fail_path).
        """
        if self.last_aborted:
            self.log.info("Previous command already aborted, no abort sent")
            return True
        if self.slew_in_progress:
            # the slew of a superseded track may still be running on TCS
            need_abort = True
            self.log.info("Previous slew may still be running")
        elif self.last_payload is None:
            # no record of the previous command (e.g. after a restart): a command
            # sent before may still be running on TCS
            need_abort = self.abort_if_unknown
            if not need_abort:
                self.log.info("No record of previous command, no abort sent")
        elif self.last_payload["type"] == "move_to":
            need_abort = True
            self.log.info("Previous move-to may still be moving")
        else:
            last_end_time = self.last_payload["start_time"] + self.last_payload["points"][-1][0]
            need_abort = last_end_time > time.time()
            if not need_abort:
                self.log.info("Previous path already ended, no abort sent")
        if not need_abort:
            return True
        try:
            self.ocs.abort()
        except Exception as e:
            self.log.error(f"Abort error: {e}")
            self._fail_path(cookie, 1)
            return False
        self.slew_in_progress = False
        self.last_aborted = True
        self.log.info("Previous command aborted")
        return True

    def _move_to_with_retry(self, az, el):
        """
        Send ocs.move_to, resending it while TCS answers busy (e.g. not yet idle
        after the abort sent just before). Returns the last response, or None if
        the track was superseded while waiting for a retry.
        """
        return self._send_with_retry(
            lambda: self.ocs.move_to(az, el),
            busy_timeout=MOVE_TO_BUSY_TIMEOUT,
            retry_interval=MOVE_TO_BUSY_RETRY_INTERVAL,
        )

    def _scan_pattern_with_retry(self, payload, busy_timeout, retry_interval):
        """
        Send payload with ocs.scan_pattern, resending it while the TCS answers
        503 (busy) for at most busy_timeout seconds, retry_interval apart.
        Returns the last response, or None if the track was superseded
        (_track_stop_event set) while waiting for a retry.
        """
        return self._send_with_retry(
            lambda: self.ocs.scan_pattern(data=payload), busy_timeout, retry_interval)

    def _send_with_retry(self, send, busy_timeout, retry_interval):
        """
        Call send() (an OCS command), calling it again while the TCS answers
        busy for at most busy_timeout seconds, retry_interval apart.
        Returns the last response, or None if the track was superseded
        (_track_stop_event set) while waiting for a retry.
        """
        deadline = time.monotonic() + busy_timeout
        attempt = 1
        while True:
            response = send()
            if not _is_busy(response):
                return response
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                self.log.warning(
                    f"TCS still busy after {attempt} attempt(s) "
                    f"within {busy_timeout} s, giving up"
                )
                return response
            self.log.info(
                f"TCS busy (attempt {attempt}), retrying in "
                f"{min(retry_interval, remaining):.2f} s"
            )
            # interruptible sleep: stop retrying as soon as a newer track arrives
            if self._track_stop_event.wait(min(retry_interval, remaining)):
                return None
            attempt += 1

    # =======================================================================
    # Cookie handshake (set by the track thread, reported by the background
    # thread in KOSMA_tel2obs.set)
    # =======================================================================

    def _fail_path(self, cookie, error, lost_track="N"):
        """
        The path for cookie will never start: return the cookie now, flagged
        as failed, so that the observer does not wait for it.

        lost_track: tel_lost_track reported for cookie. "Y" for an OTF scan:
        otf_server does not wait for its cookie (wait=0) and its quality check
        reads only tel_lost_track (tel_error is just logged), so "Y" is needed
        to flag the data taken by the backend.
        """
        if self._track_stop_event.is_set():
            # superseded by a newer obs2tel; its own thread returns the new cookie
            self.log.info(f"Track for cookie {cookie} superseded, failure not reported")
            return
        self.otf_scan_window = None
        self.slew_target = None
        self.tel_error = error
        self.tel_path_failed = True
        self.cookie_reported = True
        # the cookie may already be returned (failure after an accepted slew)
        if cookie != self.tel_return_cookie:
            self.old_tel_return_cookie = self.tel_return_cookie
            self.tel_return_cookie = cookie
        self.tel_lost_track = lost_track
        self.tel_was_on_track = False

    def _return_cookie(self, cookie, on_track_not_before, otf_scan_window=None,
                       slew_target=None):
        """
        The command for cookie is accepted by TCS: return the cookie, with
        tel_on_track "N" for ACU status times before on_track_not_before [unix sec].
        otf_scan_window: (T0, T0 + D) [unix sec] for an OTF scan, see
        self.otf_scan_window.
        slew_target: (az, el, start) for a slew before the path, see
        self.slew_target. Any other accepted command clears it.
        """
        if self._track_stop_event.is_set():
            # superseded by a newer obs2tel; its own thread returns the new cookie
            self.log.info(f"Track for cookie {cookie} superseded, cookie not returned")
            return
        self.log.info(f"returning cookie {cookie}")
        # set before the cookie: the status of the new cookie uses the new
        # window and slew target
        self.otf_scan_window = otf_scan_window
        self.slew_target = slew_target
        self.cookie_reported = True
        if cookie != self.tel_return_cookie:
            self.old_tel_return_cookie = self.tel_return_cookie
            self.tel_return_cookie = cookie
        self.on_track_not_before = on_track_not_before
        self.tel_lost_track = "N"    # initialize for the new cookie
        self.tel_was_on_track = False
        self.tel_error = 0
        self.tel_path_failed = False

    # =======================================================================
    # Coordinates and refraction (track thread and background thread)
    # =======================================================================

    def get_weather(self):
        """
        Current weather at the site, used for the refraction correction.
        This is the only place that accesses the weather source.

        Returns
        -------
        weather : dict
            pressure and temperature as astropy Quantity, relative_humidity as
            fraction (0-1). Used as keyword arguments of AltAz.
        """

        # TODO: add proper keys to get weather data, and a request timeout:
        # get_measurement_data has none, and this runs in the track thread
        # before each new path is built, so a stalled request would hold up the track
        # for now this will fail and falls back to DEFAULT_WEATHER
        self.ocs.get_measurement_data()

        # If KOSMA_wetter.status can provide the data, switch it to True
        # and add tests (for now no tests can reach the following part)
        use_KOSMA_wetter = False
        if use_KOSMA_wetter:
            files = ImportKOSMAReadWriteIntoDictionary(
                files=["KOSMA_wetter.status"])
            weather = files["KOSMA_wetter.status"]
            return {
                "pressure": weather["wet_pressure"] * u.Torr,
                # wet_temp is in K
                "temperature": (weather["wet_temp"] - 273.15) * u.deg_C,
                # wet_humidity is in percent, as used by astra (astra_track.c),
                # although the comment in some KOSMA_wetter.status templates
                # says [0..1]
                "relative_humidity": weather["wet_humidity"] / 100.0,
            }

    def _update_refraction_params(self, obs_wavelength):
        """
        Set the AltAz refraction parameters for a new track, if the refraction
        correction is ON. obs_wavelength <= 0 means no refraction, as in astra.
        DEFAULT_WEATHER is used if the weather parameters are bad or missing.

        Parameters
        ----------
        obs_wavelength : float
            Observing wavelength [micron]
        """
        if not self.refraction_correction:
            return
        if obs_wavelength <= 0:
            self.log.info(f"obs_wavelength = {obs_wavelength} micron, no refraction applied")
            self._refraction_params = {}
            return

        try:
            weather = self.get_weather()
            pressure = weather["pressure"].to_value(u.hPa)
            temperature = weather["temperature"].to_value(u.deg_C)
            humidity = weather["relative_humidity"]
            # a relative humidity above 1 would be silently clipped to 1 by ERFA
            if not (pressure > 0 and np.isfinite(temperature) and 0 <= humidity <= 1):
                raise ValueError(
                    f"pressure={pressure} hPa, temperature={temperature} deg_C, "
                    f"relative_humidity={humidity}"
                )
        except Exception as e:
            self.log.warning(
                f"Bad or missing weather parameters ({e!r}), using default values"
            )
            weather = DEFAULT_WEATHER
        self.log.info(
            f"refraction: pressure={weather['pressure'].to_value(u.hPa):.1f} hPa, "
            f"temperature={weather['temperature'].to_value(u.deg_C):.1f} deg_C, "
            f"relative_humidity={weather['relative_humidity']:.3f}, "
            f"wavelength={obs_wavelength} micron"
        )
        self._refraction_params = {**weather, "obswl": obs_wavelength * u.micron}

    def _check_acu_refraction(self, response_pointing):
        """
        Warn when the ACU RF refraction correction gets enabled while the
        translator applies refraction: it would be corrected twice.

        Parameters
        ----------
        response_pointing : dict
            ACU dataset "CmdPointingCorrection"
        """
        acu_on = bool(response_pointing.get("RF refraction correction on"))
        if self.refraction_correction and acu_on and not self._acu_refraction_on:
            self.log.warning(
                "ACU RF refraction correction is on while the translator also applies "
                "refraction: refraction is corrected twice"
            )
        self._acu_refraction_on = acu_on

    def _altaz_frame(self, obstime):
        """
        AltAz frame at obstime, with refraction if the correction is applied
        """
        return AltAz(obstime=obstime, location=self.location, **self._refraction_params)

    # =======================================================================
    # OTF, obs_otf_mode C (track thread). Pattern mode P is not implemented.
    #
    # KOSMA otf_server per scan line (see process_obs2tel):
    # - PRETP (obs_track_duration = 0, wait=1, obs_start_time is stale): with
    #   nextflg=Y a tracking path at the same position (line start) is already
    #   running and kept (keep_path_for_pre_otf). Otherwise _send_pre_otf moves
    #   to the run-up point.
    # - OTFTP (obs_track_duration = D > 0, obs_start_time = T0, wait=0, sent
    #   pre_otf_time before T0): _send_otf_scan aborts the running command and
    #   sends one path: run-up point, smooth ramp, scan from T0 to T0 + D.
    # =======================================================================

    def _send_pre_otf(self, cookie):
        """
        PRETP without a running path at its position: move-to the run-up point
        of the scan, at its az/el expected at the start of the OTF path
        (T0 ~ now + OTF_EXPECTED_PRE_OTF_TIME). tel_on_track is "Y" once the
        telescope is there. The ACU reaches the actual path start from there
        within the path lead.
        """
        obs2tel = self.obs2tel
        self._validate_coord_sys(obs2tel)
        self._update_refraction_params(obs2tel["obs_wavelength"])

        tau = -(OTF_RAMP_TIME + OTF_SETTLE_TIME)     # path start relative to T0
        t0 = time.time() + OTF_EXPECTED_PRE_OTF_TIME
        az, el = self._otf_horizon_positions(
            obs2tel, t0 + np.array([tau]),
            _otf_scan_time([tau], 0.0, OTF_RAMP_TIME, OTF_STOP_TIME))
        cmd_az, cmd_el = float(az[0]), float(el[0])
        if not (np.isfinite(cmd_az) and np.isfinite(cmd_el)):
            raise ValueError(f"run-up point not reachable: az={cmd_az}, el={cmd_el}")
        self.log.info(f"PRETP: move-to the run-up point az={cmd_az:.4f} el={cmd_el:.4f}")
        self._send_move_to_azel(cookie, cmd_az, cmd_el, obs2tel, self.track_duration)

    def _send_otf_scan(self, cookie):
        """
        OTFTP: abort the running command and send the OTF path, which passes
        obs_lam/bet_del at full speed at T0 = obs_start_time and ends at
        T0 + obs_track_duration (+ OTF_STOP_TIME).

        T0 is fixed by KOSMA (the backend starts at T0). If the path cannot be
        sent at least OTF_MIN_SEND_LEAD before its start, it is not sent and the
        cookie is returned with tel_error TEL_ERROR_OTF_TOO_LATE and
        tel_lost_track "Y".
        """
        obs2tel = self.obs2tel
        self._validate_coord_sys(obs2tel)
        t0 = float(obs2tel["obs_start_time"])
        duration = float(obs2tel["obs_track_duration"])
        self.log.info(
            f"OTF scan: T0={t0:.3f} (in {t0 - time.time():.3f} s), D={duration} s, "
            f"del=({obs2tel['obs_lam_del']}, {obs2tel['obs_bet_del']}) arcsec, "
            f"vel=({obs2tel['obs_lam_vel']}, {obs2tel['obs_bet_vel']}) arcsec/s "
            f"in {obs2tel['obs_coord_sys_del']}"
        )
        if not self._otf_lead_ok(cookie, t0 - (OTF_RAMP_TIME + OTF_SETTLE_TIME)):
            return
        self._update_refraction_params(obs2tel["obs_wavelength"])

        # The running command (PRETP path or move-to) is aborted first, so that
        # the abort does not eat into the lead. The telescope stands until the
        # path starts.
        if self._track_stop_event.is_set():
            self.log.info("Track superseded, previous command not aborted")
            return
        if not self._abort_previous(cookie):
            return

        payload = self._build_otf_payload(obs2tel, t0, duration)
        if not self._otf_lead_ok(cookie, payload["start_time"]):
            return

        first_az, first_el = payload["points"][0][1], payload["points"][0][2]
        current_az, current_el, _, _ = self._acu_position()
        distance = max(abs(first_az - current_az), abs(first_el - current_el))
        if distance >= SLEW_BEFORE_PATH_MIN_DIST:
            # no time for a slew: T0 is fixed
            self.log.warning(
                f"telescope at az={current_az:.4f} el={current_el:.4f} is {distance:.3f} deg "
                f"from the OTF path start az={first_az:.4f} el={first_el:.4f}"
            )

        if self._track_stop_event.is_set():
            self.log.info("Track superseded, OTF path not sent")
            return
        lead_send = payload["start_time"] - time.time()
        response = self._scan_pattern_with_retry(
            payload, busy_timeout=max(lead_send - OTF_MIN_SEND_LEAD, 0.0), retry_interval=0.3)
        if response is None:
            self.log.info("Track superseded while TCS was busy, OTF path not sent")
            return
        self.log.info(
            f"OTF path start lead: {lead_send:.3f} s before send, "
            f"{payload['start_time'] - time.time():.3f} s after response"
        )
        body = _response_body(response)
        if response.status_code != 200 or body.get("status") != "ok":
            self.log.error(
                f"OTF scan_pattern rejected: HTTP {response.status_code}, "
                f"status={body.get('status')}, message={body.get('message')}"
            )
            self._fail_path(cookie, _error_code(response), lost_track="Y")
            return
        self.log.info(f"OTF scan_pattern accepted: {body}")
        self.last_aborted = False
        # "type" only for bookkeeping, not part of the payload sent to TCS
        self.last_payload = {**payload, "type": "otf"}
        self.last_obs2tel = obs2tel
        self._return_cookie(cookie, t0, otf_scan_window=(t0, t0 + duration))

    def _otf_lead_ok(self, cookie, path_start):
        """
        True if a path starting at path_start [unix sec] can still be sent in
        time. Otherwise fail the OTF scan for cookie and return False.
        """
        lead = path_start - time.time()
        if lead >= OTF_MIN_SEND_LEAD:
            return True
        self.log.error(
            f"OTF path start in {lead:.3f} s < {OTF_MIN_SEND_LEAD} s: not sent "
            f"(increase pre_otf_time in KOSMA master_parameters)"
        )
        self._fail_path(cookie, TEL_ERROR_OTF_TOO_LATE, lost_track="Y")
        return False

    def _build_otf_payload(self, obs2tel, t0, duration):
        """
        OTF path for scan_pattern: points every TRACK_TIME_STEP, aligned so that
        T0 is a point, from T0 - OTF_RAMP_TIME - OTF_SETTLE_TIME (rounded to the
        step) to T0 + duration + OTF_STOP_TIME (rounded up). Each point is
        transformed in full (no interpolation), so the ramp is not smoothed away.

        Parameters
        ----------
        obs2tel : dict
            obs2tel of the OTFTP command
        t0 : float
            Scan start T0 [unix sec]
        duration : float
            Scan duration D [sec]

        Returns
        -------
        payload : dict
            start_time (unix), coordsys and points [[t, az, el, v_az, v_el], ...]
        """
        t_build = time.perf_counter()
        n_before = int(round((OTF_RAMP_TIME + OTF_SETTLE_TIME) / TRACK_TIME_STEP))
        n_after = int(np.ceil((duration + OTF_STOP_TIME) / TRACK_TIME_STEP - 1e-9))
        # at least 3 points for the second-order velocities
        n_after = max(n_after, 2 - n_before)
        tau = np.arange(-n_before, n_after + 1) * TRACK_TIME_STEP
        f = _otf_scan_time(tau, duration, OTF_RAMP_TIME, OTF_STOP_TIME)
        az, el = self._otf_horizon_positions(obs2tel, t0 + tau, f)

        # Azimuth is unwrapped so a 0/360 crossing gets the right velocity, then
        # wrapped back to [0, 360). NaN (unreachable overtop) would spread
        # through the unwrap: skip it.
        if np.all(np.isfinite(az)):
            az = np.unwrap(az, period=360.0)
        v_az = np.gradient(az, tau, edge_order=2)
        v_el = np.gradient(el, tau, edge_order=2)
        az = np.mod(az, 360.0)

        points = np.column_stack([tau - tau[0], az, el, v_az, v_el])
        payload = {
            "start_time": t0 + tau[0],
            "coordsys": "Horizon",
            "points": points.tolist(),
        }
        self.log.info(
            f"OTF payload ({len(tau)} points) computed in {time.perf_counter() - t_build:.3f} s"
        )
        return payload

    def _otf_horizon_positions(self, obs2tel, times, f):
        """
        Az/el of the OTF offset obs_lam/bet_del + obs_lam/bet_vel * f at times.
        The offset is applied as one sum in obs_coord_sys_del, with
        obs_true_angle_del, as the backend assigns it to the data.

        Parameters
        ----------
        obs2tel : dict
        times : np.ndarray
            [unix sec]
        f : np.ndarray
            Position along the scan line at times, in seconds of nominal scan
            speed (see _otf_scan_time)

        Returns
        -------
        az, el : np.ndarray
            [deg] as sent to TCS (over the top in the extended range)
        """
        lam_del = obs2tel["obs_lam_del"] + obs2tel["obs_lam_vel"] * np.asarray(f)
        bet_del = obs2tel["obs_bet_del"] + obs2tel["obs_bet_vel"] * np.asarray(f)
        altaz_frame = self._altaz_frame(Time(times, format="unix"))
        if COORD_SYS_MAP[obs2tel["obs_coord_sys_on"]] == "altaz":
            # HORIZON source: as _send_move_to, at every time
            altaz = self._offset_horizon_source(obs2tel, altaz_frame, lam_del, bet_del)
            focal_plane_az = obs2tel["obs_x_focal_plane"]
            focal_plane_el = obs2tel["obs_y_focal_plane"]
            if focal_plane_az != 0.0 or focal_plane_el != 0.0:
                altaz = _apply_offset(altaz, focal_plane_az, focal_plane_el, "Y")
        else:
            coord, horizon_offsets = self._offset_source(obs2tel, lam_del, bet_del)
            altaz = coord.transform_to(altaz_frame)
            for d_lam, d_bet, true_angle in horizon_offsets:
                altaz = _apply_offset(altaz, d_lam, d_bet, true_angle)
        if obs2tel["obs_extended_elevation_range"] == "Y":
            el, az = get_overtop_altaz_from_nominal(
                alt_nominal=altaz.alt.deg, az_nominal=altaz.az.deg)
        else:
            az, el = altaz.az.deg, altaz.alt.deg
        return np.asarray(az, dtype=float), np.asarray(el, dtype=float)


def parse_arguments():
    def_cpath = "../CCAT/observatory-control-system/tls"

    parser = argparse.ArgumentParser(description="KOSMA OCS Translator")
    parser.add_argument(
        "--certificates-path",
        type=str,
        default=def_cpath,
        help=f"Path to the certificates directory. Default is {def_cpath}",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        default="INFO",
        help="Set the logging level. Choices: DEBUG, INFO, WARNING, ERROR, CRITICAL. Default is INFO.",
    )
    # OCS or roof mount connection
    # add a true false flag for roof mount connection, if true, use roof mount connection, if false, use OCS connection
    parser.add_argument(
        "--use-roof-mount",
        action="store_true",
        help="Use roof mount connection instead of OCS connection. Default is False.",
    )
    parser.add_argument(
        "--use-local-dev",
        action="store_true",
        help="Connect to the OCS of the local development setup (127.0.0.1:5600). Default is False.",
    )
    parser.add_argument(
        "--no-abort-if-unknown",
        action="store_true",
        help="Do not abort before the first path when there is no record of the previous "
             "command (e.g. after a restart). Default is False (abort).",
    )
    parser.add_argument(
        "--refraction-correction",
        action="store_true",
        help="Apply the atmospheric refraction correction in the AltAz transforms. "
             "Default is False (no refraction).",
    )
    return parser.parse_args()


def run_write_tel2obs_file_in_background(translator):
    while True:
        translator.write_tel2obs_file()
        time.sleep(translator.obs_tel_info_update_time)


def _raise_keyboard_interrupt(signum, frame):
    """Signal handler: end the main loop in main() as on Ctrl+C."""
    raise KeyboardInterrupt


def main():
    args = parse_arguments()
    log_level = getattr(logging, args.log_level)
    logger = setup_logging(DEFAULT_LOGGER_NAME, LOG_FILE, log_level)
    ocs = setup_ocs(
        use_roof_mount=args.use_roof_mount,
        use_local_dev=args.use_local_dev,
        certificates_path=args.certificates_path,
        logger=logger,
    )
    translator = KOSMA_translator(
        ocs,
        abort_if_unknown=not args.no_abort_if_unknown,
        refraction_correction=args.refraction_correction,
    )
    # KOSMA needs the SPEM parameters currently used by the ACU
    translator.dump_spem_to_kosma()
    # and the astra pointing model in KOSMA_point.status
    translator.copy_point_status()
    background_thread = threading.Thread(
        target=run_write_tel2obs_file_in_background, args=(translator,)
    )
    background_thread.daemon = True
    background_thread.start()
    translator.log.info(
        f"obs_tel_info_update_time set to {translator.obs_tel_info_update_time} seconds"
    )

    # Ctrl+C (SIGINT) and SIGTERM (systemctl stop, docker stop, kill) both end
    # the main loop as KeyboardInterrupt; the cleanup in finally also runs if
    # the loop ends with any other exception.
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, _raise_keyboard_interrupt)

    try:
        while True:
            obs2tel_updated = translator.check_for_obs2tel_update()
            if not obs2tel_updated:
                time.sleep(OBS2TEL_POLL_INTERVAL)
                continue
            if not translator.process_obs2tel():
                time.sleep(OBS2TEL_POLL_INTERVAL)
    except KeyboardInterrupt:
        translator.log.info("Shutdown requested (Ctrl+C or SIGTERM), shutting down...")
    finally:
        translator.stop_track()
        if hasattr(ocs, "close"):
            ocs.close()
        translator.log.info("All threads closed, exiting.")


if __name__ == "__main__":
    main()

