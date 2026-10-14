"""KOSMA file io: read KOSMA_obs2tel.set, write KOSMA_tel2obs.set.

KOSMA rules (KOSMA_file_io.c): line 1 holds "date  time  sec.usec   File update time
stamp   ! writer"; readers detect new data by that stamp, not by the mtime. Every
other line is "value   name   ! comment"; a reader takes the text before the name.
"""

import os
import socket
import time
from pathlib import Path

TIMESTAMP_NAME = "File update time stamp"
TIMESTRING_LENGTH = 21  # "%d-%b-%Y  %H:%M:%S"

# The obs2tel variables the translator reads, with their OBSERVATORY.acfg types.
OBS2TEL = {
    "obs_source_name": str,
    "obs_tel_info_update_time": float,
    "obs_cookie": int,
    "obs_coord_sys_on": str,
    "obs_lam_on": float,
    "obs_bet_on": float,
    "obs_pm_lam": float,
    "obs_pm_bet": float,
    "obs_paralx": float,
    "obs_extended_elevation_range": str,
    "obs_coord_sys_del": str,
    "obs_true_angle_del": str,
    "obs_lam_del": float,
    "obs_bet_del": float,
    "obs_lam_vel": float,
    "obs_bet_vel": float,
    "obs_otf_mode": str,
    "obs_coord_sys_off": str,
    "obs_true_angle_off": str,
    "obs_lam_off": float,
    "obs_bet_off": float,
    "obs_x_focal_plane": float,
    "obs_y_focal_plane": float,
    "obs_track_duration": float,
    "obs_start_time": float,
    "obs_tolerance": float,
    "obs_amp_chop": float,
    "obs_chop_beam": int,
}


# KOSMA_tel2obs.set: the 22 lines of the CCAT template in template order, with the
# printf formats and comments of OBSERVATORY.acfg.
TEL2OBS = (
    ("tel_telescope", "%32s", "Telescope Identifier (for display and backend_name in FITS header)"),
    ("tel_on_track", "%1s", "Y if tracking on commanded position/track within tolerance [Y/N]"),
    ("tel_lost_track", "%1s", "Y if tracking got beyond tolerance since start of track [Y/N]"),
    ("tel_pos_in_range", "%1s", "Y if commanded position is within telescope range [Y/N]"),
    (
        "tel_error",
        "%12d",
        "flags telescope command syntax/consistency and functional errors (=0: ok)",
    ),
    ("tel_error_string", "%s", "Error message associated with telescope command (=NONE: OK)"),
    (
        "tel_error_level",
        "%12d",
        "Error level for message in tel_error_string [1: trace, 2: debug, 4: info, 5: message, 7: warn, 8: error, 9: fatal]",
    ),
    ("tel_return_cookie", "%12d", "return cookie to identify proper observation"),
    ("tel_plate_scale", "%12.8g", "plate scale of focal plane [arcsec/mm]"),
    ("tel_latitude", "%12.8lf", "latitude of observatory (needed for Doppler correction) [degree]"),
    (
        "tel_longitude",
        "%12.8lf",
        "longitude of observatory (needed for Doppler correction) [degree, +west (astronomical notation)]",
    ),
    ("tel_altitude", "%12.8g", "height above sea level (needed for Doppler correction) [meter]"),
    (
        "tel_angle_focal_plane",
        "%12.8g",
        "ccw angle from second coordinate of the telescope reference coordinate system (TARF,HORIZON) to second coordinate of obs_coord_sys_del [degree]",
    ),
    ("tel_los_act", "%12.8g", "actual los-angle [degree]"),
    ("tel_time_act", "%20.5f", "time when status data were valid [seconds since 1970 UTC]"),
    (
        "tel_time_del",
        "%12.8g",
        "time since start of track (may be negative, if start in the future) [seconds]",
    ),
    ("tel_azm_cmd", "%12.8g", "commanded azimuth (for tracking display) [degree,cw toward east]"),
    (
        "tel_elv_cmd",
        "%12.8g",
        "commanded elevation (for tracking display and for atmospheric calibration) [degree]",
    ),
    ("tel_azm_act", "%12.8g", "actual azimuth (for tracking display) [degree,cw toward east]"),
    ("tel_elv_act", "%12.8g", "actual elevation (for tracking display) [degree]"),
    (
        "tel_supports_ephemeris",
        "%1s",
        "Y if telescope can calculate ephemeris, N if telescope expects observing task to do this for non-inertial objects [Y/N]",
    ),
    ("tel_mjd1", "%16.10lf", "modified Julian Date UT1 [days]"),
)


def read_set(path):
    """Return (stamp, {name: value text}) of a KOSMA set file.

    stamp is None when line 1 holds no header stamp.
    """
    stamp, values = None, {}
    for line in Path(path).read_text(errors="replace").splitlines():
        data = line.split("!", 1)[0]
        if TIMESTAMP_NAME in data:
            stamp = float(data.split(TIMESTAMP_NAME)[0][TIMESTRING_LENGTH:])
            continue
        tokens = data.split()
        if tokens:
            values.setdefault(tokens[-1], data.rsplit(tokens[-1], 1)[0].strip())
    return stamp, values


def read_obs2tel(path):
    """Return (stamp, command) with the OBS2TEL variables typed.

    Raises KeyError or ValueError when a variable is missing or unreadable.
    """
    stamp, values = read_set(path)
    return stamp, {name: kind(values[name]) for name, kind in OBS2TEL.items()}


def header_time(now):
    """The date part of a header stamp, local time as KOSMA writes it."""
    return time.strftime("%d-%b-%Y  %H:%M:%S", time.localtime(now))


def format_line(fmt, value, name, comment):
    text = fmt.replace("l", "") % value
    # a "!" starts the comment and a newline ends the line for KOSMA readers
    text = " ".join(text.replace("!", ".").splitlines())
    return f"{text}   {name}   ! {comment}\n"


class SetWriter:
    """Writes one KOSMA set file: header stamp on line 1, strictly increasing,
    and a temporary file renamed onto the target, so readers never see a partial file."""

    def __init__(self, path, writer="kosma_ocs_translator"):
        self.path = Path(path)
        self.comment = f"{writer} ({socket.gethostname()}:{os.getpid()})"
        self.last_stamp = 0.0

    def write_lines(self, body, now):
        """Write body (formatted lines) under a new header; return the stamp."""
        usec = max(round(now * 1e6), round(self.last_stamp * 1e6) + 1)
        sec, usec = divmod(usec, 1_000_000)
        self.last_stamp = float(f"{sec}.{usec:06d}")
        header = f"{header_time(sec)}{sec:12d}.{usec:06d}   {TIMESTAMP_NAME}   ! {self.comment}\n"
        tmp = self.path.with_name(f"{self.path.name}_{sec}.{usec:06d}")
        tmp.write_text(header + "".join(body))
        if self.path.exists():
            os.chmod(tmp, self.path.stat().st_mode)
        os.replace(tmp, self.path)
        return self.last_stamp


class Tel2obsWriter(SetWriter):
    def write(self, values, now):
        """Write KOSMA_tel2obs.set with every TEL2OBS line; return the header stamp."""
        body = [format_line(fmt, values[name], name, comment) for name, fmt, comment in TEL2OBS]
        return self.write_lines(body, now)


def read_stamp(path):
    """The header stamp of a KOSMA set file, or None when it is missing or unreadable."""
    try:
        with open(path, errors="replace") as f:
            first = f.readline().split("!", 1)[0]
        return float(first.split(TIMESTAMP_NAME)[0][TIMESTRING_LENGTH:])
    except (OSError, ValueError):
        return None


def write_alive(writer, now):
    """The alive signal, as astra_dummy writes astra_dummy_alive (MONITORING.acfg)."""
    name = "kosma_ocs_translator_alive"
    line = format_line("%d", int(now), name, "Watch-Dog variable for kosma_ocs_translator")
    writer.write_lines([line], now)
