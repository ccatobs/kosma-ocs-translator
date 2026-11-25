#!/usr/bin/env python3
"""
Expected sky position and time of KOSMA FFTS data (OTF dumps and OFF/ON total
power integrations; for load (CAL) integrations the time only), reconstructed after the fact from the FITS headers and the
log files only, and optionally compared with ACU housekeeping az/el.

KOSMA_obs2tel.set / KOSMA_tel2obs.set are NOT used: they only hold the latest
command. Everything comes from

- the FITS header (writeheader.c / write_fits_kosma.c):
    LAMBDA/BETA/COORDSYS          source
    LAMDEL/BETDEL/COSYDEL/OTADEL  map offset (OTF: line start S at T0, full speed)
    LAMOFF/BETOFF/COSYOFF         off-position offset
    DATE-UT                       start, TRUNCATED to the second
    OBSTIME                       integration time (OTF: per dump)
    OTFNDMP                       number of dumps (only in OTF files)
    OTFSLAM/OTFSBET               (timediff + OBSTIME/2) * vel [arcsec]
    OTFVLAM/OTFVBET               vel * OBSTIME [arcsec per dump]
  where timediff = actual backend start - T0. FFTS only: CONT/CFFTS/PB8 builds
  scale OBSTIME/OTFV* by the number of dumps, which is not handled here.
- the logs, for the sub-second start time:
    translator log   "OTF scan: T0=<unix>"                    (INFO)
                     "refraction: pressure=... wavelength=..." (INFO, AltAz parameters)
    backend log      "OTF_start_time:<unix>"                  (Info)
                     "BCK_TOTPOWER: obs_start_time updated: <unix>"   (Vdebug)
                     "Started total power measurement with X seconds at <unix>." (Info)
  A log time is matched to a FITS file by its truncated second (DATE-UT).
  Without a matching log line the start is DATE-UT + 0.5 s (+-0.5 s).

Data model (backend contract): at time t within an OTF line the data are
assigned the offset LAMDEL + vel * (t - T0), vel = OTFVLAM / OBSTIME; dump i
(from 0) covers [T0 + timediff + i*OBSTIME, T0 + timediff + (i+1)*OBSTIME].
A TP/OFF integration covers [start, start + OBSTIME] at a fixed offset.
Offsets are applied as in the translator: non-HORIZON offsets (del, then off)
in their frame before AltAz, HORIZON ones after it, then the focal plane offset.

Not in the FITS header: obs_true_angle_off is taken as TRUE_ANGLE_OFF ("Y" in
all current use cases); obs_x/y_focal_plane (--focal-plane) and the site
(--site; SITELONG in the header is only printed) are options. Proper motion
and parallax are ignored.

Housekeeping (--hk): text file with time, az, el [deg]. Time is unix s, ms or
us (detected from the magnitude) or ISO 8601 UTC. Columns are selected by
index or, for files of extractDataFromInfluxDB.py ("# time name1 name2 ..."
header), by name; such files default to HK_POSITION_COLUMN_SETS. Over-
the-top positions (el > 90) are converted to nominal. The ACU current position
includes the pointing correction (SPEM, tiltmeters); it is NOT removed here.

Output
------
--out (one row per dump / integration): window, expected offsets, RA/Dec (ICRS)
and az/el at the window centre, and with --hk the statistics of the
differences: d_az = (az_hk - az_exp) cos(el), d_el, and their components along
the expected motion (as a time lag, + = telescope ahead of the data model) and
across it.
--samples-out (with --hk): one row per housekeeping sample inside a window.

Example
-------
python compare_data_timing.py /data/fits/2026-11-25/ \\
    --log kosma-ocs-translator.log --log backend_server.log \\
    --hk acu_hk.csv --out expected.csv --samples-out samples.csv
"""
import argparse
import csv
import math
import os
import re
import sys
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

import numpy as np
from astropy import units as u
from astropy.coordinates import AltAz, EarthLocation, SkyCoord
from astropy.io import fits
from astropy.time import Time

# ---------------------------------------------------------------------------
# Site (east positive), as tests/helpers.py CCAT_LOCATION
# ---------------------------------------------------------------------------
CCAT_LAT_DEG = -22.985639
CCAT_LON_DEG = -67.740278
CCAT_HEIGHT_M = 5611.8

# ---------------------------------------------------------------------------
# Coordinate systems: KOSMA name -> astropy frame name (as the translator)
# ---------------------------------------------------------------------------
COORD_SYS_MAP = {
    "J2000": "icrs",
    "B1950": "fk4",
    "GALACTIC": "galactic",
    "HORIZON": "altaz",
}
HORIZON = "HORIZON"
# obs_true_angle_off, not in the FITS header ("Y" in all current use cases)
TRUE_ANGLE_OFF = "Y"

# ---------------------------------------------------------------------------
# Log parsing
# ---------------------------------------------------------------------------
# translator log line: "2026-11-25T10:00:00.123Z - name - INFO - message" (UTC,
# kosma_ocs_lib.UTCFormatter), or in logs written before that
# "2026-11-25 10:00:00,123 - name - INFO - message" (local time, see --log-utc-offset)
TRANSLATOR_TIMESTAMP_RE = re.compile(
    r"^(\d{4}-\d{2}-\d{2})[T ](\d{2}:\d{2}:\d{2})[.,](\d{3})(Z?) - ")
TRANSLATOR_TIMESTAMP_FORMAT = "%Y-%m-%d %H:%M:%S"
# (regex, source name, data kind)
START_TIME_PATTERNS = (
    (re.compile(r"OTF scan: T0=(\d+\.\d+)"), "translator T0", "OTF"),
    (re.compile(r"OTF_start_time:\s*(\d+\.\d+)"), "backend T0", "OTF"),
    (re.compile(r"BCK_TOTPOWER: obs_start_time updated:\s*(\d+\.\d+)"), "backend start", "TP"),
    (re.compile(r"Started total power measurement with [\d.]+ seconds at (\d+\.\d+)"),
     "backend start (requested)", "TP"),
)
# preference among matching log lines, first wins
OTF_START_PREFERENCE = ("backend T0", "translator T0")
TP_START_PREFERENCE = ("backend start", "backend start (requested)")
# [sec] the requested TP start is taken before the backend starts, so it may
# fall into the second before DATE-UT
TP_REQUESTED_START_TOLERANCE = 1.0
REFRACTION_RE = re.compile(
    r"refraction: pressure=([-\d.]+) hPa, temperature=([-\d.]+) deg_C, "
    r"relative_humidity=([-\d.]+), wavelength=([-\d.eE+]+) micron")
NO_REFRACTION_RE = re.compile(r"no refraction applied")
HEADER_START_SOURCE = "DATE-UT + 0.5 s (+-0.5 s)"

# ---------------------------------------------------------------------------
# Comparison
# ---------------------------------------------------------------------------
# [sec] time step for the expected motion (lag / cross-track decomposition)
MOTION_TIME_STEP = 0.1
# [arcsec/s] below this expected speed no lag is computed
MIN_SPEED_FOR_LAG = 1.0
DEFAULT_OUT = "expected_positions.csv"

# ---------------------------------------------------------------------------
# Housekeeping time stamps: unit detected from the magnitude (unix time now is
# ~1.8e9 s = 1.8e12 ms = 1.8e15 us). (upper limit, factor to sec, unit name)
# ---------------------------------------------------------------------------
HK_TIME_UNITS = (
    (1e11, 1.0, "s"),
    (1e14, 1e-3, "ms"),
    (1e17, 1e-6, "us"),
)
# default housekeeping columns (time, az, el) for files with a "# time ..."
# header (extractDataFromInfluxDB.py): the first set whose names are all in the
# header, else HK_FALLBACK_COLUMNS. Measurements:
# Telescope.ACU-MON.DataSets.StatusGeneral8100 (1 Hz) and Telescope.ACU-POS-UDP
HK_POSITION_COLUMN_SETS = (
    ("time", "Azimuth_current_position", "Elevation_current_position"),
    ("time", "Azimuth", "Elevation"),
)
HK_VELOCITY_COLUMN_SETS = (
    ("time", "Azimuth_current_velocity", "Elevation_current_velocity"),
)
HK_FALLBACK_COLUMNS = (0, 1, 2)

# ---------------------------------------------------------------------------
# FITS file names: <scan>_<subscan>_<array>_<pixel>_<...>.fits,
# e.g. 003458_001_H_PX00_S.fits (backend_server.c: scan number, then
# bck_proc_filename and bck_name)
# ---------------------------------------------------------------------------
FITS_NAME_RE = re.compile(r"^(\d+)_(\d+)_(.+)\.fits$", re.IGNORECASE)


@dataclass
class LogStartTime:
    t: float        # [unix sec]
    source: str     # see START_TIME_PATTERNS
    kind: str       # "OTF" or "TP"


@dataclass
class RefractionEvent:
    t: float        # [unix sec] log time stamp
    params: dict    # AltAz keyword arguments, {} = no refraction
    text: str


@dataclass
class DataFile:
    path: str
    scan: int
    obsmode: str
    subobsmode: str
    utc_start_sec: int      # DATE-UT, truncated [unix sec]
    is_otf: bool
    is_load: bool           # CAL (HOT/COL/SKY/ZER): time window only
    int_time: float         # OBSTIME [sec], per dump for OTF
    n_dumps: int            # 1 for TP
    coord_sys_on: str
    lam_on: float           # [deg]
    bet_on: float
    coord_sys_del: str
    lam_del: float          # [arcsec]
    bet_del: float
    true_angle_del: str
    coord_sys_off: str
    lam_off: float          # [arcsec]
    bet_off: float
    otf_start_lam: float    # [arcsec]
    otf_start_bet: float
    otf_vel_lam: float      # [arcsec per dump]
    otf_vel_bet: float
    site_header: tuple      # (SITELAT, SITELONG, SITEALT) as written by KOSMA


@dataclass
class Windows:
    t_begin: np.ndarray     # [unix sec] one per dump / integration
    t_end: np.ndarray
    t0: float               # [unix sec] OTF: T0, TP: start
    vel_lam: float          # [arcsec/s] 0 for TP
    vel_bet: float
    timediff: float         # [sec] backend start - T0 (OTF)
    start_source: str
    start_spread: float     # [sec] spread of the matching log times


# ===========================================================================
# Input
# ===========================================================================

def parse_arguments():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("fits", nargs="+", help="FITS files or directories (*.fits inside)")
    parser.add_argument("--pixel", default=None,
                        help="only files of this pixel, e.g. PX00, or array and pixel, "
                             "e.g. H_PX00 (file name <scan>_<subscan>_<array>_<pixel>_...)")
    parser.add_argument("--scans", nargs=2, type=int, default=None, metavar=("FIRST", "LAST"),
                        help="only files with scan number FIRST..LAST (inclusive, from the "
                             "file name), e.g. --scans 3454 3458")
    parser.add_argument("--log", action="append", default=[],
                        help="translator or backend_server log file (repeatable)")
    parser.add_argument("--log-utc-offset", type=float, default=None,
                        help="UTC offset [h] of old translator log time stamps without 'Z' "
                             "(default: local time zone of this machine)")
    parser.add_argument("--hk", help="housekeeping text file: time [unix sec], az, el [deg]")
    parser.add_argument("--hk-cols", default=None,
                        help="columns of time, az, el: 0-based indices or names from a "
                             "'# time ...' header (default: "
                             f"{column_sets_text(HK_POSITION_COLUMN_SETS)} if the header "
                             "has them, "
                             "else 0,1,2)")
    parser.add_argument("--hk-delimiter", default=None,
                        help="column delimiter (default: ',' if the first data line has one, "
                             "else a single space)")
    parser.add_argument("--hk-time-offset", type=float, default=0.0,
                        help="[sec] added to the housekeeping times")
    parser.add_argument("--site", nargs=3, type=float,
                        metavar=("LAT_DEG", "LON_EAST_DEG", "HEIGHT_M"),
                        default=[CCAT_LAT_DEG, CCAT_LON_DEG, CCAT_HEIGHT_M])
    refraction = parser.add_mutually_exclusive_group()
    refraction.add_argument("--refraction", nargs=4, type=float,
                            metavar=("P_HPA", "T_DEGC", "RH", "WL_MICRON"),
                            help="refraction parameters (default: from the translator log)")
    refraction.add_argument("--no-refraction", action="store_true")
    parser.add_argument("--focal-plane", nargs=2, type=float, default=[0.0, 0.0],
                        metavar=("X_ARCSEC", "Y_ARCSEC"),
                        help="obs_x/y_focal_plane (not in FITS; default 0 0)")
    parser.add_argument("--out", default=DEFAULT_OUT, help=f"default {DEFAULT_OUT}")
    parser.add_argument("--samples-out", default=None,
                        help="per housekeeping sample output (needs --hk)")
    return parser.parse_args()


def select_by_name(path, pixel, scans):
    """
    True if the file name <scan>_<subscan>_<array>_<pixel>_<...>.fits matches
    the pixel (a "_"-separated part or sequence of parts of the name after the
    subscan, e.g. "PX00" or "H_PX00") and the scan range (FIRST, LAST).
    No filter given: always True.
    """
    if pixel is None and scans is None:
        return True
    match = FITS_NAME_RE.match(os.path.basename(path))
    if not match:
        print(f"skip {path}: file name not <scan>_<subscan>_..., cannot filter")
        return False
    scan, rest = int(match.group(1)), match.group(3)
    if scans is not None and not scans[0] <= scan <= scans[1]:
        return False
    return pixel is None or f"_{pixel}_" in f"_{rest}_"


def find_fits_files(paths, pixel=None, scans=None):
    files = []
    for path in paths:
        if os.path.isdir(path):
            files += [os.path.join(path, name) for name in sorted(os.listdir(path))
                      if name.lower().endswith(".fits")]
        else:
            files.append(path)
    return [path for path in files if select_by_name(path, pixel, scans)]


def _header_str(header, key, default=""):
    return str(header.get(key, default)).strip()


def read_data_file(path):
    """
    Header values of one KOSMA FITS file, or None if it is not a TP or OTF file.
    """
    with fits.open(path) as hdul:
        header = hdul[0].header
    obsmode = _header_str(header, "OBSMODE")
    if "OTFSW" in obsmode or "BS" in obsmode or "FS" in obsmode or "LS" in obsmode:
        print(f"skip {path}: OBSMODE {obsmode} (switched modes not handled)")
        return None
    if "DATE-UT" not in header or "LAMBDA" not in header:
        print(f"skip {path}: no DATE-UT / LAMBDA")
        return None
    is_otf = "OTFNDMP" in header
    is_load = obsmode.startswith("CAL")
    # total power integration: TP, or the OFF of an OTF map (OBSMODE e.g.
    # OTFT-OFF, written like TP without the OTF keywords)
    is_tp = "TP" in obsmode or obsmode.startswith("OTFT")
    if not (is_otf or is_load or is_tp):
        print(f"skip {path}: OBSMODE {obsmode} is neither OTF, TP, OTFT-OFF nor CAL")
        return None
    utc = Time(_header_str(header, "DATE-UT"), format="isot", scale="utc").unix
    return DataFile(
        path=path,
        scan=int(header.get("SCANNUM", -1)),
        obsmode=obsmode,
        subobsmode=_header_str(header, "SOBSMODE"),
        utc_start_sec=int(round(utc)),
        is_otf=is_otf,
        is_load=is_load,
        int_time=float(header.get("OBSTIME", 0.0)),
        n_dumps=int(header["OTFNDMP"]) if is_otf else 1,
        coord_sys_on=_header_str(header, "COORDSYS"),
        lam_on=float(header["LAMBDA"]),
        bet_on=float(header["BETA"]),
        coord_sys_del=_header_str(header, "COSYDEL"),
        lam_del=float(header.get("LAMDEL", 0.0)),
        bet_del=float(header.get("BETDEL", 0.0)),
        true_angle_del=_header_str(header, "OTADEL", "N")[:1],
        coord_sys_off=_header_str(header, "COSYOFF"),
        lam_off=float(header.get("LAMOFF", 0.0)),
        bet_off=float(header.get("BETOFF", 0.0)),
        otf_start_lam=float(header.get("OTFSLAM", 0.0)),
        otf_start_bet=float(header.get("OTFSBET", 0.0)),
        otf_vel_lam=float(header.get("OTFVLAM", 0.0)),
        otf_vel_bet=float(header.get("OTFVBET", 0.0)),
        site_header=(header.get("SITELAT"), header.get("SITELONG"), header.get("SITEALT")),
    )


def _translator_log_time(line, tz):
    match = TRANSLATOR_TIMESTAMP_RE.match(line)
    if not match:
        return None
    date, clock, msec, utc = match.groups()
    stamp = datetime.strptime(f"{date} {clock}", TRANSLATOR_TIMESTAMP_FORMAT)
    if utc:
        stamp = stamp.replace(tzinfo=timezone.utc)
    elif tz is not None:
        stamp = stamp.replace(tzinfo=tz)
    return stamp.timestamp() + int(msec) / 1000.0


def parse_logs(paths, utc_offset_hours):
    """
    Start times (any log) and refraction parameters (translator log, with its
    time stamps) found in the log files.
    """
    tz = None if utc_offset_hours is None else timezone(timedelta(hours=utc_offset_hours))
    starts, refractions = [], []
    for path in paths:
        with open(path, errors="replace") as handle:
            for line in handle:
                for regex, source, kind in START_TIME_PATTERNS:
                    match = regex.search(line)
                    if match:
                        starts.append(LogStartTime(float(match.group(1)), source, kind))
                match = REFRACTION_RE.search(line)
                no_refraction = NO_REFRACTION_RE.search(line)
                if not (match or no_refraction):
                    continue
                t = _translator_log_time(line, tz)
                if t is None:
                    continue
                if match:
                    pressure, temperature, humidity, wavelength = map(float, match.groups())
                    params = {"pressure": pressure * u.hPa,
                              "temperature": temperature * u.deg_C,
                              "relative_humidity": humidity,
                              "obswl": wavelength * u.micron}
                    refractions.append(RefractionEvent(t, params, match.group(0)))
                else:
                    refractions.append(RefractionEvent(t, {}, "no refraction"))
    refractions.sort(key=lambda event: event.t)
    return starts, refractions


def housekeeping_time_scale(t):
    """
    Factor converting the housekeeping time stamps to unix seconds, and the
    unit name, chosen from their magnitude (HK_TIME_UNITS).
    """
    median = float(np.median(t)) if len(t) else 0.0
    for limit, scale, unit in HK_TIME_UNITS:
        if median < limit:
            return scale, unit
    raise ValueError(f"housekeeping time stamps {median} too large for any known unit")


def column_sets_text(column_sets):
    """Default column sets for help texts: "a,b,c or d,e,f"."""
    return " or ".join(",".join(column_set) for column_set in column_sets)


def parse_hk_columns(text):
    """
    --hk-cols / --hk-vel-cols: comma-separated 0-based indices or column
    names (from a "# time ..." header line), or None for the default.
    """
    if text is None:
        return None
    return tuple(int(item) if item.strip().isdigit() else item.strip()
                 for item in text.split(","))


def _parse_hk_time(text):
    """
    Time stamp as a number (unit detected later) or ISO 8601 UTC
    ("2026-11-25T14:30:00.158000Z") in unix sec.

    Returns
    -------
    value, is_iso : float, bool
        value None if not a time stamp (e.g. a header line)
    """
    try:
        return float(text), False
    except ValueError:
        pass
    try:
        stamp = np.datetime64(text.rstrip("Z"), "ns")
    except ValueError:
        return None, False
    return stamp.astype(np.int64) / 1e9, True


def _read_hk_rows(path, delimiter):
    """
    Header names (from the first "#" line, or None) and the split data rows.
    delimiter None: "," if the first data line contains one, else a single
    space (extractDataFromInfluxDB.py writes space-separated *.csv files with
    missing values as empty fields, which must not be merged).
    """
    with open(path, errors="replace") as handle:
        lines = [line.strip() for line in handle if line.strip()]
    if delimiter is None:
        first_data = next((line for line in lines if not line.startswith("#")), "")
        delimiter = "," if "," in first_data else " "
    names, rows = None, []
    for line in lines:
        if line.startswith("#"):
            if names is None:
                names = [name.strip().strip('"')
                         for name in line[1:].strip().split(delimiter)]
            continue
        rows.append([field.strip().strip('"') for field in line.split(delimiter)])
    return names, rows


def _resolve_hk_columns(path, cols, default_sets, names):
    """0-based indices of the requested columns (see load_time_series)."""
    if cols is None:
        matching = [default for default in default_sets or ()
                    if names is not None and all(n in names for n in default)]
        cols = matching[0] if matching else HK_FALLBACK_COLUMNS
    indices = []
    for col in cols:
        if isinstance(col, int):
            indices.append(col)
        elif names is None:
            raise ValueError(f"{path}: column name {col!r} given, but no '# time ...' header")
        elif col not in names:
            raise ValueError(f"{path}: no column {col!r}; columns are {names}")
        else:
            indices.append(names.index(col))
    print(f"{os.path.basename(path)}: columns "
          + ", ".join(names[i] if names and i < len(names) else str(i) for i in indices))
    return indices


def load_time_series(path, cols, delimiter, time_offset, default_sets=None):
    """
    Housekeeping text file: time and value columns, sorted by time. Rows
    whose selected fields are not numbers (header, "undefined") are dropped.

    Parameters
    ----------
    cols : tuple of int or str, or None
        Time column first, then the value columns: 0-based indices or names
        from a "# time name1 name2 ..." header line (extractDataFromInfluxDB.py).
        None: the first of default_sets whose names are all in the header,
        else HK_FALLBACK_COLUMNS.
    delimiter : str or None
        None: detected from the first data line ("," or a single space)
    time_offset : float
        [sec] added to the times
    default_sets : tuple of tuple of str, or None

    Returns
    -------
    t : np.ndarray
        [unix sec] (numbers: unit detected, see housekeeping_time_scale;
        or ISO 8601 UTC)
    values : list of np.ndarray
        one array per value column
    """
    names, rows = _read_hk_rows(path, delimiter)
    indices = _resolve_hk_columns(path, cols, default_sets, names)
    times, values, iso = [], [], False
    for row in rows:
        if len(row) <= max(indices):
            continue
        t, is_iso = _parse_hk_time(row[indices[0]])
        if t is None:
            continue
        try:
            row_values = [float(row[i]) for i in indices[1:]]
        except ValueError:
            continue
        if not all(map(math.isfinite, row_values)):
            continue
        times.append(t)
        values.append(row_values)
        iso = iso or is_iso
    if not times:
        raise ValueError(f"{path}: no valid rows")
    if len(times) > 1 and min(times) == max(times):
        raise ValueError(f"{path}: all {len(times)} time stamps are {times[0]}: "
                         f"wrong time column? (columns {indices})")
    times = np.array(times)
    values = np.array(values)
    order = np.argsort(times)
    times, values = times[order], values[order]
    if iso:
        unit = "ISO 8601"
    else:
        scale, unit = housekeeping_time_scale(times)
        times = times * scale
    print(f"{os.path.basename(path)}: {len(times)} rows, time stamps in {unit}")
    return times + time_offset, [values[:, i] for i in range(values.shape[1])]


def load_housekeeping(path, cols, delimiter, time_offset):
    """
    Housekeeping time [unix sec], az, el [deg], sorted by time, over-the-top
    positions (el > 90) converted to nominal. cols: see load_time_series
    (default HK_POSITION_COLUMN_SETS).
    """
    t, (az, el) = load_time_series(path, cols, delimiter, time_offset,
                                   HK_POSITION_COLUMN_SETS)
    overtop = el > 90.0
    az = np.where(overtop, az + 180.0, az) % 360.0
    el = np.where(overtop, 180.0 - el, el)
    return t, az, el


# ===========================================================================
# Timing
# ===========================================================================

def find_start_time(data_file, log_starts):
    """
    Sub-second start (OTF: T0, TP: backend start) of a data file from the logs.

    Returns
    -------
    t, source, spread : float, str, float
        spread = max - min of the matching log times of the chosen kind
    """
    utc = data_file.utc_start_sec
    if data_file.is_otf:
        matches = [s for s in log_starts if s.kind == "OTF" and math.floor(s.t) == utc]
        preference = OTF_START_PREFERENCE
    else:
        matches = [s for s in log_starts if s.kind == "TP" and (
            math.floor(s.t) == utc if s.source == "backend start"
            else abs(s.t - (utc + 0.5)) <= TP_REQUESTED_START_TOLERANCE)]
        preference = TP_START_PREFERENCE
    for source in preference:
        chosen = [s.t for s in matches if s.source == source]
        if chosen:
            all_times = [s.t for s in matches] if data_file.is_otf else chosen
            return chosen[0], source, max(all_times) - min(all_times)
    return utc + 0.5, HEADER_START_SOURCE, float("nan")


def data_windows(data_file, log_starts):
    """
    Time windows of the dumps (OTF) or the integration (TP), and the scan
    velocity, from the header and the start time.
    """
    t_start, source, spread = find_start_time(data_file, log_starts)
    length = data_file.int_time
    if not data_file.is_otf:
        return Windows(np.array([t_start]), np.array([t_start + length]),
                       t_start, 0.0, 0.0, 0.0, source, spread)
    if length <= 0:
        raise ValueError(f"{data_file.path}: OBSTIME {length} <= 0")
    vel_lam = data_file.otf_vel_lam / length
    vel_bet = data_file.otf_vel_bet / length
    # OTFSLAM = (timediff + length/2) * vel: use the faster component
    if abs(vel_lam) >= abs(vel_bet) and vel_lam != 0.0:
        timediff = data_file.otf_start_lam / vel_lam - length / 2
    elif vel_bet != 0.0:
        timediff = data_file.otf_start_bet / vel_bet - length / 2
    else:
        print(f"warning: {data_file.path}: zero OTF velocity, timediff set to 0")
        timediff = 0.0
    i = np.arange(data_file.n_dumps)
    t_begin = t_start + timediff + i * length
    return Windows(t_begin, t_begin + length, t_start, vel_lam, vel_bet,
                   timediff, source, spread)


def offsets_at(data_file, windows, times):
    """Map offset [arcsec] assigned to the data at times [unix sec]."""
    times = np.asarray(times, dtype=float)
    lam = data_file.lam_del + windows.vel_lam * (times - windows.t0)
    bet = data_file.bet_del + windows.vel_bet * (times - windows.t0)
    return lam, bet


def refraction_at(t, refractions, override):
    """AltAz refraction parameters in use at time t."""
    if override is not None:
        return override
    before = [event for event in refractions if event.t <= t]
    return before[-1].params if before else {}


# ===========================================================================
# Positions (same order of offsets as the translator)
# ===========================================================================

def apply_offset(coord, d_lam_arcsec, d_bet_arcsec, true_angle):
    """As kosma_ocs_translator._apply_offset."""
    if true_angle == "Y":
        return coord.spherical_offsets_by(d_lam_arcsec * u.arcsec, d_bet_arcsec * u.arcsec)
    return SkyCoord(coord.data.lon + d_lam_arcsec * u.arcsec,
                    coord.data.lat + d_bet_arcsec * u.arcsec, frame=coord.frame)


def apply_offset_in_frame(coord, frame, d_lam, d_bet, true_angle):
    """Apply an offset in frame and return the result in the frame of coord."""
    if not (np.any(d_lam != 0.0) or np.any(d_bet != 0.0)):
        return coord
    base = coord.frame.replicate_without_data()
    return apply_offset(coord.transform_to(frame), d_lam, d_bet, true_angle).transform_to(base)


def expected_altaz(data_file, times, lam_del, bet_del, location, refraction, focal_plane):
    """
    Expected (commanded) AltAz position at times for the map offsets
    lam_del/bet_del [arcsec], arrays of the same length as times.
    """
    times = np.asarray(times, dtype=float)
    n = len(times)
    altaz_frame = AltAz(obstime=Time(times, format="unix"), location=location, **refraction)
    frames = {name: (altaz_frame if frame == "altaz" else frame)
              for name, frame in COORD_SYS_MAP.items()}
    lam_del = np.broadcast_to(np.asarray(lam_del, dtype=float), (n,))
    bet_del = np.broadcast_to(np.asarray(bet_del, dtype=float), (n,))
    lam_off = np.full(n, data_file.lam_off)
    bet_off = np.full(n, data_file.bet_off)
    offsets = (
        (data_file.coord_sys_del, lam_del, bet_del, data_file.true_angle_del),
        (data_file.coord_sys_off, lam_off, bet_off, TRUE_ANGLE_OFF),
    )
    source_is_horizon = data_file.coord_sys_on == HORIZON
    coord = SkyCoord(np.full(n, data_file.lam_on) * u.deg, np.full(n, data_file.bet_on) * u.deg,
                     frame=frames[data_file.coord_sys_on])
    for coord_sys, d_lam, d_bet, true_angle in offsets:
        if source_is_horizon or coord_sys != HORIZON:
            coord = apply_offset_in_frame(coord, frames[coord_sys], d_lam, d_bet, true_angle)
    coord = coord.transform_to(altaz_frame)
    if not source_is_horizon:
        for coord_sys, d_lam, d_bet, true_angle in offsets:
            if coord_sys == HORIZON:
                coord = apply_offset_in_frame(coord, altaz_frame, d_lam, d_bet, true_angle)
    if focal_plane[0] != 0.0 or focal_plane[1] != 0.0:
        coord = apply_offset(coord, np.full(n, focal_plane[0]), np.full(n, focal_plane[1]), "Y")
    return coord


def altaz_to_icrs(az, el, times, location, refraction):
    frame = AltAz(obstime=Time(times, format="unix"), location=location, **refraction)
    icrs = SkyCoord(np.asarray(az) * u.deg, np.asarray(el) * u.deg, frame=frame).icrs
    return icrs.ra.deg, icrs.dec.deg


def wrap180(angle_deg):
    return (np.asarray(angle_deg) + 180.0) % 360.0 - 180.0


# ===========================================================================
# Comparison with housekeeping
# ===========================================================================

def compare_samples(data_file, windows, hk, location, refraction, focal_plane):
    """
    Differences of the housekeeping samples inside the windows from the
    expected position at the same time.

    Returns
    -------
    dict of arrays (one entry per sample), or None if no sample is inside
    """
    hk_t, hk_az, hk_el = hk
    window = np.searchsorted(windows.t_begin, hk_t, side="right") - 1
    inside = (window >= 0) & (hk_t <= windows.t_end[np.clip(window, 0, None)])
    if not np.any(inside):
        return None
    t, az, el, window = hk_t[inside], hk_az[inside], hk_el[inside], window[inside]
    lam_del, bet_del = offsets_at(data_file, windows, t)
    expected = expected_altaz(data_file, t, lam_del, bet_del, location, refraction,
                              focal_plane)
    exp_az, exp_el = expected.az.deg, expected.alt.deg
    cos_el = np.cos(np.deg2rad(exp_el))
    d_az = wrap180(az - exp_az) * cos_el * 3600.0
    d_el = (el - exp_el) * 3600.0

    # expected motion on the sky [arcsec/s] -> along (as time lag) / across
    t_next = t + MOTION_TIME_STEP
    lam_next, bet_next = offsets_at(data_file, windows, t_next)
    expected_next = expected_altaz(data_file, t_next, lam_next, bet_next, location,
                                   refraction, focal_plane)
    v_az = wrap180(expected_next.az.deg - exp_az) * cos_el * 3600.0 / MOTION_TIME_STEP
    v_el = (expected_next.alt.deg - exp_el) * 3600.0 / MOTION_TIME_STEP
    speed = np.hypot(v_az, v_el)
    moving = speed >= MIN_SPEED_FOR_LAG
    safe_speed = np.where(moving, speed, 1.0)
    lag = np.where(moving, (d_az * v_az + d_el * v_el) / safe_speed**2, np.nan)
    cross = np.where(moving, (d_el * v_az - d_az * v_el) / safe_speed, np.nan)

    hk_ra, hk_dec = altaz_to_icrs(az, el, t, location, refraction)
    exp_icrs = expected.icrs
    return {
        "window": window, "t": t, "hk_az": az, "hk_el": el,
        "exp_az": exp_az, "exp_el": exp_el, "d_az": d_az, "d_el": d_el,
        "speed": speed, "lag": lag, "cross": cross,
        "hk_ra": hk_ra, "hk_dec": hk_dec,
        "exp_ra": exp_icrs.ra.deg, "exp_dec": exp_icrs.dec.deg,
        "lam_del": lam_del, "bet_del": bet_del,
    }


def _nan_stat(function, values):
    values = values[np.isfinite(values)]
    return float(function(values)) if values.size else float("nan")


def window_statistics(samples, i_window):
    """Statistics of the samples of one window."""
    names = ("n_hk", "d_az_mean", "d_el_mean", "d_rms", "d_max",
             "lag_mean", "cross_mean")
    if samples is None:
        return dict.fromkeys(names, float("nan")) | {"n_hk": 0}
    mask = samples["window"] == i_window
    if not np.any(mask):
        return dict.fromkeys(names, float("nan")) | {"n_hk": 0}
    d_az, d_el = samples["d_az"][mask], samples["d_el"][mask]
    distance = np.hypot(d_az, d_el)
    return {
        "n_hk": int(mask.sum()),
        "d_az_mean": float(np.mean(d_az)),
        "d_el_mean": float(np.mean(d_el)),
        "d_rms": float(np.sqrt(np.mean(distance**2))),
        "d_max": float(np.max(distance)),
        "lag_mean": _nan_stat(np.mean, samples["lag"][mask]),
        "cross_mean": _nan_stat(np.mean, samples["cross"][mask]),
    }


# ===========================================================================
# Output
# ===========================================================================

def _fmt(value, digits):
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if value is None or not np.isfinite(value):
        return ""
    return f"{value:.{digits}f}"


WINDOW_COLUMNS = (
    ("file", None), ("scan", None), ("kind", None), ("sobsmode", None), ("dump", None),
    ("t_begin", 4), ("t_mid", 4), ("t_end", 4), ("start_source", None),
    ("start_spread", 4), ("timediff", 4),
    ("lam_del", 3), ("bet_del", 3), ("lam_off", 3), ("bet_off", 3),
    ("exp_ra", 7), ("exp_dec", 7), ("exp_az", 6), ("exp_el", 6), ("refraction", None),
    ("n_hk", None), ("d_az_mean", 2), ("d_el_mean", 2), ("d_rms", 2), ("d_max", 2),
    ("lag_mean", 3), ("cross_mean", 2),
)
SAMPLE_COLUMNS = (
    ("file", None), ("scan", None), ("kind", None), ("dump", None), ("t", 4),
    ("lam_del", 3), ("bet_del", 3),
    ("hk_az", 6), ("hk_el", 6), ("exp_az", 6), ("exp_el", 6),
    ("d_az", 2), ("d_el", 2), ("speed", 2), ("lag", 3), ("cross", 2),
    ("hk_ra", 7), ("hk_dec", 7), ("exp_ra", 7), ("exp_dec", 7),
)


def write_csv(path, columns, rows):
    with open(path, "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow([name for name, _ in columns])
        for row in rows:
            writer.writerow([row[name] if digits is None else _fmt(row[name], digits)
                             for name, digits in columns])


def refraction_text(refraction):
    if not refraction:
        return "none"
    return (f"P={refraction['pressure'].to_value(u.hPa):.1f}hPa "
            f"T={refraction['temperature'].to_value(u.deg_C):.1f}C "
            f"RH={refraction['relative_humidity']:.3f} "
            f"WL={refraction['obswl'].to_value(u.micron):g}um")


# ===========================================================================
# Main
# ===========================================================================

def process_file(data_file, log_starts, refractions, refraction_override, hk,
                 location, focal_plane):
    """
    Window rows and sample rows of one data file. A load (CAL) file gets its
    time window only: the telescope position does not matter for it.
    """
    windows = data_windows(data_file, log_starts)
    refraction = refraction_at(windows.t0, refractions, refraction_override)
    t_mid = (windows.t_begin + windows.t_end) / 2
    lam_del, bet_del = offsets_at(data_file, windows, t_mid)
    samples = None
    if data_file.is_load:
        exp_ra = exp_dec = exp_az = exp_el = np.full(len(t_mid), np.nan)
    else:
        expected = expected_altaz(data_file, t_mid, lam_del, bet_del, location, refraction,
                                  focal_plane)
        expected_icrs = expected.icrs
        exp_ra, exp_dec = expected_icrs.ra.deg, expected_icrs.dec.deg
        exp_az, exp_el = expected.az.deg, expected.alt.deg
        if hk is not None:
            samples = compare_samples(data_file, windows, hk, location, refraction,
                                      focal_plane)

    kind = data_kind(data_file)
    name = os.path.basename(data_file.path)
    window_rows = []
    for i in range(len(t_mid)):
        window_rows.append({
            "file": name, "scan": data_file.scan, "kind": kind,
            "sobsmode": data_file.subobsmode, "dump": i,
            "t_begin": windows.t_begin[i], "t_mid": t_mid[i], "t_end": windows.t_end[i],
            "start_source": windows.start_source, "start_spread": windows.start_spread,
            "timediff": windows.timediff,
            "lam_del": lam_del[i], "bet_del": bet_del[i],
            "lam_off": data_file.lam_off, "bet_off": data_file.bet_off,
            "exp_ra": exp_ra[i], "exp_dec": exp_dec[i],
            "exp_az": exp_az[i], "exp_el": exp_el[i],
            "refraction": refraction_text(refraction),
            **window_statistics(samples, i),
        })
    sample_rows = []
    if samples is not None:
        for j in range(len(samples["t"])):
            sample_rows.append({
                "file": name, "scan": data_file.scan, "kind": kind,
                "dump": int(samples["window"][j]),
                **{key: samples[key][j] for key, _ in SAMPLE_COLUMNS
                   if key in samples and key != "window"},
            })
    return windows, window_rows, sample_rows


def data_kind(data_file):
    """"OTF", "TP" (OFF or ON, see SOBSMODE) or "LOAD" (CAL)."""
    if data_file.is_otf:
        return "OTF"
    return "LOAD" if data_file.is_load else "TP"


def print_summary(data_file, windows, window_rows):
    kind = data_kind(data_file)
    line = (f"scan {data_file.scan:6d} {kind:4s} {data_file.subobsmode:3s} "
            f"start {windows.t0:.3f} ({windows.start_source}")
    if np.isfinite(windows.start_spread) and windows.start_spread > 0:
        line += f", logs differ by {windows.start_spread:.3f} s"
    line += f"), {len(window_rows)} window(s) of {data_file.int_time:.3f} s"
    if data_file.is_otf:
        line += f", timediff {windows.timediff:+.3f} s"
    n_hk = sum(row["n_hk"] for row in window_rows)
    if n_hk:
        lags = np.array([row["lag_mean"] for row in window_rows], dtype=float)
        rms = np.array([row["d_rms"] for row in window_rows], dtype=float)
        line += (f", hk {n_hk} samples: rms {np.nanmax(rms):.1f}\" (worst window), "
                 f"mean lag {_nan_stat(np.mean, lags):+.3f} s")
    print(line)


def main():
    args = parse_arguments()
    location = EarthLocation(lat=args.site[0] * u.deg, lon=args.site[1] * u.deg,
                             height=args.site[2] * u.m)
    log_starts, refractions = parse_logs(args.log, args.log_utc_offset)
    print(f"logs: {len(log_starts)} start times, {len(refractions)} refraction settings")

    refraction_override = None
    if args.no_refraction:
        refraction_override = {}
    elif args.refraction:
        pressure, temperature, humidity, wavelength = args.refraction
        refraction_override = {"pressure": pressure * u.hPa,
                               "temperature": temperature * u.deg_C,
                               "relative_humidity": humidity,
                               "obswl": wavelength * u.micron}
    elif not refractions:
        print("warning: no refraction setting in the logs, positions without refraction")

    hk = None
    if args.hk:
        hk = load_housekeeping(args.hk, parse_hk_columns(args.hk_cols), args.hk_delimiter,
                               args.hk_time_offset)
        print(f"housekeeping: {len(hk[0])} samples, {hk[0][0]:.3f} .. {hk[0][-1]:.3f}")
        if len(hk[0]) > 1:
            steps = np.diff(hk[0])
            print(f"housekeeping sample interval: median {np.median(steps):.3f} s, "
                  f"min {steps.min():.3f} s, max {steps.max():.3f} s")
    elif args.samples_out:
        sys.exit("--samples-out needs --hk")

    fits_files = find_fits_files(args.fits, args.pixel, args.scans)
    data_files = [f for f in map(read_data_file, fits_files) if f is not None]
    data_files.sort(key=lambda f: (f.utc_start_sec, f.scan))
    if not data_files:
        sys.exit("no TP/OTF FITS files")
    print(f"site used: lat {args.site[0]} lon(east) {args.site[1]} height {args.site[2]}; "
          f"FITS SITELAT/SITELONG/SITEALT = {data_files[0].site_header}")

    all_window_rows, all_sample_rows = [], []
    for data_file in data_files:
        windows, window_rows, sample_rows = process_file(
            data_file, log_starts, refractions, refraction_override, hk, location,
            args.focal_plane)
        print_summary(data_file, windows, window_rows)
        all_window_rows += window_rows
        all_sample_rows += sample_rows

    write_csv(args.out, WINDOW_COLUMNS, all_window_rows)
    print(f"written {args.out} ({len(all_window_rows)} rows)")
    if args.samples_out:
        write_csv(args.samples_out, SAMPLE_COLUMNS, all_sample_rows)
        print(f"written {args.samples_out} ({len(all_sample_rows)} rows)")


if __name__ == "__main__":
    main()
