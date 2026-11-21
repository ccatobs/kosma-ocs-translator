"""
Functions shared by kosma_ocs_translator.py and update_spem_acu.py

Sections
--------
- Logging: console and log file (setup_logging)
- OCS access: connection and credentials (setup_ocs)
- KOSMA_file_io: reading and writing files in the KOSMA_file_io ReadWrite directory
- SPEM: ACU SPEM parameters stored in KOSMA_file_io
- Astra pointing model: KOSMA_point.in -> KOSMA_point.status
"""

import contextlib
import datetime
import glob
import logging
import os
import re
import socket
import stat
import subprocess
import time

from astropy import units as u
from ocs import observatory_control_system

# PH1 roof mount
try:
    from roof_mount import RoofMount
except ImportError:
    print("roof_mount package not found, make sure it is installed and in the PYTHONPATH")
    print("install from git@git.ph1.uni-koeln.de:receiver/pointing-camera-rpi.git")


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
# logger used by the functions here if no logger is passed
DEFAULT_LOGGER_NAME = "kosma-ocs-translator"

# pass as extra= to a log call to write the message to the log file only
FILE_ONLY = {"file_only": True}

# ---------------------------------------------------------------------------
# KOSMA_file_io: SPEM parameters
# ---------------------------------------------------------------------------
# KOSMA_file_io file holding the ACU SPEM parameters (used for optical pointing)
KOSMA_SPEM_FILE = "KOSMA_ccatacu_point.in"

# keys in the ACU SPEM set not used for the comparison or the update
IGNORE_KEYS_ACU = ["Time", "Year"]
IGNORE_KEYS_KOSMA = ["ccatacu_model_date"]

# ACU SPEM parameters that are compared between ACU and KOSMA_file_io and
# dumped from the ACU to KOSMA_file_io, but never sent to the ACU (set_spem).
# They do not follow the string_ccatacu_point[i] / ccatacu_point[i] scheme:
# ACU name (without "Parameter ") -> KOSMA_file_io variable
ACU_READONLY_KEYS = {"LEL velocity": "ccatacu_velelvmax"}   # [deg/s]

# ccatacu_model_type written with the SPEM parameters (put_kosma_spem): the pointing
# model of the ACU. Copied to op_tel_model_type in KOSMA_ccdpos.sim; starpoint checks
# that it starts with "CCAT-ACU" (first 8 characters). No blanks: KOSMA_file_io reads
# the value with %s and Kset_point gets it as one argument.
CCAT_ACU_MODEL_TYPE = "CCAT-ACU-VA-Error-Compensations-document-Release-1.4"

# ---------------------------------------------------------------------------
# KOSMA_file_io: astra pointing model
# ---------------------------------------------------------------------------
# Astra pointing model, edited by hand, and its status copy read by astra_track,
# astra_point and starpoint (Astra_config.acfg: written by astra or CCAT_TRANSLATOR)
KOSMA_POINT_IN_FILE = "KOSMA_point.in"
KOSMA_POINT_STATUS_FILE = "KOSMA_point.status"


# ===========================================================================
# Logging
# ===========================================================================

def setup_logging(logger_name, log_file, log_level):
    """
    Set up a logger writing to the console and to a log file.

    Parameters
    ----------
    logger_name : str
        Name of the logger
    log_file : str
        Log file, relative to the current directory
    log_level : int
        Logging level, e.g. logging.INFO

    Returns
    -------
    logger : logging.Logger
        The console shows level and message only. The log file also has the
        time stamp and the logger name. log.info(..., extra=FILE_ONLY) goes to
        the log file only, e.g. when the same information is already printed
        to the terminal as a table.
    """
    logger = logging.getLogger(logger_name)
    logger.setLevel(log_level)
    # Prevent propagation to the root logger
    logger.propagate = False

    # console handler
    ch = logging.StreamHandler()
    ch.setLevel(log_level)
    ch.setFormatter(logging.Formatter("%(levelname)s: %(message)s"))
    ch.addFilter(lambda record: not getattr(record, "file_only", False))
    logger.addHandler(ch)

    # file handler
    fh = logging.FileHandler(log_file)
    fh.setLevel(log_level)
    fh.setFormatter(logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    ))
    logger.addHandler(fh)

    return logger


# ===========================================================================
# OCS access
# ===========================================================================

def setup_ocs(use_roof_mount=False, use_local_dev=False, certificates_path=None,
              logger=None):
    """
    Set up the connection to the telescope control (OCS or roof mount).

    Parameters
    ----------
    use_roof_mount : bool
        If True, connect to the roof mount instead of OCS.
    use_local_dev : bool
        If True, connect to the OCS of the local development setup.
        If neither use_roof_mount nor use_local_dev is True, connect to the
        OCS at the observatory.
    certificates_path : str
        Path to the directory with the OCS TLS certificates.
        Required when use_roof_mount is False.
    logger : logging.Logger, optional
        Logger to use. Defaults to the DEFAULT_LOGGER_NAME logger.

    Returns
    -------
    ocs : observatory_control_system or RoofMount
    """
    if logger is None:
        logger = logging.getLogger(DEFAULT_LOGGER_NAME)
    if use_roof_mount and use_local_dev:
        raise ValueError("use_roof_mount and use_local_dev cannot both be set")
    if use_roof_mount:
        logger.info("Using roof mount connection")
        roof_mount_host = "134.95.46.51"
        roof_mount_port = 2000
        ocs = RoofMount(host=roof_mount_host, port=roof_mount_port)
        ocs.connect()
        # set master_obs_tolerance to 1 arcminute
        obs_tolerance = 8*u.arcminute.to(u.arcsecond)
        logger.info(f"Setting master_obs_tolerance to {obs_tolerance} arcseconds")
        os.system(f"Kset_master master_obs_tolerance {obs_tolerance}")
        os.system(f"KOSMA_track")
    else:
        if use_local_dev:
            logger.info("Using OCS connection (local development)")
            ocs_host = "127.0.0.1"
            ocs_port = 5600
        else:
            logger.info("Using OCS connection at the observatory")
            # TODO: placeholder, replace with the actual OCS host and port
            ocs_host = "ocs.example.invalid"
            ocs_port = 5600
        print(f"Using certificates path: {certificates_path}")
        ocs = observatory_control_system(
            url=f"https://{ocs_host}:{ocs_port}",
            server_cert=f"{certificates_path}/server.cert.pem",
            client_cert=f"{certificates_path}/client.cert.pem",
            client_key=f"{certificates_path}/client.key.pem",
        )
    return ocs


# ===========================================================================
# KOSMA_file_io
# ===========================================================================

def ImportKOSMAReadWriteIntoDictionary(files=None, variable=None, update_mod_time=True):
    log = logging.getLogger(DEFAULT_LOGGER_NAME)
    readwrite_dict = {}
    kosma_readwrite_dir = os.environ.get("WRITE_DIR", "/net/KOSMA_file_io/ReadWrite/")
    not_changed_count = 0
    if files == None:
        readwrite_files = glob.glob("%s/*" % (kosma_readwrite_dir))
    else:
        readwrite_files = []
        for filename in files:
            # check mod time here, if hasn't changed from last time.. skip it
            full_path_filename = kosma_readwrite_dir + "/" + filename
            if not os.path.exists(full_path_filename):
                log.warning("WARNING: {0} not found".format(full_path_filename))
                continue
            else:
                readwrite_files.append(kosma_readwrite_dir + "/" + filename)
    #
    if len(readwrite_files) == 0:
        # print "No files active, check servers and %s path" % (kosma_readwrite_dir)
        return {}
    # create global dictionary to store kosma_read acfg values
    for readwrite_file_full in readwrite_files:
        log.debug("reading {0}".format(readwrite_file_full))
        if os.path.isdir(readwrite_file_full):
            continue
        if not os.path.exists(readwrite_file_full):
            continue
        f = open(readwrite_file_full, "r")
        lines = f.readlines()
        f.close()
        # strip full path from readwrite_file
        readwrite_file = os.path.basename(readwrite_file_full)
        # create dictionary for each filename
        if readwrite_file not in readwrite_dict.keys():
            readwrite_dict[readwrite_file] = {}
        readwrite_dict[readwrite_file]["file_timestamp"] = os.path.getmtime(
            readwrite_file_full
        )
        for i, line in enumerate(lines):
            # check for timestamp in header
            if re.match(".+File update time stamp.+", line):
                timestamp = re.search(r".+\s(\d+\.\d+)\s+\S+.+", line).groups()[0]
                # if 'timestamp' not in readwrite_dict[readwrite_file].keys():
                readwrite_dict[readwrite_file]["timestamp"] = timestamp
                continue
            elif re.match(".+!.+", line):
                result = re.search("(.+)!(.+)", line)
                if result:
                    data, description = result.groups()
            else:
                continue
            # log.debug("found {0} {1}".format(data, description))
            # if no timestamp found in file, use file timestamp
            if "timestamp" not in readwrite_dict[readwrite_file].keys():
                readwrite_dict[readwrite_file]["timestamp"] = readwrite_dict[
                    readwrite_file
                ]["file_timestamp"]
            # populate dictionary
            try:
                result = re.search(r"(\S+)\s+(\S+)", data)
                if result:
                    value, variable_found = result.groups()
            except Exception:
                result = re.search(r"\s+(\S+)", data)
                if result:
                    (variable_found,) = result.groups()
                print("{0} variable found with no value ".format(variable_found))
                value = "None "
            # variable_found = variable_found.strip()
            # convert parse value to string, int or float

            try:
                if type(value) in [int, float]:
                    continue
                if "." in value:
                    value = float(value)
                else:
                    value = int(value)
            except ValueError:
                value = value.strip()
            except Exception as e:
                # include readwrite_file in error log
                log.error(
                    f"error parsing value {value} for variable {variable_found} from file {readwrite_file}"
                )
                log.error(e)
                continue
            #
            if (variable is not None) and (variable != variable_found):
                continue
            if variable_found not in readwrite_dict[readwrite_file]:
                readwrite_dict[readwrite_file][variable_found] = value
    # remove entries that just have a timestamp and no variable values
    # for
    return readwrite_dict


def read_first_column(path, names):
    """
    Values of the variables names in a KOSMA_file_io file, as strings.
    Lines are "<value>   <name>   ! <comment>"; the name is the last field
    before "!", the value is the first field (a value containing blanks is cut
    at the first blank). The first occurrence of a variable is used.

    Returns
    -------
    values : dict
        name -> value for the names found
    """
    values = {}
    with open(path) as f:
        for line in f:
            fields = line.partition("!")[0].split()
            if len(fields) >= 2 and fields[-1] in names and fields[-1] not in values:
                values[fields[-1]] = fields[0]
    return values



def write_atomic(path, text):
    """
    Replace the file path with text, so that a reader sees either the complete
    old or the complete new file, never an empty or partly written one.
    Same scheme as WriteVarSetList in KOSMA_file_io.c: write a temporary file
    in the same directory, copy the mode of the existing file, rename it over path.

    Parameters
    ----------
    path : str
        File to be replaced
    text : str
        New content of the file
    """
    # same directory, so that the rename stays within one filesystem (atomic)
    tmp = f"{path}_{os.getpid()}"
    try:
        with open(tmp, "w") as f:
            f.write(text)
        # keep the mode of the existing file (KOSMA processes may run as other users)
        try:
            os.chmod(tmp, stat.S_IMODE(os.stat(path).st_mode))
        except FileNotFoundError:
            pass
        os.replace(tmp, path)
    except BaseException:
        # no leftover temporary file on error
        with contextlib.suppress(FileNotFoundError):
            os.remove(tmp)
        raise


# ===========================================================================
# SPEM
# ===========================================================================

def strip_parameter_prefix(spem):
    """
    Strip "Parameter " from the keys of the ACU dataset CmdSPEMParameter
    """
    return {k.removeprefix("Parameter "): v for k, v in spem.items()}


def get_kosma_spem():
    """
    Get ACU parameters stored in KOSMA_file_io

    Returns
    -------
    spem : dict
        Dictionary of ACU SPEM parameters
    """

    spem_kosma_format = ImportKOSMAReadWriteIntoDictionary(
        files = [KOSMA_SPEM_FILE])
    if KOSMA_SPEM_FILE not in spem_kosma_format:
        raise FileNotFoundError(f"{KOSMA_SPEM_FILE} not found in KOSMA_file_io")
    spem_kosma_format = spem_kosma_format[KOSMA_SPEM_FILE]

    # Make a dictionary of key = string_ccatacu_point[i], value = ccatacu_point[i]
    spem = {}
    for var, name in spem_kosma_format.items():
        m = re.fullmatch(r"string_ccatacu_point\[(\d+)\]", var)
        if not m:
            continue
        value_var = f"ccatacu_point[{m.group(1)}]"
        if value_var not in spem_kosma_format:
            raise ValueError(f"{KOSMA_SPEM_FILE} lacks {value_var} for {name}")
        spem[name] = float(spem_kosma_format[value_var])

    # ACU read-only parameters, stored under their ACU name for the comparison.
    # If missing in the file, the comparison reports them as only in ACU.
    for acu_name, kosma_var in ACU_READONLY_KEYS.items():
        if kosma_var in spem_kosma_format:
            spem[acu_name] = float(spem_kosma_format[kosma_var])

    # also include ccatacu_model_date
    spem["ccatacu_model_date"] = spem_kosma_format.get("ccatacu_model_date")

    return spem


def put_kosma_spem(spem, logger=None):
    """
    Update ACU parameters stored in KOSMA_file_io

    Parameters
    ----------
    spem : dict
        New SPEM parameters
    logger : logging.Logger, optional
        Logger to use. Defaults to the DEFAULT_LOGGER_NAME logger.

    Returns
    -------
    spem : dict
        SPEM parameters read back from KOSMA_file_io after the update
    """
    log = logger if logger is not None else logging.getLogger(DEFAULT_LOGGER_NAME)

    # inverse of get_kosma_spem is needed to map key to the idx in string_ccatacu_point
    spem_kosma_format = ImportKOSMAReadWriteIntoDictionary(
        files = [KOSMA_SPEM_FILE]).get(KOSMA_SPEM_FILE, {})
    name2idx = {}
    for var, name in spem_kosma_format.items():
        m = re.fullmatch(r"string_ccatacu_point\[(\d+)\]", var)
        if m:
            name2idx[name] = int(m.group(1))

    commands = []
    for key, value in spem.items():
        if key in IGNORE_KEYS_ACU or key in IGNORE_KEYS_KOSMA:
            continue
        if key in ACU_READONLY_KEYS:
            commands.append(f"Kset_point {ACU_READONLY_KEYS[key]} {value:.6f}")
            continue
        if key not in name2idx:
            raise KeyError(f"{key} not found in string_ccatacu_point")
        commands.append(f"Kset_point ccatacu_point[{name2idx[key]}] {value:.6f}")
    commands.append(f"Kset_point ccatacu_model_type {CCAT_ACU_MODEL_TYPE}")
    model_date = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d:%H:%M:%SZ")
    commands.append(f"Kset_point ccatacu_model_date {model_date}")

    # use a shell command, e.g. Kset_point ccatacu_point[0] 1
    failed = []
    for cmd in commands:
        result = subprocess.run(
            cmd.split(), stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
        )
        if result.returncode != 0:
            log.error(f"Error running command {cmd}: {result.stderr}")
            failed.append(cmd)
        else:
            log.debug(f"Command {cmd} ran successfully.")
    if failed:
        raise RuntimeError(f"Failed to update {KOSMA_SPEM_FILE}: {failed}")
    log.info(
        f"{KOSMA_SPEM_FILE} updated (ccatacu_model_type = {CCAT_ACU_MODEL_TYPE}, "
        f"ccatacu_model_date = {model_date})"
    )

    return get_kosma_spem()


# ===========================================================================
# Astra pointing model
# ===========================================================================

def copy_point_in_to_status(logger=None):
    """
    Copy the astra pointing model KOSMA_point.in to KOSMA_point.status, as astra
    does at its start (astra_fileio.c). All lines are copied unchanged, only the
    "File update time stamp" header line is replaced by a new one.

    Parameters
    ----------
    logger : logging.Logger, optional
        Logger to use. Defaults to the DEFAULT_LOGGER_NAME logger.

    Raises
    ------
    FileNotFoundError
        KOSMA_point.in does not exist (KOSMA_point.status is not touched)
    """
    log = logger if logger is not None else logging.getLogger(DEFAULT_LOGGER_NAME)
    kosma_readwrite_dir = os.environ.get("WRITE_DIR", "/net/KOSMA_file_io/ReadWrite/")
    src = os.path.join(kosma_readwrite_dir, KOSMA_POINT_IN_FILE)
    dst = os.path.join(kosma_readwrite_dir, KOSMA_POINT_STATUS_FILE)

    with open(src) as f:
        lines = [line for line in f if "File update time stamp" not in line]
    header = (
        f"{time.strftime('%d-%b-%Y  %H:%M:%S')}  {time.time()}   File update time stamp"
        f"   ! ccat_translator ({socket.gethostname()}:{os.getpid()})\n"
    )
    write_atomic(dst, header + "".join(lines))
    log.info(f"{src} copied to {dst}")
