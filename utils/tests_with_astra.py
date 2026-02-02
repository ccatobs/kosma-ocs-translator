import os
import subprocess
import argparse
import re
from xml.etree.ElementPath import find
import pandas as pd
import time
import logging
import astropy.units as u
import numpy as np
from astropy.coordinates import SkyCoord, EarthLocation, AltAz, ICRS
from astropy.time import Time
from datetime import timedelta
from scipy.optimize import minimize_scalar
import matplotlib.pyplot as plt


from astropy.time import Time, TimeDelta
import astropy.units as u
import numpy as np

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s: %(message)s",
    handlers=[logging.StreamHandler(), logging.FileHandler("tests_with_astra.log")],
)
logger = logging.getLogger("tests_with_astra")


from kosma_ocs_translator import KOSMA_translator, ImportKOSMAReadWriteIntoDictionary

# generate a table of obs2tel files and export to a csv file

# save obs2tel outputs to a row in a csv file


def get_visible_sky_range_in_j2000_coordinates(location, obstime, min_alt=10):
    """
    Calculates visible RA/Dec range using Astropy objects directly.

    Parameters:
        location (EarthLocation): Astropy EarthLocation object.
        obstime (Time): Astropy Time object.
        min_alt (int/float): Minimum altitude in degrees.
    """
    # 1. Create a sampling of the horizon at the min_alt boundary
    azimuths = np.linspace(0, 360, 1000) * u.deg
    altitudes = np.full_like(azimuths, min_alt * u.deg)

    # 2. Define the AltAz frame using the passed objects
    frame_altaz = AltAz(obstime=obstime, location=location)

    # 3. Transform horizon circle to ICRS (J2000)
    horizon_coords = SkyCoord(az=azimuths, alt=altitudes, frame=frame_altaz)
    icrs_coords = horizon_coords.transform_to(ICRS())

    # 4. Process Declination
    dec_values = icrs_coords.dec.degree
    lat_deg = location.lat.degree

    # Check if a Celestial Pole is visible (circumpolar region)
    if lat_deg > 0:  # Northern Hemisphere
        dec_max = 90.0 if (lat_deg + (90 - min_alt) >= 90) else np.max(dec_values)
        dec_min = np.min(dec_values)
    else:  # Southern Hemisphere
        dec_min = -90.0 if (lat_deg - (90 - min_alt) <= -90) else np.min(dec_values)
        dec_max = np.max(dec_values)

    # 5. Process Right Ascension (with wrap-around handling)
    ra_values = icrs_coords.ra.wrap_at(180 * u.deg).degree

    return {
        "ra_min": np.min(ra_values),
        "ra_max": np.max(ra_values),
        "dec_min": dec_min,
        "dec_max": dec_max,
        "LST": obstime.sidereal_time("mean", longitude=location.lon),
    }


def is_astra_running():
    logger = logging.getLogger("tests_with_astra")
    # check is log modtime in the last 10 seconds
    log_file = "/net/KOSMA_file_io/logs/astra.log"
    if not os.path.exists(log_file):
        logger.critical(f"Astra log file {log_file} does not exist.")
        return False
    mod_time = os.path.getmtime(log_file)
    current_time = time.time()
    if current_time - mod_time > 20:
        # add the time to the message
        logger.critical(f"Log file last modified at {time.ctime(mod_time)}")
        return False
    return True

def save_data_test_outputs_to_dataframe(test_data_frames):
    logger = logging.getLogger("tests_with_astra")
    logger.info("Saving test outputs to Excel   file.")
    with pd.ExcelWriter("astra_kosma_test_outputs.xlsx") as writer:
        for file, data_list in test_data_frames.items():
            df = pd.DataFrame(data_list)
            sheet_name = os.path.splitext(os.path.basename(file))[0]
            df.to_excel(writer, sheet_name=sheet_name, index=False)


# read obs2tel using ImportKOSMAReadWriteIntoDictionary and compare values with kosma-ocs-translator outputs
def save_test_outputs_to_dataframe(test_name="default"):
    logger = logging.getLogger("tests_with_astra")
    files = ImportKOSMAReadWriteIntoDictionary(
        files=["KOSMA_obs2tel.set", "KOSMA_tel2obs.set"]
    )
    # check if there is a astra_test_output.csv
    # read each file into a pandas dataframe
    test_data = {}
    for file, values in files.items():
        #
        values["test_name"] = test_name
        #
        test_data[file] = values
    #
    return test_data


def get_track_times_from_object(
    t0, source_coord, location, horizon=15.0 * u.deg, track_step=1.0 * u.hour
):
    """
    Calculates an array of times between the next rising and setting of a source.

    Parameters:
        t0 (Time): The starting Astropy Time object.
        source_coord (SkyCoord): The target coordinates (ICRS, Galactic, etc.).
        location (EarthLocation): The observer's location.
        horizon (Quantity): Elevation angle for rise/set (default 15 deg).
        track_step (Quantity): The interval between generated time points.
    """

    # 1. Calculate rise and set times
    # Note: Assuming rise_set_times_astropy is your existing utility function
    # print args
    print(
        f"Calculating rise/set times for source at {source_coord} from location {location} starting at {t0.iso}"
    )
    rise_time, set_time = rise_set_times_astropy(
        source=source_coord,
        location=location,
        t0=t0,
        horizon=horizon,
        step=1.0 * u.hour,
        max_hours=36,
    )

    # 2. Calculate the duration and number of steps
    print(f"Rise time: {rise_time}, Set time: {set_time}")
    # get time difference in hours
    time_diff = set_time - rise_time  # hours
    print("Time difference (hours):", time_diff)

    # Ensure the set time is actually after the rise time
    # (Handles cases where the source sets the next day)
    if time_diff.sec < 0:
        print(
            "Warning: Set time calculated is before rise time. Check horizon/constraints."
        )
        return [], rise_time, set_time

    # Calculate steps based on the provided track_step
    print("Time difference:", time_diff)
    print("Track step:", track_step)
    #
    n_steps = int(np.floor(time_diff.to_value(u.hour) / track_step.to_value(u.hour)))
    print(n_steps)

    # 3. Generate the array of times
    # We use TimeDelta to increment the rise_time
    astra_times = rise_time + (np.arange(n_steps + 1) * track_step)
    print(astra_times)
    #
    return astra_times, rise_time, set_time


def get_track_times(astra_time_str, source_name):
    cmd = [f"setsource {source_name}", "KOSMA_setoffset -l 00.0"]
    obs2tel, tel2obs = run_kosma_commands(cmd, astra_time=astra_time_str)
    # extra l,b from obs2tel
    l_offset = obs2tel["obs_lam_on"]
    b_offset = obs2tel["obs_bet_on"]
    obs_system = obs2tel["obs_coord_sys_on"]
    #
    frame = coord_sys_map[obs_system]
    # calculate the time for rising and setting
    source_coord = SkyCoord(l_offset * u.deg, b_offset * u.deg, frame=frame)
    observer_location = EarthLocation(
        lon=-1 * tel2obs["tel_longitude"] * u.deg,
        lat=tel2obs["tel_latitude"] * u.deg,
        height=tel2obs["tel_altitude"] * u.m,
    )
    rise_time, set_time = rise_set_times_astropy(
        source=source_coord,
        location=observer_location,
        t0=Time(astra_time_str, format="isot", scale="utc"),
        horizon=15.0 * u.deg,
        step=1.0 * u.hour,
        max_hours=36,
    )
    track_step = 1 * u.hr
    time_diff = abs(set_time - rise_time)
    n_steps = int((time_diff / (track_step * u.s)).decompose().value)
    astra_times = [rise_time + i * track_step for i in range(n_steps + 1)]
    return astra_times, rise_time, set_time


# Update the function to parse the full TELLMOK line format
def parse_tellmok_line(line: str):
    import re

    # Define the regex pattern for the full TELLMOK line
    pattern = r"(\S+) (\S+) UTC MESSAGE: \(.*?\) TELLMOK: TEST STEER: unix-time ([\d\.]+) time\s+(\d+), pos:\s+(-?\d+)\s+(-?\d+) vel:\s+(-?\d+)\s+(-?\d+)"
    match = re.match(pattern, line)

    if match:
        return {
            "iso_timestamp": match.group(1),
            "unix_time": float(match.group(3)),
            "time": int(match.group(4)),
            "az": int(match.group(5)) / 10000,
            "el": int(match.group(6)) / 10000,
            "vel_az": int(match.group(7)),
            "vel_el": int(match.group(8)),
        }
    return None


def rise_set_times_astropy(
    source, location, t0=None, horizon=0 * u.deg, step=0.5 * u.hour, max_hours=48
):
    if t0 is None:
        t0 = Time.now()

    h = horizon.to_value(u.deg)

    # 1. Create a broad time grid to find the transitions
    n_steps = int((max_hours * u.hour / step).decompose())
    times = t0 + np.arange(n_steps + 1) * step

    altaz_frame = AltAz(obstime=times, location=location, pressure=0 * u.bar)
    alt = source.transform_to(altaz_frame).alt.to_value(u.deg)

    rise_time = None
    set_time = None

    # 2. Find the FIRST Rising event (below -> above)
    rise_indices = np.where((alt[:-1] < h) & (alt[1:] >= h))[0]

    if len(rise_indices) > 0:
        i_r = rise_indices[0]
        # Linear interpolation for precision
        frac_r = (h - alt[i_r]) / (alt[i_r + 1] - alt[i_r])
        rise_time = times[i_r] + frac_r * (times[i_r + 1] - times[i_r])

        # 3. Find the FIRST Setting event that happens AFTER the Rise
        # We only look at indices greater than i_r
        set_indices = np.where((alt[:-1] >= h) & (alt[1:] < h))[0]
        # Filter for the first setting index that is greater than the rise index
        after_rise = set_indices[set_indices >= i_r]

        if len(after_rise) > 0:
            i_s = after_rise[0]
            frac_s = (h - alt[i_s]) / (alt[i_s + 1] - alt[i_s])
            set_time = times[i_s] + frac_s * (times[i_s + 1] - times[i_s])

    return rise_time, set_time


def get_astra_log_file_between_times(start_time: str, end_time: str) -> pd.DataFrame:
    file_path = "/net/KOSMA_file_io/logs/astra.log"
    print(f"Reading Astra log file between {start_time} and {end_time}")
    start_time = start_time.replace("T", " ")
    end_time = end_time.replace("T", " ")
    # open file and check for key value pairs seperated by =
    with open(file_path, "r") as f:
        lines = f.readlines()
        rows = []
        for line in lines:
            # Check for TELLMOK line
            if "TELLMOK: TEST STEER:" in line:
                parsed_data = parse_tellmok_line(line)
                if parsed_data:
                    rows.append(parsed_data)

    # Convert to DataFrame
    df = pd.DataFrame(rows)
    # Convert mjd to datetime
    # convert iso_timestamp to datetime
    df["sys_time"] = pd.to_datetime(df["iso_timestamp"])
    #
    mask = (df["sys_time"] >= start_time) & (df["sys_time"] <= end_time)
    return df.loc[mask]


def follow(file):
    file.seek(0, 2)  # Go to end of file
    while True:
        line = file.readline()
        if not line:
            time.sleep(0.2)  # wait for new data
            continue
        yield line


def parse_log_line_to_dict(log_line):
    """
    Parses a log line and extracts key-value pairs into a dictionary.

    Parameters:
        log_line (str): The log line to parse.

    Returns:
        dict: A dictionary containing the extracted key-value pairs.
    """
    # Define the regex pattern to match key-value pairs
    pattern = r"(\w+)=\s*([^\s\[\]]+)"

    # Find all matches
    matches = re.findall(pattern, log_line)
    # Convert matches to a dictionary, see if you can convert to float if possible
    key_value_pairs = {}
    for key, value in matches:
        try:
            key_value_pairs[key] = float(value)
        except ValueError:
            key_value_pairs[key] = value

    return key_value_pairs


def run_kosma_commands_in_fast_mode(cmds, astra_time=None):
    #
    logger = logging.getLogger("tests_with_astra")
    logger.info(f"Running KOSMA commands in fast mode.")
    # check if astra_time is an Time object
    if astra_time is not None and not isinstance(astra_time, Time):
        raise ValueError("astra_time must be an astropy Time object or None.")
    #
    for cmd in cmds:
        logger.info(f"Running command: {cmd}")
        result = subprocess.run(
            cmd.split(), stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
        )
        if result.returncode != 0:
            logger.error(f"Error running command {cmd}: {result.stderr}")
            raise SystemExit
        else:
            logger.info(f"Command {cmd} ran successfully.")
    # write astra_time_array to /tmp/astra_debug_time.txt
    if astra_time is None:
        return
    #
    logger.info(f"Astra times for testing: {astra_time}")
    with open("/net/KOSMA_file_io/share/astra/astra_debug_time.inp", "w") as f:
        f.write(astra_time.iso.replace(" ", "T") + "\n")
    # grep
    time.sleep(1)  # wait for a second to ensure file is written
    # tail log file and wait for OUT_ASTRA line and print to screen
    # log file is in /net/KOSMA_file_io/logs/astra.log
    # tail and wait for the string "OUT_ASTRA"
    # run system command tail -f /net/KOSMA_file_io/logs/astra.log | grep "OUT_ASTRA"
    # proceed when found

    #
    with open("/net/KOSMA_file_io/logs/astra.log", "r") as logfile:
        loglines = follow(logfile)
        for line in loglines:
            if "OUT_ASTRA" in line:
                return parse_log_line_to_dict(line)
    logger.error("No OUT_ASTRA line found in Astra log.")
    return None


def run_kosma_commands(cmds, astra_time=None):
    # start_time in iso format for logs
    logger = logging.getLogger("tests_with_astra")
    #
    if astra_time is not None:
        # check time format matches 2024-06-16 12:30:45 format
        time_format = r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}"
        if not re.match(time_format, astra_time):
            raise ValueError(
                f"Astra time format is incorrect: {astra_time}. Should be YYYY-MM-DD HH:MM:SS"
            )
        # Round astra_time to the nearest second
        astra_time_obj = Time(astra_time, format="isot", scale="utc")
        rounded_time = astra_time_obj.datetime + timedelta(seconds=0.5)
        astra_time = rounded_time.strftime("%Y-%m-%dT%H:%M:%S")
        # write string to /tmp/astra_debug_time.txt
        with open("/net/KOSMA_file_io/share/astra/astra_debug_time.inp", "w") as f:
            f.write(astra_time)
        #
        time.sleep(1)  # wait for a second to ensure file is written
        logger.info(f"Set Astra test time to {astra_time} for testing.")
    else:
        astra_time = Time.now().iso.replace(" ", "T")
        logger.info(f"Using current Astra time {astra_time} for testing.")
    #
    for cmd in cmds:
        logger.info(f"Running command: {cmd}")
        result = subprocess.run(
            cmd.split(), stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
        )
        if result.returncode != 0:
            logger.error(f"Error running command {cmd}: {result.stderr}")
            raise SystemExit
        else:
            logger.info(f"Command {cmd} ran successfully.")
    # run KOSMA_track command
    cmd = "KOSMA_track"
    result = subprocess.run(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
    )
    if result.returncode != 0:
        logger.error(f"Error running command {cmd}: {result.stderr}")
    else:
        logger.info(f"Command {cmd} ran successfully.")
    # wait for 2 seconds for astra to process inputs
    time_to_wait = 3
    logger.info(f"Waiting for {time_to_wait} seconds for Astra to process inputs.")
    time.sleep(time_to_wait)
    # wait for it to be on track tel_on_track = Y
    # poll tel2obs.set file
    attempts = 0
    while True:
        files = ImportKOSMAReadWriteIntoDictionary(["KOSMA_tel2obs.set"])
        on_track = files["KOSMA_tel2obs.set"]["tel_on_track"]
        if on_track == "Y":
            logger.info("Telescope is on track.")
            break
        else:
            logger.info("Telescope is not on track yet. Waiting...")
            time.sleep(1)  #   seconds
            attempts += 1
            if attempts > 10:
                logger.error("Telescope did not get on track within 10 seconds.")
                return None, None
    # read tel2obs and obs2tel files and return
    files = ImportKOSMAReadWriteIntoDictionary(
        files=["KOSMA_obs2tel.set", "KOSMA_tel2obs.set"]
    )
    #
    files["KOSMA_obs2tel.set"]["astra_time"] = astra_time
    files["KOSMA_tel2obs.set"]["astra_time"] = astra_time
    # attach cmds string to files
    files["KOSMA_obs2tel.set"]["test_name"] = "; ".join(cmds)
    files["KOSMA_tel2obs.set"]["test_name"] = "; ".join(cmds)
    #
    return files["KOSMA_obs2tel.set"], files["KOSMA_tel2obs.set"]


def compare_astropy_with_kosma_fast_mode(df_astra_parameters, coord, location):
    logger = logging.getLogger("tests_with_astra")
    logger.info("Comparing Astropy with KOSMA in fast mode.")
    results_list = []
    for index, row in df_astra_parameters.iterrows():
        astra_time = row["astra_time"]
        # run astropy calculations
        observation_time = Time(
            astra_time, format="isot", scale="utc", location=location
        )
        altaz_frame = AltAz(location=location, obstime=observation_time)
        altaz = coord.transform_to(altaz_frame)
        az_astropy = altaz.az.deg
        el_astropy = altaz.alt.deg
        az_kosma = row["AZ"]
        el_kosma = row["EL"]
        az_difference = round(az_astropy - az_kosma, 6)
        el_difference = round(el_astropy - el_kosma, 6)
        logger.info(
            f"astropy Azimuth: {az_astropy:.4f}, KOSMA Azimuth: {az_kosma:.4f}, Difference: {az_difference:.4f} ({az_difference * 3600.0:.4f} arcsec) \n"
            f"astropy Elevation: {el_astropy:.4f}, KOSMA Elevation: {el_kosma:.4f}, Difference: {el_difference:.4f} ({el_difference * 3600.0:.4f} arcsec)\n"
        )
        results_list.append(
            {
                "az_astropy": az_astropy,
                "az_kosma": az_kosma,
                "az_difference": az_difference,
                "el_astropy": el_astropy,
                "el_kosma": el_kosma,
                "el_difference": el_difference,
            }
        )
    return pd.DataFrame(results_list)


def compare_astropy_with_kosma(
    obs2tel,
    tel2obs,
    df_log,
    astra_status,
    offset_time_seconds=0,
    apply_refraction=False,
):
    logger = logging.getLogger("tests_with_astra")
    #
    # run obs2tel though astropy for testing
    frame = coord_sys_map[obs2tel["obs_coord_sys_on"]]
    coord = SkyCoord(
        obs2tel["obs_lam_on"] * u.deg,
        obs2tel["obs_bet_on"] * u.deg,
        frame=frame,
    )
    # astra treats longitudes as postive for west, astropy is negative for east
    observer_location = EarthLocation(
        lon=-1 * tel2obs["tel_longitude"].values[0] * u.deg,
        lat=tel2obs["tel_latitude"].values[0] * u.deg,
        height=tel2obs["tel_altitude"].values[0] * u.m,
    )
    # set timezone to UTC
    observation_time = Time(obs2tel.astra_time, scale="utc", location=observer_location)
    #
    offset_time_seconds += args.offset_time_seconds
    # offset by 2 seconds
    observation_time += offset_time_seconds * u.s
    # compare mjd from astra_status with astropy time
    astra_status_mjd = astra_status["a_dj1"]
    observation_time.to_value("mjd", "long")
    mjd_difference = observation_time.to_value("mjd", "long") - astra_status_mjd
    print(
        f"MJD Astropy: {observation_time.to_value('mjd', 'long')}, MJD Astra: {astra_status_mjd}"
    )
    # add mjd difference to observation_time
    # observation_time -= mjd_difference * 86400.0 * u.s
    logger.info(
        f"Time difference between Astropy and Astra status MJD: {mjd_difference * 86400.0:.6f} seconds"
    )

    # set as now
    # observation_time = Time.now()
    # add now instead of fixed time
    # read astra wetter
    files = ImportKOSMAReadWriteIntoDictionary(["KOSMA_wetter.status"])
    weather = files["KOSMA_wetter.status"]
    pressure = weather["wet_pressure"] * u.Torr
    temperature = weather["wet_temp"]  # convert to celcis from K
    temperature = (temperature - 273.15) * u.deg_C
    humidity = weather["wet_humidity"]

    # get altaz
    altaz_frame = AltAz(
        location=observer_location,
        obstime=observation_time,
        pressure=pressure,
        temperature=temperature,
        relative_humidity=humidity,
    )
    #
    altaz_frame_no_refraction_correction = AltAz(
        location=observer_location, obstime=observation_time
    )
    #
    if frame != "altaz":
        altaz = coord.transform_to(altaz_frame)
        altaz_frame_no_refraction_correction = coord.transform_to(
            altaz_frame_no_refraction_correction
        )
    else:
        altaz = coord
        altaz_frame_no_refraction_correction = coord

    # log diff with and without refraction correction
    az_no_refraction = altaz_frame_no_refraction_correction.az.deg
    el_no_refraction = altaz_frame_no_refraction_correction.alt.deg
    az_refraction = altaz.az.deg
    el_refraction = altaz.alt.deg
    az_refraction_difference = round(az_refraction - az_no_refraction, 6)
    el_refraction_difference = round(el_refraction - el_no_refraction, 6)
    logger.info(
        f"Refraction correction differences for test {obs2tel['test_name']}:\n"
        f"Elevation difference: {el_refraction_difference:.6f} deg ({el_refraction_difference * 3600.0:.4f} arcsec)\n"
    )
    #
    if apply_refraction is False:
        altaz = altaz_frame_no_refraction_correction
    az_astropy = altaz.az.deg
    el_astropy = altaz.alt.deg
    # compare with tel2obs values
    az_kosma = tel2obs["tel_azm_act"].values[0]
    el_kosma = tel2obs["tel_elv_act"].values[0]
    az_cmd_kosma = tel2obs["tel_azm_cmd"].values[0]
    el_cmd_kosma = tel2obs["tel_elv_cmd"].values[0]
    # log differences
    az_difference = round(az_astropy - az_kosma, 6)
    el_difference = round(el_astropy - el_kosma, 6)
    logger.info(
        f"Test Name: {obs2tel['test_name']}\n"
        f"astropy Azimuth: {az_astropy:.4f}, KOSMA Azimuth: {az_kosma:.4f}, Difference: {az_difference:.4f} ({az_difference * 3600.0:.4f} arcsec) \n"
        f"astropy Elevation: {el_astropy:.4f}, KOSMA Elevation: {el_kosma:.4f}, Difference: {el_difference:.4f} ({el_difference * 3600.0:.4f} arcsec)\n"
    )
    # from df_log get row with closest time to obs2tel astra_time + offset_time_seconds
    # use the coordinate_time column
    df_log["time_diff"] = abs(df_log["unix_time"] - tel2obs["timestamp"].values[0])
    closest_row = df_log.loc[df_log["time_diff"].idxmin()]
    #
    # results
    results = {
        "az_difference_arc_sec": az_difference * 3600.0,
        "el_difference_arc_sec": el_difference * 3600.0,
        "tel_azm_act": az_kosma,
        "tel_elv_act": el_kosma,
        "astropy_az": az_astropy,
        "astropy_el": el_astropy,
        "test_name": obs2tel["test_name"],
        "time": obs2tel["astra_time"],
        "offset_time_seconds": offset_time_seconds,
        "astra_log_az": closest_row["az"],
        "astra_log_el": closest_row["el"],
        "log_diff_to_astropy_az_arc_sec": (az_astropy - closest_row["az"]) * 3600.0,
        "log_diff_to_astropy_el_arc_sec": (el_astropy - closest_row["el"]) * 3600.0,
        "mjd_astropy": observation_time.to_value("mjd", "long"),
        "mjd_astra": astra_status["a_dj1"],
        "mjd_difference_seconds": mjd_difference * 86400.0,
    }
    return results


def zero_pointing_model():
    logger = logging.getLogger("tests_with_astra")
    logger.info("Checking pointing model parameters.")
    # check if pointing model is set to zero
    files = ImportKOSMAReadWriteIntoDictionary(["KOSMA_point.in", "KOSMA_point.sim"])
    # loop over a_point variables and check if they are zero
    kosma_point_in = files["KOSMA_point.in"]
    kosma_point_sim = files["KOSMA_point.sim"]
    non_zero_points = []
    for values in [kosma_point_in, kosma_point_sim]:
        for var, value in values.items():
            if var.startswith("a_point"):
                if value != 0.0:
                    non_zero_points.append(var)
    #
    if non_zero_points:
        # set pointing model to zero
        for var in non_zero_points:
            # run system command Kset_point var 0.0
            cmd = f"Kset_point {var} 0.0"
            result = subprocess.run(
                cmd.split(), stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
            )
            if result.returncode != 0:
                logger.error(f"Error running command {cmd}: {result.stderr}")
            else:
                logger.info(f"Command {cmd} ran successfully.")
        raise SystemExit(
            "You shoud restart Astra with the new pointing model set to zero."
        )


def compare_kosma_tests_with_astropy(
    make_plots=True, figure_tag="", plot_logs=False, fixed_offset=0.0
):
    logger = logging.getLogger("tests_with_astra")
    logger.info("Comparing KOSMA tests with Astropy outputs.")
    #
    df_obs2tel = pd.read_excel(
        "astra_kosma_test_outputs.xlsx", sheet_name="KOSMA_obs2tel"
    )
    df_tel2obs = pd.read_excel(
        "astra_kosma_test_outputs.xlsx", sheet_name="KOSMA_tel2obs"
    )
    df_astra_status = pd.read_excel(
        "astra_kosma_test_outputs.xlsx", sheet_name="KOSMA_astra"
    )
    # check each test row and compare to astropy outputs
    # range of 20
    # fixed_offset = -0.03
    # fixed_offset = 0.0
    # time_offset_range = [fixed_offset + offset / 10 for offset in range(-10, 10, 1)]
    time_offset_range = []
    # get the max and min time
    start_time = df_obs2tel["timestamp"].min()
    end_time = df_obs2tel["timestamp"].max()
    # convert from unix timestamp to iso time strings
    start_time_iso = pd.to_datetime(start_time, unit="s").isoformat()
    end_time_iso = pd.to_datetime(end_time, unit="s").isoformat()
    # get log data between these times
    df_log = get_astra_log_file_between_times(start_time_iso, end_time_iso)
    # save log data to excel sheet
    #
    results_list = []
    for index, row in df_obs2tel.iterrows():
        # process each row
        obs2tel = row
        # get cookies and find tel2obs return
        cookie = obs2tel["obs_cookie"]
        tel2obs = df_tel2obs[df_tel2obs["tel_return_cookie"] == cookie]
        astra_status = df_astra_status.loc[index]
        # check that there is only one matching tel2obs
        if len(tel2obs) != 1:
            logger.error(
                f"Error: Found {len(tel2obs)} matching tel2obs entries for cookie {cookie}"
            )
            # raise SystemExit
            continue
        #
        if len(time_offset_range) > 1:
            plot_axis = "offset_time_seconds"
            for offset_time_seconds in time_offset_range:
                results = compare_astropy_with_kosma(
                    obs2tel,
                    tel2obs,
                    df_log,
                    astra_status,
                    offset_time_seconds=offset_time_seconds,
                )
                results_list.append(results)
        else:
            plot_axis = "time_dt"
            results = compare_astropy_with_kosma(
                obs2tel, tel2obs, df_log, astra_status, fixed_offset
            )
            results_list.append(results)
        #
    results_df = pd.DataFrame(results_list)
    # add as sheet to the existing excel file
    logger.info(
        "Saving comparison results to Excel file astra_kosma_test_outputs.xlsx."
    )
    with pd.ExcelWriter(
        "astra_kosma_test_outputs.xlsx", mode="a", engine="openpyxl"
    ) as writer:
        # if sheet exists, remove it
        if "comparison_results" in writer.book.sheetnames:
            std = writer.book["comparison_results"]
            writer.book.remove(std)
            writer.book.save("astra_kosma_test_outputs.xlsx")
        #
        results_df.to_excel(writer, sheet_name="comparison_results", index=False)
    #
    if make_plots:
        make_comparison_plots(
            results_df, plot_axis=plot_axis, plot_logs=plot_logs, figure_tag=figure_tag
        )
    return results_df


def make_comparison_plots(
    results_df, plot_axis="time_dt", plot_logs=False, figure_tag=""
):
    # make a 2x2 panel plot: absolute values on top, residuals on the bottom
    import matplotlib.pyplot as plt

    fig, axs = plt.subplots(2, 2, figsize=(14, 10), sharex=False)
    fig.suptitle(f"Astra  vs KOSMA-OCS-Translator Pointing Analysis\n{figure_tag}")

    # Convert time strings to datetime objects
    results_df["time_dt"] = pd.to_datetime(results_df["time"])

    # Calculate statistics for residuals from log data
    fixed_offset = results_df["offset_time_seconds"].iloc[0]
    az_residual_mean = results_df["az_difference_arc_sec"].mean()
    az_residual_std = results_df["az_difference_arc_sec"].std()
    el_residual_mean = results_df["el_difference_arc_sec"].mean()
    el_residual_std = results_df["el_difference_arc_sec"].std()

    # Plot Azimuth absolute values (top-left)
    axs[0, 0].plot(
        results_df["time_dt"],
        results_df["tel_azm_act"],
        "-o",
        label="KOSMA Azimuth",
        color="blue",
        linewidth=4,  # Thicker line for KOSMA
    )
    axs[0, 0].plot(
        results_df["time_dt"],
        results_df["astropy_az"],
        "-o",
        label="Astropy Azimuth",
        color="red",
        linewidth=2,  # Thinner line for Astropy
    )
    if plot_logs:
        # plot log data azimuth as black dots
        axs[0, 0].plot(
            results_df["time_dt"],
            results_df["astra_log_az"],
            "k.",
            label="Astra Log Azimuth",
            markersize=8,
        )
    axs[0, 0].set_ylabel("Azimuth (deg)")
    axs[0, 0].legend()
    axs[0, 0].grid()

    # Plot Elevation absolute values (top-right)
    axs[0, 1].plot(
        results_df["time_dt"],
        results_df["tel_elv_act"],
        "-o",
        label="KOSMA Elevation",
        color="blue",
        linewidth=4,  # Thicker line for KOSMA
    )
    axs[0, 1].plot(
        results_df["time_dt"],
        results_df["astropy_el"],
        "-o",
        label="Astropy Elevation",
        color="red",
        linewidth=2,  # Thinner line for Astropy
    )
    if plot_logs:
        # plot log data elevation as black dots
        axs[0, 1].plot(
            results_df["time_dt"],
            results_df["astra_log_el"],
            "k.",
            label="Astra Log Elevation",
            markersize=8,
        )
    axs[0, 1].set_ylabel("Elevation (deg)")
    axs[0, 1].legend()
    axs[0, 1].grid()

    # Plot Azimuth residuals (bottom-left)
    time_axis = plot_axis
    if time_axis == "time_dt":
        # add fig suptitle with offset time seconds
        fig.suptitle(
            f"Astra vs KOSMA-OCS-Translator Pointing Analysis, coords {figure_tag}\n(Offset Time: {fixed_offset} seconds)"
        )
    axs[1, 0].plot(
        results_df[time_axis],
        results_df["az_difference_arc_sec"],
        marker="o",
        linestyle="-",
        color="blue",  # Same color as KOSMA
    )
    if plot_logs:
        # plot log residuals as black dots
        axs[1, 0].plot(
            results_df[time_axis],
            results_df["log_diff_to_astropy_az_arc_sec"],
            "k.",
            label="Astra Log Residuals",
            markersize=8,
        )
    axs[1, 0].set_ylabel("Azimuth Residual (arcsec)")
    axs[1, 0].set_xlabel("Time")
    axs[1, 0].grid()
    # Add annotation for azimuth residuals
    axs[1, 0].annotate(
        f"Mean: {az_residual_mean:.2f} arcsec\nStddev: {az_residual_std:.2f} arcsec",
        xy=(0.02, 0.95),
        xycoords="axes fraction",
        fontsize=10,
        color="blue",
        verticalalignment="top",
    )

    # Plot Elevation residuals (bottom-right)
    axs[1, 1].plot(
        results_df[time_axis],
        results_df["el_difference_arc_sec"],
        marker="o",
        linestyle="-",
        color="red",  # Same color as Astropy
    )
    if plot_logs:
        # plot log residuals as black dots
        axs[1, 1].plot(
            results_df[time_axis],
            results_df["log_diff_to_astropy_el_arc_sec"],
            "k.",
            label="Astra Log Residuals",
            markersize=8,
        )
    axs[1, 1].set_ylabel("Elevation Residual (arcsec)")
    axs[1, 1].set_xlabel("Time")
    axs[1, 1].grid()
    # Add annotation for elevation residuals
    axs[1, 1].annotate(
        f"Mean: {el_residual_mean:.2f} arcsec\nStddev: {el_residual_std:.2f} arcsec",
        xy=(0.02, 0.95),
        xycoords="axes fraction",
        fontsize=10,
        color="red",
        verticalalignment="top",
    )

    # Adjust layout
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    figure_filename = f"figures/astra_kosma_pointing_analysis_{figure_tag}.png"
    logger.info(f"Saving pointing analysis plot to {figure_filename}")
    plt.savefig(figure_filename)
    # plt.show()


def run_track_test(source_name, coord, location, obs_time):
    cmd_create_source = (
        f"setsource {source_name} -l {coord.ra.deg} -b {coord.dec.deg} -C J2000"
    )
    cmd = [cmd_create_source, "KOSMA_setoffset -l 00.0", "setpoint -p L"]
    # get rise and set times for this source
    try:
        astra_times, rise_time, set_time = get_track_times_from_object(
            obs_time, coord, location
        )
    except Exception as e:
        logger.error(
            f"Error getting track times for source {source_name} at RA: {ra}, Dec: {dec}: {e}"
        )
        return None
    start_time = Time.now().iso.replace(" ", "T")
    test_data_frames = {}
    test_data_frames["KOSMA_obs2tel.set"] = []
    test_data_frames["KOSMA_tel2obs.set"] = []
    test_data_frames["KOSMA_astra.status"] = []
    # commands to run for testing

    cmd = [cmd_create_source, "KOSMA_setoffset -l 00.0", "setpoint -p L"]
    # make an array of times from rise to set time

    # collect obs2tel and get coord
    logger.info(
        f"Running track tests from rise time {rise_time.iso} to set time {set_time.iso}"
    )
    logger.info(f"Total of {len(astra_times)} track tests to run.")
    for astra_time in astra_times:
        # round out to nearest second
        astra_time = Time(
            astra_time.datetime.strftime("%Y-%m-%dT%H:%M:%S"), location=location
        )
        #
        astra_time_str = astra_time.iso.replace(" ", "T")
        logger.info(f"Running track test at Astra time: {astra_time_str}")
        obs2tel, tel2obs = run_kosma_commands(cmd, astra_time=astra_time_str)
        if obs2tel is None or tel2obs is None:
            logger.error("Error running KOSMA commands. Skipping this test.")
            continue
        test_data_frames["KOSMA_obs2tel.set"].append(obs2tel)
        test_data_frames["KOSMA_tel2obs.set"].append(tel2obs)
        # read KOSMA_astra.status file
        files = ImportKOSMAReadWriteIntoDictionary(["KOSMA_astra.status"])
        test_data_frames["KOSMA_astra.status"].append(files["KOSMA_astra.status"])
    # make datraframe for each file for later analysis
    # save output to an excel file in different sheets
    logger.info("Saving test outputs to Excel file.")
    with pd.ExcelWriter("astra_kosma_test_outputs.xlsx") as writer:
        for file, data_list in test_data_frames.items():
            df = pd.DataFrame(data_list)
            sheet_name = os.path.splitext(os.path.basename(file))[0]
            df.to_excel(writer, sheet_name=sheet_name, index=False)
    logger.info("Test outputs saved successfully.")
    #
    date_tag = astra_time.datetime.strftime("%Y-%m-%d")
    figure_tag = f"RA{coord.ra.deg:03.0f}_DEC{coord.dec.deg:03.0f}_{date_tag}"
    results_df = compare_kosma_tests_with_astropy(
        make_plots=True,
        figure_tag=figure_tag,
        fixed_offset=0.0,
    )
    # add figure tag to results_df
    results_df["figure_tag"] = figure_tag
    return results_df


# Define the objective function to minimize
def minimize_azimith_residuals(time_offset):
    # Call compare_kosma_tests_with_astropy with the given time_offset
    results_df = compare_kosma_tests_with_astropy(
        make_plots=False,  # Disable plotting for optimization
        fixed_offset=time_offset,
    )
    # Extract the residual_az_std value from the results
    residual_az_std = results_df["az_difference_arc_sec"].std()
    return residual_az_std


# Add argument parsing
parser = argparse.ArgumentParser(description="Run Astra tests and compare results.")
parser.add_argument(
    "--run-astra-tests",
    action="store_true",
    help="Run Astra tests before proceeding with the script.",
)
parser.add_argument(
    "--run-track-tests-J2000",
    action="store_true",
    help="Run track tests for J2000 coordinate system.",
)

parser.add_argument(
    "--run-track-tests-galactic",
    action="store_true",
    help="Run track tests for Galactic coordinate system.",
)

parser.add_argument(
    "--run-horizon-tests",
    action="store_true",
    help="Run track tests for Horizon coordinate system.",
)

# add a compare only option
parser.add_argument(
    "--compare-data-only",
    action="store_true",
    help="Just run comparison and not run any tests.",
)

# offtime option
parser.add_argument(
    "--offset-time-seconds",
    type=float,
    default=0.0,
    help="Offset time in seconds to apply when comparing Astra and Astropy times.",
)

# parametric j2000 option
parser.add_argument(
    "--parametric-j2000-tests",
    action="store_true",
    help="Run parametric tests for J2000 coordinate and galactic system.",
)

parser.add_argument(
    "--find-optimal-time-offset",
    action="store_true",
    help="Find the optimal time offset to minimize azimuth residuals.",
)

parser.add_argument(
    "--run-single-ra-dec-test",
    action="store_true",
    help="Run a single test for a specific RA and Dec.",
)

parser.add_argument(
    "--run-track-tests-J2000-every-month",
    action="store_true",
    help="Run track tests for J2000 coordinate system every month.",
)

global args
args = parser.parse_args()

#
coord_sys_map = {
    "J2000": "icrs",
    "B1950": "fk4",
    "GALACTIC": "galactic",
    "HORIZON": "altaz",
}


zero_pointing_model()
if not is_astra_running():
    raise SystemExit("Astra is not running. Please start Astra and try again.")

files = ImportKOSMAReadWriteIntoDictionary(["KOSMA_tel2obs.set", "measurement.set"])
tel2obs = files["KOSMA_tel2obs.set"]
location = EarthLocation(
    lon=-1 * tel2obs["tel_longitude"] * u.deg,
    lat=tel2obs["tel_latitude"] * u.deg,
    height=tel2obs["tel_altitude"] * u.m,
)
# check obs wavelength is zero to avoid refraction
measrement = files["measurement.set"]
if measrement["wavelength"] != 0.0:
    print("Wavelength is not zero. Refraction correction may be applied.")


if args.run_single_ra_dec_test:
    ra = -180
    dec = -50
    source_name = f"TEST_RA{ra:03.0f}_DEC{dec:03.0f}"
    cmd_create_source = f"setsource {source_name} -l {ra} -b {dec} -C J2000"
    coord = SkyCoord(ra * u.deg, dec * u.deg, frame="icrs")
    obs_time = Time(
        "2026-02-10T16:00:00",
        location=location,
        scale="utc",
    )
    #
    astra_times, rise_time, set_time = get_track_times_from_object(
        obs_time, coord, location
    )
    # get rise and set times for this source
    cmd = [cmd_create_source, "KOSMA_setoffset -l 00.0", "setpoint -p L"]
    cmd.append("KOSMA_track")
    #
    astra_parameters = []
    for astra_time in astra_times:
        print(f"Astra time: {astra_time.iso}")
        astra_parameter = run_kosma_commands_in_fast_mode(cmd, astra_time=astra_time)
        # compare with astropy
        astra_parameter.update({"astra_time": astra_time})
        astra_parameter.update({"source_name": source_name})
        astra_parameters.append(astra_parameter)

    # make dataframe
    df_astra_parameters = pd.DataFrame(astra_parameters)
    #
    results_df = compare_astropy_with_kosma_fast_mode(
        df_astra_parameters, coord, location
    )


if args.run_track_tests_J2000:
    #
    ra = 180
    dec = -50
    source_name = f"TEST_RA{ra:03.0f}_DEC{dec:03.0f}"
    coord = SkyCoord(ra * u.deg, dec * u.deg, frame="icrs")
    obs_time = Time(
        "2026-06-23T10:00:00",
        location=location,
        scale="utc",
    )
    # print the utc-ut1 difference
    ut1_utc_difference = obs_time.ut1 - obs_time.utc
    print(
        f"########## UT1-UTC difference at observation time: {ut1_utc_difference.value:.10f} seconds"
    )
    #
    results_df = run_track_test(source_name, coord, location, obs_time)
    #
    figure_tag = f"RA{coord.ra.deg:03.0f}_DEC{coord.dec.deg:03.0f}_mjd_fixed"
    results_df = compare_kosma_tests_with_astropy(
        make_plots=True, figure_tag=figure_tag, fixed_offset=0.0
    )
    # save dataframe to excel with figure_tag
    results_df.to_excel(
        f"figures/astra_kosma_test_results_{figure_tag}.xlsx", index=False
    )

if args.run_track_tests_J2000_every_month:
    #
    ra = 180
    dec = -50
    residuals = []
    source_name = f"TEST_RA{ra:03.0f}_DEC{dec:03.0f}"
    coord = SkyCoord(ra * u.deg, dec * u.deg, frame="icrs")
    obs_time_start = Time(
        "2026-02-24T16:00:00",
        location=location,
        scale="utc",
    )
    # make a ran of every week for a year
    # Define the start and end of the year
    start_time = obs_time_start
    end_time = obs_time_start + TimeDelta(300 * u.day)

    # Calculate the number of weeks (inclusive)
    day_step = 7
    n_weeks = int(np.ceil((end_time - start_time) / TimeDelta(day_step * u.day))) + 1

    # Generate the weekly times
    weekly_times = start_time + TimeDelta(day_step * u.day) * np.arange(n_weeks)
    for obs_time in weekly_times:
        print("######### running test for obs_time:", obs_time.iso)
        results_df = run_track_test(source_name, coord, location, obs_time)

        if results_df is None:
            continue
        # if results_df is None:
        #    continue
        # find optimal time offset
        fit_result = minimize_scalar(
            minimize_azimith_residuals, bounds=(-0.6, 0.6), method="bounded"
        )
        residual = {
            "optimal_time_offset_seconds": fit_result.x,
            "minimized_azimuth_residual_std_arcsec": fit_result.fun,
            "azimuth_residual_std_arcsec": results_df["az_difference_arc_sec"].std(),
            "elevation_residual_std_arcsec": results_df["el_difference_arc_sec"].std(),
            "azimuth_residual_mean_arcsec": results_df["az_difference_arc_sec"].mean(),
            "elevation_residual_mean_arcsec": results_df[
                "el_difference_arc_sec"
            ].mean(),
            "number_of_iterations": fit_result.nit,
            "obstime": obs_time.datetime.isoformat(),
            "source": source_name,
        }
        #
        figure_tag = results_df["figure_tag"].unique()[0] + "_minimized"
        #
        results_df = compare_kosma_tests_with_astropy(
            make_plots=True, figure_tag=figure_tag, fixed_offset=fit_result.x
        )
        residual["azimuth_residual_std_arcsec_minimized"] = results_df[
            "az_difference_arc_sec"
        ].std()
        residual["elevation_residual_std_arcsec_minimized"] = results_df[
            "el_difference_arc_sec"
        ].std()
        residual["azimuth_residual_mean_arcsec_minimized"] = results_df[
            "az_difference_arc_sec"
        ].mean()
        residual["elevation_residual_mean_arcsec_minimized"] = results_df[
            "el_difference_arc_sec"
        ].mean()
        residuals.append(residual)
    # make dataframe
    df_residuals = pd.DataFrame(residuals)
    # save to excel
    df_residuals.to_excel(
        f"astra_kosma_monthly_optimal_time_offsets_RA{ra}_DEC{dec}.xlsx", index=False
    )
    # make a plot of optimal time offsets vs month

    # convert obstime to datetime
    df_residuals["obstime_dt"] = pd.to_datetime(df_residuals["obstime"])

    plt.figure(figsize=(10, 6))
    plt.plot_date(
        df_residuals["obstime_dt"], df_residuals["optimal_time_offset_seconds"], "o"
    )
    plt.xlabel("Time")
    plt.ylabel("Optimal Time Offset (seconds)")
    plt.title("Optimal Time Offset vs Month for KOSMA Astra Tests")
    plt.suptitle(f"Source: {source_name}, RA: {ra} deg, Dec: {dec} deg")
    plt.grid()
    plt.savefig(f"figures/optimal_time_offset_vs_month_RA{ra}_DEC{dec}.png")


if args.run_track_tests_galactic:
    #
    test_data_frames = {}
    test_data_frames["KOSMA_obs2tel.set"] = []
    test_data_frames["KOSMA_tel2obs.set"] = []
    test_data_frames["KOSMA_astra.status"] = []
    # commands to run for testing
    astra_time_str = "2026-02-10T14:00:00"
    source_name = "G347_PeakB"
    astra_times, rise_time, set_time = get_track_times(astra_time_str, source_name)
    # make an array of times from rise to set time

    # collect obs2tel and get coord
    logger.info(
        f"Running track tests from rise time {rise_time.iso} to set time {set_time.iso}",
    )
    logger.info(f"Total of {len(astra_times)} track tests to run.")
    for astra_time in astra_times:
        #
        logger.info(f"Running track test at Astra time: {astra_time.iso}")
        astra_time_str = astra_time.iso.replace(" ", "T")
        cmd = [f"setsource {source_name}", "KOSMA_setoffset -l 00.0"]
        obs2tel, tel2obs = run_kosma_commands(cmd, astra_time=astra_time_str)
        if obs2tel is None or tel2obs is None:
            logger.error("Error running KOSMA commands. Skipping this test.")
            continue
        test_data_frames["KOSMA_obs2tel.set"].append(obs2tel)
        test_data_frames["KOSMA_tel2obs.set"].append(tel2obs)
        # read KOSMA_astra.status file
        files = ImportKOSMAReadWriteIntoDictionary(["KOSMA_astra.status"])
        test_data_frames["KOSMA_astra.status"].append(files["KOSMA_astra.status"])
    # make datraframe for each file for later analysis
    # save output to an excel file in different sheets
    logger.info("Saving test outputs to Excel file.")
    with pd.ExcelWriter("astra_kosma_test_outputs.xlsx") as writer:
        for file, data_list in test_data_frames.items():
            df = pd.DataFrame(data_list)
            sheet_name = os.path.splitext(os.path.basename(file))[0]
            df.to_excel(writer, sheet_name=sheet_name, index=False)
    logger.info("Test outputs saved successfully.")
    #
    compare_kosma_tests_with_astropy(make_plots=True)

# Check if Astra tests should be run
if args.run_astra_tests:
    logger.info("Running Astra tests...")
    # Add logic to run Astra tests here
    # Example: subprocess.run(["astra", "test-command"]
    #
    #
    cmds = []
    cmds += [["setsource H40_40 -l 40 -b 40 -C HORIZON"]]  # HORIZONAL
    cmds += [["setsource H30_30 -l 30 -b 30 -C HORIZON"]]  # HORIZONAL
    cmds += [["setsource H30_30 -l 20 -b 20 -C HORIZON"]]  # HORIZONAL
    cmds += [["setsource H30_30 -l 10 -b 10 -C HORIZON"]]  # HORIZONAL
    test_data_frames = {}
    test_data_frames["KOSMA_obs2tel.set"] = []
    test_data_frames["KOSMA_tel2obs.set"] = []
    test_data_frames["KOSMA_astra.status"] = []
    # commands to run for testing
    astra_time = "2026-02-10T10:00:00"
    for cmd in cmds:
        obs2tel, tel2obs = run_kosma_commands(cmd, astra_time=astra_time)
        test_data_frames["KOSMA_obs2tel.set"].append(obs2tel)
        test_data_frames["KOSMA_tel2obs.set"].append(tel2obs)
        # read KOSMA_astra.status file
        files = ImportKOSMAReadWriteIntoDictionary(["KOSMA_astra.status"])
        test_data_frames["KOSMA_astra.status"].append(files["KOSMA_astra.status"])
        test_data = save_test_outputs_to_dataframe(test_name=str(cmd))
        # test_data["astra_time"] = astra_time
        for file, data in test_data.items():
            data["astra_time"] = astra_time
            if file not in test_data_frames:
                test_data_frames[file] = []
            test_data_frames[file].append(data)
    # make datraframe for each file for later analysis
    # save output to an excel file in different sheets
    logger.info("Saving test outputs to Excel file.")
    with pd.ExcelWriter("astra_kosma_test_outputs.xlsx") as writer:
        for file, data_list in test_data_frames.items():
            df = pd.DataFrame(data_list)
            sheet_name = os.path.splitext(os.path.basename(file))[0]
            df.to_excel(writer, sheet_name=sheet_name, index=False)
    logger.info("Test outputs saved successfully.")
    #
    compare_kosma_tests_with_astropy(make_plots=True)


# Check if Astra tests should be run
if args.run_horizon_tests:
    logger.info("Running Astra tests...")
    # Add logic to run Astra tests here
    # Example: subprocess.run(["astra", "test-command"]
    #
    if not is_astra_running():
        logger.critical(
            "Astra is not running. Please start Astra before running the tests."
        )
        raise SystemExit
    else:
        logger.info("Astra is running. Proceeding with tests.")
    #
    cmds = []
    # cmds += [["setsource G347_PeakB"]]  # GALACTIC
    # cmds += [["setsource G300_10 -l 300 -b 10 -C GALACTIC"]]  # GALACTIC
    # cmds += [["setsource G300_20 -l 300 -b 20 -C GALACTIC"]]  # GALACTIC
    # cmds += [["setsource G300_30 -l 300 -b 30 -C GALACTIC"]]  # GALACTIC
    cmds += [["setsource H40_40 -l 40 -b 40 -C HORIZON"]]  # HORIZONAL
    cmds += [["setsource H30_30 -l 30 -b 30 -C HORIZON"]]  # HORIZONAL
    # cmds += [["setsource H30_30 -l 20 -b 20 -C HORIZON"]]  # HORIZONAL
    # cmds += [["setsource H30_30 -l 10 -b 10 -C HORIZON"]]  # HORIZONAL
    # cmds += [["KOSMA_setoffset -l 10.0"]]
    # cmds += [["KOSMA_setoffset -l 20.0"]]
    # cmds += [["setsource W43_OFF", "KOSMA_setoffset -l 00.0"]]
    # cmds += [["setsource GC2_J2000", "KOSMA_setoffset -l 00.0"]]
    # cmds += [["KOSMA_setoffset -l 10.0"]]
    # cmds += [["KOSMA_setoffset -l 20.0"]]
    test_data_frames = {}
    # commands to run for testing
    astra_time = "2026-02-10T10:00:00"
    for cmd in cmds:
        run_kosma_commands(cmd, astra_time=astra_time)
        test_data = save_test_outputs_to_dataframe(test_name=str(cmd))
        # test_data["astra_time"] = astra_time
        for file, data in test_data.items():
            data["astra_time"] = astra_time
            if file not in test_data_frames:
                test_data_frames[file] = []
            test_data_frames[file].append(data)
    # make datraframe for each file for later analysis
    # save output to an excel file in different sheets
    logger.info("Saving test outputs to Excel file.")
    with pd.ExcelWriter("astra_kosma_test_outputs.xlsx") as writer:
        for file, data_list in test_data_frames.items():
            df = pd.DataFrame(data_list)
            sheet_name = os.path.splitext(os.path.basename(file))[0]
            df.to_excel(writer, sheet_name=sheet_name, index=False)
    logger.info("Test outputs saved successfully.")
    #
    compare_kosma_tests_with_astropy(make_plots=True)
else:
    logger.info("Skipping Astra tests.")


if args.parametric_j2000_tests:
    logger.info("Running parametric J2000 tests...")
    # TODO implement parametric J2000 tests
    # read tel2obs and extract location
    # get range of ra and dec available
    obs_time = Time("2026-02-10T10:00:00", scale="utc")
    results = get_visible_sky_range_in_j2000_coordinates(location, obs_time, min_alt=20)
    # log results
    logger.info(
        f"Visible sky range in J2000 coordinates at time {obs_time.iso} from location"
        f" (lon: {location.lon.deg}, lat: {location.lat.deg}, height: {location.height.value} m):\n"
        f"RA range: {results['ra_min']:.2f} deg to {results['ra_max']:.2f} deg\n"
        f"Dec range: {results['dec_min']:.2f} deg to {results['dec_max']:.2f} deg"
    )
    # generate test points in this range
    ra_values = np.linspace(results["ra_min"], results["ra_max"], 5)
    dec_values = np.linspace(results["dec_min"], results["dec_max"], 5)
    # make a array of tuples of ra and dec
    ra_dec_tuples = [(ra, dec) for ra in ra_values for dec in dec_values]
    #
    residuals = []
    for ra, dec in ra_dec_tuples:
        #  loop over ra and dec values and run tests
        test_data_frames = {}
        test_data_frames["KOSMA_obs2tel.set"] = []
        test_data_frames["KOSMA_tel2obs.set"] = []
        test_data_frames["KOSMA_astra.status"] = []
        #
        source_name = f"TEST_RA{ra:03.0f}_DEC{dec:03.0f}"
        cmd_create_source = f"setsource {source_name} -l {ra} -b {dec} -C J2000"
        coord = SkyCoord(ra * u.deg, dec * u.deg, frame="icrs")
        # get rise and set times for this source
        try:
            astra_times, rise_time, set_time = get_track_times_from_object(
                obs_time, coord, location
            )
        except Exception as e:
            logger.error(
                f"Error getting track times for source {source_name} at RA: {ra}, Dec: {dec}: {e}"
            )
            continue
        logger.info(f"Running test for source {source_name} at RA: {ra}, Dec: {dec}")
        for astra_time in astra_times:
            logger.info(f"Running track test at Astra time: {astra_time.iso}")
            # create source in Astra
            obs2tel, tel2obs = run_kosma_commands(
                [cmd_create_source], astra_time=astra_time.iso.replace(" ", "T")
            )
            #
            if obs2tel is None or tel2obs is None:
                logger.error("Error running KOSMA commands. Skipping this test.")
                continue
            #
            test_data_frames["KOSMA_obs2tel.set"].append(obs2tel)
            test_data_frames["KOSMA_tel2obs.set"].append(tel2obs)
            # read KOSMA_astra.status file
            files = ImportKOSMAReadWriteIntoDictionary(["KOSMA_astra.status"])
            test_data_frames["KOSMA_astra.status"].append(files["KOSMA_astra.status"])
        save_data_test_outputs_to_dataframe(test_data_frames)
        #
        if args.find_optimal_time_offset:
            logger.info(
                f"Finding optimal time offset for source {source_name} at RA: {ra}, Dec: {dec}"
            )
            result = minimize_scalar(
                minimize_azimith_residuals, bounds=(-0.3, 0.3), method="bounded"
            )
            logger.info(
                f"Optimal time offset for source {source_name} at RA: {ra}, Dec: {dec}: {result.x} seconds"
            )
            logger.info(f"Minimized azimuth residual std: {result.fun} arcsec")
            results_df = compare_kosma_tests_with_astropy(
                make_plots=True,
                figure_tag=f"RA{ra:02.0f}_DEC{dec:02.0f}_optimized",
                fixed_offset=result.x,
            )
            residual = {
                "optimal_time_offset_seconds": result.x,
                "minimized_azimuth_residual_std_arcsec": result.fun,
                "minimized_elevation_residual_std_arcsec": results_df[
                    "el_difference_arc_sec"
                ].std(),
                "number_of_iterations": result.nit,
                "source_name": source_name,
                "ra_deg": ra,
                "dec_deg": dec,
                "number_of_track_points": len(results_df),
            }
            residuals.append(residual)
        else:
            logger.info(
                f"Comparing KOSMA tests with Astropy for source {source_name} at RA: {ra}, Dec: {dec}"
            )
            results_df = compare_kosma_tests_with_astropy(
                make_plots=True, figure_tag=f"RA{ra:.2f}_DEC{dec:.2f}"
            )
    # save residuals to a dataframe and excel file
    if len(residuals) > 0:
        residuals_df = pd.DataFrame(residuals)
        logger.info("Saving residuals summary to Excel file.")
        with pd.ExcelWriter(
            "astra_kosma_parametric_j2000_residuals.xlsx", mode="w"
        ) as writer:
            residuals_df.to_excel(writer, sheet_name="residuals_summary", index=False)

    # make datraframe for each file for later analysis
    # save output to an excel file in different sheets

if args.compare_data_only:
    logger.info("Running comparison only as per user request.")
    results_df = compare_kosma_tests_with_astropy(
        make_plots=True, figure_tag="comparison_only", fixed_offset=0.0
    )


if args.find_optimal_time_offset:
    logger.info("Minimizing azimuth residuals by optimizing time offset.")
    result = minimize_scalar(
        minimize_azimith_residuals, bounds=(-0.1, 0.1), method="bounded"
    )
    residual = {
        "optimal_time_offset_seconds": result.x,
        "minimized_azimuth_residual_std_arcsec": result.fun,
        "minimized_elevation_residual_std_arcsec": results_df[
            "el_difference_arc_sec"
        ].std(),
        "number_of_iterations": result.nit,
    }
    #
    results_df = compare_kosma_tests_with_astropy(
        make_plots=True, figure_tag="optimized_fit", fixed_offset=result.x
    )
    #
    print("Optimal time offset:", result.x)
    print("Minimized residual_az_std:", result.fun)
    print(
        "Minimized elevation residual std:", results_df["el_difference_arc_sec"].std()
    )
    print("Number of iterations:", result.nit)
    #


# TODO check a track and monitor offsets
# TODO introduce coord offsets
# TODO introduce instrument/pixel offsets
# TODO check refraction corrections
# TODO check OTF implemnentation
