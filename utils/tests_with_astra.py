import os
import subprocess
import argparse
import re
import pandas as pd
import time
import logging
import astropy.units as u
import numpy as np
from astropy.coordinates import SkyCoord, EarthLocation, AltAz
from astropy.time import Time
from datetime import timedelta

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
    source, location, t0=None, horizon=0 * u.deg, step=1 * u.hour, max_hours=36
):
    """
    Compute rise and set times of a celestial source using Astropy only.

    Parameters
    ----------
    source : SkyCoord
        Source coordinates (any frame, e.g. Galactic).
    location : EarthLocation
        Observer location.
    t0 : Time, optional
        Start time (UTC). Defaults to now.
    horizon : Quantity
        Altitude defining rise/set (default 0 deg).
    step : Quantity
        Time sampling step (default 1 hour).
    max_hours : float
        Search window (default 36 hours).
        (>24h handles late set / early rise cases)

    Returns
    -------
    rise_time : Time or None
    set_time : Time or None
    """

    if t0 is None:
        t0 = Time.now()

    # Ensure equatorial frame
    source_icrs = source.icrs

    # Time grid
    n_steps = int((max_hours * u.hour / step).decompose())
    times = t0 + np.arange(n_steps + 1) * step

    # AltAz frame (no refraction)
    altaz_frame = AltAz(obstime=times, location=location, pressure=0 * u.bar)

    # Transform and extract altitude
    alt = source_icrs.transform_to(altaz_frame).alt.to_value(u.deg)
    h = horizon.to_value(u.deg)

    rise_time = None
    set_time = None

    # Rise: below -> above
    rise_idx = np.where((alt[:-1] < h) & (alt[1:] >= h))[0]
    if len(rise_idx) > 0:
        i = rise_idx[0]
        frac = (h - alt[i]) / (alt[i + 1] - alt[i])
        rise_time = times[i] + frac * (times[i + 1] - times[i])

    # Set: above -> below
    set_idx = np.where((alt[:-1] >= h) & (alt[1:] < h))[0]
    if len(set_idx) > 0:
        i = set_idx[0]
        frac = (h - alt[i]) / (alt[i + 1] - alt[i])
        set_time = times[i] + frac * (times[i + 1] - times[i])

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
        with open("/tmp/astra_debug_time.txt", "w") as f:
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


def compare_astropy_with_kosma(
    obs2tel, tel2obs, df_log, astra_status, offset_time_seconds=0
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
    # set timezone to UTC
    observation_time = Time(obs2tel.astra_time, scale="utc")
    # offset by 2 seconds
    observation_time += offset_time_seconds * u.s
    # offset args off
    observation_time += args.offset_time_seconds * u.s
    # compare mjd from astra_status with astropy time
    astra_status_mjd = astra_status["a_dj1"]
    observation_time.to_value("mjd", "long")
    mjd_difference = observation_time.to_value("mjd", "long") - astra_status_mjd
    print(
        f"MJD Astropy: {observation_time.to_value('mjd', 'long')}, MJD Astra: {astra_status_mjd}"
    )
    # add mjd difference to observation_time
    # observation_time += mjd_difference * 86400.0 * u.s
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

    # astra treats longitudes as postive for west, astropy is negative for east
    observer_location = EarthLocation(
        lon=-1 * tel2obs["tel_longitude"].values[0] * u.deg,
        lat=tel2obs["tel_latitude"].values[0] * u.deg,
        height=tel2obs["tel_altitude"].values[0] * u.m,
    )
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
    apply_refraction = False
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


def compare_kosma_tests_with_astropy(make_plots=True, plot_logs=False):
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
    fixed_offset = -0.03
    fixed_offset = 0.0
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
        make_comparison_plots(results_df, plot_axis=plot_axis, plot_logs=plot_logs)


def make_comparison_plots(results_df, plot_axis="time_dt", plot_logs=False):
    # make a 2x2 panel plot: absolute values on top, residuals on the bottom
    import matplotlib.pyplot as plt

    fig, axs = plt.subplots(2, 2, figsize=(14, 10), sharex=False)
    fig.suptitle("Astra  vs KOSMA-OCS-Translator Pointing Analysis")

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
        # add fig suptile with offset time seconds
        fig.suptitle(
            f"Astra vs KOSMA-OCS-Translator Pointing Analysis (Offset Time: {fixed_offset} seconds)"
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
    logger.info("Saving pointing analysis plot to astra_kosma_pointing_analysis.png")
    plt.savefig("astra_kosma_pointing_analysis.png")
    plt.show()


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
is_astra_running()

if args.run_track_tests_J2000:
    start_time = Time.now().iso.replace(" ", "T")
    test_data_frames = {}
    test_data_frames["KOSMA_obs2tel.set"] = []
    test_data_frames["KOSMA_tel2obs.set"] = []
    test_data_frames["KOSMA_astra.status"] = []
    # commands to run for testing

    cmd = ["setsource W43_OFF", "KOSMA_setoffset -l 00.0", "setpoint -p L"]
    astra_time_str = "2026-02-10T16:00:00"
    source_name = "W43_OFF"
    astra_times, rise_time, set_time = get_track_times(astra_time_str, source_name)
    # make an array of times from rise to set time

    # collect obs2tel and get coord
    logger.info(
        f"Running track tests from rise time {rise_time.iso} to set time {set_time.iso}"
    )
    logger.info(f"Total of {len(astra_times)} track tests to run.")
    for astra_time in astra_times:
        # round out to nearest second
        astra_time = Time(astra_time.datetime.strftime("%Y-%m-%dT%H:%M:%S"))
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
    compare_kosma_tests_with_astropy(make_plots=True)

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
        f"Running track tests from rise time {rise_time.iso} to set time {set_time.iso}"
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

if args.compare_data_only:
    logger.info("Running comparison only as per user request.")
    compare_kosma_tests_with_astropy(make_plots=True)
# TODO check a track and monitor offsets
# TODO introduce coord offsets
# TODO introduce instrument/pixel offsets
# TODO check refraction corrections
# TODO check OTF implemnentation
