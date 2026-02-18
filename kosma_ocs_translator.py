import urllib.request, json, requests
import time, datetime
import random
import logging
import os, glob, re
from ocs import observatory_control_system
from astropy.coordinates import SkyCoord, EarthLocation, AltAz
from astropy import units as u
from astropy.time import Time
import numpy as np
import argparse
import threading
import signal,sys
# PH1 root mount
try:
    from roof_mount import RoofMount
except:
    print("roof_mount package not found, make sure it is installed and in the PYTHONPATH")
    print("install from git@git.ph1.uni-koeln.de:receiver/pointing-camera-rpi.git")
    

coord_sys_map = {
    "J2000": "icrs",
    "B1950": "fk4",
    "GALACTIC": "galactic",
    "HORIZON": "altaz",
}

def ImportKOSMAReadWriteIntoDictionary(files=None, variable=None, update_mod_time=True):
    log = logging.getLogger("kosma-ocs-translator")
    readwrite_dict = {}
    kosma_readwrite_dir = os.environ.get("WRITE_DIR", "/net/KOSMA_file_io/ReadWrite/")
    # log.debug("polling for changes from {0}".format(kosma_readwrite_dir))
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
        # strip fill path from readwrite_file
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
                timestamp = re.search(".+\s(\d+\.\d+)\s+\S+.+", line).groups()[0]
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
                result = re.search("(\S+)\s+(\S+)", data)
                if result:
                    value, variable_found = result.groups()
            except:
                result = re.search("\s+(\S+)", data).groups()
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
                # includ readwrite_file in error log
                log.error(
                    f"error parsing value {value} for variable {variable_found} from file {readwrite_file}"
                )
                log.error(e)
                continue
            #
            if (variable != None) & (variable != variable_found):
                continue
            if variable not in readwrite_dict[readwrite_file].keys():
                readwrite_dict[readwrite_file][variable_found] = value
    # remove entries that just have a timestamp and no variable values
    # for
    return readwrite_dict


class KOSMA_translator:
    def __init__(self, ocs):
        self.ocs = ocs
        #
        self.log = logging.getLogger("kosma-ocs-translator")
        #
        self.read_write_dir = os.environ.get(
            "WRITE_DIR", "/net/KOSMA_file_io/ReadWrite/"
        )
        self.log.info(f"Using KOSMA ReadWrite directory: {self.read_write_dir}")
        #
        self.load_tel2obs_template()
        #
        self.tel_return_cookie = 0
        self.old_tel_return_cookie = 0
        self.read_obs2tel_file()
        # get the telescope position from the ocs
        self.ocs.get_telescope_position()
        self.location = ocs.earth_location
        self.log.info(f"Telescope location: {self.location}")
        # Add tracking thread management
        self._track_thread = None
        self._track_stop_event = threading.Event()

    def check_for_obs2tel_update(self):
        # check modification time of obs2tel file and store

        current_mod_time = os.path.getmtime(f"{self.read_write_dir}/KOSMA_obs2tel.set")
        # on start up set mod time to 0 and return false
        if not hasattr(self, "obs2tel_mod_time"):
            self.obs2tel_mod_time = current_mod_time
            return False
        current_mod_time = os.path.getmtime(f"{self.read_write_dir}/KOSMA_obs2tel.set")
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
        self.old_tel_return_cookie = self.tel_return_cookie
        self.tel_return_cookie = self.kio_files["KOSMA_obs2tel.set"]["obs_cookie"]
        self.obs_tel_info_update_time = self.kio_files["KOSMA_obs2tel.set"][
            "obs_tel_info_update_time"
        ]
        # read all variables from obs2tel file into the object
        self.obs2tel = self.kio_files["KOSMA_obs2tel.set"]

    def load_tel2obs_template(self):
        self.tel2obs = """
        {0[timestring]}  {0[timestamp]}   File update time stamp   ! ccat_translator (test_computer:1000)
        EMU_   tel_telescope   ! Telescope Identifier (for display and backend_name in FITS header)
        {0[tel_on_track]}   tel_on_track   ! Y if tracking on commanded position/track within tolerance [Y/N]
        {0[tel_lost_track]}   tel_lost_track   ! Y if tracking got beyond tolerance since start of track [Y/N]
        {0[tel_pos_in_range]}   tel_pos_in_range   ! Y if commanded position is within telescope range [Y/N]
                0   tel_error   ! flags telescope command syntax/consistency and functional errors (=0: ok)
        NONE   tel_error_string   ! Error message associated with telescope command (=NONE: OK)
                5   tel_error_level   ! Error level for message in tel_error_string [1: trace, 2: debug, 4: info, 5: message, 7: warn, 8: error, 9: fatal]
            {0[tel_return_cookie]}  tel_return_cookie   ! return cookie to identify proper observation
            {0[tel_plate_scale]}  tel_plate_scale   ! plate scale of focal plane [arcsec/mm]
            {0[tel_latitude]}   tel_latitude   ! latitude of observatory (needed for Doppler correction) [degree]
            {0[tel_longitude]}  tel_longitude   ! longitude of observatory (needed for Doppler correction) [degree, +west (astronomical notation)]
            {0[tel_altitude]}  tel_altitude   ! height above sea level (needed for Doppler correction) [meter]
            {0[tel_angle_focal_plane]}   tel_angle_focal_plane   ! ccw angle from second coordinate of the telescope reference coordinate system (TARF,HORIZON) to second coordinate of obs_coord_sys_del [degree]
        #           0   tel_los_act   ! actual los-angle [degree]
        #    1557237163.99149   tel_time_act   ! time when status data were valid [seconds since 1970 UTC]
        #   87.091738   tel_time_del   ! time since start of track (may be negative, if start in the future) [seconds]
        {0[tel_azm_cmd]:3.4f}   tel_azm_cmd   ! commanded azimuth (for tracking display) [degree,cw toward east]
        {0[tel_elv_cmd]:3.4f}   tel_elv_cmd   ! commanded elevation (for tracking display and for atmospheric calibration) [degree]
        {0[tel_azm_act]:3.4f}   tel_azm_act   ! actual azimuth (for tracking display) [degree,cw toward east]
        {0[tel_elv_act]:3.4f}   tel_elv_act   ! actual elevation (for tracking display) [degree]
        {0[tel_supports_ephemeris]}   tel_supports_ephemeris   ! Telescope supports ephemeris tracking [Y/N] !
        0.0    tel_mjd1 ! MJD of integer part of current time [days] 
        """

    def set_tel2obs_dict(self):
        response = self.ocs.get_status()
        input_dict = {}
        input_dict["tel_azm_act"] = response["Azimuth current position"]
        input_dict["tel_elv_act"] = response["Elevation current position"]
        input_dict["tel_azm_cmd"] = response["Azimuth commanded position"]
        input_dict["tel_elv_cmd"] = response["Elevation commanded position"]
        input_dict["tel_latitude"] = self.ocs.earth_location.lat.deg
        input_dict["tel_longitude"] = self.ocs.earth_location.lon.deg
        input_dict["tel_altitude"] = self.ocs.earth_location.height.to(u.m).value
        input_dict["tel_telescope"] = "CCAT"
        input_dict["tel_plate_scale"] = "1"
        input_dict["tel_angle_focal_plane"] = 0.0
        input_dict["tel_on_track"] = "N"
        input_dict["tel_lost_track"] = "N"
        input_dict["tel_return_cookie"] = self.tel_return_cookie
        input_dict["tel_error"] = "0"
        input_dict["tel_supports_ephemeris"] = "N"
        #
        current_azi = response["Azimuth current position"]
        current_ele = response["Elevation current position"]
        cmd_azi = response["Azimuth commanded position"]
        cmd_ele = response["Elevation commanded position"]
        actdiff = (
            3600 * ((current_azi - cmd_azi) ** 2 + (current_ele - cmd_ele) ** 2) ** 0.5
        )
        if actdiff < self.obs_tolerance:
            input_dict["tel_pos_in_range"] = "Y"
            input_dict["tel_on_track"] = "Y"
        else:
            input_dict["tel_pos_in_range"] = "Y"
            input_dict["tel_on_track"] = "N"
            input_dict["tel_lost_track"] = "N"
    
        #
        self.log.info(f"on track: {input_dict['tel_on_track']} az:{response['Azimuth current position']:3.2f} ele:{response['Elevation current position']:3.2f} cmd_az:{response['Azimuth commanded position']:3.2f} cmd_ele:{response['Elevation commanded position']:3.2f} obs_tolerance: {self.obs_tolerance/3600.0:3.2f} deg ")        
        fmt = "%d-%b-%Y  %H:%I:%S"
        current = time.localtime()
        input_dict["timestamp"] = time.time()
        input_dict["timestring"] = date = time.strftime(fmt, current)
        return input_dict

    def write_tel2obs_file(self):
        #
        input_dict = self.set_tel2obs_dict()
        #
        self.log.debug(
            f"writing tel2obs file with obs cookie: {self.tel_return_cookie}"
        )
        tel2obs_handle = open("/net/KOSMA_file_io/ReadWrite/KOSMA_tel2obs.set", "w")
        tel2obs_handle.write(self.tel2obs.format(input_dict))
        tel2obs_handle.close()
        

    def track(self):

        # 1. If a track is already running → stop it
        if self._track_thread and self._track_thread.is_alive():
            self.log.info("Stopping previous track before starting new one")
            self.stop_track()

        # 2. Reset stop event
        self._track_stop_event.clear()

        # 3. Start new thread
        self._track_thread = threading.Thread(
            target=self._track_loop,
            daemon=True
        )
        self._track_thread.start()

    def _track_loop(self):
        # commanded position
        cmd_lam = self.obs2tel["obs_lam_on"]
        cmd_bet = self.obs2tel["obs_bet_on"]
        cmd_coord_sys_on = self.obs2tel["obs_coord_sys_on"]
        # track details
        track_duration = 600  # seconds
        # make into an astropy coordinate object
        frame = coord_sys_map[cmd_coord_sys_on]
        coord = SkyCoord(cmd_lam * u.deg, cmd_bet * u.deg, frame=frame)
        #
        self.log.info(f"tracking to {cmd_lam} {cmd_bet} in {cmd_coord_sys_on} frame")
        # 
        if frame.lower() == "altaz":
            self.ocs.move_to(cmd_lam, cmd_bet)
            return
        # make an array of times from now to now + track_duration, with 1 second steps
        n_steps = int(track_duration) + 1
        time_array = Time.now() + np.linspace(0, track_duration, n_steps) * u.second
        # calculate the altaz coordinates for each time step
        altaz_frames = AltAz(obstime=time_array, location=self.ocs.earth_location)
        altaz = coord.transform_to(altaz_frames)
        # add in focal plane offset from obs2tel file, convert from arcseconds to degrees
        focal_plane_offset_az = self.obs2tel.get("obs_x_focal_plane") / 3600.0
        focal_plane_offset_el = self.obs2tel.get("obs_y_focal_plane") / 3600.0
        # add to altaz coordinates
        az_with_focal_plane_offset = altaz.az.deg + focal_plane_offset_az
        el_with_focal_plane_offset = altaz.alt.deg + focal_plane_offset_el
        # calculate the velocities in azimuth and elevation using np.gradient
        dt = np.gradient(time_array.unix)
        az_velocities = np.gradient(az_with_focal_plane_offset) / dt
        el_velocities = np.gradient(el_with_focal_plane_offset) / dt
        # program track mode, see defintion here ICD-1000000-32000-02-00 VA Webserver - Remote Protocol
        mode = 0  #
        mode_arr = np.full_like(az_with_focal_plane_offset, mode)
        # Assuming all arrays are 1D and of the same length
        points = np.column_stack(
            [
                time_array.unix - time_array.unix[0],  # time in seconds since 1970
                az_with_focal_plane_offset,
                el_with_focal_plane_offset,
                az_velocities,
                el_velocities,
            ]
        )
        points_list = points.tolist()
        payload = {
            "start_time": time_array[0].unix,
            "coordsys": "Horizon",
            "points": points_list,
        }
        #
        try:
            response = self.ocs.scan_pattern(data=payload, stop_event=self._track_stop_event)
            self.log.info(f"scan pattern response: {response}")
        except Exception as e:
            self.log.error(f"Track error: {e}")


    def stop_track(self):
        if self._track_thread and self._track_thread.is_alive():
            self.log.info("Signaling track thread to stop")

            self._track_stop_event.set()   # signal thread

            self._track_thread.join()      # WAIT until thread exits

            self.log.info("Track thread stopped")

            self._track_thread = None

def setup_logging(log_level):
    # setup logging to file and to screen
    logger = logging.getLogger("kosma-ocs-translator")
    logger.setLevel(log_level)

    # Remove existing handlers to avoid conflicts
    # if logger.hasHandlers():
    #    logger.handlers.clear()

    # Prevent propagation to the root logger
    logger.propagate = False

    # create console handler and set level to debug
    ch = logging.StreamHandler()
    ch.setLevel(log_level)
    # create simpler formatter for console output
    console_formatter = logging.Formatter("%(levelname)s: %(message)s")
    ch.setFormatter(console_formatter)
    logger.addHandler(ch)

    # file handler
    fh = logging.FileHandler("kosma-ocs-translator.log")
    fh.setLevel(log_level)
    # detailed formatter for file output
    file_formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    fh.setFormatter(file_formatter)
    logger.addHandler(fh)

    return logger


def parse_arguments():
    parser = argparse.ArgumentParser(description="KOSMA OCS Translator")
    parser.add_argument(
        "--certificates-path",
        type=str,
        default="../CCAT/observatory-control-system/tls",
        help="Path to the certificates directory.",
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
    return parser.parse_args()


def run_write_tel2obs_file_in_background(translator):
    while True:
        translator.write_tel2obs_file()
        time.sleep(translator.obs_tel_info_update_time)


ocs_host = "127.0.0.1"
ocs_port = 5600
# certificates_path = "../observatory-control-system/tls"
certificates_path = None
ocs = None


def main():
    args = parse_arguments()
    certificates_path = args.certificates_path
    log_level = getattr(logging, args.log_level)
    logger = setup_logging(log_level)
    if args.use_roof_mount:
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
        logger.info("Using OCS connection")    
        print(f"Using certificates path: {certificates_path}")
        ocs = observatory_control_system(
            url=f"https://{ocs_host}:{ocs_port}",
            server_cert=f"{certificates_path}/server.cert.pem",
            client_cert=f"{certificates_path}/client.cert.pem",
            client_key=f"{certificates_path}/client.key.pem",
        )
    translator = KOSMA_translator(ocs)
    background_thread = threading.Thread(
        target=run_write_tel2obs_file_in_background, args=(translator,)
    )
    background_thread.daemon = True
    background_thread.start()
    translator.log.info(
        f"obs_tel_info_update_time set to {translator.obs_tel_info_update_time} seconds"
    )
    
    # Define signal handler with closure
    def signal_handler(signum, frame):
        translator.log.info("Ctrl+C received, shutting down...")
        translator.stop_track()
        if hasattr(ocs, 'close'):
            ocs.close()
        translator.log.info("All threads closed, exiting.")
        sys.exit(0)
    
    # Register signal handler for Ctrl+C
    signal.signal(signal.SIGINT, signal_handler)
    
    # get telescope position

    while True:
        obs2tel_updated = translator.check_for_obs2tel_update()
        if not obs2tel_updated:
            time.sleep(translator.obs_tel_info_update_time)
            #translator.log.debug(
            #    f"obs2tel file has not changed, waiting {translator.obs_tel_info_update_time} seconds"
            #)
            continue
        translator.read_obs2tel_file()
        if translator.tel_return_cookie == translator.old_tel_return_cookie:
            translator.log.info(
                f"obs2tel cookie has not changed ({translator.tel_return_cookie}), no action taken"
            )
            time.sleep(translator.obs_tel_info_update_time)
            continue
        if translator.obs2tel["obs_otf_mode"] in ["C", "P"]:
            translator.log.info("OTF mode command received, not yet implemented")
            time.sleep(translator.obs_tel_info_update_time)
            continue
        if translator.obs2tel["obs_otf_mode"] == "N":
            translator.log.info("track mode command received")
            translator.track()


if __name__ == "__main__":
    main()

# TODO
# check if obs2tel has changed
# read obs2tel file
# move telescope to commanded position
# if obs2tel had changed, abort previous command and send new one


"""
tel_return_cookie = kio_files["KOSMA_obs2tel.set"]["obs_cookie"]
obs_tolerance = kio_files["KOSMA_obs2tel.set"]["obs_tolerance"]
obs_tel_info_update_time = kio_files["KOSMA_obs2tel.set"]["obs_tel_info_update_time"]


# commanded position
cmd_lam = kio_files["KOSMA_obs2tel.set"]["obs_lam_on"]
cmd_bet = kio_files["KOSMA_obs2tel.set"]["obs_bet_on"]
cmd_coord_sys_on = kio_files["KOSMA_obs2tel.set"]["obs_coord_sys_on"]

#
coord_sys_translator = {
    "J2000": "icrs",
    "B1950": "fk4",
    "GALACTIC": "galactic",
    "ECLIPTIC": "geocentrictrueecliptic",
    "HORIZON": "altaz",
}

if cmd_coord_sys_on not in coord_sys_translator.keys():
    logger.error("coordinate system {0} not recognized".format(cmd_coord_sys_on))
    raise SystemExit

if coord_sys_translator[cmd_coord_sys_on].lower() not in ocs.supported_coord_systems:
    logger.info("coordinate system not {0} supported by OCS".format(cmd_coord_sys_on))
    logger.info("following systems are supported:   ")
    logger.info(ocs.supported_coord_systems)
    raise SystemExit

star_position = SkyCoord(
    cmd_lam, cmd_bet, unit=(u.deg, u.deg), frame=coord_sys_translator[cmd_coord_sys_on]
)
logger.info(
    "tracking to {0} {1} in {2} frame".format(cmd_lam, cmd_bet, cmd_coord_sys_on)
)
#
if coord_sys_translator[cmd_coord_sys_on].lower() == "horizon":
    ocs.move_to(azimuth=cmd_lam, elevation=cmd_bet)
elif coord_sys_translator[cmd_coord_sys_on].lower() in "icrs":
    ocs.track(star_position)

# wait for telescope to move and get to position
time.sleep(1)
response = ocs.get_status()
current_azi = response["Azimuth current position"]
current_ele = response["Elevation current position"]
print("current pos. az: {0:3.0f} ele: {1:3.0f}".format(current_azi, current_ele))
"""
