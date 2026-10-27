import urllib.request, json, requests
import time, datetime
import random
import logging
import os, glob, re
import socket
from ocs import observatory_control_system
from astropy.coordinates import SkyCoord, EarthLocation, AltAz, Distance
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
# sun avoidance
try:
    from sun_avoidance.telescope import get_overtop_altaz_from_nominal
except:
    print("sun-avoidance not found. Install from git@github.com:ccatobs/sun-avoidance.git")
    sys.exit(1)


coord_sys_map = {
    "J2000": "icrs",
    "B1950": "fk4",
    "GALACTIC": "galactic",
    "HORIZON": "altaz",
}

# Proper-motion keyword names in astropy SkyCoord differ by frame
_frame_pm_keys = {
    "icrs":     ("pm_ra_cosdec", "pm_dec"),
    "fk4":      ("pm_ra_cosdec", "pm_dec"),
    "galactic": ("pm_l_cosb",    "pm_b"),
}

# Reference epoch for catalogue coordinates by frame
# only matters for proper motion correction
_frame_epochs = {
    "icrs":     Time("J2000"),
    "fk4":      Time(1950.0, format="byear"),
    "galactic": Time("J2000"),
}


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
            except:
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


class KOSMA_translator:
    def __init__(self, ocs):
        self.ocs = ocs
        #
        self.log = logging.getLogger("kosma-ocs-translator")
        self._host_pid = f"({socket.gethostname()}:{os.getpid()})"
        #
        self.read_write_dir = os.environ.get(
            "WRITE_DIR", "/net/KOSMA_file_io/ReadWrite/"
        )
        self.log.info(f"Using KOSMA ReadWrite directory: {self.read_write_dir}")
        #
        self.load_tel2obs_template()
        self.load_ccdpossim_template()
        #
        self.tel_return_cookie = 0
        self.old_tel_return_cookie = 0
        self.next_tel_return_cookie = 0
        # cookie of an accepted path that has not started yet; published as
        # tel_return_cookie only once the ACU is executing the new path
        self.pending_tel_return_cookie = None
        # [sec] after path start_time before the cookie is returned, so that the
        # ACU commanded position has switched to the new path
        self.cookie_start_margin = 1.0
        # True if the path for tel_return_cookie was rejected or failed to be sent
        self.tel_path_failed = False
        self.tel_error = 0
        self.tel_lost_track = "N"
        self.read_obs2tel_file()
        # For now, ocs.get_telescope_position() always fall back to
        # a hard coded earth_location.
        # If this is solved on the OCS side and wants to use values
        # read from the ACU is prefered, use get_telescope_position
        # instead of ocs.earth_location
        # self.ocs.get_telescope_position()
        self.location = ocs.earth_location
        self.log.info(f"Telescope location: {self.location}")
        # Add tracking thread management
        self._track_thread = None
        self._track_stop_event = threading.Event()
        # Keep the record of the last payload and corresponding obs2tel
        self.last_payload = None
        self.last_obs2tel = None

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
            "obs_track_duration",
        ]
        missing = [k for k in required_keys if k not in self.obs2tel]
        if missing:
            self.log.error(f"Missing required obs2tel keys: {missing}")
            raise KeyError(f"Missing required obs2tel keys: {missing}")

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
        {0[tel_angle_focal_plane]}   tel_angle_focal_plane   ! ccw angle from second coordinate of the telescope reference coordinate system (TARF,HORIZON) to second coordinate of obs_coord_sys_del [degree]
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
        {0[op_oazm]}  op_oazm   ! azimuth pointing offset [rad]; optical pointing (HORIZON system)
        {0[op_oelv]}  op_oelv   ! elevation pointing offset [rad]; optical pointing (HORIZON system)
        {0[del_tel_tot_azm]}   del_tel_tot_azm   ! azimuth  pointing offset [rad] from telescope pointing model (including non repeatable terms)
        {0[del_tel_tot_elv]}   del_tel_tot_elv   ! elevation pointing offset [rad] from telescope pointing model (including non repeatable terms)
        {0[del_tel_rep_azm]}   del_tel_rep_azm   ! azimuth  pointing offset [rad] from telescope pointing model (only repeatable terms)
        {0[del_tel_rep_elv]}   del_tel_rep_elv   ! elevation pointing offset [rad] from telescope pointing model (only repeatable terms)
        """

    def _build_tel2obs_params(self, response) -> dict:
        """
        Derive the KOSMA_tel2obs parameter dictionary from a single ACU status response.
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

        # time stamp of the status, relative to the start of the last sent path
        status_time = _get_time_from_acu_format(response["Time"], response["Year"])
        if self.last_payload is None:
            # no path accepted yet; only reached when the first path failed
            t_rel = 0.0
        else:
            t_rel = status_time.unix - self.last_payload["start_time"]

        # Return the new cookie only once the ACU is executing the new path
        if self.pending_tel_return_cookie is not None and t_rel >= self.cookie_start_margin:
            self.log.info(
                f"path started (t_rel={t_rel:.2f} s), "
                f"returning cookie {self.pending_tel_return_cookie}"
            )
            self.old_tel_return_cookie = self.tel_return_cookie
            self.tel_return_cookie = self.pending_tel_return_cookie
            self.pending_tel_return_cookie = None
            self.tel_lost_track = "N"    # initialize for the new cookie
            self.tel_error = 0
            self.tel_path_failed = False

        # values set elsewhere
        input_dict["tel_return_cookie"] = self.tel_return_cookie
        input_dict["tel_error"] = self.tel_error
        # starpoint ignores tel_error; "N" makes it skip the target of a failed path
        input_dict["tel_pos_in_range"] = "N" if self.tel_path_failed else "Y"

        current_azi = response["Azimuth current position"]
        current_ele = response["Elevation current position"]
        cmd_azi = response["Azimuth commanded position"]
        cmd_ele = response["Elevation commanded position"]
        acu_azm_del = current_azi - cmd_azi
        acu_elv_del = current_ele - cmd_ele

        actdiff = 3600 * ((acu_azm_del*np.cos(np.deg2rad(cmd_ele)))**2 + acu_elv_del**2) ** 0.5

        if self.tel_path_failed:
            # the ACU is not executing the commanded path
            input_dict["tel_on_track"] = "N"
        elif actdiff < self.obs_tolerance:
            input_dict["tel_on_track"] = "Y"
        else:
            input_dict["tel_on_track"] = "N"
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

        # time of the status since the start time of the table sent
        input_dict["tel_time_del"] = t_rel

        if self.last_payload is None:
            # no path to interpolate: report the ACU's own positions
            input_dict["tel_azm_cmd"] = cmd_azi
            input_dict["tel_elv_cmd"] = cmd_ele
        else:
            pts = np.array(self.last_payload["points"])   # shape (N, 5): dt, az, el, vaz, vel
            # TODO: apply refraction correction to tel_elev_cmd
            input_dict["tel_azm_cmd"] = np.interp(t_rel, pts[:, 0], pts[:, 1])
            input_dict["tel_elv_cmd"] = np.interp(t_rel, pts[:, 0], pts[:, 2])
        input_dict["tel_azm_act"] = input_dict["tel_azm_cmd"] + acu_azm_del
        input_dict["tel_elv_act"] = input_dict["tel_elv_cmd"] + acu_elv_del

        # Calculate angle between coordinate systems
        if self.last_obs2tel is None or self.last_obs2tel["obs_coord_sys_del"] == "HORIZON":
            input_dict["tel_angle_focal_plane"] = 0.0
        else:
            cmd_coord_sys_del = self.last_obs2tel["obs_coord_sys_del"]
            # Calculation is needed:
            # Calculate north (+epsilon in the second axis cmd_coord_sys_del) in Horizon
            # then csys_del -> horizon (opposite sign)

            altaz_frame = AltAz(obstime=status_time, location=self.location)

            pos = SkyCoord(
                az=input_dict["tel_azm_cmd"], alt=input_dict["tel_elv_cmd"],
                unit='deg', frame=altaz_frame)

            epsilon = 1e-3  # about 4 arcsec, small angle approximation is sufficient
            north = _apply_offset(
                pos.transform_to(coord_sys_map[cmd_coord_sys_del]), 0, epsilon, 'Y'
            ).transform_to(altaz_frame)

            angle = np.arctan2(
                north.alt.deg - pos.alt.deg,
                (north.az.deg - pos.az.deg) * np.cos(pos.alt.rad)
            )
            tel_angle_focal_plane = np.rad2deg(angle) - 90.   # from second axis in degrees
            if tel_angle_focal_plane < -180:
                tel_angle_focal_plane += 360.

            input_dict["tel_angle_focal_plane"] = tel_angle_focal_plane * (-1)

        self.log.info(f"on track: {input_dict['tel_on_track']} az:{input_dict['tel_azm_act']:.4f} ele:{input_dict['tel_elv_act']:.4f} cmd_az:{input_dict['tel_azm_cmd']:.4f} cmd_ele:{input_dict['tel_elv_cmd']:.4f} obs_tolerance: {self.obs_tolerance/3600.0:.4f} deg ")

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
        tilt1_azm = response_pointing["Tiltmeter Az correction AZ"]
        tilt2_azm = response_pointing["Tiltmeter El correction AZ"]
        tilt1_elv = response_pointing["Tiltmeter Az correction EL"]
        tilt2_elv = response_pointing["Tiltmeter El correction EL"]

        ccd_input_dict = {}
        ccd_input_dict["op_azm"] = np.deg2rad(input_dict["tel_azm_cmd"])
        ccd_input_dict["op_elv"] = np.deg2rad(input_dict["tel_elv_cmd"])

        if self.last_obs2tel["obs_coord_sys_on"] == "HORIZON":
            ccd_input_dict["op_oazm"] = np.deg2rad(self.last_obs2tel["obs_lam_del"])
            ccd_input_dict["op_oelv"] = np.deg2rad(self.last_obs2tel["obs_bet_del"])
        else:
            # these values are irrelevant if not Horizon
            ccd_input_dict["op_oazm"] = 0
            ccd_input_dict["op_oelv"] = 0

        ccd_input_dict["del_tel_tot_azm"] = np.deg2rad(spem_azm + tilt1_azm + tilt2_azm)
        ccd_input_dict["del_tel_tot_elv"] = np.deg2rad(spem_elv + tilt1_elv + tilt2_elv)
        ccd_input_dict["del_tel_rep_azm"] = np.deg2rad(spem_azm)
        ccd_input_dict["del_tel_rep_elv"] = np.deg2rad(spem_elv)

        ccd_input_dict["timestamp"] = input_dict["timestamp"]
        ccd_input_dict["timestring"] = input_dict["timestring"]
        ccd_input_dict["host_pid"] = input_dict["host_pid"]

        return ccd_input_dict

    def set_tel2obs_dict(self):
        """
        Set dictionary of KOSMA_tel2obs and KOSMA_ccdpos.sim
        """
        if (self.last_payload is None or self.last_obs2tel is None) and not self.tel_path_failed:
            self.log.info('No record of successfully sent command to telescope. Wait...')
            return None, None

        response = self.ocs.get_status()
        input_dict = self._build_tel2obs_params(response)

        # TODO: implement this get command on OCS and TCS
        #response_pointing = self.ocs.get_cmd_pointing()
        #ccd_input_dict = self._build_ccdpos_params(input_dict, response_pointing)
        ccd_input_dict = {}
        
        return input_dict, ccd_input_dict

    def write_tel2obs_file(self):
        input_dict, ccd_input_dict = self.set_tel2obs_dict()
        if input_dict is None:
            return

        self.log.debug(
            f"writing tel2obs file with obs cookie: {self.tel_return_cookie}"
        )
        tel2obs_handle = open(f"{self.read_write_dir}/KOSMA_tel2obs.set", "w")
        tel2obs_handle.write(self.tel2obs.format(input_dict))
        tel2obs_handle.close()

        if ccd_input_dict:
            ccdpossim_handle = open(f"{self.read_write_dir}/KOSMA_ccdpos.sim", "w")
            ccdpossim_handle.write(self.ccdpossim .format(ccd_input_dict))
            ccdpossim_handle.close()


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
        """
        Thread target. Any unexpected exception fails the path, so that the
        cookie is still returned to the observer.
        """
        # the cookie this thread works for; next_tel_return_cookie may already be
        # overwritten by a newer obs2tel while this thread is being stopped
        cookie = self.next_tel_return_cookie
        try:
            self._send_track(cookie)
        except Exception as e:
            self.log.exception(f"Track error: {e}")
            self._fail_path(cookie, 1)

    def _send_track(self, cookie):
        """
        This routine is called every time when KOSMA_obs2tel.set is updated. If a 
        track is already running, the previous one is stopped and the new one starts.
        Therefore practically the duration of the track in this routine is overwritten
        by a new track triggered by the KOSMA_obs2tel update.
        This justifies to use the fixed obs_x/y_focal_plane, which is a function of
        elevation and changes over time: once the value changes more than a tolerance,
        a new track is triggered with updated obs_x/y_focal_plane values.
        """

        # commanded position
        cmd_lam = self.obs2tel["obs_lam_on"]
        cmd_bet = self.obs2tel["obs_bet_on"]
        cmd_coord_sys_on = self.obs2tel["obs_coord_sys_on"]

        # sky/map offsets (arcsec, applied before AltAz transform unless HORIZON frame)
        lam_del = self.obs2tel["obs_lam_del"]
        bet_del = self.obs2tel["obs_bet_del"]
        coord_sys_del = self.obs2tel["obs_coord_sys_del"]
        true_angle_del = self.obs2tel["obs_true_angle_del"]  # Y/N

        # off-position offsets (arcsec, applied before AltAz transform unless HORIZON frame)
        lam_off = self.obs2tel["obs_lam_off"]
        bet_off = self.obs2tel["obs_bet_off"]
        coord_sys_off = self.obs2tel["obs_coord_sys_off"]
        true_angle_off = self.obs2tel["obs_true_angle_off"]  # Y/N

        # focal plane offsets (arcsec, always applied after AltAz transform,
        # and always apply cos-correction (corresponding true_angle=Y).
        # for CCAT obs_coord_sys_focal_plane is always set to HORIZON
        focal_plane_az = self.obs2tel["obs_x_focal_plane"]
        focal_plane_el = self.obs2tel["obs_y_focal_plane"]

        # proper motion (arcsec/yr) and parallax (arcsec) of the source
        # obs_pm_lam/obs_pm_bet are raw coordinate rates dlam/dt, not cos(bet)*dlam/dt;
        # slalib h2fk5.c note 1: "proper motions in RA are dRA/dt 
        #                         rather than cos(Dec)*dRA/dt"
        pm_lam = self.obs2tel["obs_pm_lam"]
        pm_bet = self.obs2tel["obs_pm_bet"]
        paralx = self.obs2tel["obs_paralx"]

        # extended elevation range
        if self.obs2tel["obs_extended_elevation_range"] == "Y":
            if_extended = True
        else:
            if_extended = False

        # obs wavelength (needs to be passed for refraction correction)
        obs_wavelength = self.obs2tel["obs_wavelength"]

        # validate coordinate systems
        for label, csys in [
            ("obs_coord_sys_on", cmd_coord_sys_on),
            ("obs_coord_sys_del", coord_sys_del),
            ("obs_coord_sys_off", coord_sys_off),
        ]:
            if csys not in coord_sys_map:
                msg = f"Unknown coordinate system {label}={csys!r}, must be one of {list(coord_sys_map)}"
                self.log.error(msg)
                raise ValueError(msg)

        # track details
        track_duration = 600  # seconds
        frame = coord_sys_map[cmd_coord_sys_on]

        # If coordinate system of ON is Horizon, use move-to.
        # For this case, it does not make much sense if map or off offset is given
        # by other coordinate system but such a case is also covered below.
        if frame.lower() == "altaz":
            now_frame = AltAz(obstime=Time.now(), location=self.location)
            coord = SkyCoord(cmd_lam * u.deg, cmd_bet * u.deg, frame=now_frame)
            if lam_del != 0.0 or bet_del != 0.0:
                if coord_sys_del == "HORIZON":
                    coord = _apply_offset(coord, lam_del, bet_del, true_angle_del)
                else:
                    coord = _apply_offset(
                        coord.transform_to(coord_sys_map[coord_sys_del]),
                        lam_del, bet_del, true_angle_del,
                    ).transform_to(now_frame)
            if lam_off != 0.0 or bet_off != 0.0:
                if coord_sys_off == "HORIZON":
                    coord = _apply_offset(coord, lam_off, bet_off, true_angle_off)
                else:
                    coord = _apply_offset(
                        coord.transform_to(coord_sys_map[coord_sys_off]),
                        lam_off, bet_off, true_angle_off,
                    ).transform_to(now_frame)

            self.log.info(
                f"tracking to {coord.az.deg} {coord.alt.deg} in {cmd_coord_sys_on} frame"
            )

            # Apply focal plane offsets
            if focal_plane_az != 0.0 or focal_plane_el != 0.0:
                coord = _apply_offset(coord, focal_plane_az, focal_plane_el, "Y")

            # command move-to
            if if_extended:
                cmd_az, cmd_el = get_overtop_altaz_from_nominal(
                    coord.az.deg, coord.alt.deg)
            else:
                cmd_az = coord.az.deg
                cmd_el = coord.alt.deg

            if self._track_stop_event.is_set():
                self.log.info("Track superseded, move-to not sent")
                return
            # ocs.move_to aborts the previous command itself
            response = self.ocs.move_to(cmd_az, cmd_el)
            body = response.json()
            if response.status_code != 200 or body.get("status") != "ok":
                self.log.error(
                    f"move_to rejected: HTTP {response.status_code}, "
                    f"status={body.get('status')}, message={body.get('message')}"
                )
                self._fail_path(cookie, response.status_code)
            else:
                self.log.info(f"move_to accepted: {body}")
                # Bookkeeping only (not sent): the ACU starts moving on receipt
                # and holds the position, so the commanded position is constant.
                self.last_payload = {
                    "type": "move_to",
                    "start_time": time.time(),
                    "coordsys": "Horizon",
                    "points": [[0.0, cmd_az, cmd_el, 0.0, 0.0],
                               [float(track_duration), cmd_az, cmd_el, 0.0, 0.0]],
                }
                self.last_obs2tel = self.obs2tel
                self.pending_tel_return_cookie = cookie
            return

        # non-HORIZON source: make into an astropy coordinate object,
        # applying proper motion and parallax propagation when non-zero.
        if pm_lam != 0.0 or pm_bet != 0.0 or paralx != 0.0:
            # Convert raw coordinate rate to true angular rate for astropy 
            # (multiply by cos(beta))
            cos_bet = np.cos(np.deg2rad(cmd_bet))
            pm_lon_key, pm_lat_key = _frame_pm_keys[frame]
            pm_kwargs = {
                pm_lon_key: pm_lam * cos_bet * u.arcsec/u.yr,
                pm_lat_key: pm_bet * u.arcsec/u.yr,
                "obstime": _frame_epochs[frame],
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
        if coord_sys_del != "HORIZON" and (lam_del != 0.0 or bet_del != 0.0):
            if coord_sys_del == cmd_coord_sys_on:
                coord = _apply_offset(coord, lam_del, bet_del, true_angle_del)
            else:
                coord = _apply_offset(
                    coord.transform_to(coord_sys_map[coord_sys_del]),
                    lam_del, bet_del, true_angle_del,
                ).transform_to(frame)

        # Apply (non-Horizon) off-position offsets in obs_coord_sys_off frame 
        # before AltAz transform
        if coord_sys_off != "HORIZON" and (lam_off != 0.0 or bet_off != 0.0):
            if coord_sys_off == cmd_coord_sys_on:
                coord = _apply_offset(coord, lam_off, bet_off, true_angle_off)
            else:
                coord = _apply_offset(
                    coord.transform_to(coord_sys_map[coord_sys_off]),
                    lam_off, bet_off, true_angle_off,
                ).transform_to(frame)

        self.log.info(
            f"tracking to {coord.data.lon.deg} {coord.data.lat.deg} in {cmd_coord_sys_on} frame"
        )

        # Abort the previous path first, if it has not ended yet.
        # This must come before time_array is built: TCS rejects a path starting
        # less than 9.8 sec from its arrival, so the abort must not eat into the
        # overhead below. The telescope is not tracking from here until the new
        # path starts.
        if self._track_stop_event.is_set():
            self.log.info("Track superseded, previous path not aborted")
            return
        if self.last_payload is not None and self.last_payload["type"] == "move_to":
            self.log.info("Previous command was move-to, no abort sent")
        elif self.last_payload is not None:
            last_end_time = self.last_payload["start_time"] + self.last_payload["points"][-1][0]
            if last_end_time > time.time():
                try:
                    self.ocs.abort()
                except Exception as e:
                    self.log.error(f"Abort error: {e}")
                    self._fail_path(cookie, 1)
                    return
            else:
                self.log.info("Previous path already ended, no abort sent")

        # make an array of times from now + overhead to now + overhead + track_duration,
        # with 1 second steps
        # Overhead [sec] to account for computation and communication delay
        # TCS requires > 10 sec ahead
        track_overhead = 11
        n_steps = int(track_duration) + 1
        t_build = time.perf_counter()
        time_array = Time.now() + track_overhead * u.second + np.linspace(
            0, track_duration, n_steps) * u.second

        # calculate the altaz coordinates for each time step
        altaz_frames = AltAz(obstime=time_array, location=self.location)
        altaz = coord.transform_to(altaz_frames)

        # Apply HORIZON-frame sky/map offsets after AltAz transform
        if coord_sys_del == "HORIZON" and (lam_del != 0.0 or bet_del != 0.0):
            altaz = _apply_offset(altaz, lam_del, bet_del, true_angle_del)

        # Apply HORIZON-frame off-position offsets after AltAz transform
        if coord_sys_off == "HORIZON" and (lam_off != 0.0 or bet_off != 0.0):
            altaz = _apply_offset(altaz, lam_off, bet_off, true_angle_off)

        # Apply focal plane offsets
        if focal_plane_az != 0.0 or focal_plane_el != 0.0:
            altaz = _apply_offset(altaz, focal_plane_az, focal_plane_el, "Y")

        if if_extended:
            az_array, el_array = get_overtop_altaz_from_nominal(
                altaz.az.deg, altaz.alt.deg)
        else:
            az_array = altaz.az.deg
            el_array = altaz.alt.deg

        # calculate the velocities in azimuth and elevation using np.gradient
        dt = np.gradient(time_array.unix)
        az_velocities = np.gradient(az_array) / dt
        el_velocities = np.gradient(el_array) / dt

        # program track mode, see definition here ICD-1000000-32000-02-00 VA Webserver
        # - Remote Protocol
        mode = 0  #
        mode_arr = np.full_like(az_array, mode)
        # Assuming all arrays are 1D and of the same length
        points = np.column_stack(
            [
                time_array.unix - time_array.unix[0],  # time offset in seconds since start of track
                az_array,
                el_array,
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
        self.log.info(f"payload computed in {time.perf_counter() - t_build:.3f} s")

        try:
            # TODO: find out a way to pass obs_wavelength together with payload
            # A newer track has superseded this one while the payload was computed
            if self._track_stop_event.is_set():
                self.log.info("Track superseded, payload not sent")
                return
            # TCS answers 503 (busy) until it is idle after the abort. Retry only
            # within the margin left by track_overhead (11 sec vs 9.8 sec required
            # by TCS); a later retry would be rejected as starting too soon anyway.
            lead_send = payload["start_time"] - time.time()
            response = self.ocs.scan_pattern(data=payload, busy_timeout=1, retry_interval=0.3)
            lead_done = payload["start_time"] - time.time()
            self.log.info(
                f"start_time lead: {lead_send:.3f} s before send, "
                f"{lead_done:.3f} s after response (TCS needs >= 9.8 s)"
            )
            body = response.json()
            if response.status_code != 200 or body.get("status") != "ok":
                self.log.error(
                    f"scan_pattern rejected: HTTP {response.status_code}, "
                    f"status={body.get('status')}, message={body.get('message')}"
                )
                self._fail_path(cookie, response.status_code)
            else:
                self.log.info(f"scan_pattern accepted: {body}")
                # "type" only for bookkeeping, not part of the payload sent to TCS
                self.last_payload = {**payload, "type": "path"}
                self.last_obs2tel = self.obs2tel
                # Do not return the cookie yet: until start_time the ACU still holds
                # the previous target and tel_on_track would be a stale "Y".
                self.pending_tel_return_cookie = cookie
        except Exception as e:
            self.log.error(f"Track error: {e}")
            self._fail_path(cookie, 1)

    def _fail_path(self, cookie, error):
        """
        The path for cookie will never start: return the cookie now, flagged
        as failed, so that the observer does not wait for it.
        """
        if self._track_stop_event.is_set():
            # superseded by a newer obs2tel; its own thread returns the new cookie
            self.log.info(f"Track for cookie {cookie} superseded, failure not reported")
            return
        self.tel_error = error
        self.tel_path_failed = True
        self.pending_tel_return_cookie = None
        self.old_tel_return_cookie = self.tel_return_cookie
        self.tel_return_cookie = cookie
        self.tel_lost_track = "N"

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
    return parser.parse_args()


def run_write_tel2obs_file_in_background(translator):
    while True:
        translator.write_tel2obs_file()
        time.sleep(translator.obs_tel_info_update_time)


def setup_ocs(use_roof_mount=False, use_local_dev=False, certificates_path=None,
              logger=None):
    """
    Set up the connection to the telescope control (OCS or roof mount).
    Can be imported by other scripts to share the same connection treatment.

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
        Logger to use. Defaults to the "kosma-ocs-translator" logger.

    Returns
    -------
    ocs : observatory_control_system or RoofMount
    """
    if logger is None:
        logger = logging.getLogger("kosma-ocs-translator")
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


def main():
    args = parse_arguments()
    log_level = getattr(logging, args.log_level)
    logger = setup_logging(log_level)
    ocs = setup_ocs(
        use_roof_mount=args.use_roof_mount,
        use_local_dev=args.use_local_dev,
        certificates_path=args.certificates_path,
        logger=logger,
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
        if translator.next_tel_return_cookie in (
            translator.tel_return_cookie, translator.pending_tel_return_cookie
        ):
            translator.log.info(
                f"obs2tel cookie has not changed ({translator.next_tel_return_cookie}), no action taken"
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
