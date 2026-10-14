"""The translator loop: fills the KOSMA telescope seat at FYST.

One loop owns all state. It polls the header stamp of KOSMA_obs2tel.set every 0.1 s,
validates a new command, echoes its cookie and writes KOSMA_tel2obs.set. A single
command worker sends the OCS requests, so a slow request never delays tel2obs.
"""

import logging
import math
import queue
import threading
import time
from pathlib import Path

import numpy as np
import ocs as ocs_client

from . import geometry, kio

log = logging.getLogger("kosma_ocs_translator")

POLL = 0.1  # s between looks at the obs2tel header stamp
LEAD = 10.3  # s from now to the window start: the TCS minimum of 9.8 s plus 0.5 s margin
SPACING = 1.0  # s between path points
WINDOW_POINTS = 1500  # points per window: 1499 s, about 82 KB of request body
STATUS_PERIOD = 0.5  # s between ACU status reads
UNKNOWN = "unknown"  # the TCS reports no command id


class Worker:
    """The single command worker: runs OCS requests one at a time, off the loop."""

    def __init__(self):
        self.jobs = queue.Queue()
        self.results = queue.Queue()
        threading.Thread(target=self._run, daemon=True).start()

    def submit(self, kind, fn, *args):
        self.jobs.put((kind, fn, args))

    def _run(self):
        while True:
            kind, fn, args = self.jobs.get()
            try:
                self.results.put((kind, fn(*args), None))
            except Exception as error:  # noqa: BLE001 - every failure goes back to the loop
                self.results.put((kind, None, error))
            self.jobs.task_done()

    def finished(self):
        """The (kind, result, error) of every job done since the last call."""
        while True:
            try:
                yield self.results.get_nowait()
            except queue.Empty:
                return

    def join(self):
        self.jobs.join()


class Translator:
    def __init__(self, ocs, read_dir, write_dir, eop_path, clock=time.time):
        self.ocs = ocs
        self.read_dir = Path(read_dir)
        self.clock = clock
        self.eop = geometry.read_bulletin_a(eop_path)
        self.worker = Worker()
        self.tel2obs = kio.Tel2obsWriter(Path(write_dir) / "KOSMA_tel2obs.set")
        self.obs2tel = self.read_dir / "KOSMA_obs2tel.set"
        self.stamp = None
        self.cookie = 0
        self.last_cookie = None
        self.update_time = 1.0
        self.written = None  # (time, values) of the last tel2obs write
        self.site = None
        self.track = None
        self.tolerance = 10.0
        self.window = None  # {"start", "end", "id"} of the uploaded window
        self.status = None  # (time, TelescopeStatus, running command id)
        self.status_pending = False

    def start(self):
        """Take the command already in KOSMA_obs2tel.set as seen; report cookie 0."""
        site = self.ocs.get_telescope_position()
        self.site = geometry.Site(site["latitude"], site["longitude"], site["height"])
        self.stamp = kio.read_stamp(self.obs2tel)
        try:
            self.last_cookie = kio.read_obs2tel(self.obs2tel)[1]["obs_cookie"]
        except (OSError, KeyError, ValueError):
            pass
        self.write(self.clock())

    def step(self):
        now = self.clock()
        for kind, result, error in self.worker.finished():
            getattr(self, f"_{kind}_done")(result, error)
        stamp = kio.read_stamp(self.obs2tel)
        if stamp is not None and stamp != self.stamp:
            self.stamp = stamp
            _, command = kio.read_obs2tel(self.obs2tel)
            if command["obs_cookie"] != self.last_cookie:
                self.new_command(command, now)
        if not self.status_pending and (
            self.status is None or now - self.status[0] >= STATUS_PERIOD
        ):
            self.status_pending = True
            self.worker.submit("status", self._read_status)
        if now - self.written[0] >= self.update_time:
            self.write(now)

    def new_command(self, command, now):
        self.last_cookie = self.cookie = command["obs_cookie"]
        self.tolerance = command["obs_tolerance"]
        self.track = geometry.Track.from_command(command)
        self.write(now)
        start = now + LEAD
        t = start + SPACING * np.arange(WINDOW_POINTS)
        points = geometry.window(self.track, t, np.zeros_like(t), self.site, self.eop)
        self.window = {"start": start, "end": t[-1], "id": None}
        self.worker.submit("upload", self._upload, start, points)

    def _upload(self, start, points):
        """Worker: abort what the TCS runs, then send the window; return the command id."""
        if self.ocs.get_running_command() is not None:
            try:
                self.ocs.abort()
            except ocs_client.OCSRejected as error:
                if error.status_code != 409:  # 409: nothing left to abort
                    raise
        return self.ocs.path(start, points, "Horizon").get("id", UNKNOWN)

    def _upload_done(self, command_id, error):
        self.window["id"] = command_id

    def _read_status(self):
        """Worker: the ACU status and the id of the command the TCS runs."""
        now = self.clock()
        status = self.ocs.get_status()
        running = self.ocs.get_running_command()
        return now, status, None if running is None else running.get("id", UNKNOWN)

    def _status_done(self, status, error):
        self.status_pending = False
        if error is None:
            self.status = status

    def on_track(self, now):
        if self.window is None or self.window["id"] is None or self.status is None:
            return False
        _, acu, running = self.status
        if running != self.window["id"] or not self.window["start"] <= now <= self.window["end"]:
            return False
        error = math.hypot(
            (acu.azimuth - acu.azimuth_commanded) * math.cos(math.radians(acu.elevation)),
            acu.elevation - acu.elevation_commanded,
        )
        return error * 3600 <= self.tolerance

    def values(self, now):
        return {
            "tel_telescope": "CCAT",
            "tel_on_track": "Y" if self.on_track(now) else "N",
            "tel_lost_track": "N",
            "tel_pos_in_range": "Y",
            "tel_error": 0,
            "tel_error_string": "NONE",
            "tel_error_level": 5,
            "tel_return_cookie": self.cookie,
            "tel_plate_scale": 13.89,
            "tel_latitude": 0.0,
            "tel_longitude": 0.0,
            "tel_altitude": 0.0,
            "tel_angle_focal_plane": 0.0,
            "tel_los_act": 0.0,
            "tel_time_act": now,
            "tel_time_del": 0.0,
            "tel_azm_cmd": 0.0,
            "tel_elv_cmd": 0.0,
            "tel_azm_act": 0.0,
            "tel_elv_act": 0.0,
            "tel_supports_ephemeris": "Y",
            "tel_mjd1": 0.0,
        }

    def write(self, now):
        values = self.values(now)
        self.tel2obs.write(values, now)
        self.written = (now, values)
