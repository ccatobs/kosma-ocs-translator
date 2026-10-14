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
STATUS_STALE = 2.0  # s after which an ACU status no longer tells on-track
SETTLE = 3.0  # s the ACU needs on the track before an OTF line starts
UNKNOWN = "unknown"  # the TCS reports no command id


class CommandError(Exception):
    """A command the translator refuses, with its tel_error code."""

    def __init__(self, code, message):
        super().__init__(message)
        self.code = code


def _line_time(t, otf_start, duration):
    """OTF line time: the map offset is del + v * clip(t - S, -10 s, D); 0 in mode N."""
    if otf_start is None:
        return np.zeros_like(t)
    return np.clip(np.asarray(t) - otf_start, -10.0, duration)


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
                self.results.put((kind, args, fn(*args), None))
            except Exception as error:  # noqa: BLE001 - every failure goes back to the loop
                self.results.put((kind, args, None, error))
            self.jobs.task_done()

    def finished(self):
        """The (kind, args, result, error) of every job done since the last call."""
        while True:
            try:
                yield self.results.get_nowait()
            except queue.Empty:
                return

    def join(self):
        self.jobs.join()


class Translator:
    def __init__(
        self,
        ocs,
        read_dir,
        write_dir,
        eop_path,
        clock=time.time,
        supports_ephemeris=True,
        window_points=WINDOW_POINTS,
    ):
        self.ocs = ocs
        self.supports_ephemeris = supports_ephemeris
        self.window_points = window_points
        self.read_dir = Path(read_dir)
        self.clock = clock
        self.eop = geometry.read_bulletin_a(eop_path)
        self.worker = Worker()
        self.tel2obs = kio.Tel2obsWriter(Path(write_dir) / "KOSMA_tel2obs.set")
        self.alive = kio.SetWriter(Path(write_dir) / "kosma_ocs_translator_alive")
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
        self.on = False  # tel_on_track
        self.lost = False  # tel_lost_track, latched until the next command
        self.error = (0, "NONE", 5)  # tel_error, tel_error_string, tel_error_level
        self.scan = None  # (S, D): the scan window [S, S + D)
        self.duration = 0.0
        self.otf_start = None  # S of an OTF line, inf for a pre-OTF command
        self.refusals = 0  # windows the TCS accepted but did not run, for this command

    def start(self):
        """Take the command already in KOSMA_obs2tel.set as seen; report cookie 0."""
        site = self.ocs.get_telescope_position()
        self.site = geometry.Site(site["latitude"], site["longitude"], site["height"])
        _, astra = kio.read_set(self.read_dir / "KOSMA_astra.in")
        kosma = (
            float(astra["a_latitude"]),
            -float(astra["a_longitude"]),
            float(astra["a_altitude"]),
        )
        if not np.allclose(
            kosma, (site["latitude"], site["longitude"], site["height"]), atol=1e-5, rtol=0
        ):
            raise ValueError(
                f"site of the OCS {site} disagrees with KOSMA_astra.in {kosma} (+east)"
            )
        self.telescope = astra["a_sitename"]
        self.plate_scale = float(astra["a_platescale"])
        self.stamp = kio.read_stamp(self.obs2tel)
        try:
            _, command = kio.read_obs2tel(self.obs2tel)
            self.last_cookie, self.update_time = command["obs_cookie"], _update_time(command)
        except (OSError, KeyError, ValueError):
            pass
        self.write(self.clock())

    def step(self):
        now = self.clock()
        for kind, args, result, error in self.worker.finished():
            getattr(self, f"_{kind}_done")(*args, result, error)
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
        self.check_window(now)
        self.update(now)
        if now - self.alive.last_stamp >= 1.0:
            kio.write_alive(self.alive, now)
        due = now - self.written[0] >= self.update_time - POLL / 2
        if due or self.state() != self.written_state:
            self.write(now)

    def update(self, now):
        """Evaluate on-track and latch lost track inside the scan window."""
        self.on, note = self.on_track(now)
        if self.error[0] == 0:  # warnings set only the string and the level
            self.error = (0, note, 7) if note else (0, "NONE", 5)
        acu = self.status[1] if self.status else None
        if (
            self.track
            and acu
            and (acu.azimuth_fault or acu.elevation_fault)
            and self.error[0] in (0, 199)
        ):
            self.on = False
            self.error = (212, "ACU summary fault", 8)
        if self.on and self.scan is None and self.duration > 0:
            self.scan = (now, self.duration)  # mode N: the scan starts on track
        if self.scan and self.scan[0] <= now < sum(self.scan) and not self.on and not self.lost:
            self.lost = True
            self.error = (199, f"lost track {now - self.scan[0]:.1f} s into the scan window", 8)

    def state(self):
        return self.cookie, self.on, self.lost, self.error

    def new_command(self, command, now):
        """Validate a command with a new cookie, echo the cookie, send its window."""
        self.last_cookie = self.cookie = command["obs_cookie"]
        self.update_time = _update_time(command)
        self.on = self.lost = False
        self.scan = None
        self.refusals = 0
        try:
            self.accept(command, now)
            self.error = (0, "NONE", 5)
        except CommandError as refused:
            log.warning("cookie %s refused: %s", self.cookie, refused)
            self.error = (refused.code, str(refused), 8)
        self.write(now)

    def accept(self, command, now):
        if command["obs_chop_beam"] != 0 or command["obs_amp_chop"] != 0:
            raise CommandError(1, "chopper requested; chopped modes are not supported")
        mode = command["obs_otf_mode"]
        if mode == "P":
            raise CommandError(201, "pattern OTF (mode P) is not supported")
        if mode not in ("N", "C"):
            raise CommandError(202, f"obs_otf_mode {mode} is not N, C or P")
        name = command["obs_source_name"]
        if name.startswith("@") and not self.supports_ephemeris:
            raise CommandError(
                202, f"{name}: tel_supports_ephemeris is N, kephem resolves @ sources"
            )
        if name.startswith("@") and name[1:].lower() not in geometry.BODIES:
            raise CommandError(213, f"unknown solar-system body {name}")
        try:
            track = geometry.Track.from_command(command, self.pointing_offsets())
        except ValueError as error:
            raise CommandError(202, str(error)) from None
        duration = command["obs_track_duration"]
        otf_start = None  # mode N: the line time is 0
        if mode == "C":
            otf_start = command["obs_start_time"] if duration > 0 else math.inf  # pre-OTF
            if otf_start - now < LEAD + SETTLE:
                raise CommandError(
                    200, f"OTF start {otf_start - now:.1f} s ahead, needs {LEAD + SETTLE:.1f} s"
                )
        try:
            window, points = self.make_window(track, otf_start, duration, now)
        except ValueError as error:
            raise CommandError(198, str(error)) from None
        self.track, self.otf_start, self.duration = track, otf_start, duration
        self.tolerance = command["obs_tolerance"]
        if mode == "C" and duration > 0:
            self.scan = (otf_start, duration)
        self.send_window(window, points)

    def make_window(self, track, otf_start, duration, now):
        """The next window of track, starting LEAD seconds from now; raises ValueError."""
        start = now + LEAD
        t = start + SPACING * np.arange(self.window_points)
        az_ref = self.status[1].azimuth if self.status else 180.0
        tau = _line_time(t, otf_start, duration)
        points = geometry.window(track, t, tau, self.site, self.eop, az_ref)
        return {"start": start, "end": t[-1], "id": None}, points

    def send_window(self, window, points):
        self.window = window
        self.worker.submit("upload", self._upload, window, points)

    def check_window(self, now):
        """Renew a window that ran out or that the TCS does not run (TCS Ready() refusal)."""
        window = self.window
        if window is None or window["id"] is None:
            return
        if now >= window["end"]:
            self.renew(now)
            return
        if self.status is None or self.status[0] < window["sent"] + STATUS_PERIOD:
            return
        running = self.status[2]
        if running in (window["id"], UNKNOWN):
            self.refusals = 0
        elif running is not None:
            self.window = None
            self.error = (211, f"the TCS runs command {running}, not path {window['id']}", 8)
        elif self.refusals < 2:
            self.refusals += 1
            log.warning("the TCS does not run path %s, sending it again", window["id"])
            self.renew(now)
        else:
            self.window = None
            self.error = (211, f"the TCS accepted but did not run path {window['id']} 3 times", 8)

    def renew(self, now):
        """A renewal: the window of the current command from now on."""
        try:
            self.send_window(*self.make_window(self.track, self.otf_start, self.duration, now))
        except ValueError as error:
            self.window = None
            self.error = (198, str(error), 8)

    def pointing_offsets(self):
        """Observer pointing offsets (arcsec) from KOSMA_track_point.set, read per command."""
        try:
            _, values = kio.read_set(self.read_dir / "KOSMA_track_point.set")
            return float(values["track_azpoi"]), float(values["track_elpoi"])
        except (OSError, KeyError, ValueError):
            return 0.0, 0.0

    def line_time(self, t):
        return _line_time(t, self.otf_start, self.duration)

    def _upload(self, window, points):
        """Worker: abort what the TCS runs, then send the window; return the command id."""
        if self.ocs.get_running_command() is not None:
            try:
                self.ocs.abort()
            except ocs_client.OCSRejected as error:
                if error.status_code != 409:  # 409: nothing left to abort
                    raise
        return self.ocs.path(window["start"], points, "Horizon").get("id", UNKNOWN)

    def _upload_done(self, window, points, command_id, error):
        if window is not self.window:
            return  # the window of an earlier command
        if error is None:
            window["id"], window["sent"] = command_id, self.clock()
            return
        log.error("upload for cookie %s failed: %s", self.cookie, error)
        self.window = None
        if isinstance(error, ocs_client.OCSConnectionError):
            self.error = (210, f"OCS unreachable: {error}", 8)
        else:
            self.error = (211, f"OCS or TCS rejected the path: {error}", 8)

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
        """(on track, development note): the TCS runs our window, the ACU tracks within
        obs_tolerance, and an OTF line has started."""
        window = self.window
        if self.error[0] not in (0, 199) or window is None or window["id"] is None:
            return False, None  # a refused command is never on track
        if self.status is None:
            return False, None
        read, acu, running = self.status
        if now - read > STATUS_STALE:
            return False, f"no ACU status for {now - read:.1f} s"
        # before the window starts the ACU stands still, commanded = actual
        if not window["start"] <= read <= now <= window["end"]:
            return False, None
        note = None
        if UNKNOWN in (running, window["id"]):
            note = "on track inferred: the TCS reports no command id (development only)"
        elif running != window["id"]:
            return False, None
        if self.duration > 0 and self.otf_start is not None and now < self.otf_start:
            return False, note  # an OTF line is on track from S, a pre-OTF command at once
        if _valid(acu.azimuth_commanded) and _valid(acu.elevation_commanded):
            az, el = acu.azimuth_commanded, acu.elevation_commanded
        else:  # ACU simulator v0.6.0: compare with the target, raw = sky without pointing model
            note = "on track from ACU actual vs target: the ACU reports no commanded position (development only)"
            az, el = (
                float(x[0])
                for x in geometry.target(
                    self.track, read, self.line_time(read), self.site, self.eop
                )
            )
        error = math.hypot(
            ((acu.azimuth - az + 180) % 360 - 180) * math.cos(math.radians(acu.elevation)),
            acu.elevation - el,
        )
        return error * 3600 <= self.tolerance, note

    def values(self, now):
        acu = self.status[1] if self.status else None
        if self.track:
            tau = self.line_time(now)
            az, el = (
                float(x[0]) for x in geometry.target(self.track, now, tau, self.site, self.eop)
            )
            angle = geometry.focal_plane_angle(self.track, now, tau, self.site, self.eop)
        else:
            az, el = (acu.azimuth, acu.elevation) if acu else (0.0, 0.0)
            angle = 0.0
        az_act, el_act = az, el
        if acu and _valid(acu.azimuth_commanded) and _valid(acu.elevation_commanded):
            az_act += (acu.azimuth - acu.azimuth_commanded + 180) % 360 - 180
            el_act += acu.elevation - acu.elevation_commanded
        time_act = self.status[0] if self.status else now
        dut1 = self.eop.at(time_act / 86400 + 40587)[2]
        return {
            "tel_telescope": self.telescope,
            "tel_on_track": "Y" if self.on else "N",
            "tel_lost_track": "Y" if self.lost else "N",
            "tel_pos_in_range": "N" if self.error[0] == 198 else "Y",
            "tel_error": self.error[0],
            "tel_error_string": self.error[1],
            "tel_error_level": self.error[2],
            "tel_return_cookie": self.cookie,
            "tel_plate_scale": self.plate_scale,
            "tel_latitude": self.site.latitude,
            "tel_longitude": -self.site.longitude,  # +west, as the CCAT template
            "tel_altitude": self.site.height,
            "tel_angle_focal_plane": angle,
            "tel_los_act": 0.0,
            "tel_time_act": time_act,
            "tel_time_del": 0.0,
            "tel_azm_cmd": az,
            "tel_elv_cmd": el,
            "tel_azm_act": az_act,
            "tel_elv_act": el_act,
            "tel_supports_ephemeris": "Y" if self.supports_ephemeris else "N",
            "tel_mjd1": (time_act + dut1) / 86400 + 40587,
        }

    def write(self, now):
        values = self.values(now)
        self.tel2obs.write(values, now)
        self.written = (now, values)
        self.written_state = self.state()


def _update_time(command):
    """tel2obs cycle: obs_tel_info_update_time, 1 s when it is 0 (no continuous update)."""
    return command["obs_tel_info_update_time"] if command["obs_tel_info_update_time"] > 0 else 1.0


def _valid(position):
    """An ACU position, not the TCS NaN placeholder (-1e9)."""
    return position is not None and math.isfinite(position) and position > -1e8
