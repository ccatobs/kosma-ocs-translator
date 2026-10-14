"""ACU stub: an in-process stand-in for the OCS, the TCS and the ACU, on a fake clock.

It answers the OCS client calls the translator makes. A path runs from its start
time to its last point; the ACU follows it after `settle` seconds. Switches:
down (OCS unreachable), fault (ACU summary fault), track_error (arcsec added to the
actual elevation), command_ids (False: the TCS reports no command id), drop_paths
(accept that many paths but never run them, as the TCS does when Ready() fails).
"""

import math

import erfa
import numpy as np
import ocs

SITE = {"latitude": -22.985639, "longitude": -67.740278, "height": 5611.8}


class Clock:
    def __init__(self, now=1792040400.0):  # 2026-10-15T05:00:00Z, LST 31.0 deg
        self.now = now

    def __call__(self):
        return self.now


class AcuStub:
    def __init__(self, clock, settle=2.0):
        self.clock = clock
        self.settle = settle
        self.down = False
        self.fault = False
        self.track_error = 0.0
        self.command_ids = True
        self.commanded = True  # False: NaN commanded positions (ACU simulator v0.6.0)
        self.drop_paths = 0  # accept this many paths without running them (TCS Ready() refusal)
        self.paths = []  # every accepted path: {"id", "start", "points"}
        self.aborts = 0
        self.running = None
        self.az, self.el = 180.0, 45.0  # where the ACU stands when no path moves it

    def _reach(self):
        if self.down:
            raise ocs.OCSConnectionError("OCS unreachable (ACU stub)")

    def _commanded(self, now):
        p = self.running["points"]
        dt = now - self.running["start"]
        return np.interp(dt, p[:, 0], p[:, 1]), np.interp(dt, p[:, 0], p[:, 2])

    def _update(self):
        now = self.clock()
        if self.running and now > self.running["start"] + self.running["points"][-1, 0]:
            self.az, self.el = self._commanded(now)
            self.running = None

    def get_telescope_position(self):
        self._reach()
        return dict(SITE)

    def get_running_command(self):
        self._reach()
        self._update()
        if not self.running:
            return None
        command = {"endpoint": "/path", "started_at": self.running["start"]}
        if self.command_ids:
            command["id"] = self.running["id"]
        return command

    def abort(self):
        self._reach()
        self._update()
        if not self.running:
            raise ocs.OCSRejected("nothing to abort", 409)
        self.aborts += 1
        now = self.clock()
        if now >= self.running["start"] + self.settle:
            self.az, self.el = self._commanded(now)
        self.running = None
        return {"status": "ok"}

    def path(self, start_time, points, coordsys="Horizon"):
        self._reach()
        # the client's own checks, with the lead taken on the fake clock
        ocs.check_path(start_time - self.clock(), points, coordsys)
        self._update()
        if self.running:
            raise ocs.OCSBusy("busy", 503)
        path = {"id": len(self.paths) + 1, "start": start_time, "points": np.array(points)}
        self.paths.append(path)
        if self.drop_paths:
            self.drop_paths -= 1
        else:
            self.running = path
        return {"status": "ok", "id": path["id"]}

    def get_status(self):
        self._reach()
        self._update()
        now = self.clock()
        az = az_cmd = self.az
        el = el_cmd = self.el
        if self.running and now >= self.running["start"]:
            az_cmd, el_cmd = self._commanded(now)
            if now >= self.running["start"] + self.settle:
                az, el = az_cmd, el_cmd + self.track_error / 3600
        raw = {
            "Azimuth current position": az,
            "Elevation current position": el,
            "Azimuth commanded position": az_cmd if self.commanded else -1e9,
            "Elevation commanded position": el_cmd if self.commanded else -1e9,
            "Azimuth Mode": "ProgramTrack" if self.running else "Stop",
            "Elevation Mode": "ProgramTrack" if self.running else "Stop",
            "Azimuth summary fault": "Fault" if self.fault else "No Fault",
            "Elevation summary fault": "No Fault",
        }
        return ocs.TelescopeStatus.from_raw(raw)


def separation(az1, el1, az2, el2):
    """Great-circle angle between two az/el positions (deg) in arcsec."""
    return math.degrees(erfa.seps(*np.radians([az1, el1, az2, el2]))) * 3600
