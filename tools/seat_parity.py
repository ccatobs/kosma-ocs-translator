"""Seat parity: run the same KOSMA scenarios with astra and with the translator in the
telescope seat, and compare the two tel2obs streams.

Runs on a KOSMA host with the local KOSMA simulation (kosma_bridge) and the OCS stack
of run.bash (ACU simulator). Never in CI.

    python tools/seat_parity.py --astra ASTRA --astra-io DIR --tls-dir OCS_TLS --report FILE

Scenarios, each a real KOSMA command or the command sequence of control_tel:
  track     setsource (J2000 star) + KOSMA_track -o -w
  off       KOSMA_track -O -w, OFF -1800" in J2000
  map       KOSMA_setoffset -l 60 -b -30 + KOSMA_track -o -w
  otf       PRETP, on track, then OTFTP 15 s ahead (pre_otf_time), as otf_server sends
            them through control_tel; written directly, so no backend takes data
The seat in turn: astra (astra -t -n -r: test telescope, no refraction, as ADR 0001
puts refraction in the OCS) and the translator with the ACU simulator. The tool stops
astra_dummy for the run and restores it and the measurement setup afterwards.
For every tel2obs write it compares the reported target with the translator geometry
for the command in force, at tel_time_act.
"""

import argparse
import json
import os
import random
import statistics
import subprocess
import sys
import threading
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from kosma_ocs_translator import geometry, kio

RW = Path("/net/KOSMA_file_io/ReadWrite")
SITE = geometry.Site(-22.9856389, -67.7402778, 5611.8)  # KOSMA_astra.in


def sh(rc, command, timeout=200):
    """Run a KOSMA command in the KOSMA environment; return (exit code, output)."""
    p = subprocess.run(
        ["bash", "-c", f"source {rc} >/dev/null 2>&1; {command}"],
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )
    return p.returncode, p.stdout + p.stderr


class Recorder(threading.Thread):
    """Every tel2obs write and every obs2tel command, with the time seen."""

    def __init__(self):
        super().__init__(daemon=True)
        self.writes, self.commands, self.running = [], {}, True

    def run(self):
        last_tel = last_obs = None
        while self.running:
            try:
                stamp = kio.read_stamp(RW / "KOSMA_obs2tel.set")
                if stamp != last_obs:
                    last_obs = stamp
                    _, command = kio.read_obs2tel(RW / "KOSMA_obs2tel.set")
                    self.commands.setdefault(str(command["obs_cookie"]), (time.time(), command))
                stamp, values = kio.read_set(RW / "KOSMA_tel2obs.set")
                if stamp != last_tel:
                    last_tel = stamp
                    self.writes.append((time.time(), values))
            except (OSError, ValueError):
                pass
            time.sleep(0.05)


def write_command(changes):
    """Write a command into KOSMA_obs2tel.set as control_tel does; return its cookie."""
    lines = (RW / "KOSMA_obs2tel.set").read_text().splitlines()
    cookie = random.randint(1, 9_999_999)
    changes = {**changes, "obs_cookie": cookie}
    for i, line in enumerate(lines[1:], 1):
        name = line.split("!")[0].split()[-1] if line.split("!")[0].split() else ""
        if name in changes:
            lines[i] = f"{changes[name]}   {name}   ! seat_parity"
    now = time.time()
    lines[0] = (
        f"{kio.header_time(now)}{int(now):12d}.{int(now % 1 * 1e6):06d}"
        "   File update time stamp   ! seat_parity (control_tel sequence)"
    )
    tmp = RW / "KOSMA_obs2tel.set_seat_parity"
    tmp.write_text("\n".join(lines) + "\n")
    tmp.replace(RW / "KOSMA_obs2tel.set")
    return cookie


def wait_on_track(recorder, cookie, timeout):
    t0 = time.time()
    while time.time() - t0 < timeout:
        for _, v in recorder.writes[::-1][:3]:
            if v["tel_return_cookie"] == str(cookie) and v["tel_on_track"] == "Y":
                return True
        time.sleep(0.2)
    return False


def scenarios(rc, ra, dec, recorder):
    """Run the scenarios; return {name: [cookies]}."""
    out = {}
    sh(rc, f'setsource SEATPARITY "" -l {ra:.6f} -b {dec:.6f} -C J2000 -d 0')
    sh(rc, "KOSMA_setoffset -l 0 -b 0 -L -1800 -B 0 -m J2000 -o J2000 --vlam 12 --vbet 0")

    def kosma(name, command):
        before = set(recorder.commands)
        code, text = sh(rc, command)
        time.sleep(3)
        out[name] = {
            "cookies": [c for c in recorder.commands if c not in before],
            "exit": code,
            "kosma": [x for x in text.splitlines() if "TEL_TIMING" in x or "track" in x.lower()][
                -4:
            ],
        }

    kosma("track", "KOSMA_track -o -w")
    kosma("off", "KOSMA_track -O -w")
    sh(rc, "KOSMA_setoffset -l 60 -b -30")
    kosma("map", "KOSMA_track -o -w")
    sh(rc, "KOSMA_setoffset -l 0 -b 0")
    # otf_server's sequence: PRETP (wait on track), OTFTP pre_otf_time = 15 s ahead
    otf = {
        "obs_otf_mode": "C",
        "obs_lam_del": -120.0,
        "obs_bet_del": 0.0,
        "obs_lam_vel": 12.0,
        "obs_bet_vel": 0.0,
        "obs_lam_off": 0.0,
        "obs_bet_off": 0.0,
    }
    pre = write_command({**otf, "obs_track_duration": 0.0})
    ok = wait_on_track(recorder, pre, 100)
    start = time.time() + 15.0
    line = write_command({**otf, "obs_track_duration": 20.0, "obs_start_time": f"{start:.5f}"})
    time.sleep(start + 25 - time.time())
    out["otf"] = {"cookies": [str(pre), str(line)], "exit": 0 if ok else 1, "kosma": []}
    return out


def evaluate(recorder, result, eop):
    """Per scenario: echo and on-track delays, errors, target residuals."""
    report = {}
    for name, run in result.items():
        rows = {"exit": run["exit"], "kosma": run["kosma"], "commands": []}
        for cookie in run["cookies"]:
            sent, command = recorder.commands[cookie]
            writes = [(t, v) for t, v in recorder.writes if t >= sent]
            echo = next((t for t, v in writes if v["tel_return_cookie"] == cookie), None)
            mine = [(t, v) for t, v in writes if v["tel_return_cookie"] == cookie]
            on = next((t for t, v in mine if v["tel_on_track"] == "Y"), None)
            track = geometry.Track.from_command(command)
            otf_start = None
            if command["obs_otf_mode"] == "C":
                otf_start = (
                    command["obs_start_time"] if command["obs_track_duration"] > 0 else np.inf
                )
            residuals, angles = [], []
            for _, v in mine:
                t = float(v["tel_time_act"])
                tau = (
                    0.0
                    if otf_start is None
                    else float(np.clip(t - otf_start, -10, command["obs_track_duration"]))
                )
                az, el = geometry.target(track, t, tau, SITE, eop)
                daz = (float(v["tel_azm_cmd"]) - az[0] + 180) % 360 - 180
                residuals.append(
                    float(
                        np.hypot(daz * np.cos(np.radians(el[0])), float(v["tel_elv_cmd"]) - el[0])
                        * 3600
                    )
                )
                angle = geometry.focal_plane_angle(track, t, tau, SITE, eop)
                angles.append(abs((float(v["tel_angle_focal_plane"]) - angle + 180) % 360 - 180))
            last = mine[-1][1] if mine else {}
            rows["commands"].append(
                {
                    "cookie": cookie,
                    "mode": command["obs_otf_mode"]
                    + (
                        ""
                        if command["obs_otf_mode"] == "N"
                        else (" line" if command["obs_track_duration"] > 0 else " pre")
                    ),
                    "echo_s": None if echo is None else round(echo - sent, 2),
                    "on_track_s": None if on is None else round(on - sent, 2),
                    "errors": sorted({v["tel_error"] for _, v in mine}),
                    "lost_track": sorted({v["tel_lost_track"] for _, v in mine}),
                    "writes": len(mine),
                    "target_residual_arcsec": None
                    if not residuals
                    else {
                        "median": round(statistics.median(residuals), 3),
                        "max": round(max(residuals), 3),
                    },
                    "angle_residual_deg": None if not angles else round(max(angles), 5),
                    "static": {
                        k: last.get(k)
                        for k in (
                            "tel_telescope",
                            "tel_plate_scale",
                            "tel_latitude",
                            "tel_longitude",
                            "tel_altitude",
                            "tel_supports_ephemeris",
                        )
                    },
                }
            )
        report[name] = rows
    return report


def run_seat(seat, args, ra, dec, eop):
    proc = None
    if seat == "astra":
        env = dict(os.environ, READ_DIR=str(RW), WRITE_DIR=str(RW))
        proc = subprocess.Popen(
            [args.astra, "-t", "-n", "-r"],
            cwd=args.astra_io,
            env=env,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    else:
        # a fresh ACU simulator: its Clear Stack keeps the last point of an earlier path
        subprocess.run(["docker", "restart", args.acu_container], check=True, capture_output=True)
        time.sleep(4)
        proc = subprocess.Popen(
            [
                str(REPO / ".venv" / "bin" / "kosma_ocs_translator"),
                "--tls-dir",
                args.tls_dir,
                "--window-points",
                str(args.window_points),
                "--iers",
                args.iers,
                "--log-file",
                str(Path(args.report).with_suffix(".translator.log")),
            ],
            env=dict(os.environ, READ_DIR=str(RW), WRITE_DIR=str(RW)),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    time.sleep(4)
    recorder = Recorder()
    recorder.start()
    try:
        result = scenarios(args.rc, ra, dec, recorder)
    finally:
        recorder.running = False
        proc.terminate()
        proc.wait(timeout=10)
        if seat == "astra":
            Path("/tmp/astra_lock").unlink(missing_ok=True)
    return evaluate(recorder, result, eop)


def markdown(report, ra, dec):
    lines = [
        "# Seat parity: astra vs translator\n",
        f"Source: J2000 {ra:.4f} {dec:.4f} (hour angle -1 h at start).\n",
    ]
    for name in report["astra"]:
        lines.append(f"## {name}\n")
        lines.append(
            "| Seat | Command | Echo [s] | On track [s] | tel_error | Lost | Writes | Target - geometry [arcsec] median / max | Angle [deg] max |"
        )
        lines.append("|---|---|---|---|---|---|---|---|---|")
        for seat in ("astra", "translator"):
            for c in report[seat][name]["commands"]:
                r = c["target_residual_arcsec"] or {}
                lines.append(
                    f"| {seat} | {c['mode']} | {c['echo_s']} | {c['on_track_s']} | {','.join(c['errors'])} | "
                    f"{','.join(c['lost_track'])} | {c['writes']} | {r.get('median')} / {r.get('max')} | {c['angle_residual_deg']} |"
                )
        lines.append("")
    lines.append("## Static tel2obs fields (last write)\n")
    for seat in ("astra", "translator"):
        first = next(iter(report[seat].values()))["commands"][0]["static"]
        lines.append(f"- {seat}: " + ", ".join(f"{k}={v}" for k, v in first.items()))
    return "\n".join(lines) + "\n"


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--astra", required=True, help="astra binary (OBS_CCAT, IOPATH = --astra-io)")
    ap.add_argument("--astra-io", required=True, help="astra IOPATH: logs/ and share/astra/")
    ap.add_argument("--tls-dir", required=True, help="OCS client certificates of the OCS stack")
    ap.add_argument("--rc", default="/opt/kosma_control/.kosmacontrolrc")
    ap.add_argument(
        "--acu-container", default="ocs-stack-epic5-acu-1", help="ACU simulator container"
    )
    ap.add_argument(
        "--window-points",
        type=int,
        default=30,
        help="short windows: the ACU simulator ignores a new path until the last window ends",
    )
    ap.add_argument("--iers", help="IERS Bulletin A for both seats; default: the one astra reads")
    ap.add_argument("--report", required=True)
    args = ap.parse_args()
    args.iers = args.iers or str(Path(args.astra_io) / "share" / "astra" / "bulletinA")
    eop = geometry.read_bulletin_a(args.iers)
    from astropy import units as u
    from astropy.time import Time

    ra = (Time.now().sidereal_time("mean", longitude=SITE.longitude * u.deg).deg + 15) % 360
    dec = -30.0
    saved = {n: (RW / n).read_bytes() for n in ("measurement.set",)}
    sh(args.rc, "kosma servers stop astra_dummy")
    report = {}
    try:
        for seat in ("astra", "translator"):
            print(f"seat: {seat}", flush=True)
            report[seat] = run_seat(seat, args, ra, dec, eop)
    finally:
        for name, data in saved.items():
            tmp = RW / f"{name}_seat_parity"
            tmp.write_bytes(data)
            tmp.replace(RW / name)
        sh(args.rc, "kosma servers start astra_dummy")
    Path(args.report).with_suffix(".json").write_text(json.dumps(report, indent=1))
    Path(args.report).write_text(markdown(report, ra, dec))
    print(Path(args.report).read_text())


if __name__ == "__main__":
    main()
