"""Make tests/fixtures/golden.json: target positions from astra (astra -t -f 4 -n -r).

Runs on a KOSMA development host, never in CI. astra must be built with OBS_CCAT and
an IOPATH that points to IO_DIR (logs/, share/astra/ with bulletinA and
astra_debug_time.inp); it reads READ_DIR/WRITE_DIR, here IO_DIR/ReadWrite, and
locks /tmp/astra_lock.

    python tools/make_golden.py --astra ASTRA_BINARY --io IO_DIR --sha KOSMA_SHA

For each scenario the script writes the fixed times, starts astra, writes one obs2tel
command, waits for astra to step through the times and reads, per cycle, the
target from the ASTRA_POINT line (after the focal-plane and pointing offsets, before
the pointing model) and the time from the ASTRA_STEER line (fixed time + 2 s).
tau is the OTF line time astra used: its map offset is del + v * tau.
"""

import argparse
import json
import math
import os
import re
import subprocess
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
from astropy import units as u
from astropy.coordinates import FK4, FK5, Galactic, SkyCoord
from astropy.time import Time

REPO = Path(__file__).resolve().parent.parent
TEMPLATE = REPO / "tests" / "fixtures" / "ccat_chai" / "KOSMA_obs2tel.set"
LON_WEST = 67.7402778  # KOSMA_astra.in
PAD = 8  # cycles at the first time while astra starts and reads the command


def lst_deg(utc):
    return Time(utc).sidereal_time("mean", longitude=-LON_WEST * u.deg).deg


def equatorial(utc, ha_h, dec):
    """J2000 (ra, dec) of a source at hour angle ha_h at utc."""
    return (lst_deg(utc) - 15 * ha_h) % 360, dec


def times(start, n, step):
    t0 = datetime.fromisoformat(start)
    return [(t0 + timedelta(seconds=step * i)).strftime("%Y-%m-%dT%H:%M:%S") for i in range(n)]


def scenarios():
    out = []

    def add(name, utcs, **command):
        out.append(
            {
                "name": name,
                "utcs": utcs,
                "track_point": command.pop("track_point", (0, 0)),
                "start_offset": command.pop("start_offset", None),
                "command": command,
            }
        )

    t = times("2026-10-15T02:00:00", 3, 600)
    ra, dec = equatorial(t[0], -2.0, -30.0)
    add("j2000", t, obs_coord_sys_on="J2000", obs_lam_on=ra, obs_bet_on=dec)

    t = times("2026-10-15T03:00:00", 2, 900)
    ra, dec = equatorial(t[0], 1.0, -50.0)
    b = SkyCoord(ra * u.deg, dec * u.deg, frame=FK5(equinox="J2000")).transform_to(
        FK4(equinox="B1950")
    )
    add(
        "b1950_pm",
        t,
        obs_coord_sys_on="B1950",
        obs_lam_on=b.ra.deg,
        obs_bet_on=b.dec.deg,
        obs_pm_lam=0.8,
        obs_pm_bet=-0.4,
    )

    t = times("2026-10-16T04:00:00", 2, 900)
    ra, dec = equatorial(t[0], 0.5, -40.0)
    g = SkyCoord(ra * u.deg, dec * u.deg, frame=FK5(equinox="J2000")).transform_to(Galactic())
    add(
        "galactic",
        t,
        obs_coord_sys_on="GALACTIC",
        obs_lam_on=g.l.deg,
        obs_bet_on=g.b.deg,
        obs_coord_sys_del="GALACTIC",
        obs_coord_sys_off="GALACTIC",
    )

    t = times("2026-10-16T05:00:00", 2, 900)
    ra, dec = equatorial(t[0], -1.0, 0.5)
    add(
        "hipparcos_pm_parallax",
        t,
        obs_coord_sys_on="HIPPARCOS",
        obs_lam_on=ra,
        obs_bet_on=dec,
        obs_pm_lam=0.5,
        obs_pm_bet=-0.25,
        obs_paralx=0.4,
    )

    t = times("2026-10-17T06:00:00", 2, 900)
    ra, dec = equatorial(t[0], 1.5, -60.0)
    add(
        "j2000_proper_motion",
        t,
        obs_coord_sys_on="J2000",
        obs_lam_on=ra,
        obs_bet_on=dec,
        obs_pm_lam=-0.75,
        obs_pm_bet=10.3,
    )

    add(
        "horizon_offsets",
        times("2026-10-17T07:00:00", 2, 900),
        obs_coord_sys_on="HORIZON",
        obs_lam_on=200.0,
        obs_bet_on=45.0,
        obs_coord_sys_del="HORIZON",
        obs_lam_del=30.0,
        obs_bet_del=20.0,
        obs_coord_sys_off="HORIZON",
        obs_lam_off=-600.0,
        obs_true_angle_off="N",
    )

    t = times("2026-10-18T02:30:00", 2, 900)
    ra, dec = equatorial(t[0], -1.5, -25.0)
    add(
        "map_and_off_j2000",
        t,
        obs_coord_sys_on="J2000",
        obs_lam_on=ra,
        obs_bet_on=dec,
        obs_lam_del=120.0,
        obs_bet_del=-60.0,
        obs_lam_off=-1800.0,
        obs_bet_off=300.0,
        obs_true_angle_off="N",
    )

    t = times("2026-10-18T03:30:00", 2, 900)
    ra, dec = equatorial(t[0], 2.0, -10.0)
    add(
        "map_galactic_off_horizon",
        t,
        obs_coord_sys_on="J2000",
        obs_lam_on=ra,
        obs_bet_on=dec,
        obs_coord_sys_del="GALACTIC",
        obs_lam_del=300.0,
        obs_bet_del=100.0,
        obs_coord_sys_off="HORIZON",
        obs_lam_off=600.0,
        obs_bet_off=-30.0,
    )

    t = times("2026-10-18T04:30:00", 2, 900)
    ra, dec = equatorial(t[0], -1.0, 20.0)
    add(
        "map_b1950",
        t,
        obs_coord_sys_on="J2000",
        obs_lam_on=ra,
        obs_bet_on=dec,
        obs_coord_sys_del="B1950",
        obs_lam_del=-200.0,
        obs_bet_del=150.0,
    )

    t = times("2026-10-19T05:00:00", 3, 300)
    ra, dec = equatorial(t[1], 0.0, -13.0)
    add(
        "focal_plane_near_80_deg",
        t,
        obs_coord_sys_on="J2000",
        obs_lam_on=ra,
        obs_bet_on=dec,
        obs_x_focal_plane=150.0,
        obs_y_focal_plane=-80.0,
        track_point=(12.5, -7.3),
    )

    t = times("2026-10-19T06:00:00", 2, 900)
    ra, dec = equatorial(t[0], 1.0, -45.0)
    add(
        "extended_range",
        t,
        obs_coord_sys_on="J2000",
        obs_lam_on=ra,
        obs_bet_on=dec,
        obs_extended_elevation_range="Y",
        obs_x_focal_plane=40.0,
        obs_y_focal_plane=25.0,
        track_point=(9.0, -4.0),
    )

    t = times("2026-10-20T07:00:00", 5, 60)
    ra, dec = equatorial(t[2], 0.0, 5.0)
    add("north_transit", t, obs_coord_sys_on="J2000", obs_lam_on=ra, obs_bet_on=dec)

    t = times("2026-10-20T08:00:00", 2, 60)
    ra, dec = equatorial(t[0], -0.5, -35.0)
    otf = {
        "obs_coord_sys_on": "J2000",
        "obs_lam_on": ra,
        "obs_bet_on": dec,
        "obs_otf_mode": "C",
        "obs_lam_del": -300.0,
        "obs_bet_del": 60.0,
        "obs_lam_vel": 30.0,
        "obs_bet_vel": -5.0,
        "obs_track_duration": 60.0,
    }
    add("otf_line_standby", t, start_offset=3600.0, **otf)
    add("otf_line_end", t, start_offset=-3600.0, **otf)
    add("otf_line_running", times("2026-10-20T08:00:00", 12, 30), start_offset=6.0, **otf)
    return out


def write_obs2tel(path, command, cookie):
    lines = TEMPLATE.read_text().splitlines()
    for name, value in command.items():
        for i, line in enumerate(lines):
            if re.match(rf"^\s*\S+\s+{name}\s", line):
                lines[i] = f"{value}   {name}   ! x"
                break
        else:
            raise KeyError(name)
    for i, line in enumerate(lines):
        if re.match(r"^\s*\S+\s+obs_cookie\s", line):
            lines[i] = f"{cookie}   obs_cookie   ! x"
    now = time.time()
    lines[0] = (
        time.strftime("%d-%b-%Y  %H:%M:%S", time.localtime(now))
        + f"{int(now):12d}.{int(now % 1 * 1e6):06d}   File update time stamp   ! make_golden"
    )
    tmp = path.with_suffix(".tmp")
    tmp.write_text("\n".join(lines) + "\n")
    os.replace(tmp, path)


def write_track_point(path, azpoi, elpoi):
    now = time.time()
    path.write_text(
        time.strftime("%d-%b-%Y  %H:%M:%S", time.localtime(now))
        + f"{int(now):12d}.{int(now % 1 * 1e6):06d}   File update time stamp   ! make_golden\n"
        f"{azpoi:.4f}   track_azpoi   ! Az Pointing Offset [arcsec]\n"
        f"{elpoi:.4f}   track_elpoi   ! El Pointing Offset [arcsec]\n"
    )


def run(astra, io, scenario):
    rw = io / "ReadWrite"
    utcs = scenario["utcs"]
    (io / "share" / "astra" / "astra_debug_time.inp").write_text(
        "\n".join([utcs[0]] * PAD + utcs + [utcs[-1]] * 3) + "\n"
    )
    write_track_point(rw / "KOSMA_track_point.set", *scenario["track_point"])
    log = io / "logs" / "astra.log"
    log.unlink(missing_ok=True)
    command = dict(scenario["command"], obs_source_name="GOLD_" + scenario["name"][:20])
    env = dict(os.environ, READ_DIR=str(rw), WRITE_DIR=str(rw))
    proc = subprocess.Popen(
        [astra, "-t", "-f", "4", "-n", "-r"],
        cwd=io,
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    time.sleep(2.5)
    start = None
    if scenario["start_offset"] is not None:
        start = math.floor(time.time()) + scenario["start_offset"]
        command["obs_start_time"] = f"{start:.5f}"
    write_obs2tel(rw / "KOSMA_obs2tel.set", command, cookie=int(time.time()) % 10**7)
    proc.wait(timeout=PAD + len(utcs) + 30)
    return parse(log, command["obs_source_name"], scenario, start), read_parang(rw)


def parse(log, source, scenario, start):
    command = scenario["command"]
    duration = float(command.get("obs_track_duration", 0))
    samples, ktime, point = [], None, None
    for line in log.read_text(errors="replace").splitlines():
        if "ASTRA_TIMES: Using fixed time" in line:
            ktime = math.floor(float(line.split()[1]))
            point = None
        elif "ASTRA_POINT:" in line and "Az,El" in line:
            words = line.split("ASTRA_POINT:")[1].split()
            point = (words[0], float(words[2]), float(words[3]))
        elif "ASTRA_STEER (UTC+2" in line and point and point[0] == source:
            utc = float(line.split("-->")[1].split(",")[0])
            tau = 0.0
            if command.get("obs_otf_mode") == "C":
                tau = float(np.clip(ktime - start - 2, -10, duration) + 2)
            sample = {"utc": utc, "tau": tau, "az": point[1], "el": point[2]}
            if sample not in samples:
                samples.append(sample)
    return samples


def read_parang(rw):
    for line in (rw / "KOSMA_tel2obs.set").read_text().splitlines():
        if " tel_angle_focal_plane " in line:
            return float(line.split()[0])
    raise ValueError("no tel_angle_focal_plane")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--astra", required=True)
    ap.add_argument("--io", required=True, type=Path)
    ap.add_argument("--sha", required=True, help="kosma_software commit astra was built from")
    ap.add_argument("--only", help="run one scenario")
    args = ap.parse_args()
    fixtures = []
    for scenario in scenarios():
        if args.only and scenario["name"] != args.only:
            continue
        samples, parang = run(args.astra, args.io, scenario)
        if not samples:
            raise SystemExit(f"{scenario['name']}: astra logged no target for the command")
        print(f"{scenario['name']}: {len(samples)} samples, tel_angle_focal_plane {parang}")
        fixtures.append(
            {
                "name": scenario["name"],
                "command": scenario["command"],
                "track_point": scenario["track_point"],
                "samples": samples,
                "tel_angle_focal_plane": {"utc": samples[-1]["utc"], "deg": parang},
            }
        )
    golden = {
        "provenance": {
            "astra": f"kosma_software feat/ocs-translator-support @ {args.sha}, built with OBS_CCAT",
            "command": "astra -t -f 4 -n -r",
            "made": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "iers": "astra/share/bulletinA of the same commit (tests/fixtures/bulletinA)",
            "site": "KOSMA_astra.in of ccat_chai: -22.9856389 deg, 67.7402778 deg W, 5611.8 m",
            "target": "ASTRA_POINT Az,El: after focal-plane and pointing offsets, before the pointing model",
            "utc": "ASTRA_STEER time: the fixed time of astra_debug_time.inp + 2 s",
            "tau": "OTF line time astra applied: map offset = del + v * tau",
        },
        "fixtures": fixtures,
    }
    out = REPO / "tests" / "fixtures" / "golden.json"
    out.write_text(json.dumps(golden, indent=1) + "\n")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
