# kosma-ocs-translator

The translator fills the KOSMA telescope seat at FYST. It reads `KOSMA_obs2tel.set`,
drives the telescope through the OCS (observatory control system), then the TCS
(telescope control system), then the ACU (antenna control unit), and writes
`KOSMA_tel2obs.set`. To KOSMA it looks like astra.

Design: PRD 0001 (KOSMA OCS translator) and ADR 0001 (topocentric targets) in
`docs/`; the terms are in `CONTEXT.md`.

## Install and run

```sh
pip install .            # Python >= 3.11; pulls the OCS client release over ssh
kosma_ocs_translator     # as kosma_bridge starts it (server registry: kosma_ocs_translator)
```

| Setting | Flag | Environment | Default |
|---|---|---|---|
| KOSMA file io | `--read-dir`, `--write-dir` | `READ_DIR`, `WRITE_DIR` | `/net/KOSMA_file_io/ReadWrite` |
| OCS | `--ocs-url` | `OCS_URL` | `https://127.0.0.1:5600` |
| OCS certificates | `--tls-dir` (`server.cert.pem`, `client.cert.pem`, `client.key.pem`) | `OCS_TLS_DIR` | none; `--insecure` for development |
| IERS Bulletin A | `--iers` | | `<file io>/share/astra/bulletinA`, the file astra reads |
| Ephemeris | `--no-ephemeris` reports `tel_supports_ephemeris` = N | | Y |
| Window | `--window-points` | | 1500 points at 1 s |
| Log | `--log-file`, `--log-level` | | `<file io>/logs/kosma_ocs_translator.log` |

- The translator refuses to start when the site of the OCS and `KOSMA_astra.in`
  differ by more than 1e-5 deg or 1 m, or when the IERS file does not cover today.
- A cron job refreshes the IERS file every day from
  https://maia.usno.navy.mil/ser7/ser7.dat; the translator reads it again when it
  changes.
- The alive signal `kosma_ocs_translator_alive` is written every second.

## What the translator does

- It polls the header stamp of `KOSMA_obs2tel.set` every 0.1 s. At start it does not
  act on the command already there: it reports cookie 0 and waits for a new cookie.
- For each command it computes the target (topocentric az/el without refraction) and
  sends one window: abort what the TCS runs, then a `Horizon` path of 1500 points
  1 s apart, starting 10.3 s ahead. A window that runs out before the next command,
  or that the TCS accepted but does not run, is sent again (renewal); tel2obs
  reports `tel_on_track` = N during the gap.
- `tel_return_cookie`: the new cookie, once the command passes validation (also
  with the error of a refused command), with `tel_on_track` = N.
- `tel_on_track` = Y when the TCS runs our path (command id), an ACU status read
  after the window start shows the tracking error within `obs_tolerance`
  (hypot((az - az_cmd) cos el, el - el_cmd), raw coordinates), and an OTF line has
  reached its start S.
- `tel_lost_track` latches inside the scan window [S, S + D) and resets with the
  next command. An OTF line has S = `obs_start_time`; a track in mode N starts its
  scan window when it is first on track.
- Continuous OTF: map offset = del + v * clip(t - S, -10 s, D). A pre-OTF command
  (mode C, duration 0) parks on the standby point, S = infinity.
- `tel_azm_cmd`, `tel_elv_cmd`: the target now. `tel_azm_act`, `tel_elv_act`:
  target + (ACU actual - ACU commanded). `tel_angle_focal_plane`: astra's
  parallactic angle (bearing of the map frame's beta axis in az/el).
- Development only, said in `tel_error_string` (level 7): without a TCS command id
  on-track is inferred; without an ACU commanded position (ACU simulator) the
  ACU actual position is compared with the target.

## Error codes

A non-zero `tel_error` means "abort this command". Warnings set only
`tel_error_string` and `tel_error_level`.

| Code | Meaning |
|---|---|
| 1 | chopper requested (`obs_chop_beam` or `obs_amp_chop` not 0) |
| 198 | position out of range: the window leaves the TCS limits (az -180..360 deg, el -90..180 deg, 3 and 1.5 deg/s) |
| 199 | lost track during the scan window |
| 200 | OTF start unreachable: S is closer than lead (10.3 s) plus settle time (3 s) |
| 201 | mode not supported (pattern OTF, mode P) |
| 202 | invalid command: unknown frame or mode, unreadable variable, `@` source with `--no-ephemeris` |
| 210 | OCS unreachable |
| 211 | OCS or TCS rejected the command, or the TCS did not run or stopped our path |
| 212 | ACU fault (azimuth or elevation summary fault) |
| 213 | unknown solar-system body |

## ACU simulator measurements

Measured on 2026-10-14 on the OCS stack of `run.bash` in
ccatobs/observatory-control-system @ 24c56f6 (TCS image 2026-10-07, ACU
simulator `antenna-control-unit:v0.6.0`), on one host.

| Quantity | Measured | Translator value |
|---|---|---|
| Lead the TCS accepts | 9.85 s accepted, 9.75 s refused (client clock at send) | |
| Translator clock to TCS start of a path | median 0.041 s, max 0.095 s (n = 22, 1500 points) | |
| `path` request, 1500 points (52.7 KB) | median 0.044 s, max 0.095 s | |
| ACU - host clock | -0.7 ms (same host); TCS calibration -0.13 ms | |
| Lead | | 10.3 s = 9.8 s + 0.5 s margin |
| Point spacing: tracking error, curved test track | 2 s: median 0.19", max 85" (spline overshoot); 1 s: median 0.16", max 0.75"; 0.5 s: 0.19"/0.51"; 0.2 s: 0.23"/0.52" | 1 s |
| Settle after the window start, jumps of 1', 1 deg, 10 deg | on target before the start (the simulator moves during the lead and has no drive limits) | 3 s (starting value; the simulator cannot measure it) |

The 0.2-0.5" floor is the timing jitter of the status reads. Limits of the
ACU simulator v0.6.0 that a test on it must allow for:

- it reports NaN commanded positions in ProgramTrack
  (ccatobs/antenna-control-unit#9, commanded positions in ProgramTrack), so the
  translator falls back to target vs actual (development only, flagged);
- `Clear Stack` keeps the last point of the previous path, and points earlier
  than that point are dropped: a new path is ignored until the previous window
  would have ended. Use short windows there (`--window-points 60`).

The TCS refuses an accepted path now and then with "ACU & TCS clock mismatch"
when two commands arrive within its 1 s status cycle (2 of 25 abort + path
pairs); the translator sends such a path again.

## Accuracy

- Against astra (`astra -t -f 4 -n -r`, 15 fixtures: every frame, proper motion
  and parallax, map and OFF offsets in J2000, B1950, GALACTIC and HORIZON,
  focal-plane offsets at 80 deg elevation, extended range, a north transit, an OTF
  line): at most 0.064". A HORIZON source agrees to 0.002". The rest is astra's:
  SLALIB with IAU 1976/1980 precession-nutation and no celestial pole offsets
  (Bulletin A: dpsi = -123 mas); the translator uses ERFA, IAU 2006/2000A.
- Solar-system bodies (`@NAME`, `tel_supports_ephemeris` = Y) use astropy's
  builtin ephemeris (ERFA epv00, plan94, moon98): Mars 1.3" from JPL Horizons
  airless apparent az/el at the test epoch; a `#` source with Horizons
  astrometric coordinates agrees to 0.36".
- ADR 0001 trade-off: the OCS refracts the focal-plane offset at the offset
  elevation (at most 0.1" above 30 deg for 3' offsets).

## Tests

```sh
pip install ".[test]"
pytest                     # contract (kio), geometry golden, handshake (ACU stub, fake clock)
```

`tests/fixtures/golden.json` comes from astra on a KOSMA host
(`tools/make_golden.py`, provenance in the file and in `tests/fixtures/PROVENANCE`).

The seat-parity tests run on a KOSMA host with the local KOSMA simulation and the OCS
stack of `run.bash`:

```sh
python tools/seat_parity.py --astra ASTRA --astra-io ASTRA_IOPATH \
    --tls-dir OCS_TLS_DIR --report seat_parity_report.md
```

The command stops `astra_dummy`, runs the same scenarios (KOSMA_track ON, OFF, a map
offset, an OTF line) with astra and then with the translator, restores the
measurement setup and `astra_dummy`, and writes the report. It compares each
reported target with the translator geometry at `tel_time_act`.
