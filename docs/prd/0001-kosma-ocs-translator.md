---
title: KOSMA OCS translator
status: draft
epic: https://github.com/ccatobs/kosma-ocs-translator/issues/5
repos:
  - ccatobs/kosma-ocs-translator
  - ccatobs/observatory-control-system
  - ccatobs/telescope-control-system
---

# PRD 0001: KOSMA OCS translator

> Source of truth for this epic. Tracker: [epic issue](https://github.com/ccatobs/kosma-ocs-translator/issues/5).
> The epic links here; this file is not embedded in the issue body.
> The KOSMA changes go to `kosma_software/kosma_software` on GitLab (git.ph1.uni-koeln.de) as a reviewed merge request.
> ADR 0001 (topocentric targets) records the split of geometry between the translator and the OCS.

## Problem

- KOSMA controls the CHAI receiver through file io. One program fills the telescope seat: it reads `KOSMA_obs2tel.set` and writes `KOSMA_tel2obs.set`.
- At FYST, the only way to reach the telescope is the OCS (observatory control system). No program fills the telescope seat for the real telescope.
- The current translator in this repository has not tracked through the OCS once. Its call to the OCS client raises a `TypeError`, which the translator logs and ignores.
- The current translator echoes a cookie before it runs the command. It compares the ACU (antenna control unit) with the ACU's own commanded position. It ends every track after 600 s.
- The OCS Python client exits the process on network errors. It has no timeouts and ignores error replies.

## Solution

- A rewritten translator fills the telescope seat at FYST. To KOSMA it looks like astra.
- The translator tracks sidereal sources, solar-system bodies and continuous OTF (on-the-fly) lines. It applies map, OFF, focal-plane and observer pointing offsets.
- A test suite shows parity with astra. It has golden positions, handshake scenarios against an ACU stub, and seat-parity runs on a KOSMA host.
- The OCS client and the OCS API get the changes the translator depends on.

## Implementation decisions

### Translator shape

- The translator is a Python package with three modules: file io (`kio`), geometry, and the translator loop.
- Runtime dependencies are astropy, numpy and a tagged release of the OCS client.
- One loop owns all state. A single command worker sends the OCS requests, so a slow request does not delay tel2obs.
- The translator polls the header stamp of `KOSMA_obs2tel.set` every 0.1 s. It does not use inotify, because inotify misses events on NFS.
- kosma_bridge lists the translator in the server registry as the counterpart of astra_dummy. The translator writes the alive signal `kosma_ocs_translator_alive`.

### File io contract

- Each write contains all 22 `tel_*` lines of the CCAT template, in template order, with the printf formats of `OBSERVATORY.acfg`.
- Line 1 holds the header stamp with a strictly increasing `sec.usec`. KOSMA readers compare the stamp, not the mtime, to detect new data.
- The translator writes a temporary file in the same directory and renames it onto `KOSMA_tel2obs.set`.
- The translator writes tel2obs at least every `obs_tel_info_update_time` and on every state change.

### Geometry

- The translator computes the target: topocentric az/el without refraction. The OCS applies refraction and the pointing model to every path sample.
- The pipeline follows astra stage by stage:
  - source frame to mean J2000 (J2000, B1950, GALACTIC, HIPPARCOS; HORIZON skips ahead), with proper motion and parallax;
  - map offset in `obs_coord_sys_del` and OFF offset in `obs_coord_sys_off`, with their true-angle flags;
  - mean to apparent to topocentric with ERFA, including DUT1 and polar motion;
  - HORIZON-frame offsets, then focal-plane offsets, then observer pointing offsets from `KOSMA_track_point.set`, each as a true-angle az/el offset;
  - extended range, then one azimuth branch for the whole window inside −180…360°.
- Continuous OTF follows astra: map offset = del + v · clamp(t − S, −10 s, D). A pre-OTF command has no start yet, so the translator sets S to infinity. The clamp then parks the telescope on the standby point.
- Solar-system bodies: with `tel_supports_ephemeris` = Y, the translator resolves `@NAME` itself. A configuration flag switches to N, and kephem then sends J2000 coordinates.
- Site constants come from the OCS. The translator refuses to start when `KOSMA_astra.in` disagrees.
- IERS (International Earth Rotation and Reference Systems Service) data come from a local file that a cron job refreshes every day.
- Measurements on the ACU simulator set the point spacing, the settle time and the lead margin. The starting values are 1 s spacing and a 3 s settle time.

### OCS use

- For each command, the translator aborts the running OCS command and uploads one window.
- A window starts at least 9.8 s plus a margin in the future. It holds about 1500 points at 1 s spacing, which fits the 100 KB body limit of the OCS nginx.
- When a window runs out before the next command, the translator uploads a renewal. tel2obs reports `tel_on_track` = N during the gap.
- A track longer than one window has a gap, and its scan gets flagged as lost track.

### tel2obs state

- `tel_return_cookie`: the translator echoes the new cookie once the command passes validation, with `tel_on_track` = N. focalplane needs the cookie within 1 s.
- `tel_on_track` is Y when three conditions hold:
  - the TCS (telescope control system) runs our command, by command id;
  - the ACU tracking error is within `obs_tolerance`;
  - for OTF, t ≥ S.
- Without a command id, the translator infers on-track and says so in `tel_error_string`. This mode is for development only.
- `tel_lost_track` latches inside [S, S + D) and resets with the next command.
- After D, the translator keeps tracking: on the source in mode N, on the end point in OTF.
- `tel_azm_act` and `tel_elv_act` report target + (ACU actual − ACU commanded).
- At start, the translator does not act on the command already in `KOSMA_obs2tel.set`. It reports cookie 0 and waits for a new cookie.

### Error codes

- A non-zero `tel_error` means "abort this command". Warnings set only `tel_error_string` and `tel_error_level`.
- The table lives in the translator:

| Code | Meaning |
|---|---|
| 1 | chopper requested |
| 198 | position out of range |
| 199 | lost track during the scan window |
| 200 | OTF start unreachable |
| 201 | mode not supported |
| 202 | invalid command |
| 210 | OCS unreachable |
| 211 | OCS or TCS rejected the command |
| 212 | ACU fault |
| 213 | unknown solar-system body |

### OCS client and API (observatory-control-system)

- Client: one exception family, timeouts on every request, retries only for status reads and explicit "busy", local path checks, typed status.
- Client: no network I/O in the constructor, TLS that fails closed, a session lock, quiet logging, declared dependencies, tagged releases, mocked-transport tests.
- API: pass the TCS status code and command id through. Proxy TCS `/status`, `/acu/failure-reset` and `/clear-track`. Repair `/telescope-position`.
- Deployment: move to a TCS image with `/status` and command ids.

### TCS (telescope-control-system)

- A proposal PR adds an endpoint that appends points to the running program track. The TCS team may decline it, and the translator works without it.

### KOSMA (kosma_software)

- astra applies the OTF velocities for mode C, and its IERS table covers the current date.
- CCAT `pre_otf_time` is 15 s until the TCS accepts appends.
- The CCAT tel2obs template carries the OCS site constants.
- kosma_bridge: a registry entry for the translator; the TUI (terminal user interface) shows OTF for modes C and P and shows `tel_error_string`.

## Testing decisions

- **Contract tests:** kio parses real fixtures. The written tel2obs reads back under KOSMA rules. A test cross-checks names and formats against `OBSERVATORY.acfg`.
- **Geometry golden tests:** the target matches golden positions from `astra -t -f 4 -n -r` within 0.1″. Fixtures cover each frame, proper motion, map and OFF offsets, focal-plane offsets near 80° elevation, extended range, a north transit and a C-mode line.
- **Handshake tests:** the translator runs against a temporary `READ_DIR`/`WRITE_DIR` and the ACU stub, with a fake clock. Scenarios cover:
  - track, boresight update, OFF;
  - PRETP → OTFTP with 15 s and with 5 s of lead;
  - renewal, startup with a stale cookie;
  - chopper, invalid frame, OCS down.
- **Seat-parity tests:** one command runs the same KOSMA scenarios twice. One run uses astra; the other uses the translator with the ACU simulator. The command then compares the two tel2obs streams.
- GitHub Actions run the contract, geometry golden and handshake tests. The seat-parity tests run on a KOSMA host.
- Prior art: the astra log parsers and `compute_focal_plane` in the old `utils/`; the `tmp_path` file tests of kosma_bridge.

## Out of scope

- Pattern OTF (mode P).
- Chopped modes.
- Tracks longer than one window without a gap.
- On-sky commissioning at FYST.
- TCS changes beyond the append proposal.
- RoofMount support.
