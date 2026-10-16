# 0001 — The translator sends topocentric targets; the OCS applies refraction and the pointing model

- **Status:** Accepted
- **Date:** 2026-10-14
- **Deciders:** Christof Buchbender
- **Relates to:** PRD 0001 (KOSMA OCS translator), epic ccatobs/kosma-ocs-translator#5 (KOSMA OCS translator)

## Holds

- **Default:** The translator computes the target as topocentric az/el without refraction and sends it as `Horizon` path points.
- **Default:** The translator adds focal-plane offsets and observer pointing offsets to the target, before the OCS applies refraction.
- **Default:** The translator decides on-track from the ACU tracking error in raw coordinates and from the TCS command id.
- **Incidental:** ERFA, through astropy, computes the target. The point spacing is 1 s.

## Revisit if

- The OCS stops applying refraction or the pointing model to `Horizon` path points.
- The TCS offers its clients a conversion from sky coordinates to raw coordinates.
- The pointing budget drops below 0.2″. That value is the error from the order of offsets and refraction.

## Context

- KOSMA expects the program in the telescope seat to turn source coordinates into telescope positions. astra does the full chain, refraction included.
- At FYST the OCS (observatory control system) sits between the translator and the ACU (antenna control unit).
- The TCS (telescope control system) inside the OCS passes every path sample through its pointing transform. The transform adds refraction and pointing offsets (`telescope.go:142` and `pointing.go:18-27` in ccatobs/telescope-control-system).
- The TCS sky transform for ICRS paths takes no offsets, sends zero velocities and does not unwrap azimuth.
- The ACU interface control document is not available to the team.

## Decision

The translator owns the sky geometry up to topocentric az/el. The OCS owns refraction and the pointing model.

## Consequences / Trade-offs

- Easier: each correction has one owner. The pointing model and refraction live next to the drive positions.
- Easier: the geometry golden tests compare with astra run without refraction (`astra -r`).
- Harder: the translator cannot compare its target with the ACU position, because the ACU reports raw positions. On-track needs the TCS command id.
- Harder: the OCS refracts the focal-plane offset at the offset elevation, while astra adds the offset after refraction. The difference is 0.1″ or less above 30° elevation for offsets up to 3′, and about 0.2″ at 20°.
- Harder: the translator reports `tel_azm_act` and `tel_elv_act` as target + (ACU actual − ACU commanded).

## Alternatives considered

- **The translator applies refraction as well.** Rejected. The OCS would refract a second time, unless the OCS turned its refraction off for KOSMA paths. The translator would also need its own weather feed.
- **The translator sends ICRS paths, and the TCS computes az/el.** Rejected. The TCS ICRS path takes no map, OFF or focal-plane offsets. It drops velocities and does not unwrap azimuth.

## Status

Accepted on 2026-10-14.

## Amendments

- 2026-10-14: On TCS master (commit 8359639), `NewPointing()` returns `Pointing{}` (`pointing.go:14-16`). Its refraction constants and az/el offsets are zero, and `NewRefraction` has no caller. Issue ccatobs/telescope-control-system#23 (where refraction and the pointing model apply) asks the TCS team which component applies them. The decision holds until the TCS team answers.
