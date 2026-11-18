# 0002 — The translator sends `Horizon` paths; the ACU applies the pointing model

- **Status:** Accepted
- **Date:** 2026-11-18
- **Deciders:** Yoko Okada
- **Supersedes:** [ADR 0001](0001-topocentric-targets.md)
- **Relates to:** PRD 0001 (KOSMA OCS translator), epic ccatobs/kosma-ocs-translator#5 (KOSMA OCS translator)

## Holds

- **Default:** The translator computes the target as topocentric az/el without refraction and sends it as `Horizon` path points.
- **Default:** The translator adds the map, OFF and focal-plane offsets to the target. It does not apply the pointing offsets `track_azpoi` and `track_elpoi` in `KOSMA_track_point.set`, written by `setpoint -a` and `setpoint -e`, because there is no use case for them at CCAT for now.
- **Default:** The ACU (antenna control unit) applies the pointing model and reports the corrections it applied in the dataset `CmdPointingCorrection`.
- **Default:** The translator reports the positions of a perfect telescope to KOSMA, with refraction and without the pointing model:
  - `tel_azm_cmd` and `tel_elv_cmd` are the path position.
  - `tel_azm_act` and `tel_elv_act` are the ACU current position minus the pointing correction (SPEM + tiltmeters).
- **Default:** The tracking error is `tel_*_act` − `tel_*_cmd`. The translator decides on-track from this error, with azimuth scaled by cos(elevation), and from whether the path failed.
- **Workaround:** With `--refraction-correction` (default OFF), the translator adds refraction to the path points, because the TCS applies none today.
- **Incidental:** ERFA, through astropy, computes the target. The point spacing is not fixed. It starts at 0.1 s, to reduce the risk of wild interpolation between points, and is to be reviewed once the spline interpolation of the ACU is understood.

## Revisit if

- The ACU stops reporting the corrections it applies, or their meaning changes.
- `track_azpoi` and `track_elpoi` are used at CCAT. The translator ignores them today, so `setpoint -a` and `setpoint -e` have no effect on the telescope. astra adds them to its commanded position as an on-sky offset, after the extended-range conversion.
- The TCS starts applying refraction. The translator workaround must then be removed, and the refraction must be added to `tel_*_cmd`, not to the pointing correction.
- The TCS can track a sky position with a point away from the optical center, such as an off-center pixel. The TCS converts sky coordinates to `Horizon` today, but it does not add such offsets. To keep an off-center pixel on a fixed RA/Dec, the RA/Dec that the optical center tracks has to change over time, because the focal plane rotates relative to the sky during the track (parallactic angle and elevation).
- The TCS applies refraction and the allowed pointing error drops below 0.2″. If the TCS refracts positions that already include the focal-plane offset, the result differs from astra, which adds the offset after refraction. The difference is 0.1″ or less above 30° elevation for offsets up to 3′, and about 0.2″ at 20°.

## Context

- KOSMA expects the program in the telescope seat to turn source coordinates into telescope positions. astra does the full chain, refraction included.
- At FYST the OCS (observatory control system) sits between the translator and the ACU.
- The ACU current position is the encoder position. It includes the pointing corrections that the ACU applied.
- The ACU azimuth corrections are axis differences, as the pointing correction equations work on the encoders. KOSMA and astra define the azimuth correction on sky, so `del_tel_tot_azm` and `del_tel_rep_azm` in `KOSMA_ccdpos.sim` are the ACU values times cos(elevation).
- The TCS (telescope control system) inside the OCS passes every path sample through its pointing transform (`telescope.go:154` and `pointing.go:18-27` in ccatobs/telescope-control-system). As of 2026-11-18, `NewPointing()` returns `Pointing{}`, the az/el offsets are set to zero, and `NewRefraction` has no caller, so the transform adds no refraction and no pointing offsets.
- The TCS sky transform for ICRS paths takes no offsets and does not unwrap azimuth.
- The ACU interface control document is not available to the team.

## Decision

The translator owns the sky geometry up to topocentric az/el. The ACU owns the pointing model and reports what it applied, so that the translator can recover the nominal positions. Refraction belongs in the TCS. Until the TCS applies it, the translator can apply it as a workaround.

## Consequences / Trade-offs

- Easier: the translator can compare the path with the ACU position, so on-track does not need the TCS command id or the ACU commanded position.
- Easier: the pointing model lives next to the drive positions.
- Easier: the geometry golden tests compare with astra run without refraction (`astra -r`) while the workaround is OFF.
- Harder: the translator depends on the ACU reporting its corrections correctly.
- Harder: only one component may apply refraction. The translator logs a warning if the ACU reports `RF refraction correction on` while `--refraction-correction` is ON. The option must stay OFF once the TCS applies refraction.
- Harder: the workaround needs a weather feed. `KOSMA_translator.get_weather()` is the only place that reads it, and default values are used, with a warning, if the weather is bad or missing.
- Harder: with `--refraction-correction` OFF, nothing applies refraction today, so the telescope does not follow the refracted position.
- Harder: a 600 s path at 0.1 s spacing has 6001 points, which is expected to exceed the 100 KB body limit of the OCS nginx. The limit is an open issue, raised with the ACU simulator developer.

## Alternatives considered

- **The OCS applies the pointing model.** This was the decision of ADR 0001. It is replaced: the ACU applies the pointing model internally.
- **The translator reports `tel_azm_act` and `tel_elv_act` as target + (ACU actual − ACU commanded), as in ADR 0001.** Replaced by ACU actual − pointing correction. In theory the two forms match, but the ACU commanded position has two problems:
  - It does not work for program tracks in the ACU simulator. During path commands, the status shows the placeholder −999 instead of a position. It seems to work on the real telescope, but this could not be tested.
  - In the ACU status it is reported to have a 5 ms delay, and the OCS team recommended not using it. For the slow scanning of optical pointing and CHAI observing, this delay should not make a significant position difference.
- **The translator applies refraction permanently.** Not chosen. The translator would need its own weather feed for good, and the refraction would be corrected twice if another component applied it too.
- **The translator sends ICRS paths, and the TCS computes az/el.** Rejected. The TCS ICRS path takes no map, OFF or focal-plane offsets, and it does not unwrap azimuth.

## Status

Accepted on 2026-11-18. Supersedes ADR 0001.
