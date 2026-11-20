# 0003 — The cookie is returned on acceptance; `tel_on_track` waits for the path start

- **Status:** Accepted
- **Date:** 2026-11-20
- **Deciders:** Yoko Okada
- **Relates to:** PRD 0001 (KOSMA OCS translator), [ADR 0002](0002-acu-pointing-model.md) (on-track from path and ACU position)

## Holds

- **Default:** The translator returns the obs2tel cookie as `tel_return_cookie` as soon as the TCS accepts the command (`scan_pattern` for a path, `move_to` for a `HORIZON` source, or the slew `move_to` before a path whose start is far away). Returning the cookie resets `tel_error` to 0, `tel_lost_track` to `N` and the path-failed state.
- **Default:** `tel_on_track` is `N` while the ACU status time is before `on_track_not_before`:
  - the path `start_time` for a path (no margin);
  - the start time of the bookkeeping payload for a `move_to` to a `HORIZON` source;
  - infinity during a slew before a path, until the path sent after the slew is accepted.
- **Default:** A failure after the cookie is returned (e.g. the path rejected after an accepted slew, or a slew timeout) keeps the same cookie and sets `tel_error` and `tel_on_track = N`.
- **Default:** `tel_lost_track` becomes `Y` only after `tel_on_track` was `Y` at least once for the current cookie (`tel_was_on_track`). The time the telescope needs to settle onto a new path is not lost track. Once `Y`, it stays `Y` until the next cookie.
- **Default:** If `reuse_last_track` keeps a running path for a new cookie with the same target, the cookie is returned at once, `tel_lost_track` and `tel_was_on_track` start afresh, and a hold of a path that has not started yet stays in place.
- **Assumption:** The ACU and the translator host are NTP-synchronized, so the ACU status time can be compared with `start_time` from the translator clock without a margin.

## Revisit if

- The TCS lead requirement (`start_time` ≥ 9.8 s ahead) is reduced substantially. The early cookie is then less important, but the design still holds.
- A KOSMA client is found that waits for the cookie only and then assumes the telescope is on the new target, without checking `tel_on_track`.
- A KOSMA client decides on data quality from `tel_lost_track` alone. A telescope that never reaches the path keeps `tel_lost_track = N` and shows only `tel_on_track = N`.
- The ACU and the translator host are not synchronized (e.g. the ACU simulator on a machine without NTP). Then a margin on `on_track_not_before` is needed again.
- The ACU or the TCS reports which command it is executing. The translator could then decide when the path started from that instead of from time.
- obsdop needs the cookie later than ~1 s after `KOSMA_track` regularly. Then `obsdop_cookie_timeout` (2 s in the CCAT `master_parameters`) has to be raised.

## Context

- The TCS rejects a path that starts less than 9.8 s after it receives it. The translator sets `start_time` to 11 s after it starts computing the path (`TRACK_OVERHEAD`) and rebuilds the path if less than 10.3 s are left before sending (`TRACK_MIN_SEND_LEAD`).
- Until `start_time`, the ACU still executes the previous command. A position-based `tel_on_track` before then compares against the new target while the telescope is still on the old one, and can be a stale `Y` (e.g. if the old and new targets are close).
- Before this decision, the translator held the cookie until the ACU status time was 1 s after `start_time`, to avoid this stale `Y`. The cookie therefore came back about 11 s after `KOSMA_track`.
- `observing_loop.py` runs `setsource`, `KOSMA_track`, then `tune`. obsdop (`obsdop.c`) waits for `obs_cookie == tel_return_cookie` for at most `obsdop_cookie_timeout` (2 s in the CCAT `master_parameters`), then skips, and `tune` fails. With the cookie held for about 11 s, `tune` failed whenever the loop was fast.
- `control_tel_wait_or_cancel` (`control_tel.c`) waits for a cookie match **and** `tel_on_track = Y` on `tel_handshake` consecutive reads. While it waits, it keeps checking `tel_pos_in_range` (→ −2) and `tel_error` (→ −4). starpoint also waits for the cookie and `tel_on_track = Y`, and stops on `tel_error ≠ 0`.
- obsdop needs only the source and the time for the Doppler correction, not the telescope position.
- `tel_lost_track` latches for the whole cookie. Without `tel_was_on_track`, any `N` after the path start latched it, including the time to settle onto the path at the start.

## Decision

The cookie means "the command for this cookie is accepted". Whether the telescope is on the new target is reported only by `tel_on_track`, which is held at `N` until the ACU executes the new path. Losing track means leaving the path after having been on it.

## Consequences / Trade-offs

- Easier: `tune` (via obsdop) gets the cookie about 1 s after `KOSMA_track` (obs2tel polling, payload computation < 0.2 s, TCS response) and tunes during the 10 s lead instead of after it.
- Easier: a long slew before a path does not block clients that wait only for the cookie.
- Easier: no time margin to tune for the stale `Y` or for settling; settling takes as long as it needs and is not reported as lost track.
- Easier: errors after the cookie still reach `control_tel` and starpoint, because they keep checking `tel_error` while they wait for `tel_on_track = Y`.
- Harder: clients must not take a cookie match as "on target". All known clients that need the position (`control_tel`, starpoint) check `tel_on_track`.
- Harder: between the cookie return and the path start, `tel_azm_cmd` and `tel_elv_cmd` show the first point of the new path, or during a slew the previous path. `tel_on_track` is `N` meanwhile.
- Harder: a telescope that never reaches the path is visible only from `tel_on_track = N`, not from `tel_lost_track`.
- Harder: the hold depends on synchronized clocks between the ACU and the translator host.

## Alternatives considered

- **Hold the cookie until 1 s after `start_time` (previous design).** Replaced: obsdop waits only 2 s for the cookie, so `tune` failed when the loop was fast, and tuning could not overlap the lead time.
- **Keep the held cookie and raise `obsdop_cookie_timeout` to ~15 s.** Not chosen: no translator change, but `tune` would always wait the full lead, and a slew before a path (up to `SLEW_TIMEOUT` = 300 s) would exceed any reasonable timeout. Raising it moderately (~5 s) as a safety margin for the early cookie is deferred.
- **Reduce the TCS lead requirement of 9.8 s.** Not possible for now.
- **Keep a 1 s margin after `start_time` for `tel_on_track`.** Not chosen: it was introduced when `tel_on_track` used the ACU commanded position, which is no longer used (ADR 0002). With NTP-synchronized clocks, its remaining purpose was to avoid latching `tel_lost_track` while settling, which `tel_was_on_track` handles without a guessed time.

## Status

Accepted on 2026-11-20.
