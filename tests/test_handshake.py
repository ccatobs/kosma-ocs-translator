"""Handshake tests: the translator loop against the ACU stub, on a fake clock.

The tests write KOSMA_obs2tel.set as control_tel does and read every
KOSMA_tel2obs.set the translator writes, under KOSMA rules (kio).
"""

import itertools
import json
import shutil
import threading
import time
from pathlib import Path

import ocs
import pytest
from acu_stub import AcuStub, Clock, separation

from kosma_ocs_translator import geometry, kio
from kosma_ocs_translator.translator import Translator

FIXTURES = Path(__file__).parent / "fixtures"
STAR = {
    "obs_source_name": "STAR",
    "obs_coord_sys_on": "J2000",
    "obs_lam_on": 46.0,
    "obs_bet_on": -30.0,
}


class Seat:
    """The telescope seat in a temporary KOSMA file io, with the translator in it."""

    def __init__(self, tmp_path, now=None, **options):
        self.dir = tmp_path
        for name in ("KOSMA_astra.in", "KOSMA_track_point.set", "KOSMA_obs2tel.set"):
            shutil.copy(FIXTURES / "ccat_chai" / name, tmp_path / name)
        self.clock = Clock() if now is None else Clock(now)
        self.acu = AcuStub(self.clock)
        self.cookie = 1000
        self.history = []  # (time, tel2obs values) of every write
        self.translator = Translator(
            self.acu,
            read_dir=tmp_path,
            write_dir=tmp_path,
            clock=self.clock,
            **{"eop_path": FIXTURES / "bulletinA", **options},
        )
        self.translator.start()
        self._collect()

    def send(self, **changes):
        """Write a new command into KOSMA_obs2tel.set, as control_tel does; return its cookie."""
        self.cookie += 1
        changes = {**STAR, **changes, "obs_cookie": self.cookie}
        lines = (FIXTURES / "ccat_chai" / "KOSMA_obs2tel.set").read_text().splitlines()
        for i, line in enumerate(lines[1:], 1):
            name = line.split("!")[0].split()[-1]
            if name in changes:
                lines[i] = f"{changes[name]}   {name}   ! test"
        now = self.clock()
        lines[0] = (
            f"{kio.header_time(now)}{int(now):12d}.{int(now % 1 * 1e6):06d}   File update time stamp   ! test"
        )
        (self.dir / "KOSMA_obs2tel.set").write_text("\n".join(lines) + "\n")
        return self.cookie

    def run(self, seconds, dt=0.1):
        for _ in range(round(seconds / dt)):
            self.clock.now += dt
            self.translator.step()
            self.translator.worker.join()
            self._collect()

    def _collect(self):
        stamp, values = kio.read_set(self.dir / "KOSMA_tel2obs.set")
        if not self.history or stamp != self.history[-1][2]:
            self.history.append((self.clock(), values, stamp))

    @property
    def tel2obs(self):
        return self.history[-1][1]

    def first(self, condition):
        """Time of the first tel2obs write for which condition(values) holds."""
        return next((t for t, values, _ in self.history if condition(values)), None)


@pytest.fixture
def seat(tmp_path):
    return Seat(tmp_path)


def test_echoes_a_new_cookie_within_half_a_second(seat):
    sent = seat.clock()
    cookie = seat.send()

    seat.run(0.5)

    echoed = seat.first(lambda v: v["tel_return_cookie"] == str(cookie))
    assert echoed is not None and echoed - sent <= 0.5
    assert seat.tel2obs["tel_on_track"] == "N"
    assert seat.tel2obs["tel_error"] == "0"


def test_at_start_ignores_the_command_already_in_obs2tel_and_reports_cookie_0(seat):
    stale = (FIXTURES / "ccat_chai" / "KOSMA_obs2tel.set").read_text()
    seat.run(2.0)
    # the same command again, under a new header stamp
    (seat.dir / "KOSMA_obs2tel.set").write_text(
        stale.replace("1723217707.338880", "1792040401.000000")
    )
    seat.run(2.0)

    assert {values["tel_return_cookie"] for _, values, _ in seat.history} == {"0"}
    assert seat.acu.paths == []


def test_reports_on_track_while_the_tcs_runs_the_window_within_tolerance(seat):
    sent = seat.clock()
    cookie = seat.send()

    seat.run(20.0)

    [path] = seat.acu.paths
    assert path["start"] - sent >= 9.8
    on = seat.first(lambda v: v["tel_on_track"] == "Y")
    assert on is not None and on >= path["start"] + seat.acu.settle
    assert seat.tel2obs["tel_on_track"] == "Y"
    assert seat.tel2obs["tel_return_cookie"] == str(cookie)


def test_reports_target_actual_site_and_ut1_date(seat):
    seat.send()
    seat.acu.track_error = 3.6  # arcsec, ACU actual above ACU commanded
    seat.run(20.0)

    v = seat.tel2obs
    now = float(v["tel_time_act"])  # every position refers to the time of the ACU status
    assert seat.history[-1][0] - 1.0 < now <= seat.history[-1][0]
    track = seat.translator.track
    az, el = geometry.target(track, now, 0.0, seat.translator.site, seat.translator.eop)
    assert separation(float(v["tel_azm_cmd"]), float(v["tel_elv_cmd"]), az[0], el[0]) < 0.05
    assert float(v["tel_elv_act"]) - float(v["tel_elv_cmd"]) == pytest.approx(0.001, abs=1e-5)
    assert float(v["tel_azm_act"]) == pytest.approx(float(v["tel_azm_cmd"]), abs=1e-5)
    assert (v["tel_latitude"], v["tel_longitude"], v["tel_altitude"]) == (
        "-22.98563900",
        "67.74027800",
        "5611.8",
    )
    assert (v["tel_telescope"], v["tel_plate_scale"]) == ("CCAT", "13.89")
    angle = geometry.focal_plane_angle(track, now, 0.0, seat.translator.site, seat.translator.eop)
    assert float(v["tel_angle_focal_plane"]) == pytest.approx(angle, abs=1e-5)
    dut1 = seat.translator.eop.at(float(v["tel_time_act"]) / 86400 + 40587)[2]
    assert float(v["tel_mjd1"]) == pytest.approx(
        (float(v["tel_time_act"]) + dut1) / 86400 + 40587, abs=1e-9
    )


def test_lost_track_inside_the_scan_window_latches_until_the_next_command(seat):
    seat.send(obs_track_duration=30)
    seat.run(15.0)
    assert seat.tel2obs["tel_on_track"] == "Y"

    seat.acu.track_error = 20.0  # beyond obs_tolerance (10 arcsec)
    seat.run(2.0)
    seat.acu.track_error = 0.0
    seat.run(3.0)

    assert seat.tel2obs["tel_on_track"] == "Y"
    assert seat.tel2obs["tel_lost_track"] == "Y"
    assert seat.tel2obs["tel_error"] == "199"

    seat.send(obs_track_duration=30)
    seat.run(0.5)
    assert (seat.tel2obs["tel_lost_track"], seat.tel2obs["tel_error"]) == ("N", "0")


def test_leaving_tolerance_after_the_scan_window_is_no_lost_track(seat):
    seat.send(obs_track_duration=5)
    seat.run(20.0)

    seat.acu.track_error = 20.0
    seat.run(2.0)

    assert seat.tel2obs["tel_on_track"] == "N"
    assert seat.tel2obs["tel_lost_track"] == "N"


OTF = {"obs_otf_mode": "C", "obs_lam_del": -300.0, "obs_lam_vel": 30.0, "obs_bet_vel": -5.0}


def pretp_then_otftp(seat, lead, duration=60.0):
    """control_tel's OTF sequence: PRETP, wait for on track, OTFTP lead seconds ahead."""
    seat.send(obs_track_duration=0, **OTF)
    seat.run(15.0)
    assert seat.tel2obs["tel_on_track"] == "Y"
    start = seat.clock() + lead
    cookie = seat.send(obs_track_duration=duration, obs_start_time=f"{start:.5f}", **OTF)
    return start, cookie


def test_pretp_parks_on_the_standby_point_and_otftp_with_15_s_of_lead_runs_the_line(seat):
    start, cookie = pretp_then_otftp(seat, lead=15.0)
    seat.run(80.0)

    standby, line = seat.acu.paths
    translator = seat.translator
    site, eop = translator.site, translator.eop
    t0 = standby["start"]
    az, el = geometry.target(translator.track, t0, -10.0, site, eop)
    assert separation(standby["points"][0][1], standby["points"][0][2], az[0], el[0]) < 0.01
    k = round(start - line["start"])  # the window point next to the OTF start S
    t = line["start"] + k
    az, el = geometry.target(translator.track, t, t - start, site, eop)
    assert separation(line["points"][k][1], line["points"][k][2], az[0], el[0]) < 0.01
    on = seat.first(lambda v: v["tel_return_cookie"] == str(cookie) and v["tel_on_track"] == "Y")
    assert on is not None and on >= start
    assert {v["tel_error"] for _, v, _ in seat.history} == {"0"}
    assert seat.tel2obs["tel_lost_track"] == "N"
    # tel_time_del: the line time, -10 s before S up to D after
    v = next(
        v for t, v, _ in seat.history if v["tel_return_cookie"] == str(cookie) and t > start + 5
    )
    assert float(v["tel_time_del"]) == pytest.approx(float(v["tel_time_act"]) - start, abs=1e-3)
    assert float(seat.tel2obs["tel_time_del"]) == 60.0


def test_otf_start_closer_than_lead_plus_settle_is_error_200(seat):
    _, cookie = pretp_then_otftp(seat, lead=5.0)
    seat.run(0.5)

    assert seat.tel2obs["tel_return_cookie"] == str(cookie)
    assert seat.tel2obs["tel_error"] == "200"
    assert len(seat.acu.paths) == 1


@pytest.mark.parametrize(
    "changes, code",
    [
        ({"obs_chop_beam": 1, "obs_amp_chop": 60.0}, "1"),
        ({"obs_otf_mode": "P", "obs_track_duration": 30}, "201"),
        ({"obs_coord_sys_on": "ECLIPTIC"}, "202"),
        ({"obs_lam_on": 33.51, "obs_bet_on": -23.11}, "198"),  # through the zenith, az > 3 deg/s
        ({"obs_source_name": "@VULCAN"}, "213"),
    ],
    ids=["chopper", "pattern-otf", "invalid-frame", "out-of-range", "unknown-body"],
)
def test_a_refused_command_echoes_its_cookie_with_the_error_and_sends_nothing(seat, changes, code):
    cookie = seat.send(**changes)
    seat.run(0.5)

    v = seat.tel2obs
    assert (v["tel_return_cookie"], v["tel_error"], v["tel_error_level"]) == (
        str(cookie),
        code,
        "8",
    )
    assert v["tel_error_string"] != "NONE"
    assert v["tel_pos_in_range"] == ("N" if code == "198" else "Y")
    assert seat.acu.paths == []


HORIZONS = json.loads((FIXTURES / "horizons_mars.json").read_text())["samples"]
NO_FOCAL_PLANE = {"obs_x_focal_plane": 0.0, "obs_y_focal_plane": 0.0}
MARS_EPOCH = 1792062400.0  # Mars 0.7 h before transit; windows start at the Horizons times


def test_with_ephemeris_support_tracks_at_mars_without_kephem(tmp_path):
    seat = Seat(tmp_path, now=MARS_EPOCH)
    # control_tel with tel_supports_ephemeris = Y: name @MARS, coordinates 0 0 J2000
    seat.send(obs_source_name="@MARS", obs_lam_on=0.0, obs_bet_on=0.0, **NO_FOCAL_PLANE)
    seat.run(1.0)

    assert seat.tel2obs["tel_supports_ephemeris"] == "Y"
    [path] = seat.acu.paths
    for sample in HORIZONS:
        point = path["points"][round(sample["utc"] - path["start"])]
        # astropy's builtin ephemeris (ERFA plan94): 1.3 arcsec from JPL Horizons here
        assert separation(point[1], point[2], sample["az"], sample["el"]) < 2.0


def test_without_ephemeris_support_tracks_hash_sources_from_kephem_coordinates(tmp_path):
    seat = Seat(tmp_path, now=MARS_EPOCH, supports_ephemeris=False)
    reference = HORIZONS[0]
    # kephem turned @MARS into #MARS with topocentric astrometric J2000 coordinates
    seat.send(
        obs_source_name="#MARS",
        obs_lam_on=reference["ra"],
        obs_bet_on=reference["dec"],
        **NO_FOCAL_PLANE,
    )
    seat.run(1.0)

    assert seat.tel2obs["tel_supports_ephemeris"] == "N"
    [path] = seat.acu.paths
    point = path["points"][0]
    assert path["start"] == pytest.approx(reference["utc"])
    # 0.36 arcsec between the ERFA pipeline and Horizons' airless apparent az/el
    assert separation(point[1], point[2], reference["az"], reference["el"]) < 0.5


def test_an_at_source_without_ephemeris_support_is_invalid(tmp_path):
    seat = Seat(tmp_path, supports_ephemeris=False)
    seat.send(obs_source_name="@MARS")
    seat.run(0.5)

    assert seat.tel2obs["tel_error"] == "202"
    assert seat.acu.paths == []


def test_ocs_down_is_error_210_for_the_command(seat):
    seat.acu.down = True
    cookie = seat.send()
    seat.run(2.0)

    v = seat.tel2obs
    assert (v["tel_return_cookie"], v["tel_error"], v["tel_on_track"]) == (str(cookie), "210", "N")
    assert "unreachable" in v["tel_error_string"]


def test_boresight_update_replaces_the_window_with_the_new_focal_plane_offsets(seat):
    seat.send(obs_x_focal_plane=0.0, obs_y_focal_plane=0.0)
    seat.run(14.0)
    assert seat.tel2obs["tel_on_track"] == "Y"
    cookie = seat.send(
        obs_x_focal_plane=0.0, obs_y_focal_plane=30.0
    )  # control_tel's second command
    seat.run(14.0)

    first, second = seat.acu.paths
    assert seat.acu.aborts == 1
    k = round(second["start"] - first["start"])
    assert second["points"][0][2] - first["points"][k][2] == pytest.approx(
        30 / 3600, abs=0.3 / 3600
    )
    assert seat.tel2obs["tel_return_cookie"] == str(cookie)
    assert seat.tel2obs["tel_on_track"] == "Y"


def test_the_result_of_an_old_upload_does_not_touch_the_next_command(seat):
    gate = threading.Event()
    path = seat.acu.path

    def first_path_fails_late(*args):
        if not gate.is_set():
            gate.wait(5)
            raise ocs.OCSRejected("rejected late", 400)
        return path(*args)

    seat.acu.path = first_path_fails_late
    seat.send()
    seat.clock.now += 0.1
    seat.translator.step()  # the worker now hangs in the first upload
    cookie = seat.send(obs_y_focal_plane=30.0)
    seat.clock.now += 0.1
    seat.translator.step()  # the second command arrives meanwhile
    gate.set()
    seat.translator.worker.join()
    seat.run(14.0)

    assert (seat.tel2obs["tel_return_cookie"], seat.tel2obs["tel_error"]) == (str(cookie), "0")
    assert seat.tel2obs["tel_on_track"] == "Y"


def test_off_command_moves_the_target_by_the_off_offset(seat):
    seat.send()
    seat.run(14.0)
    seat.send(obs_lam_off=-1800.0, obs_bet_off=0.0, obs_true_angle_off="Y")
    seat.run(14.0)

    on, off = seat.acu.paths
    k = round(off["start"] - on["start"])
    assert separation(
        off["points"][0][1], off["points"][0][2], on["points"][k][1], on["points"][k][2]
    ) == (pytest.approx(1800.0, abs=1.0))
    assert seat.tel2obs["tel_on_track"] == "Y"


def test_renewal_when_the_window_runs_out_with_on_track_n_during_the_gap(tmp_path):
    seat = Seat(tmp_path, window_points=60)
    seat.send()
    seat.run(14.0)
    assert seat.tel2obs["tel_on_track"] == "Y"

    seat.run(75.0)  # the window ends at 69.4 s, the renewal starts 10.3 s later

    first, renewal = seat.acu.paths
    end = first["start"] + first["points"][-1][0]
    assert renewal["start"] - end >= 9.8
    gap = [v["tel_on_track"] for t, v, _ in seat.history if end < t < renewal["start"]]
    assert gap and set(gap) == {"N"}
    assert seat.tel2obs["tel_on_track"] == "Y"


def test_a_path_the_tcs_accepts_but_does_not_run_is_sent_again(seat):
    seat.acu.drop_paths = 1
    seat.send()
    seat.run(16.0)

    assert len(seat.acu.paths) == 2
    assert (seat.tel2obs["tel_on_track"], seat.tel2obs["tel_error"]) == ("Y", "0")


def test_a_tcs_that_never_runs_the_path_is_error_211(seat):
    seat.acu.drop_paths = 99
    cookie = seat.send()
    seat.run(10.0)

    assert len(seat.acu.paths) == 3
    assert (seat.tel2obs["tel_return_cookie"], seat.tel2obs["tel_error"]) == (str(cookie), "211")


def test_without_command_ids_on_track_is_inferred_and_said_so(seat):
    seat.acu.command_ids = False
    seat.send()
    seat.run(14.0)

    v = seat.tel2obs
    assert (v["tel_on_track"], v["tel_error"], v["tel_error_level"]) == ("Y", "0", "7")
    assert "no command id" in v["tel_error_string"]


def test_without_commanded_positions_on_track_compares_actual_and_target_and_says_so(seat):
    seat.acu.commanded = False
    seat.send()
    seat.run(14.0)
    v = seat.tel2obs
    assert (v["tel_on_track"], v["tel_error"], v["tel_error_level"]) == ("Y", "0", "7")
    assert "no commanded position" in v["tel_error_string"]

    seat.acu.track_error = 20.0
    seat.run(1.0)
    assert seat.tel2obs["tel_on_track"] == "N"


def test_acu_fault_is_error_212(seat):
    cookie = seat.send()
    seat.run(14.0)
    seat.acu.fault = True
    seat.run(1.0)

    v = seat.tel2obs
    assert (v["tel_return_cookie"], v["tel_error"], v["tel_on_track"]) == (str(cookie), "212", "N")


def test_refuses_to_start_when_kosma_astra_in_disagrees_with_the_ocs_site(tmp_path):
    shutil.copy(FIXTURES / "ccat_chai" / "KOSMA_obs2tel.set", tmp_path)
    astra_in = (FIXTURES / "ccat_chai" / "KOSMA_astra.in").read_text()
    (tmp_path / "KOSMA_astra.in").write_text(astra_in.replace("67.7402778", "67.7002778"))
    clock = Clock()
    translator = Translator(AcuStub(clock), tmp_path, tmp_path, FIXTURES / "bulletinA", clock=clock)

    with pytest.raises(ValueError, match="KOSMA_astra.in"):
        translator.start()


def test_writes_tel2obs_at_least_every_obs_tel_info_update_time(seat):
    seat.send(obs_tel_info_update_time=0.5)
    seat.run(20.0)

    times = [t for t, _, _ in seat.history if t > seat.history[1][0]]
    gaps = [b - a for a, b in itertools.pairwise(times)]
    assert max(gaps) <= 0.5 + 1e-6


def test_a_slow_ocs_request_does_not_delay_tel2obs(tmp_path):
    shutil.copy(FIXTURES / "ccat_chai" / "KOSMA_obs2tel.set", tmp_path)
    shutil.copy(FIXTURES / "ccat_chai" / "KOSMA_astra.in", tmp_path)
    acu = AcuStub(time.time)
    path = acu.path

    def slow_path(*args):
        time.sleep(1.5)
        return path(*args)

    acu.path = slow_path
    translator = Translator(acu, tmp_path, tmp_path, FIXTURES / "bulletinA")
    translator.start()
    seat = Seat.__new__(Seat)  # only its command writer, on the real clock
    seat.dir, seat.clock, seat.cookie = tmp_path, time.time, 1000
    sent = time.time()
    cookie = seat.send()
    echoed = None
    while time.time() - sent < 1.0:
        translator.step()
        if kio.read_set(tmp_path / "KOSMA_tel2obs.set")[1]["tel_return_cookie"] == str(cookie):
            echoed = echoed or time.time() - sent
        time.sleep(0.05)

    assert echoed is not None and echoed < 0.5
    assert acu.paths == []  # the upload still hangs in the worker


def test_writes_the_alive_signal_every_second(seat):
    seat.run(3.0)

    _, values = kio.read_set(seat.dir / "kosma_ocs_translator_alive")
    assert int(seat.clock()) - 1 <= int(values["kosma_ocs_translator_alive"]) <= int(seat.clock())


def test_without_fresh_acu_status_the_translator_reports_off_track(seat):
    seat.send()
    seat.run(14.0)
    assert seat.tel2obs["tel_on_track"] == "Y"

    seat.acu.down = True
    seat.run(3.0)

    v = seat.tel2obs
    assert v["tel_on_track"] == "N"
    assert "no ACU status" in v["tel_error_string"]

    seat.acu.down = False
    seat.run(2.0)
    assert (seat.tel2obs["tel_on_track"], seat.tel2obs["tel_error_string"]) == ("Y", "NONE")


def test_a_command_with_an_unreadable_variable_is_invalid(seat):
    cookie = seat.send(obs_tolerance="ten")
    seat.run(0.5)

    v = seat.tel2obs
    assert (v["tel_return_cookie"], v["tel_error"]) == (str(cookie), "202")
    assert "obs_tolerance" in v["tel_error_string"]


def test_reads_the_iers_file_again_when_the_daily_job_replaced_it(tmp_path):
    bulletin = tmp_path / "bulletinA"
    shutil.copy(FIXTURES / "bulletinA", bulletin)
    seat = Seat(tmp_path, eop_path=bulletin)
    first = seat.translator.eop
    text = bulletin.read_text().replace(
        "2026 10 15  61328       0.1614", "2026 10 15  61328       0.1714"
    )
    bulletin.write_text(text)
    seat.send()
    seat.run(0.5)

    assert seat.translator.eop is not first
    assert seat.translator.eop.x[seat.translator.eop.mjd == 61328] == pytest.approx(0.1714)


def test_refuses_to_start_when_the_iers_file_does_not_cover_today(tmp_path):
    with pytest.raises(ValueError, match="IERS"):
        Seat(tmp_path, now=1830000000.0)  # 2027-12-28, beyond the bulletin


def test_a_path_stopped_by_someone_else_is_error_211_and_not_sent_again(seat):
    seat.send()
    seat.run(14.0)
    seat.acu.abort()  # e.g. an operator in the OCS web interface
    seat.run(2.0)

    assert len(seat.acu.paths) == 1
    assert (seat.tel2obs["tel_error"], seat.tel2obs["tel_on_track"]) == ("211", "N")
