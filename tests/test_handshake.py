"""Handshake tests: the translator loop against the ACU stub, on a fake clock.

The tests write KOSMA_obs2tel.set as control_tel does and read every
KOSMA_tel2obs.set the translator writes, under KOSMA rules (kio).
"""

import shutil
from pathlib import Path

import pytest
from acu_stub import AcuStub, Clock

from kosma_ocs_translator import kio
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

    def __init__(self, tmp_path, **options):
        self.dir = tmp_path
        for name in ("KOSMA_astra.in", "KOSMA_track_point.set", "KOSMA_obs2tel.set"):
            shutil.copy(FIXTURES / "ccat_chai" / name, tmp_path / name)
        self.clock = Clock()
        self.acu = AcuStub(self.clock)
        self.cookie = 1000
        self.history = []  # (time, tel2obs values) of every write
        self.translator = Translator(
            self.acu,
            read_dir=tmp_path,
            write_dir=tmp_path,
            eop_path=FIXTURES / "bulletinA",
            clock=self.clock,
            **options,
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
