"""Unit tests for update_spem_acu.py."""
import logging
import re
import subprocess
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

# update_spem_acu.py is a script at the repository root, not an installed module
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import update_spem_acu as usa


# names in string_ccatacu_point[i], in the order of KOSMA_ccatacu_point.in
KOSMA_NAMES = [
    "IA", "CA", "NPAE", "AN", "AW", "AES", "AEC", "IE", "EEC", "EES",
    "AW2", "AN2", "EES2", "AES2", "EEC2", "AEC2", "LELA", "LELE", "LEVA", "LEVE",
    "TFNC", "TFNS", "TFEC", "TFES", "TFN", "TFE", "NRX", "NRY",
]
ACU_NAMES = [usa.kosma2acu_names.get(name, name) for name in KOSMA_NAMES]

MODEL_DATE = "2026-10-30:14:44:00Z"

# SPEM set used at the time of the pointing session (ACU names, mdeg)
CURRENT_SPEM = {name: 0.0 for name in ACU_NAMES}
CURRENT_SPEM.update({
    "NRX": 1.5, "NRY": -2.0,
    "TFC": 0.3, "TFS": 0.4, "TF": 0.5, "TFEC": 0.6, "TFES": 0.7, "TFE": 0.8,
})


def write_ccatacu_point(directory, spem, model_date=MODEL_DATE):
    """Write KOSMA_ccatacu_point.in to *directory* from an ACU-name -> value dict."""
    lines = [
        "30-Oct-2026  14:46:42  1793368002.442420   File update time stamp"
        "   ! Kset_point (fips:1091816)\n"
    ]
    for i, name in enumerate(ACU_NAMES):
        lines.append(
            f"{spem[name]}   ccatacu_point[{i}]"
            "   ! CCAT-ACU pointing correction coefficient [mdeg]\n"
        )
    for i, name in enumerate(KOSMA_NAMES):
        lines.append(
            f"{name}   string_ccatacu_point[{i}]"
            " ! description text for CCAT-ACU pointing correction coefficients\n"
        )
    lines.append(
        "CCAT-ACU VA Error Compensations-document Release 1.3   ccatacu_model_type"
        "   ! identifier string for CCAT-ACU pointing model type [CCAT-ACU_*]\n"
    )
    lines.append(
        f"{model_date}   ccatacu_model_date"
        "   ! creation date of CCAT_ACU pointing model values\n"
    )
    path = Path(directory) / usa.KOSMA_SPEM_FILE
    path.write_text("".join(lines))
    return path


def write_point_in(path, model_type, apoint_mdeg, skip=()):
    """Write KOSMA_point.in with a_point[i] given in mdeg (converted to arcsec)."""
    lines = [
        "! comment line\n",
        "\n",
        f"{model_type}   model_type   ! pointing model type\n",
    ]
    for i, value in enumerate(apoint_mdeg):
        if i in skip:
            continue
        lines.append(f"{value * 3.6:.6f}   a_point[{i}]   ! [arcsec]\n")
    Path(path).write_text("".join(lines))
    return path


class FakeKsetPoint:
    """
    Replacement for subprocess.run that emulates Kset_point by rewriting
    the value of the variable in KOSMA_ccatacu_point.in
    """

    def __init__(self, directory, fail_on=None):
        self.path = Path(directory) / usa.KOSMA_SPEM_FILE
        self.fail_on = fail_on
        self.commands = []

    def __call__(self, cmd, **kwargs):
        self.commands.append(" ".join(cmd))
        if self.fail_on is not None and self.fail_on in cmd[1]:
            return subprocess.CompletedProcess(cmd, 1, stdout="", stderr="failed")
        assert cmd[0] == "Kset_point"
        var, value = cmd[1], cmd[2]
        new_lines = []
        for line in self.path.read_text().splitlines(keepends=True):
            data, _, comment = line.partition("!")
            fields = data.split()
            if len(fields) >= 2 and fields[-1] == var:
                line = f"{value}   {var}   !{comment}"
            new_lines.append(line)
        self.path.write_text("".join(new_lines))
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")


def answers(*replies):
    """Return a fake input() that gives *replies* in order."""
    it = iter(replies)
    return lambda prompt="": next(it)


def no_input(prompt=""):
    raise AssertionError("input() must not be called")


@pytest.fixture(autouse=True)
def reset_logger():
    yield
    log = logging.getLogger(usa.LOGGER_NAME)
    for handler in list(log.handlers):
        handler.close()
        log.removeHandler(handler)


@pytest.fixture
def kosma_dir(tmp_path, monkeypatch):
    """KOSMA ReadWrite directory containing KOSMA_ccatacu_point.in = CURRENT_SPEM."""
    write_ccatacu_point(tmp_path, CURRENT_SPEM)
    monkeypatch.setenv("WRITE_DIR", str(tmp_path))
    return tmp_path


@pytest.fixture
def acu_spem():
    """SPEM set as returned by the ACU (after stripping "Parameter ")."""
    spem = {"Time": 123.456, "Year": 2026}
    spem.update(CURRENT_SPEM)
    spem["LEL velocity"] = 0.1
    return spem


class TestAskContinue:

    @pytest.mark.parametrize("reply,expected", [
        ("y", True), ("yes", True), ("Y", True), ("  yes ", True),
        ("n", False), ("no", False), ("N", False),
    ])
    def test_valid_answers(self, monkeypatch, reply, expected):
        monkeypatch.setattr("builtins.input", answers(reply))
        assert usa.ask_continue("Continue?") is expected

    def test_invalid_answer_asks_again(self, monkeypatch, capsys):
        monkeypatch.setattr("builtins.input", answers("maybe", "", "n"))
        assert usa.ask_continue("Continue?") is False
        out = capsys.readouterr().out
        assert out.count("Please answer y (continue) or n (abort).") == 2


class TestFormatValue:

    def test_float(self):
        assert usa.format_value(1.23456) == "       1.235"

    def test_int(self):
        assert usa.format_value(2) == "       2.000"

    def test_string_right_aligned(self):
        assert usa.format_value("N/A") == "         N/A"

    def test_none(self):
        assert usa.format_value(None) == "        None"


class TestSetupLogging:

    def test_file_only_messages_not_on_console(self, tmp_path, monkeypatch, capsys):
        monkeypatch.chdir(tmp_path)
        log = usa.setup_logging(logging.INFO)
        log.info("to both")
        log.info("to file only", extra=usa.FILE_ONLY)
        for handler in log.handlers:
            handler.flush()

        console = capsys.readouterr().err
        assert "to both" in console
        assert "to file only" not in console
        logfile = (tmp_path / "kosma-ocs-translator.log").read_text()
        assert "to both" in logfile
        assert "to file only" in logfile


class TestStripParameterPrefix:

    def test_strip(self):
        spem = {"Parameter IA": 1.0, "Time": 2.0, "LEL velocity": 3.0}
        assert usa.strip_parameter_prefix(spem) == {
            "IA": 1.0, "Time": 2.0, "LEL velocity": 3.0,
        }


class TestGetKosmaSpem:

    def test_reads_all_coefficients_with_acu_names(self, kosma_dir):
        spem = usa.get_kosma_spem()
        for name in ACU_NAMES:
            assert spem[name] == pytest.approx(CURRENT_SPEM[name]), name

    def test_kosma_names_are_mapped(self, kosma_dir):
        spem = usa.get_kosma_spem()
        for kosma_name, acu_name in usa.kosma2acu_names.items():
            assert kosma_name not in spem
            assert acu_name in spem

    def test_values_are_float(self, kosma_dir):
        spem = usa.get_kosma_spem()
        assert all(isinstance(spem[name], float) for name in ACU_NAMES)

    def test_model_date(self, kosma_dir):
        assert usa.get_kosma_spem()["ccatacu_model_date"] == MODEL_DATE

    def test_no_extra_keys(self, kosma_dir):
        # ccatacu_model_type is not read
        assert set(usa.get_kosma_spem()) == set(ACU_NAMES) | {"ccatacu_model_date"}

    def test_template_with_zeros(self, tmp_path, monkeypatch):
        # values written as "0" are parsed as int by ImportKOSMAReadWriteIntoDictionary
        write_ccatacu_point(tmp_path, {name: 0 for name in ACU_NAMES})
        monkeypatch.setenv("WRITE_DIR", str(tmp_path))
        spem = usa.get_kosma_spem()
        assert all(spem[name] == 0.0 for name in ACU_NAMES)

    def test_missing_file_raises(self, tmp_path, monkeypatch):
        monkeypatch.setenv("WRITE_DIR", str(tmp_path))
        with pytest.raises(FileNotFoundError):
            usa.get_kosma_spem()

    def test_missing_value_raises(self, kosma_dir):
        path = kosma_dir / usa.KOSMA_SPEM_FILE
        lines = path.read_text().splitlines(keepends=True)
        # leading space so that string_ccatacu_point[3] is kept
        path.write_text("".join(l for l in lines if " ccatacu_point[3] " not in l))
        with pytest.raises(ValueError, match=r"ccatacu_point\[3\]"):
            usa.get_kosma_spem()


class TestPutKosmaSpem:

    def new_spem(self):
        return {name: 10.0 + i for i, name in enumerate(ACU_NAMES)}

    def test_commands_use_correct_index(self, kosma_dir, monkeypatch):
        fake = FakeKsetPoint(kosma_dir)
        monkeypatch.setattr(usa.subprocess, "run", fake)
        usa.put_kosma_spem(self.new_spem())
        for i, name in enumerate(ACU_NAMES):
            assert f"Kset_point ccatacu_point[{i}] {10.0 + i:.6f}" in fake.commands

    def test_model_date_is_set(self, kosma_dir, monkeypatch):
        fake = FakeKsetPoint(kosma_dir)
        monkeypatch.setattr(usa.subprocess, "run", fake)
        usa.put_kosma_spem(self.new_spem())
        date_cmds = [c for c in fake.commands if "ccatacu_model_date" in c]
        assert len(date_cmds) == 1
        assert re.fullmatch(
            r"Kset_point ccatacu_model_date \d{4}-\d{2}-\d{2}:\d{2}:\d{2}:\d{2}Z",
            date_cmds[0],
        )

    def test_round_trip(self, kosma_dir, monkeypatch):
        monkeypatch.setattr(usa.subprocess, "run", FakeKsetPoint(kosma_dir))
        new_spem = self.new_spem()
        updated = usa.put_kosma_spem(new_spem)
        for name in ACU_NAMES:
            assert updated[name] == pytest.approx(new_spem[name]), name
        assert updated["ccatacu_model_date"] != MODEL_DATE
        assert updated == usa.get_kosma_spem()

    def test_ignored_keys_are_skipped(self, kosma_dir, monkeypatch):
        fake = FakeKsetPoint(kosma_dir)
        monkeypatch.setattr(usa.subprocess, "run", fake)
        spem = self.new_spem()
        spem.update({"Time": 1.0, "Year": 2026, "LEL velocity": 0.1,
                     "ccatacu_model_date": "old"})
        usa.put_kosma_spem(spem)
        # one command per coefficient plus one for the date
        assert len(fake.commands) == len(ACU_NAMES) + 1
        assert not any("old" in c for c in fake.commands)

    def test_unknown_key_raises_before_any_command(self, kosma_dir, monkeypatch):
        fake = FakeKsetPoint(kosma_dir)
        monkeypatch.setattr(usa.subprocess, "run", fake)
        spem = self.new_spem()
        spem["UNKNOWN"] = 1.0
        with pytest.raises(KeyError, match="UNKNOWN"):
            usa.put_kosma_spem(spem)
        assert fake.commands == []

    def test_failed_command_raises(self, kosma_dir, monkeypatch):
        fake = FakeKsetPoint(kosma_dir, fail_on="ccatacu_point[5]")
        monkeypatch.setattr(usa.subprocess, "run", fake)
        with pytest.raises(RuntimeError, match=r"ccatacu_point\[5\]"):
            usa.put_kosma_spem(self.new_spem())
        # the other commands are still executed
        assert len(fake.commands) == len(ACU_NAMES) + 1


class TestCheckSpemSets:

    def kosma_spem(self):
        spem = dict(CURRENT_SPEM)
        spem["ccatacu_model_date"] = MODEL_DATE
        return spem

    def test_identical_sets_do_not_ask(self, acu_spem, monkeypatch, capsys):
        monkeypatch.setattr("builtins.input", no_input)
        usa.check_spem_sets(acu_spem, self.kosma_spem())
        out = capsys.readouterr().out
        assert "differs" not in out
        # ignored keys are not listed
        for key in usa.ignore_keys_acu + usa.ignore_keys_kosma:
            assert key not in out

    def test_difference_within_tolerance_matches(self, acu_spem, monkeypatch):
        monkeypatch.setattr("builtins.input", no_input)
        kosma = self.kosma_spem()
        kosma["IA"] += 0.5 * usa.spem_abs_tol
        usa.check_spem_sets(acu_spem, kosma)

    def test_mismatch_continue(self, acu_spem, monkeypatch, capsys):
        monkeypatch.setattr("builtins.input", answers("y"))
        kosma = self.kosma_spem()
        kosma["CA"] = 5.0
        usa.check_spem_sets(acu_spem, kosma)
        out = capsys.readouterr().out
        differ_lines = [l for l in out.splitlines() if "differs" in l]
        assert len(differ_lines) == 1
        assert differ_lines[0].startswith("CA")
        assert MODEL_DATE in out

    def test_mismatch_abort(self, acu_spem, monkeypatch):
        monkeypatch.setattr("builtins.input", answers("n"))
        kosma = self.kosma_spem()
        kosma["CA"] = 5.0
        with pytest.raises(SystemExit):
            usa.check_spem_sets(acu_spem, kosma)

    def test_key_only_in_acu_is_a_mismatch(self, acu_spem, monkeypatch, capsys):
        monkeypatch.setattr("builtins.input", answers("y"))
        kosma = self.kosma_spem()
        del kosma["NRY"]
        usa.check_spem_sets(acu_spem, kosma)
        out = capsys.readouterr().out
        assert any(l.startswith("NRY") and "differs" in l for l in out.splitlines())

    def test_key_only_in_kosma_is_a_mismatch(self, acu_spem, monkeypatch, capsys):
        monkeypatch.setattr("builtins.input", answers("y"))
        kosma = self.kosma_spem()
        kosma["EXTRA"] = 1.0
        usa.check_spem_sets(acu_spem, kosma)
        out = capsys.readouterr().out
        assert any(l.startswith("EXTRA") and "differs" in l for l in out.splitlines())


class TestKosma2Spem:

    # a_point[i] = i + 1 mdeg
    APOINT = [i + 1.0 for i in range(26)]

    COMMON = {
        "IA": 1 + 1.5, "CA": -2, "NPAE": -(3 - 2.0), "AN": -4, "AW": 5,
        "AES": 6, "AEC": -7, "AW2": 11, "AN2": -12, "EES2": 13, "AES2": 14,
        "EEC2": 15, "AEC2": 16, "LELA": 17, "LELE": 18, "LEVA": 19, "LEVE": 20,
        "NRX": 1.5, "NRY": -2.0,
    }
    CURRENT_TILT = {"TFC": 0.3, "TFS": 0.4, "TF": 0.5,
                    "TFEC": 0.6, "TFES": 0.7, "TFE": 0.8}

    EXPECTED = {
        "CCAT_NORMAL": {
            "IE": 8 - 0.3, "EEC": 9 + 2.0 - 0.5, "EES": 10 + 1.5 - 0.4,
            **CURRENT_TILT,
        },
        "CCAT_EXTENDED": {
            "IE": 8 - 0.6, "EEC": 9 + 2.0 + 0.8, "EES": 10 + 1.5 - 0.7,
            **CURRENT_TILT,
        },
        "CCAT_SYMMETRIC": {
            "IE": 8, "EEC": 9 + 2.0 - 0.5, "EES": 10 + 1.5,
            "TFC": 21, "TFS": 22, "TFEC": -21, "TFES": -22,
            "TF": 0.5, "TFE": -0.5,
        },
        "CCAT_ASYMMETRIC": {
            "IE": 8, "EEC": 9 + 2.0, "EES": 10 + 1.5,
            "TFC": 21, "TFS": 22, "TF": 23, "TFEC": 24, "TFES": 25, "TFE": 26,
        },
    }

    @pytest.mark.parametrize("model_type", list(EXPECTED))
    def test_model_types(self, tmp_path, model_type):
        point_in = write_point_in(tmp_path / "KOSMA_point.in", model_type, self.APOINT)
        spem = usa.kosma2spem(str(point_in), CURRENT_SPEM)
        expected = {**self.COMMON, **self.EXPECTED[model_type]}
        assert set(spem) == set(ACU_NAMES)
        for key, value in expected.items():
            assert spem[key] == pytest.approx(value), key

    def test_arcsec_to_mdeg(self, tmp_path):
        apoint = [0.0] * 26
        apoint[4] = 1.0  # AW = a4, written as 3.6 arcsec
        point_in = write_point_in(tmp_path / "KOSMA_point.in", "CCAT_ASYMMETRIC", apoint)
        assert usa.kosma2spem(str(point_in), CURRENT_SPEM)["AW"] == pytest.approx(1.0)

    def test_values_rounded_to_0p001_mdeg(self, tmp_path):
        apoint = [0.0] * 26
        apoint[4] = 100 / 3.6  # AW = a4, written as 100 arcsec = 27.777... mdeg
        current = {**CURRENT_SPEM, "NRX": 1.23456789}
        point_in = write_point_in(tmp_path / "KOSMA_point.in", "CCAT_ASYMMETRIC", apoint)
        spem = usa.kosma2spem(str(point_in), current)
        assert spem["AW"] == 27.778
        assert spem["NRX"] == 1.235
        for key, value in spem.items():
            assert value == round(value, usa.spem_decimals), key

    def test_missing_coefficient_raises(self, tmp_path):
        point_in = write_point_in(
            tmp_path / "KOSMA_point.in", "CCAT_NORMAL", self.APOINT, skip=(5,)
        )
        with pytest.raises(ValueError, match="a5"):
            usa.kosma2spem(str(point_in), CURRENT_SPEM)

    def test_missing_model_type_raises(self, tmp_path):
        path = tmp_path / "KOSMA_point.in"
        path.write_text("".join(
            f"{v * 3.6}   a_point[{i}]   ! [arcsec]\n" for i, v in enumerate(self.APOINT)
        ))
        with pytest.raises(ValueError, match="model_type"):
            usa.kosma2spem(str(path), CURRENT_SPEM)

    def test_unknown_model_type_raises(self, tmp_path):
        point_in = write_point_in(tmp_path / "KOSMA_point.in", "CCAT_UNKNOWN", self.APOINT)
        with pytest.raises(ValueError, match="CCAT_UNKNOWN"):
            usa.kosma2spem(str(point_in), CURRENT_SPEM)


class TestConfirmUpdate:

    def test_confirm(self, acu_spem, monkeypatch, capsys):
        monkeypatch.setattr("builtins.input", answers("y"))
        new_spem = dict(CURRENT_SPEM)
        new_spem["IA"] = 2.0
        usa.confirm_update(acu_spem, new_spem)
        out = capsys.readouterr().out
        ia_line = next(l for l in out.splitlines() if l.startswith("IA "))
        assert ia_line.split() == ["IA", "0.000", "2.000", "2.000"]
        for key in usa.ignore_keys_acu:
            assert key not in out

    def test_abort(self, acu_spem, monkeypatch):
        monkeypatch.setattr("builtins.input", answers("n"))
        with pytest.raises(SystemExit):
            usa.confirm_update(acu_spem, dict(CURRENT_SPEM))

    def test_key_missing_on_one_side(self, acu_spem, monkeypatch, capsys):
        monkeypatch.setattr("builtins.input", answers("y"))
        new_spem = dict(CURRENT_SPEM)
        new_spem["EXTRA"] = 1.0
        usa.confirm_update(acu_spem, new_spem)
        out = capsys.readouterr().out
        extra_line = next(l for l in out.splitlines() if l.startswith("EXTRA"))
        assert extra_line.split() == ["EXTRA", "None", "1.000", "N/A"]


class TestChangedSpem:

    def test_only_changed_keys(self):
        new = dict(CURRENT_SPEM)
        new["IA"] += 1.0
        new["CA"] += 0.5 * usa.spem_abs_tol   # within tolerance -> unchanged
        assert usa.changed_spem(new, CURRENT_SPEM) == {"IA": new["IA"]}

    def test_missing_in_current_is_changed(self):
        current = dict(CURRENT_SPEM)
        del current["NRY"]
        assert usa.changed_spem(dict(CURRENT_SPEM), current) == {"NRY": CURRENT_SPEM["NRY"]}

    def test_ignored_keys_are_skipped(self, acu_spem):
        new = {**CURRENT_SPEM, "Time": 1.0, "Year": 2000, "LEL velocity": 9.0,
               "ccatacu_model_date": "new"}
        assert usa.changed_spem(new, acu_spem) == {}


class TestMain:

    APOINT = [i + 1.0 for i in range(26)]

    @staticmethod
    def apoint_reproducing_current():
        """CCAT_ASYMMETRIC point model that reproduces CURRENT_SPEM."""
        apoint = [0.0] * 26
        apoint[0] = -CURRENT_SPEM["NRX"]
        apoint[2] = -CURRENT_SPEM["NRY"]
        apoint[8] = CURRENT_SPEM["NRY"]
        apoint[9] = -CURRENT_SPEM["NRX"]
        for i, name in zip(range(20, 26), ["TFC", "TFS", "TF", "TFEC", "TFES", "TFE"]):
            apoint[i] = CURRENT_SPEM[name]
        return apoint

    def run_main(self, tmp_path, monkeypatch, acu_spem, replies, apoint=APOINT,
                 status_code=200, acu_applies=True):
        """
        Run main() against a fake ACU holding acu_spem. set_spem updates the
        fake ACU only if acu_applies (False: accepted but silently not applied).
        """
        write_ccatacu_point(tmp_path, CURRENT_SPEM)
        point_in = write_point_in(
            tmp_path / "KOSMA_point.in", "CCAT_ASYMMETRIC", apoint,
        )
        monkeypatch.setenv("WRITE_DIR", str(tmp_path))
        # setup_logging writes kosma-ocs-translator.log into the cwd
        monkeypatch.chdir(tmp_path)
        monkeypatch.setattr(sys, "argv", ["update_spem_acu.py", "-i", str(point_in)])
        monkeypatch.setattr("builtins.input", answers(*replies))

        acu = dict(acu_spem)

        def get_dataset(name):
            assert name == "CmdSPEMParameter"
            return {(k if k in usa.ignore_keys_acu else f"Parameter {k}"): v
                    for k, v in acu.items()}

        def set_spem(**params):
            if acu_applies and status_code == 200:
                acu.update(params)
            return MagicMock(status_code=status_code, text="{}")

        ocs = MagicMock()
        ocs.get_dataset.side_effect = get_dataset
        ocs.set_spem.side_effect = set_spem
        monkeypatch.setattr(usa, "setup_ocs", lambda **kwargs: ocs)
        fake = FakeKsetPoint(tmp_path)
        monkeypatch.setattr(usa.subprocess, "run", fake)

        usa.main()
        return ocs, fake

    @staticmethod
    def point_cmds(fake):
        return [c for c in fake.commands if "ccatacu_point[" in c]

    def test_full_update(self, tmp_path, monkeypatch, acu_spem):
        ocs, fake = self.run_main(tmp_path, monkeypatch, acu_spem, replies=["y"])

        expected = usa.kosma2spem(str(tmp_path / "KOSMA_point.in"), CURRENT_SPEM)
        ocs.set_spem.assert_called_once()
        assert ocs.set_spem.call_args.args == ()
        sent = ocs.set_spem.call_args.kwargs
        # NRX, NRY are taken over from the current set and therefore not resent
        assert set(sent) == set(ACU_NAMES) - {"NRX", "NRY"}
        assert sent == pytest.approx({k: expected[k] for k in sent})
        assert ocs.get_dataset.call_count == 2

        updated = usa.get_kosma_spem()
        for name in ACU_NAMES:
            assert updated[name] == pytest.approx(expected[name], abs=1e-6), name
        assert updated["ccatacu_model_date"] != MODEL_DATE
        assert (tmp_path / "kosma-ocs-translator.log").exists()

    def test_only_changed_parameters_sent_to_acu(self, tmp_path, monkeypatch, acu_spem):
        apoint = self.apoint_reproducing_current()
        apoint[4] = 2.0   # only AW changes
        ocs, fake = self.run_main(tmp_path, monkeypatch, acu_spem, replies=["y"],
                                  apoint=apoint)

        assert ocs.set_spem.call_args.kwargs == pytest.approx({"AW": 2.0})
        # KOSMA gets the full read-back set from the ACU
        assert len(self.point_cmds(fake)) == len(ACU_NAMES)
        assert (f"Kset_point ccatacu_point[{ACU_NAMES.index('AW')}] 2.000000"
                in self.point_cmds(fake))

    def test_kosma_follows_acu_when_set_spem_not_applied(self, tmp_path, monkeypatch,
                                                         acu_spem, caplog):
        apoint = self.apoint_reproducing_current()
        apoint[4] = 2.0
        # the script logger does not propagate, attach caplog directly
        logging.getLogger(usa.LOGGER_NAME).addHandler(caplog.handler)
        self.run_main(tmp_path, monkeypatch, acu_spem, replies=["y"],
                      apoint=apoint, acu_applies=False)

        # KOSMA holds what the ACU really has (old AW), not the uploaded value
        updated = usa.get_kosma_spem()
        for name in ACU_NAMES:
            assert updated[name] == pytest.approx(CURRENT_SPEM[name], abs=1e-6), name
        assert "differ from the new ones" in caplog.text
        assert "AW" in caplog.text

    def test_nothing_changed_not_sent_but_kosma_dumped(self, tmp_path, monkeypatch,
                                                       acu_spem):
        # no replies: input() would raise StopIteration if called
        ocs, fake = self.run_main(tmp_path, monkeypatch, acu_spem, replies=[],
                                  apoint=self.apoint_reproducing_current())

        ocs.set_spem.assert_not_called()
        assert len(self.point_cmds(fake)) == len(ACU_NAMES)
        assert usa.get_kosma_spem()["ccatacu_model_date"] != MODEL_DATE

    def test_set_spem_failure_does_not_update_kosma(self, tmp_path, monkeypatch, acu_spem):
        with pytest.raises(RuntimeError, match="503"):
            self.run_main(tmp_path, monkeypatch, acu_spem, replies=["y"], status_code=503)
        assert usa.get_kosma_spem()["ccatacu_model_date"] == MODEL_DATE

    def test_abort_does_not_update(self, tmp_path, monkeypatch, acu_spem):
        with pytest.raises(SystemExit):
            self.run_main(tmp_path, monkeypatch, acu_spem, replies=["n"])
        assert usa.get_kosma_spem()["ccatacu_model_date"] == MODEL_DATE
