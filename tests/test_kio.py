"""Contract tests: kio reads KOSMA files and writes tel2obs as KOSMA readers expect."""

from pathlib import Path

from kosma_ocs_translator import kio

FIXTURES = Path(__file__).parent / "fixtures"


def test_reads_the_command_of_a_real_obs2tel_file():
    stamp, command = kio.read_obs2tel(FIXTURES / "ccat_chai" / "KOSMA_obs2tel.set")

    assert stamp == 1723217707.338880
    assert command["obs_cookie"] == 2177073
    assert command["obs_source_name"] == "M17_SAO"
    assert command["obs_lam_on"] == 275.115
    assert command["obs_bet_on"] == -16.20025
    assert command["obs_otf_mode"] == "N"
    assert command["obs_tolerance"] == 10.0
    assert command["obs_start_time"] == 0.0


TEL2OBS_VALUES = {
    "tel_telescope": "CCAT",
    "tel_on_track": "Y",
    "tel_lost_track": "N",
    "tel_pos_in_range": "Y",
    "tel_error": 0,
    "tel_error_string": "NONE",
    "tel_error_level": 5,
    "tel_return_cookie": 2177073,
    "tel_plate_scale": 13.89,
    "tel_latitude": -22.985639,
    "tel_longitude": 67.740278,
    "tel_altitude": 5611.8,
    "tel_angle_focal_plane": 40.158577,
    "tel_los_act": 0.0,
    "tel_time_act": 1723217849.15293,
    "tel_time_del": 141.06987,
    "tel_azm_cmd": 16.935555,
    "tel_elv_cmd": 40.158011,
    "tel_azm_act": 16.936224,
    "tel_elv_act": 40.158577,
    "tel_supports_ephemeris": "Y",
    "tel_mjd1": 61327.75,
}


def template_names():
    _, values = kio.read_set(FIXTURES / "ccat_chai" / "KOSMA_tel2obs.set")
    return list(values)


def test_writes_every_template_line_in_template_order_after_a_header_stamp(tmp_path):
    path = tmp_path / "KOSMA_tel2obs.set"

    stamp = kio.Tel2obsWriter(path).write(TEL2OBS_VALUES, now=1791990000.25)

    lines = path.read_text().splitlines()
    assert len(lines) == 23
    assert lines[0].startswith(kio.header_time(1791990000.25))
    assert "File update time stamp" in lines[0]
    assert [line.split("!")[0].split()[-1] for line in lines[1:]] == template_names()
    read_stamp, values = kio.read_set(path)
    assert read_stamp == stamp == 1791990000.25
    assert values["tel_return_cookie"] == "2177073"
    assert values["tel_on_track"] == "Y"
    assert values["tel_longitude"] == "67.74027800"
    assert float(values["tel_mjd1"]) == 61327.75


def acfg_variables():
    """{name: (type, format, comment)} of OBSERVATORY.acfg (first declaration wins)."""
    lines = (FIXTURES / "OBSERVATORY.acfg").read_text(errors="replace").splitlines()
    found = {}
    for i, line in enumerate(lines):
        words = line.split()
        if len(words) == 3 and words[0] in ("INT_VAR", "DOUBLE_VAR", "STRING_VAR"):
            comment = lines[i + 3].lstrip("!").strip()
            found.setdefault(words[1], (words[0], words[2], comment))
    return found


ACFG_TYPES = {"INT_VAR": int, "DOUBLE_VAR": float, "STRING_VAR": str}


def test_obs2tel_names_and_types_match_observatory_acfg():
    acfg = acfg_variables()
    for name, kind in kio.OBS2TEL.items():
        assert ACFG_TYPES[acfg[name][0]] is kind, name


def test_tel2obs_names_formats_and_comments_match_observatory_acfg():
    acfg = acfg_variables()
    for name, fmt, comment in kio.TEL2OBS:
        assert (fmt, comment) == acfg[name][1:], name
    assert len(kio.TEL2OBS) == 22


def test_header_stamp_increases_on_every_write_even_when_the_clock_does_not(tmp_path):
    writer = kio.Tel2obsWriter(tmp_path / "KOSMA_tel2obs.set")

    stamps = [writer.write(TEL2OBS_VALUES, now=t) for t in (100.0, 100.0, 99.5)]

    assert stamps[0] < stamps[1] < stamps[2]
    assert kio.read_set(tmp_path / "KOSMA_tel2obs.set")[0] == stamps[2]


def test_a_reader_with_the_old_file_open_still_reads_it_whole(tmp_path):
    path = tmp_path / "KOSMA_tel2obs.set"
    writer = kio.Tel2obsWriter(path)
    writer.write(TEL2OBS_VALUES, now=100.0)
    old = path.read_text()

    with open(path) as reader:
        writer.write({**TEL2OBS_VALUES, "tel_return_cookie": 42}, now=101.0)
        assert reader.read() == old

    assert [p.name for p in tmp_path.iterdir()] == ["KOSMA_tel2obs.set"]


def test_an_error_string_with_comment_tag_or_newline_stays_one_value(tmp_path):
    path = tmp_path / "KOSMA_tel2obs.set"
    text = "OCS rejected: bad path!\nsecond line"

    kio.Tel2obsWriter(path).write({**TEL2OBS_VALUES, "tel_error_string": text}, now=1.0)

    _, values = kio.read_set(path)
    assert values["tel_error_string"] == "OCS rejected: bad path. second line"
    assert values["tel_error_level"] == "5"


def test_reads_an_obs2tel_file_written_by_kosma_track():
    stamp, command = kio.read_obs2tel(FIXTURES / "live" / "KOSMA_obs2tel.set")

    assert stamp == kio.read_stamp(FIXTURES / "live" / "KOSMA_obs2tel.set") == 1791548191.05828
    assert command["obs_source_name"] == "IC1623"
    assert command["obs_cookie"] == 5481910


def test_alive_signal_is_written_like_astra_dummy_writes_its_own(tmp_path):
    kio.write_alive(kio.SetWriter(tmp_path / "kosma_ocs_translator_alive"), now=1791987409.03)

    lines = (tmp_path / "kosma_ocs_translator_alive").read_text().splitlines()
    assert len(lines) == 2
    assert (
        lines[1]
        == "1791987409   kosma_ocs_translator_alive   ! Watch-Dog variable for kosma_ocs_translator"
    )
