"""Unit tests for ImportKOSMAReadWriteIntoDictionary and _write_atomic."""
import os
import stat
import time
import pytest
from pathlib import Path

import kosma_ocs_translator
from kosma_ocs_translator import ImportKOSMAReadWriteIntoDictionary, _write_atomic
from helpers import BASE_OBS2TEL, DATA_DIR, write_obs2tel


class TestImportKOSMAReadWriteIntoDictionary:

    def test_template_file_parses_without_error(self, tmp_path, monkeypatch):
        # Smoke test: the checked-in template file must be valid.
        import shutil
        shutil.copy(DATA_DIR / "KOSMA_obs2tel.set", tmp_path / "KOSMA_obs2tel.set")
        monkeypatch.setenv("WRITE_DIR", str(tmp_path))
        result = ImportKOSMAReadWriteIntoDictionary(files=["KOSMA_obs2tel.set"])
        assert "KOSMA_obs2tel.set" in result

    def test_reads_all_required_keys(self, tmp_path, monkeypatch):
        write_obs2tel(tmp_path, BASE_OBS2TEL)
        monkeypatch.setenv("WRITE_DIR", str(tmp_path))
        d = ImportKOSMAReadWriteIntoDictionary(files=["KOSMA_obs2tel.set"])["KOSMA_obs2tel.set"]
        for key in BASE_OBS2TEL:
            assert key in d, f"missing key: {key}"

    def test_float_value_parsed_as_float(self, tmp_path, monkeypatch):
        write_obs2tel(tmp_path, BASE_OBS2TEL)
        monkeypatch.setenv("WRITE_DIR", str(tmp_path))
        d = ImportKOSMAReadWriteIntoDictionary(files=["KOSMA_obs2tel.set"])["KOSMA_obs2tel.set"]
        assert isinstance(d["obs_lam_on"], float)

    def test_int_value_parsed_as_int(self, tmp_path, monkeypatch):
        write_obs2tel(tmp_path, BASE_OBS2TEL)
        monkeypatch.setenv("WRITE_DIR", str(tmp_path))
        d = ImportKOSMAReadWriteIntoDictionary(files=["KOSMA_obs2tel.set"])["KOSMA_obs2tel.set"]
        assert isinstance(d["obs_cookie"], int)

    def test_string_value_parsed_as_str(self, tmp_path, monkeypatch):
        write_obs2tel(tmp_path, BASE_OBS2TEL)
        monkeypatch.setenv("WRITE_DIR", str(tmp_path))
        d = ImportKOSMAReadWriteIntoDictionary(files=["KOSMA_obs2tel.set"])["KOSMA_obs2tel.set"]
        assert d["obs_coord_sys_on"] == "J2000"

    def test_variable_filter_keeps_only_requested(self, tmp_path, monkeypatch):
        write_obs2tel(tmp_path, BASE_OBS2TEL)
        monkeypatch.setenv("WRITE_DIR", str(tmp_path))
        d = ImportKOSMAReadWriteIntoDictionary(
            files=["KOSMA_obs2tel.set"], variable="obs_tolerance"
        )["KOSMA_obs2tel.set"]
        assert "obs_tolerance" in d
        data_keys = [k for k in d if k not in ("timestamp", "file_timestamp")]
        assert data_keys == ["obs_tolerance"]

    def test_missing_file_returns_empty(self, tmp_path, monkeypatch):
        monkeypatch.setenv("WRITE_DIR", str(tmp_path))
        result = ImportKOSMAReadWriteIntoDictionary(files=["nonexistent.set"])
        assert result == {}

    def test_empty_directory_returns_empty(self, tmp_path, monkeypatch):
        monkeypatch.setenv("WRITE_DIR", str(tmp_path))
        result = ImportKOSMAReadWriteIntoDictionary()
        assert result == {}

    def test_timestamp_extracted_from_header(self, tmp_path, monkeypatch):
        write_obs2tel(tmp_path, BASE_OBS2TEL)
        monkeypatch.setenv("WRITE_DIR", str(tmp_path))
        d = ImportKOSMAReadWriteIntoDictionary(files=["KOSMA_obs2tel.set"])["KOSMA_obs2tel.set"]
        assert "timestamp" in d

    def test_duplicate_variable_first_value_wins(self, tmp_path, monkeypatch):
        ts = time.time()
        (tmp_path / "KOSMA_obs2tel.set").write_text(
            f"19-Oct-2026  14:00:00   {ts:.3f}   File update time stamp   ! test\n"
            "1   obs_cookie   ! first occurrence\n"
            "2   obs_cookie   ! second occurrence, must be ignored\n"
        )
        monkeypatch.setenv("WRITE_DIR", str(tmp_path))
        d = ImportKOSMAReadWriteIntoDictionary(files=["KOSMA_obs2tel.set"])["KOSMA_obs2tel.set"]
        assert d["obs_cookie"] == 1

    @pytest.mark.parametrize("key,value", [
        ("obs_coord_sys_on", "GALACTIC"),
        ("obs_otf_mode", "C"),
        ("obs_extended_elevation_range", "Y"),
    ])
    def test_various_string_values(self, tmp_path, monkeypatch, key, value):
        data = dict(BASE_OBS2TEL)
        data[key] = value
        write_obs2tel(tmp_path, data)
        monkeypatch.setenv("WRITE_DIR", str(tmp_path))
        d = ImportKOSMAReadWriteIntoDictionary(files=["KOSMA_obs2tel.set"])["KOSMA_obs2tel.set"]
        assert d[key] == value


class TestWriteAtomic:

    def test_creates_new_file(self, tmp_path):
        path = tmp_path / "KOSMA_tel2obs.set"
        _write_atomic(str(path), "new\n")
        assert path.read_text() == "new\n"
        # no temporary file left behind
        assert os.listdir(tmp_path) == ["KOSMA_tel2obs.set"]

    def test_replaces_file_by_rename(self, tmp_path):
        path = tmp_path / "KOSMA_tel2obs.set"
        path.write_text("old\n")
        old_inode = path.stat().st_ino
        # a reader that opened the file before the update
        with open(path) as reader:
            _write_atomic(str(path), "new\n")
            # still sees the complete old content, not an emptied file
            assert reader.read() == "old\n"
        assert path.read_text() == "new\n"
        # replaced by a new file (rename), not truncated and rewritten in place
        assert path.stat().st_ino != old_inode
        assert os.listdir(tmp_path) == ["KOSMA_tel2obs.set"]

    @pytest.mark.parametrize("mode", [0o644, 0o664, 0o640])
    def test_keeps_mode_of_existing_file(self, tmp_path, mode):
        path = tmp_path / "KOSMA_tel2obs.set"
        path.write_text("old\n")
        os.chmod(path, mode)
        _write_atomic(str(path), "new\n")
        assert stat.S_IMODE(path.stat().st_mode) == mode

    def test_write_error_keeps_old_file(self, tmp_path):
        path = tmp_path / "KOSMA_tel2obs.set"
        path.write_text("old\n")
        with pytest.raises(TypeError):
            _write_atomic(str(path), None)   # f.write(None) fails
        assert path.read_text() == "old\n"
        assert os.listdir(tmp_path) == ["KOSMA_tel2obs.set"]

    def test_rename_error_keeps_old_file(self, tmp_path, monkeypatch):
        path = tmp_path / "KOSMA_tel2obs.set"
        path.write_text("old\n")

        def failing_replace(src, dst):
            raise OSError("rename failed")

        monkeypatch.setattr(kosma_ocs_translator.os, "replace", failing_replace)
        with pytest.raises(OSError, match="rename failed"):
            _write_atomic(str(path), "new\n")
        assert path.read_text() == "old\n"
        assert os.listdir(tmp_path) == ["KOSMA_tel2obs.set"]


def test_write_tel2obs_file_is_atomic(translator_factory, tmp_path):
    t = translator_factory()
    (tmp_path / "KOSMA_tel2obs.set").write_text("old\n")
    old_inode = (tmp_path / "KOSMA_tel2obs.set").stat().st_ino

    t.write_tel2obs_file()

    assert (tmp_path / "KOSMA_tel2obs.set").stat().st_ino != old_inode
    assert "tel_return_cookie" in (tmp_path / "KOSMA_tel2obs.set").read_text()
    assert "del_tel_tot_azm" in (tmp_path / "KOSMA_ccdpos.sim").read_text()
    # only the obs2tel input and the two output files, no temporary files
    assert sorted(os.listdir(tmp_path)) == [
        "KOSMA_ccdpos.sim", "KOSMA_obs2tel.set", "KOSMA_tel2obs.set",
    ]
