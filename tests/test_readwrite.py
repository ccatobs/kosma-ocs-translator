"""Unit tests for ImportKOSMAReadWriteIntoDictionary."""
import time
import pytest
from pathlib import Path

from kosma_ocs_translator import ImportKOSMAReadWriteIntoDictionary
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
