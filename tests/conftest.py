import sys
from pathlib import Path

# Make tests/helpers.py importable as "helpers" from any test file
sys.path.insert(0, str(Path(__file__).parent))

import pytest
from unittest.mock import MagicMock

from helpers import (
    CCAT_LOCATION,
    BASE_OBS2TEL,
    DEFAULT_STATUS,
    DEFAULT_POINTING,
    write_obs2tel,
    make_payload,
)


@pytest.fixture
def mock_ocs():
    """Minimal mock of the OCS object used by KOSMA_translator."""
    ocs = MagicMock()
    ocs.earth_location = CCAT_LOCATION
    ocs.get_telescope_position.return_value = None
    ocs.get_status.return_value = dict(DEFAULT_STATUS)
    ocs.get_cmd_pointing.return_value = dict(DEFAULT_POINTING)
    return ocs


@pytest.fixture
def translator_factory(mock_ocs, tmp_path, monkeypatch):
    """
    Returns a factory that builds a KOSMA_translator with given overrides.

    Both obs2tel_overrides and status_overrides are shallow-merged on top of
    the defaults in helpers.py, so a test only needs to specify what differs.

    The factory also pre-populates last_payload and last_obs2tel so that
    set_tel2obs_dict() does not return early.  The payload is a static
    pointing at the mock commanded az/el, with the start time chosen so
    that the mock status time falls 30 s into the window.

    Usage::

        def test_something(translator_factory):
            t = translator_factory(
                obs2tel_overrides={"obs_coord_sys_on": "HORIZON"},
                status_overrides={"Azimuth current position": 182.0},
            )
            result, ccd = t.set_tel2obs_dict()
    """
    from kosma_ocs_translator import KOSMA_translator, _get_time_from_acu_format

    def _make(obs2tel_overrides=None, status_overrides=None):
        obs2tel = dict(BASE_OBS2TEL)
        if obs2tel_overrides:
            obs2tel.update(obs2tel_overrides)
        write_obs2tel(tmp_path, obs2tel)
        monkeypatch.setenv("WRITE_DIR", str(tmp_path))

        if status_overrides:
            status = dict(mock_ocs.get_status.return_value)
            status.update(status_overrides)
            mock_ocs.get_status.return_value = status

        translator = KOSMA_translator(mock_ocs)

        # Build a consistent payload: start 30 s before the mock status time
        # so that t_rel = 30 s lies within the 600 s window.
        status = mock_ocs.get_status.return_value
        status_time = _get_time_from_acu_format(status["Time"], status["Year"])
        t_start = status_time.unix - 30
        az_cmd = status["Azimuth commanded position"]
        el_cmd = status["Elevation commanded position"]
        translator.last_payload = make_payload(az=az_cmd, el=el_cmd, t_start=t_start)
        translator.last_obs2tel = dict(translator.obs2tel)
        return translator

    return _make
