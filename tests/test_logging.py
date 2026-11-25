"""Unit tests for the log file time stamp (kosma_ocs_lib.UTCFormatter / setup_logging)."""
import logging
import os
import re
import time

from kosma_ocs_lib import UTCFormatter, setup_logging

UTC_STAMP_RE = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{3}Z$")


def test_utc_formatter_time_stamp():
    # 2026-11-25 10:00:00.123 UTC
    record = logging.LogRecord("test", logging.INFO, __file__, 1, "message", None, None)
    record.created = 1795600800.123
    record.msecs = 123.0
    stamp = UTCFormatter("%(asctime)s").format(record)
    assert stamp == "2026-11-25T10:00:00.123Z"
    assert UTC_STAMP_RE.match(stamp)


def test_utc_formatter_ignores_local_time_zone():
    old_tz = os.environ.get("TZ")
    os.environ["TZ"] = "Europe/Berlin"
    time.tzset()
    try:
        record = logging.LogRecord("test", logging.INFO, __file__, 1, "message", None, None)
        record.created = 1795600800.0
        record.msecs = 0.0
        assert UTCFormatter("%(asctime)s").format(record) == "2026-11-25T10:00:00.000Z"
    finally:
        if old_tz is None:
            del os.environ["TZ"]
        else:
            os.environ["TZ"] = old_tz
        time.tzset()


def test_setup_logging_file_uses_utc_stamp(tmp_path):
    log_file = tmp_path / "test.log"
    logger = setup_logging("test-utc-stamp", str(log_file), logging.INFO)
    try:
        logger.info("hello")
        for handler in logger.handlers:
            handler.flush()
        line = log_file.read_text().splitlines()[-1]
        stamp, rest = line.split(" - ", 1)
        assert UTC_STAMP_RE.match(stamp)
        assert rest == "test-utc-stamp - INFO - hello"
    finally:
        for handler in list(logger.handlers):
            handler.close()
            logger.removeHandler(handler)
