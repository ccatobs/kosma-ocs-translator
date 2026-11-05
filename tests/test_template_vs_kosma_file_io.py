"""
Compare the variables in the translator's output templates (load_*_template)
with the files that currently exist in KOSMA_file_io.

The translator overwrites these files completely, so a variable added on the
KOSMA side (ACFG) but missing in the template is wiped out at the next write.
Run this check against a file written by KOSMA itself, e.g. before starting
the translator, or point WRITE_DIR to a reference copy:

    pytest tests/test_template_vs_kosma_file_io.py
    WRITE_DIR=/path/to/reference pytest tests/test_template_vs_kosma_file_io.py

The test is skipped if the file does not exist (e.g. on a development machine
without KOSMA_file_io).
"""
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from kosma_ocs_translator import KOSMA_translator

KOSMA_READWRITE_DIR = os.environ.get("WRITE_DIR", "/net/KOSMA_file_io/ReadWrite/")

# (method that sets the template, attribute holding it, KOSMA_file_io file)
TEMPLATES = [
    ("load_tel2obs_template", "tel2obs", "KOSMA_tel2obs.set"),
    ("load_ccdpossim_template", "ccdpossim", "KOSMA_ccdpos.sim"),
]


def variable_names(text):
    """
    Variable names in KOSMA_file_io format, "<value>   <name>   ! <comment>",
    in the order of appearance. The value may contain blanks, so the name is the
    last field before "!". The "File update time stamp" header is skipped.
    """
    names = []
    for line in text.splitlines():
        data, sep, _ = line.partition("!")
        if not sep or "File update time stamp" in data:
            continue
        fields = data.split()
        if fields:
            names.append(fields[-1])
    return names


def template_text(method, attribute):
    # the template methods only set an attribute: no translator (and no OCS) needed
    obj = SimpleNamespace()
    getattr(KOSMA_translator, method)(obj)
    return getattr(obj, attribute)


@pytest.mark.parametrize("method,attribute,filename", TEMPLATES,
                         ids=[t[2] for t in TEMPLATES])
def test_template_matches_kosma_file(method, attribute, filename):
    path = Path(KOSMA_READWRITE_DIR) / filename
    if not path.exists():
        pytest.skip(f"{path} not found")

    template_vars = variable_names(template_text(method, attribute))
    file_vars = variable_names(path.read_text())

    only_in_file = [v for v in file_vars if v not in template_vars]
    only_in_template = [v for v in template_vars if v not in file_vars]

    messages = []
    if only_in_file:
        messages.append(
            f"in {path} but missing in {method} (would be wiped out): "
            + ", ".join(only_in_file)
        )
    if only_in_template:
        messages.append(
            f"in {method} but not in {path}: " + ", ".join(only_in_template)
        )
    assert not messages, "\n".join(messages)


@pytest.mark.parametrize("method,attribute,filename", TEMPLATES,
                         ids=[t[2] for t in TEMPLATES])
def test_template_has_no_duplicate_variables(method, attribute, filename):
    names = variable_names(template_text(method, attribute))
    duplicates = sorted({n for n in names if names.count(n) > 1})
    assert not duplicates, f"duplicate variables in {method}: {', '.join(duplicates)}"
