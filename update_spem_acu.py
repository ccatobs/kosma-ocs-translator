"""
Script to translate the KOSMA starpoint fit results to the SPEM parameters
and update them on ACU
"""

import argparse
import logging
import math
import os
import re
import subprocess  # used by put_kosma_spem in kosma_ocs_translator (patched in tests)

from kosma_ocs_translator import (
    setup_ocs,
    KOSMA_SPEM_FILE,
    ignore_keys_acu,
    ignore_keys_kosma,
    strip_parameter_prefix,
    get_kosma_spem,
    put_kosma_spem,
)

# logger name follows the script name (without .py), also when run as __main__
LOGGER_NAME = os.path.splitext(os.path.basename(__file__))[0]

# tolerance [mdeg] to treat two SPEM parameter values as identical
# (0.01 mdeg = 0.036 arcsec)
spem_abs_tol = 1e-2

# SPEM parameters derived from the pointing fit are rounded to 0.001 mdeg
spem_decimals = 3

# pass as extra= to a log call to write the message to the log file only
FILE_ONLY = {"file_only": True}


def ask_continue(question):
    """
    Ask the user to continue or abort

    Parameters
    ----------
    question : str
        Question shown to the user

    Returns
    -------
    bool
        True if the user chose to continue
    """
    while True:
        answer = input(f"{question} [y(continue)/n(abort)]: ").strip().lower()
        if answer in ("y", "yes"):
            return True
        if answer in ("n", "no"):
            return False
        print("Please answer y (continue) or n (abort).")


def format_value(value):
    if isinstance(value, (int, float)):
        return f"{value:12.3f}"
    return f"{str(value):>12s}"


def check_spem_sets(spem_acu, spem_kosma):
    """
    Check wether the two SPEM sets from ACU and KOSMA are identical.
    If not, user has to explicitly confirm.

    Parameters
    ----------
    spem_acu, spem_kosma : dict
        SPEM sets to compare
    """
    log = logging.getLogger(LOGGER_NAME)

    # Compare the keys first. Ignore ignore_keys_acu and ignore_keys_kosma
    keys_acu = [k for k in spem_acu if k not in ignore_keys_acu]
    keys_kosma = [k for k in spem_kosma if k not in ignore_keys_kosma]
    only_acu = [k for k in keys_acu if k not in keys_kosma]
    only_kosma = [k for k in keys_kosma if k not in keys_acu]
    if only_acu:
        log.warning(f"Keys only in ACU SPEM set: {', '.join(only_acu)}")
    if only_kosma:
        log.warning(f"Keys only in KOSMA SPEM set: {', '.join(only_kosma)}")

    # Print the list of key - value from spem_acu - value from spem_kosma
    keys = keys_acu + only_kosma
    differ = []
    print("")
    print(f"{'key':<8s} {'ACU [mdeg]':>12s} {'KOSMA_ccatacu_point.in [mdg]':>30s}")
    for key in keys:
        value_acu = spem_acu.get(key)
        value_kosma = spem_kosma.get(key)
        match = (
            value_acu is not None
            and value_kosma is not None
            and math.isclose(value_acu, value_kosma, abs_tol=spem_abs_tol)
        )
        if not match:
            differ.append(key)
        mark = "" if match else "  <-- differs"
        print(f"{key:<8s} {format_value(value_acu)} {format_value(value_kosma)}{mark}")

    if not differ:
        log.info("SPEM parameters in ACU and KOSMA_file_io match")
        return

    print(f"ccatacu_model_date in KOSMA: {spem_kosma.get('ccatacu_model_date')}")
    log.warning(
        "SPEM parameters in ACU and KOSMA_file_io differ for: " + ", ".join(differ)
    )
    if ask_continue("SPEM sets in ACU and KOSMA differ. Continue?"):
        log.info("User chose to continue despite the SPEM mismatch")
        return
    log.info("User aborted due to the SPEM mismatch")
    raise SystemExit("Aborted by user")


def kosma2spem(point_in, current_spem):
    """
    read KOSMA_point_in and derive SPEM parameters

    Parameters
    ----------
    point_in : str
        File name of KOSMA_point.in
    current_spem : dict
        ACU SPEM parameters used at the time of pointing session

    Returns
    -------
    spem : dict
        Dictionary of ACU SPEM parameters
    """
    n_coeff = 26
    apoint = {}

    # apoint in KOSMA is in arcsec, ACU parameters are in mdeg
    arcsec2mdeg = 1000./3600.

    with open(point_in) as f:
        for line in f:
            # format: "<value> <name> ! <comment>"
            fields = line.split("!")[0].split()
            if len(fields) < 2:
                continue
            value, name = fields[0], fields[1]
            if name == "model_type":
                apoint["model_type"] = value
            else:
                m = re.fullmatch(r"a_point\[(\d+)\]", name)
                if m:
                    apoint[f"a{m.group(1)}"] = float(value) * arcsec2mdeg

    expected = ["model_type"] + [f"a{i}" for i in range(n_coeff)]
    missing = [k for k in expected if k not in apoint]
    if missing:
        raise ValueError(f"{point_in} lacks entries for: {', '.join(missing)}")

    spem = {}

    spem["IA"] = apoint['a0'] + current_spem["NRX"]
    spem["CA"] = -apoint['a1']
    spem["NPAE"] = -(apoint['a2'] + current_spem["NRY"])
    spem["AN"] = -apoint['a3']
    spem["AW"] = apoint['a4']
    spem["AES"] = apoint['a5']
    spem["AEC"] = -apoint['a6']
    spem["AW2"] = apoint['a10']
    spem["AN2"] = -apoint['a11']
    spem["EES2"] = apoint['a12']
    spem["AES2"] = apoint['a13']
    spem["EEC2"] = apoint['a14']
    spem["AEC2"] = apoint['a15']
    spem["LELA"] = apoint['a16']
    spem["LELE"] = apoint['a17']
    spem["LEVA"] = apoint['a18'] * 1000    # scaled in KOSMA
    spem["LEVE"] = apoint['a19'] * 1000
    spem["NRX"] = current_spem["NRX"]
    spem["NRY"] = current_spem["NRY"]

    if apoint["model_type"] == "CCAT_NORMAL":
        spem["IE"] = apoint['a7'] - current_spem["TFC"]
        spem["EEC"] = apoint['a8'] - current_spem["NRY"] - current_spem["TF"]
        spem["EES"] = apoint['a9'] + current_spem["NRX"] - current_spem["TFS"]
        spem["TFC"] = current_spem["TFC"]
        spem["TFS"] = current_spem["TFS"]
        spem["TF"] = current_spem["TF"]
        spem["TFEC"] = current_spem['TFEC']
        spem["TFES"] = current_spem['TFES']
        spem["TFE"] = current_spem['TFE']

    elif apoint["model_type"] == "CCAT_EXTENDED":
        spem["IE"] = apoint['a7'] - current_spem["TFEC"]
        spem["EEC"] = apoint['a8'] - current_spem["NRY"] + current_spem["TFE"]
        spem["EES"] = apoint['a9'] + current_spem["NRX"] - current_spem["TFES"]
        spem["TFC"] = current_spem["TFC"]
        spem["TFS"] = current_spem["TFS"]
        spem["TF"] = current_spem["TF"]
        spem["TFEC"] = current_spem['TFEC']
        spem["TFES"] = current_spem['TFES']
        spem["TFE"] = current_spem['TFE']

    elif apoint["model_type"] == "CCAT_SYMMETRIC":
        spem["IE"] = apoint['a7']
        spem["EEC"] = apoint['a8'] - current_spem["NRY"] - current_spem["TF"]
        spem["EES"] = apoint['a9'] + current_spem["NRX"]
        spem["TFC"] = apoint['a20']
        spem["TFS"] = apoint['a21']
        spem["TFEC"] = -apoint['a20']
        spem["TFES"] = -apoint['a21']
        spem["TF"] = current_spem["TF"]
        spem["TFE"] = -current_spem["TF"]

    elif apoint["model_type"] == "CCAT_ASYMMETRIC":
        spem["IE"] = apoint['a7']
        spem["EEC"] = apoint['a8'] - current_spem["NRY"]
        spem["EES"] = apoint['a9'] + current_spem["NRX"]
        spem["TFC"] = apoint['a20']
        spem["TFS"] = apoint['a21']
        spem["TF"] = apoint['a22']
        spem["TFEC"] = apoint['a23']
        spem["TFES"] = apoint['a24']
        spem["TFE"] = apoint['a25']

    else:
        raise ValueError(f"Unknown model_type {apoint['model_type']} in {point_in}")

    # e.g. 100 arcsec = 27.77777777777778 mdeg -> 27.778 mdeg
    spem = {key: round(value, spem_decimals) for key, value in spem.items()}

    return spem


def changed_spem(new_spem, current_spem):
    """
    Select the SPEM parameters that differ from the current set

    Parameters
    ----------
    new_spem : dict
        SPEM parameters to be uploaded
    current_spem : dict
        SPEM parameters currently set

    Returns
    -------
    changed : dict
        Entries of new_spem missing in current_spem or differing by more than
        spem_abs_tol [mdeg]
    """
    changed = {}
    for key, value in new_spem.items():
        if key in ignore_keys_acu or key in ignore_keys_kosma:
            continue
        current = current_spem.get(key)
        if current is None or not math.isclose(value, current, abs_tol=spem_abs_tol):
            changed[key] = value
    return changed


def confirm_update(current_spem_acu, new_spem):
    """
    Print the current ACU spem and new SPEM to be uploaded and ask for user confirmation

    Parameters
    ----------
    current_spem_acu : dict
       SPEM parameters currently in the ACU
    new_spem : dict
       SPEM parameters to be uploaded
    """
    log = logging.getLogger(LOGGER_NAME)

    # Print key - current - new for all keys except for ignore_keys_acu
    keys = [k for k in current_spem_acu if k not in ignore_keys_acu]
    keys += [k for k in new_spem if k not in keys and k not in ignore_keys_acu]
    print("")
    print(f"{'key':<8s} {'current [mdeg]':>12s} {'new [mdeg]':>12s} {'diff':>12s}")
    for key in keys:
        current = current_spem_acu.get(key)
        new = new_spem.get(key)
        if current is not None and new is not None:
            diff = format_value(new - current)
        else:
            diff = format_value("N/A")
        print(f"{key:<8s} {format_value(current)} {format_value(new)} {diff}")
    log.info(f"New SPEM parameters to upload: {new_spem}", extra=FILE_ONLY)

    # Ask for user confirmation - continue or abort
    if ask_continue("Upload the new SPEM parameters to ACU?"):
        log.info("User confirmed the SPEM update")
        return
    log.info("User aborted the SPEM update")
    raise SystemExit("Aborted by user")


def print_spem(spem):
    """
    Print SPEM parameters

    Parameters
    ----------
    spem : dict
        SPEM parameters
    """
    log = logging.getLogger(LOGGER_NAME)
    for key, value in spem.items():
        print(f"{key:<16s} {format_value(value)}")
    log.info(f"SPEM parameters: {spem}", extra=FILE_ONLY)


def setup_logging(log_level):
    # setup logging to file and to screen
    log = logging.getLogger(LOGGER_NAME)
    log.setLevel(log_level)

    # Remove existing handlers to avoid conflicts
    # if log.hasHandlers():
    #    log.handlers.clear()

    # Prevent propagation to the root logger
    log.propagate = False

    # create console handler and set level to debug
    ch = logging.StreamHandler()
    ch.setLevel(log_level)
    # create simpler formatter for console output
    console_formatter = logging.Formatter("%(levelname)s: %(message)s")
    ch.setFormatter(console_formatter)
    # log.info(..., extra=FILE_ONLY) goes to the log file only, e.g. when the
    # same information is already printed to the terminal as a table
    ch.addFilter(lambda record: not getattr(record, "file_only", False))
    log.addHandler(ch)

    # file handler
    fh = logging.FileHandler("update-spem-acu.log")
    fh.setLevel(log_level)
    # detailed formatter for file output
    file_formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    fh.setFormatter(file_formatter)
    log.addHandler(fh)

    return log


def parse_arguments():
    kosma_readwrite_dir = os.environ.get("WRITE_DIR", "/net/KOSMA_file_io/ReadWrite/")
    def_point_in = f"{kosma_readwrite_dir}/KOSMA_point.in"

    def_cpath = "../CCAT/observatory-control-system/tls"

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "-i",
        "--point_in",
        type=str,
        default=def_point_in,
        help=f"File name of pointing model in kosma. Default is {def_point_in}."
    )
    parser.add_argument(
        "--certificates-path",
        type=str,
        default=def_cpath,
        help=f"Path to the certificates directory. Default is {def_cpath}",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        default="INFO",
        help="Set the logging level. Choices: DEBUG, INFO, WARNING, ERROR, CRITICAL. Default is INFO.",
    )
    parser.add_argument(
        "--use-local-dev",
        action="store_true",
        help="Connect to the OCS of the local development setup (127.0.0.1:5600). Default is False.",
    )
    return parser.parse_args()


def main():
    args = parse_arguments()
    log_level = getattr(logging, args.log_level)
    log = setup_logging(log_level)
    ocs = setup_ocs(
        use_local_dev=args.use_local_dev,
        certificates_path=args.certificates_path,
        logger=log,
    )

    # get current SPEM in ACU, strip "Parameter " from the acu keys
    current_spem_acu = strip_parameter_prefix(ocs.get_dataset("CmdSPEMParameter"))

    # get current SPEM in KOSMA-file-io
    current_spem_kosma = get_kosma_spem()

    # check that current_spem_acu and current_spem_kosma is identical
    check_spem_sets(spem_acu=current_spem_acu, spem_kosma=current_spem_kosma)

    # read KOSMA_point_in and derive SPEM parameters
    new_spem = kosma2spem(point_in=args.point_in, current_spem=current_spem_kosma)

    # only parameters that actually change are sent to the ACU
    changed_acu = changed_spem(new_spem, current_spem_acu)
    if changed_acu:
        # print and ask confirmation
        confirm_update(current_spem_acu, new_spem)

        log.info(f"Updating ACU SPEM parameters: {', '.join(changed_acu)}")
        response = ocs.set_spem(**changed_acu)
        log.info(f"Response from set_spem: {response.status_code} {response.text}")
        if response.status_code != 200:
            # do not touch KOSMA_file_io, it should stay in sync with the ACU
            raise RuntimeError(f"set_spem failed with HTTP {response.status_code}")
    else:
        log.info("New SPEM parameters are identical to the current ones in ACU, not sent")

    # read back what the ACU actually uses
    updated_spem_acu = strip_parameter_prefix(ocs.get_dataset("CmdSPEMParameter"))
    print_spem(updated_spem_acu)
    not_applied = changed_spem(new_spem, updated_spem_acu)
    if not_applied:
        log.warning(
            "SPEM parameters in ACU differ from the new ones after set_spem: "
            + ", ".join(not_applied)
        )
        if not ask_continue("Continue anyway to mirror the ACU values in KOSMA?"):
            log.info("User aborted, KOSMA_file_io not updated")
            raise SystemExit("Aborted by user")

    # KOSMA_file_io mirrors the ACU: dump the full read-back set (not new_spem),
    # so that KOSMA never holds values the ACU did not apply
    updated_spem_kosma = put_kosma_spem(updated_spem_acu, logger=log)
    print_spem(updated_spem_kosma)


if __name__ == "__main__":
    main()
