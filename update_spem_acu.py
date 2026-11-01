"""
Script to translate the KOSMA starpoint fit results to the SPEM parameters
and update them on ACU
"""

import argparse
import logging
import os
import re

from kosma_ocs_translator import setup_ocs, ImportKOSMAReadWriteIntoDictionary

def get_kosma_spem():
    """
    Get ACU parameters stored in KOSMA_file_io

    Returns
    -------
    spem : dict
        Dictionary of ACU SPEM parameters
    """

    spem_kosma_format = ImportKOSMAReadWriteIntoDictionary(
        files = ["KOSMA_ccatacu_point.in"])

    # Make a dictionary of key = string_ccatacu_point[i], value = ccatacu_point[i]
    # also include ccatacu_model_type and ccatacu_model_date
    # for ccatacu_model_type, read the entire string before "ccatacu_model_type"
    # but then replace space by -.


def put_kosma_spem(spem):
    """
    Update ACU parameters stored in KOSMA_file_io

    Parameters
    ----------
    spem : dict
        New SPEM parameters
    """

    # inverse of get_kosma_spem is needed to map key to the idx in string_ccatacu_point
    # and use a shell command, e.g. Kset_point ccatacu_point[0] 1
    # and Kset_point ccatacu_model_date as a current time

def check_spem_sets(spem_acu, spem_kosma):
    """
    Check wether the two SPEM sets from ACU and KOSMA are identical.
    If not, user has to explicitly confirm.

    Parameters
    ----------
    spem_acu, spem_kosma : dict
        SPEM sets to compare
    """

    # 1. compare the keys first. Ignore the following keys on either set
    # leave the comment below
    # LEL velocity may be needed later if the pointing model turns out to be
    # dependent of the elevation velocity. For now not used in the fit
    ignore_keys_acu = ["Time", "Year", "LEL velocity"]
    ignore_keys_kosma = ["ccatacu_model_type", "ccatacu_model_date"]
    
    # 2. print the list of key - value from spem_acu - value from spem_kosma for
    # all keys (except for the ignore_keys above) to the terminal
    # if all values match, message both in terminal and log that they match and return
    # if any values do not match, highlight the difference,
    # print in addition ccat_acu_model_date from spem_kosma
    # and ask for user input - either continue or abort
    # log the decision and return or abort



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
                    apoint[f"a{m.group(1)}"] = float(value) * argsec2mdeg

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
    spem["LEVA"] = apoint['a18']
    spem["LEVE"] = apoint['a19']
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

    return spem

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

    # 1. print key - current - new for all keys except for ignore_keys_acu 
    # listed in check_spem_sets (ignore_keys_acu)
    # 2. ask for user confirmation - continue or abort
    # log the decision and continue or abort



def setup_logging(log_level):
    # setup logging to file and to screen
    logger = logging.getLogger("update_spem_acu")
    logger.setLevel(log_level)

    # Remove existing handlers to avoid conflicts
    # if logger.hasHandlers():
    #    logger.handlers.clear()

    # Prevent propagation to the root logger
    logger.propagate = False

    # create console handler and set level to debug
    ch = logging.StreamHandler()
    ch.setLevel(log_level)
    # create simpler formatter for console output
    console_formatter = logging.Formatter("%(levelname)s: %(message)s")
    ch.setFormatter(console_formatter)
    logger.addHandler(ch)

    # file handler
    fh = logging.FileHandler("kosma-ocs-translator.log")
    fh.setLevel(log_level)
    # detailed formatter for file output
    file_formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    fh.setFormatter(file_formatter)
    logger.addHandler(fh)

    return logger


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
    logger = setup_logging(log_level)
    ocs = setup_ocs(
        use_local_dev=args.use_local_dev,
        certificates_path=args.certificates_path,
        logger=logger,
    )
    
    # get current SPEM in ACU, TODO: implement on the OCS side
    current_spem_acu = ocs.get_spem()

    # strip "Parameter " from the acu keys

    # get current SPEM in KOSMA-file-io
    current_spem_kosma = get_kosma_spem()

    # check that current_spem_acu and current_spem_kosma is identical
    check_spem_sets(spem_acu=current_spem_acu, spem_kosma=current_spem_kosma)

    # read KOSMA_point_in and derive SPEM parameters
    new_spem = kosma2spem(point_in=args.point_in, current_spem=current_spem_kosma)

    # print and ask confirmation
    confirm_update(current_spem_acu, new_spem)

    # push to ACU, TODO: implement on the OCS side
    response = ocs.put_spem(new_spem)

    # verify the update
    updated_spem_acu = ocs.get_spem()
    print_spem(updated_spem_acu)

    # push to KOSMA_file_io
    updated_spem_kosma = put_kosma_spem(new_spem)


if __name__ == "__main__":
    main()
