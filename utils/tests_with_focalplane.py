#!/usr/bin/env python3
"""
Pure geometric focal-plane model (no rotator hardware, no SOFIA blocks).

Transforms telescope + instrument parameters into:
- Focal plane X/Y offsets (arcsec)
- Boresight position (mm)
- Effective focal length (mm)
- Total geometric rotation angle rho (deg)

Author: Generated conversion from legacy control code
"""

from dataclasses import dataclass
from kosma_ocs_translator import KOSMA_translator, ImportKOSMAReadWriteIntoDictionary
import math
import subprocess
import os
from datetime import datetime
import time
    

DEG2RAD = math.pi / 180.0

# rot_dx = rot_dx/(focal_length*M_PI)*180.0*3600.0;
def mm_to_arcsec(mm, focal_length_mm):
    return mm / (focal_length_mm * math.pi) * 180.0 * 3600.0


@dataclass
class TelescopeData:
    tel_telescope: str
    tel_angle_focal_plane: float   # ξ_s
    tel_elv_act: float             # elevation angle
    tel_plate_scale: float         # arcsec/mm


@dataclass
class InstrumentData:
    instr_name: str
    instr_boresight_offset_x: float
    instr_boresight_offset_y: float
    instr_elevation_axis_x: float
    instr_elevation_axis_y: float
    instr_flange_rotation: float
    instr_focal_length_correction: float
    instr_port_name: str
    instr_rotator_axis_x: float
    instr_rotator_axis_y: float
    instr_focal_plane_rotation: float   # α_p
    instr_reference_pos_x: float
    instr_reference_pos_y: float
    boresight_reference_pos_x: float
    boresight_reference_pos_y: float

@dataclass
class FocalPlaneComparison:
    obs_x_focal_plane_kio: float
    obs_x_focal_plane_computed: float
    diff_x: float
    obs_y_focal_plane_kio: float
    obs_y_focal_plane_computed: float
    diff_y: float


def compute_focal_plane(tel: TelescopeData, inst: InstrumentData):
    # Plate scale → focal length
    focal_length = (
        (180.0 * 3600.0) / (tel.tel_plate_scale * math.pi)
        + inst.instr_focal_length_correction
    )
    print(f"Computed focal length: {focal_length} mm")

    # Telescope → Instrument rotation (α_IM)
    if inst.instr_port_name.startswith("Left"):
        angle_if = -tel.tel_elv_act
    elif inst.instr_port_name.startswith("Right"):
        angle_if = tel.tel_elv_act
    elif inst.instr_port_name.startswith("Cass"):
        angle_if = inst.instr_flange_rotation
    else:
        angle_if = 0.0

    # Nasmyth geometry offsets (mm)
    ns_dx = inst.instr_rotator_axis_x + inst.instr_boresight_offset_x
    ns_dy = inst.instr_rotator_axis_y + inst.instr_boresight_offset_y

    ns_dx0 = inst.instr_elevation_axis_x
    ns_dy0 = inst.instr_elevation_axis_y

    if_x = ns_dx * math.cos(angle_if * DEG2RAD) - ns_dy * math.sin(angle_if * DEG2RAD)
    if_y = ns_dx * math.sin(angle_if * DEG2RAD) + ns_dy * math.cos(angle_if * DEG2RAD)

    if_x += ns_dx0
    if_y += ns_dy0

    if_x = mm_to_arcsec(if_x, focal_length)
    if_y = mm_to_arcsec(if_y, focal_length)

    # Total geometric rotation (ρ)
    rho = angle_if  + inst.instr_focal_plane_rotation

    # Reference offsets mm → arcsec
    rot_dx_arc = mm_to_arcsec(inst.instr_reference_pos_x, focal_length)
    rot_dy_arc = mm_to_arcsec(inst.instr_reference_pos_y, focal_length)
    print(f"Reference offsets (arcsec): rot_dx_arc={rot_dx_arc}, rot_dy_arc={rot_dy_arc}")

    # Focal plane offsets    
    #rho = rho
    #print(f"rho minus= {rho}, rho * DEG2RAD =", rho * DEG2RAD)    
    #rho = -28.727250
    #print(f"rho= {rho}, rho * DEG2RAD =", rho * DEG2RAD)
    fp_x = rot_dx_arc * math.cos(rho * DEG2RAD) - rot_dy_arc * math.sin(rho * DEG2RAD)
    fp_y = rot_dx_arc * math.sin(rho * DEG2RAD) + rot_dy_arc * math.cos(rho * DEG2RAD)
    print(f"Focal plane offsets before IF addition (arcsec): fp_x={fp_x}, fp_y={fp_y} if_x={if_x}, if_y={if_y}")

    fp_offset_x = fp_x + if_x
    fp_offset_y = fp_y + if_y

    # Boresight in mm
    angle_im = rho - inst.instr_focal_plane_rotation

    fp_bx = (
        inst.instr_reference_pos_x * math.cos(angle_im * DEG2RAD)
        + inst.instr_reference_pos_y * math.sin(angle_im * DEG2RAD)
        + inst.instr_rotator_axis_x
    )
    fp_by = (
        -inst.instr_reference_pos_x * math.sin(angle_im * DEG2RAD)
        + inst.instr_reference_pos_y * math.cos(angle_im * DEG2RAD)
        + inst.instr_rotator_axis_y
    )

    fp_boresight_x = fp_bx + inst.boresight_reference_pos_x
    fp_boresight_y = fp_by + inst.boresight_reference_pos_y

    return {
        "focal_length_mm": focal_length,
        "rho_deg": rho,
        "fp_offset_x_arcsec": fp_offset_x,
        "fp_offset_y_arcsec": fp_offset_y,
        "fp_boresight_x_mm": fp_boresight_x,
        "fp_boresight_y_mm": fp_boresight_y,
    }



def test_focal_plane_static():
    telescope = TelescopeData(
        tel_telescope="CCAT",
        tel_angle_focal_plane=91.677518,
        tel_elv_act=28.72725,
        tel_plate_scale=13.89
    )

    instrument = InstrumentData(
        instr_name="CHAI",
        instr_boresight_offset_x=0.0,
        instr_boresight_offset_y=0.0,
        instr_elevation_axis_x=0.0,
        instr_elevation_axis_y=0.0,
        instr_flange_rotation=0.0,
        instr_focal_length_correction=0.0,
        instr_port_name="Left",
        instr_rotator_axis_x=0.0,
        instr_rotator_axis_y=0.0,
        instr_focal_plane_rotation=0.0,
        instr_reference_pos_x=10.1,
        instr_reference_pos_y=-20.0,
        boresight_reference_pos_x=10.1,
        boresight_reference_pos_y=10.0,
    )

    result = compute_focal_plane(telescope, instrument)

    print("\n=== FOCAL PLANE SOLUTION ===")
    for k, v in result.items():
        print(f"{k:25s}: {v}")
        

def test_focal_plane_with_kio():                
    # map KIO variables to dataclass fields
    kio_files = ImportKOSMAReadWriteIntoDictionary(files=["KOSMA_tel2obs.set",
                                                        "KOSMA_obs2tel.set",
                                                        "TEL_hardware.status",
                                                        "KOSMA_focalplane.status",
                                                        "KOSMA_angle_fp.set"])
    #
    telescope_data = TelescopeData(
        tel_telescope=kio_files["KOSMA_tel2obs.set"]["tel_telescope"],
        tel_angle_focal_plane=kio_files["KOSMA_tel2obs.set"]["tel_angle_focal_plane"],
        tel_elv_act=kio_files["KOSMA_tel2obs.set"]["tel_elv_act"],
        tel_plate_scale=kio_files["KOSMA_tel2obs.set"]["tel_plate_scale"]
    )
    #
    instrument_data = InstrumentData(
        instr_name=kio_files["TEL_hardware.status"]["instr_name[0]"],
        instr_boresight_offset_x=kio_files["TEL_hardware.status"]["instr_boresight_offset_x[0]"],
        instr_boresight_offset_y=kio_files["TEL_hardware.status"]["instr_boresight_offset_y[0]"],
        instr_elevation_axis_x=kio_files["TEL_hardware.status"]["instr_elevation_axis_x[0]"],
        instr_elevation_axis_y=kio_files["TEL_hardware.status"]["instr_elevation_axis_y[0]"],
        instr_flange_rotation=kio_files["TEL_hardware.status"]["instr_flange_rotation[0]"],
        instr_focal_length_correction=kio_files["TEL_hardware.status"]["instr_focal_length_correction[0]"],
        instr_port_name=kio_files["TEL_hardware.status"]["instr_port_name[0]"],
        instr_rotator_axis_x=kio_files["TEL_hardware.status"]["instr_rotator_axis_x[0]"],
        instr_rotator_axis_y=kio_files["TEL_hardware.status"]["instr_rotator_axis_y[0]"],
        instr_focal_plane_rotation=kio_files["KOSMA_focalplane.status"]["instr_focal_plane_rotation[0]"],
        instr_reference_pos_x=kio_files["KOSMA_focalplane.status"]["instr_reference_pos_x[0]"],
        instr_reference_pos_y=kio_files["KOSMA_focalplane.status"]["instr_reference_pos_y[0]"],
        boresight_reference_pos_x=kio_files["KOSMA_focalplane.status"]["boresight_reference_pos_x[0]"],
        boresight_reference_pos_y=kio_files["KOSMA_focalplane.status"]["boresight_reference_pos_y[0]"]
    )
    # focal plane offset from KIO variables
    obs_x_focal_plane_obs2tel = kio_files["KOSMA_obs2tel.set"]["obs_x_focal_plane"]
    obs_y_focal_plane_obs2tel = kio_files["KOSMA_obs2tel.set"]["obs_y_focal_plane"]
    # calculate obs_x_focal_plane and obs_y_focal_plane from geometric model
    # pass data into focal plane computation
    focal_plane_result = compute_focal_plane(telescope_data, instrument_data)
    # look at difference between python and focalplane code
    diff_x = obs_x_focal_plane_obs2tel - focal_plane_result["fp_offset_x_arcsec"]
    diff_y = obs_y_focal_plane_obs2tel - focal_plane_result["fp_offset_y_arcsec"]
    # print comparison
    print("\n=== FOCAL PLANE TEST WITH KIO DATA ===")
    comparison = FocalPlaneComparison(
        obs_x_focal_plane_kio=obs_x_focal_plane_obs2tel,
        obs_x_focal_plane_computed=focal_plane_result["fp_offset_x_arcsec"],
        diff_x=diff_x,
        obs_y_focal_plane_kio=obs_y_focal_plane_obs2tel,
        obs_y_focal_plane_computed=focal_plane_result["fp_offset_y_arcsec"],
        diff_y=diff_y
    )
    #
    #
    return comparison

def run_system_commands_and_apply(cmds):
    # use subprocess
    cmds.append("KOSMA_track")
    cmds.append("KOSMA_track")  # run twice to ensure KIO updates    
    for cmd in cmds:
        print(f"Running command: {cmd}")
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"Error running command: {result.stderr}")
        else:
            print(f"Command output: {result.stdout}")
    time.sleep(2)  # wait a bit for KIO to update
 
def check_if_focalplane_server_running():
    # check modtime on the focalplane status file
    proc_file = "/net/KOSMA_file_io/ReadWrite/focalplane_alive"
    # check mod time using python
    if not os.path.exists(proc_file):
        return False
    mod_time = os.path.getmtime(proc_file)
    current_time = time.time()
    # if mod time is older than 2 seconds, assume not running
    if current_time - mod_time > 2:
        return False
    return True
    

if not check_if_focalplane_server_running():
    print("Focalplane server not running. Please start the focalplane server before running this test.")
    raise SystemExit(1)

comparison = test_focal_plane_with_kio()
print(f"diff_x={comparison.diff_x:3.4f}, diff_y={comparison.diff_y:3.4f}")
# try zeroing reference positions
cmds = []
cmds.append("Kset_hardware instr_reference_pos_x[0] 0.0")
cmds.append("Kset_hardware instr_reference_pos_y[0] 0.0")
cmds.append("Kset_hardware boresight_reference_pos_x[0] 0.0")
cmds.append("Kset_hardware boresight_reference_pos_y[0] 0.0")
run_system_commands_and_apply(cmds)
comparison = test_focal_plane_with_kio()
print(f"After zeroing reference positions: diff_x={comparison.diff_x:3.4f}, diff_y={comparison.diff_y:3.4f}")

#
cmds = []
cmds.append("Kset_hardware instr_reference_pos_x[0] 0.0")
cmds.append("Kset_hardware instr_reference_pos_y[0] 0.0")
cmds.append("Kset_hardware boresight_reference_pos_x[0] 0.0")
cmds.append("Kset_hardware boresight_reference_pos_y[0] 0.0")
cmds.append("Kset_hardware Rx_px[0] 10.0")
cmds.append("Kset_hardware Rx_py[0] -1000.0")
cmds.append("Kset_hardware Rx_cx[0] 1e2.0")
cmds.append("Kset_hardware Rx_cy[0] 0.0")
cmds.append("Kset_hardware Rx_tilt[0] 0")
cmds.append("setpoint -p L_PX00 -t") ## applies the changes to the hardware
cmds.append("setsource TEST_GAL -l 294.04987 -b 12.21678 -C GALACTIC")
run_system_commands_and_apply(cmds)
comparison = test_focal_plane_with_kio()
print(comparison)
print(f"After setting instr_reference_pos_y to 10.0: diff_x={comparison.diff_x:3.4f}, diff_y={comparison.diff_y:3.4f}")

