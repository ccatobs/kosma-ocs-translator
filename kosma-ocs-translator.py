import urllib.request, json, requests
import time,datetime
import random
import logging
import os,glob,re
from ocs import observatory_control_system
from astropy.coordinates import SkyCoord
from astropy import units as u

def get_status(url):
    cmd = "acu-status"
    print(url)
    with urllib.request.urlopen(url+cmd) as url_open:
        response = url_open.read().decode()
        if response=="":
            print("no response received from {0}".format(cmd))
    return response

def get_status(url):
    cmd = "acu-status"
    response = requests.get(url+cmd)
    if response.text=="":
        print("no response received from {0}".format(cmd))
    return response.json()

def abort(url):
    cmd="abort"
    r = requests.post(url+cmd, allow_redirects=True)
    response = json.loads(r.content.decode())
    print(response)


def move_to(url,azimuth=90,elevation=60):
    abort(url)
    time.sleep(2)
    # move to 
    dt = datetime.datetime.now() + datetime.timedelta(seconds=10)
    # azimuth scab example
    cmd = "move-to"
    data ={
      "azimuth": azimuth,
      "elevation": elevation
    }
    r = requests.post(url+cmd, data=json.dumps(data), allow_redirects=True)
    print(r.json())


def azimuth_scan():
    # track telescope
    dt = datetime.datetime.now() + datetime.timedelta(seconds=10)
    # azimuth scab example
    cmd = "azimuth-scan"
    data ={
      "azimuth_range": [110,130],
      "elevation": 60,
      "num_scans": 20,
      "start_time": dt.strftime("%Y-%m-%dT%H:%M:%SZ"),
      "turnaround_time": 30,
      "speed": 0.8
    }
    r = requests.post(url+cmd, data=json.dumps(data), allow_redirects=True)
    print(r.json())
#
def source_track(url,ra=120,dec=45):
    # track telescope
    dt = datetime.datetime.now() + datetime.timedelta(seconds=10)
    # azimuth scan example
    cmd = "track"
    dtstart = datetime.datetime.now() + datetime.timedelta(seconds=10)
    dtend = datetime.datetime.now() + datetime.timedelta(seconds=30)
    data ={
      "start_time": dtstart.strftime("%Y-%m-%dT%H:%M:%SZ"),
      "stop_time": dtend.strftime("%Y-%m-%dT%H:%M:%SZ"),
      "ra": ra,
      "dec": dec
    }
    print(data)
    r = requests.post(url+cmd, data=json.dumps(data), allow_redirects=True)
    print(r.json())
#curl 'localhost:5600/track' -d@- <<___
#{
#    "start_time": "2019-04-01T20:00:00Z",
#    "stop_time": "2019-04-01T21:00:00Z",
#    "ra": 120,
#    "dec": 45
#}


def ImportKOSMAReadWriteIntoDictionary(files=None, variable=None,update_mod_time=True,):
    log = logging.getLogger("kosma_db_hk_open_mon.out")
    readwrite_dict = {}
    if 'write_dir' not in vars():
        kosma_readwrite_dir = os.environ.get('WRITE_DIR',"/net/KOSMA_file_io/ReadWrite/")
    else:
        kosma_readwrite_dir = write_dir
    log.info("polling for changes from {0}".format(kosma_readwrite_dir))
    not_changed_count = 0
    if files==None:
        readwrite_files=glob.glob("%s/*" % (kosma_readwrite_dir))        
    else:
        readwrite_files=[]
        for filename in files:            
            # check mod time here, if hasn't changed from last time.. skip it
            full_path_filename = kosma_readwrite_dir+"/"+filename
            if not os.path.exists(full_path_filename):
                log.warning("WARNING: {0} not found".format(full_path_filename))
                continue
            else:
                readwrite_files.append(kosma_readwrite_dir+"/"+filename)
    #
    if len(readwrite_files)==0:
        #print "No files active, check servers and %s path" % (kosma_readwrite_dir)
        return {}
    # create global dictionary to store kosma_read acfg values
    for readwrite_file_full in readwrite_files:
       log.debug("reading {0}".format(readwrite_file_full))
       if os.path.isdir(readwrite_file_full):
          continue
       if not os.path.exists(readwrite_file_full):
          continue
       f = open(readwrite_file_full, "r")
       lines=f.readlines()
       f.close()       
       # strip fill path from readwrite_file
       readwrite_file = os.path.basename(readwrite_file_full)
       # create dictionary for each filename
       if readwrite_file not in readwrite_dict.keys():
           readwrite_dict[readwrite_file] = {}
       readwrite_dict[readwrite_file]['file_timestamp']=os.path.getmtime(readwrite_file_full)
       for i,line in enumerate(lines):
          # check for timestamp in header
          if re.match('.+File update time stamp.+', line):
            timestamp = re.search('.+\s(\d+\.\d+)\s+\S+.+', line).groups()[0]
            #if 'timestamp' not in readwrite_dict[readwrite_file].keys():
            readwrite_dict[readwrite_file]['timestamp']=timestamp             
            continue        
          elif re.match('.+!.+', line):
            result = re.search('(.+)!(.+)', line)
            if result:
               data, description = result.groups()
          else:
            continue
          #log.debug("found {0} {1}".format(data, description))
          # if no timestamp found in file, use file timestamp
          if 'timestamp' not in readwrite_dict[readwrite_file].keys():
              readwrite_dict[readwrite_file]['timestamp'] = readwrite_dict[readwrite_file]['file_timestamp']
          # populate dictionary
          try:
             result = re.search('(\S+)\s+(\S+)', data)
             if result:
                value, variable_found = result.groups()
          except: 
             result =  re.search('\s+(\S+)', data).groups()
             if result:
                 variable_found, = result.groups()
             print("{0} variable found with no value ".format(variable_found))
             value="None "
          #variable_found = variable_found.strip()
          # convert parse value to string, int or float
          
          try:
             if type(value)==int:
                continue
             if "." in value:
               value=float(value)
             else:
               value=int(value)
          except:
               value=value.strip()
          #
          if (variable!=None) & (variable!=variable_found):
              continue
          if variable not in readwrite_dict[readwrite_file].keys():
               readwrite_dict[readwrite_file][variable_found]=value
    # remove entries that just have a timestamp and no variable values
    #for
    return readwrite_dict    



ocs_host="127.0.0.1"
ocs_port=5600
certificates_path = "../observatory-control-system/tls"
ocs = observatory_control_system(url=f'https://{ocs_host}:{ocs_port}',
                                   server_cert=f'{certificates_path}/server.cert.pem',
                                   client_cert=f'{certificates_path}/client.cert.pem',
                                   client_key=f'{certificates_path}/client.key.pem')



kio_files = ImportKOSMAReadWriteIntoDictionary(files=["KOSMA_obs2tel.set","KOSMA_tel2obs.set"], variable=None,update_mod_time=True)

#print(kio_files["KOSMA_obs2tel.set"].keys())

tel_return_cookie =  kio_files["KOSMA_obs2tel.set"]["obs_cookie"]
obs_tolerance =  kio_files["KOSMA_obs2tel.set"]["obs_tolerance"]
obs_tel_info_update_time = kio_files["KOSMA_obs2tel.set"]["obs_tel_info_update_time"]
#
#cmd_ele = 90- 70*random.uniform(0, 1)
#cmd_azi = 90+ 70*random.uniform(0, 1)
#move_to(url, azimuth=cmd_azi,elevation=cmd_ele)
#print("moving to az: {0:3.0f} ele: {1:3.0f}".format(cmd_azi,cmd_ele))
#
cmd_lam = kio_files["KOSMA_obs2tel.set"]["obs_lam_on"]
cmd_bet = kio_files["KOSMA_obs2tel.set"]["obs_bet_on"]
cmd_coord_sys_on = kio_files["KOSMA_obs2tel.set"]["obs_coord_sys_on"]
#
coord_sys_translator = {"J2000":"icrs","B1950":"fk4","GALACTIC":"galactic","ECLIPTIC":"geocentrictrueecliptic"}

if cmd_coord_sys_on not in coord_sys_translator.keys():
   logger.error("coordinate system {0} not recognized".format(cmd_coord_sys_on))
   raise SystemExit
star_position = SkyCoord(cmd_lam,cmd_bet,unit=(u.deg,u.deg),
                         frame=coord_sys_translator[cmd_coord_sys_on])
ocs.track(star_position)
#
time.sleep(1)
success=False
while (success==False):
   try:
       response = ocs.get_status()
       success=True
       time.sleep(1)
   except:
       time.sleep(1)
       pass
print(response)
current_azi = response["Azimuth current position"]
current_ele = response["Elevation current position"]
print("current pos. az: {0:3.0f} ele: {1:3.0f}".format(current_azi,current_ele))


def get_tel2obs_dict(ocs, obs_tolerance):
    response = ocs.get_status()
    print(response)
    input_dict ={}
    input_dict["tel_azm_act"] = response["Azimuth current position"] 
    input_dict["tel_elv_act"] = response["Elevation current position"] 
    input_dict["tel_azm_cmd"] = response["Azimuth commanded position"] 
    input_dict["tel_elv_cmd"] = response["Elevation commanded position"] 
    input_dict["tel_latitude"] = '-22.96995611'
    input_dict["tel_longitude"] = '67.70308139'
    input_dict["tel_altitude"] = '4863.85'
    input_dict["tel_telescope"] = 'NANTEN2'
    input_dict["tel_plate_scale"] = '1'
    input_dict["tel_angle_focal_plane"] = 53.97   
    input_dict["tel_on_track"]='Y'
    input_dict["tel_lost_track"] = 'N'
    input_dict["tel_return_cookie"] = kio_files["KOSMA_obs2tel.set"]["obs_cookie"]
    input_dict["tel_error"] = '0'
    current_azi = response["Azimuth current position"]
    current_ele = response["Elevation current position"]
    cmd_azi = response["Azimuth commanded position"]
    cmd_ele = response["Elevation commanded position"]
    actdiff = 3600*((current_azi-cmd_azi)**2+(current_ele-cmd_ele)**2)**0.5
    if (actdiff < obs_tolerance):
     input_dict["tel_pos_in_range"] = 'Y'
     input_dict["tel_on_track"] = 'Y'
    else:
     input_dict["tel_pos_in_range"] = 'Y'
     input_dict["tel_on_track"] = 'N'
    input_dict["tel_lost_track"] = 'N'
    #
    fmt = '%d-%b-%Y  %H:%I:%S'
    current = time.localtime()
    input_dict['timestamp'] = time.time()
    input_dict['timestring'] = date = time.strftime(fmt,current)
    return input_dict

def get_obs2tel_dict(obs_tolerance):
    response = get_status(url)
    input_dict ={}
    input_dict["tel_azm_act"] = response["Azimuth current position"] 
    input_dict["tel_elv_act"] = response["Elevation current position"] 
    input_dict["tel_azm_cmd"] = response["Azimuth desired position"] 
    input_dict["tel_elv_cmd"] = response["Elevation desired position"] 
    input_dict["tel_latitude"] = '-22.96995611'
    input_dict["tel_longitude"] = '67.70308139'
    input_dict["tel_altitude"] = '4863.85'
    input_dict["tel_telescope"] = 'NANTEN2'
    input_dict["tel_plate_scale"] = '1'
    input_dict["tel_angle_focal_plane"] = 53.97   
    input_dict["tel_on_track"]='Y'
    input_dict["tel_lost_track"] = 'N'
    input_dict["tel_return_cookie"] = kio_files["KOSMA_obs2tel.set"]["obs_cookie"]
    input_dict["tel_error"] = '0'
    current_azi = response["Azimuth current position"]
    current_ele = response["Elevation current position"]
    cmd_azi = response["Azimuth desired position"]
    cmd_ele = response["Elevation desired position"]
    actdiff = 3600*((current_azi-cmd_azi)**2+(current_ele-cmd_ele)**2)**0.5
    if (actdiff < obs_tolerance):
     input_dict["tel_pos_in_range"] = 'Y'
     input_dict["tel_on_track"] = 'Y'
    else:
     input_dict["tel_pos_in_range"] = 'Y'
     input_dict["tel_on_track"] = 'N'
    input_dict["tel_lost_track"] = 'N'
    #
    fmt = '%d-%b-%Y  %H:%I:%S'
    current = time.localtime()
    input_dict['timestamp'] = time.time()
    input_dict['timestring']  = time.strftime(fmt,current)
    return input_dict

tel2obs='''
{0[timestring]}  {0[timestamp]}   File update time stamp   ! ccat_translator (test_computer:1000)
EMU_OCS   tel_telescope   ! Telescope Identifier (for display and backend_name in FITS header)
{0[tel_on_track]}   tel_on_track   ! Y if tracking on commanded position/track within tolerance [Y/N]
{0[tel_lost_track]}   tel_lost_track   ! Y if tracking got beyond tolerance since start of track [Y/N]
{0[tel_pos_in_range]}   tel_pos_in_range   ! Y if commanded position is within telescope range [Y/N]
           0   tel_error   ! flags telescope command syntax/consistency and functional errors (=0: ok)
NONE   tel_error_string   ! Error message associated with telescope command (=NONE: OK)
           5   tel_error_level   ! Error level for message in tel_error_string [1: trace, 2: debug, 4: info, 5: message, 7: warn, 8: error, 9: fatal]
      {0[tel_return_cookie]}  tel_return_cookie   ! return cookie to identify proper observation
      {0[tel_plate_scale]}  tel_plate_scale   ! plate scale of focal plane [arcsec/mm]
      {0[tel_latitude]}   tel_latitude   ! latitude of observatory (needed for Doppler correction) [degree]
      {0[tel_longitude]}  tel_longitude   ! longitude of observatory (needed for Doppler correction) [degree, +west (astronomical notation)]
      {0[tel_altitude]} 3141   tel_altitude   ! height above sea level (needed for Doppler correction) [meter]
      {0[tel_angle_focal_plane]}   tel_angle_focal_plane   ! ccw angle from second coordinate of the telescope reference coordinate system (TARF,HORIZON) to second coordinate of obs_coord_sys_del [degree]
#           0   tel_los_act   ! actual los-angle [degree]
#    1557237163.99149   tel_time_act   ! time when status data were valid [seconds since 1970 UTC]
#   87.091738   tel_time_del   ! time since start of track (may be negative, if start in the future) [seconds]
   {0[tel_azm_cmd]:3.4f}   tel_azm_cmd   ! commanded azimuth (for tracking display) [degree,cw toward east]
   {0[tel_elv_cmd]:3.4f}   tel_elv_cmd   ! commanded elevation (for tracking display and for atmospheric calibration) [degree]
   {0[tel_azm_act]:3.4f}   tel_azm_act   ! actual azimuth (for tracking display) [degree,cw toward east]
   {0[tel_elv_act]:3.4f}   tel_elv_act   ! actual elevation (for tracking display) [degree]
'''

while True:
    kio_files = ImportKOSMAReadWriteIntoDictionary(files=["KOSMA_obs2tel.set"], variable=None,update_mod_time=True)
    obs_tolerance =  kio_files["KOSMA_obs2tel.set"]["obs_tolerance"]
    obs_tel_info_update_time = kio_files["KOSMA_obs2tel.set"]["obs_tel_info_update_time"]
    input_dict = get_tel2obs_dict(ocs, obs_tolerance)
    time.sleep(obs_tel_info_update_time)
    tel2obs_handle = open("/net/KOSMA_file_io/ReadWrite/KOSMA_tel2obs.set","w")
    tel2obs_handle.write(tel2obs.format(input_dict))
    tel2obs_handle.close()
