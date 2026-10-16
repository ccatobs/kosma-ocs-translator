# KOSMA OCS Translator

The server that fills the telescope seat of KOSMA file io at FYST. It turns KOSMA telescope commands into OCS program tracks and reports the telescope state back to KOSMA.

## Language

### Seat and commands

**Telescope seat**:
The role in KOSMA file io that reads `KOSMA_obs2tel.set` and writes `KOSMA_tel2obs.set`. astra, astra_dummy or the translator fills it.
_Avoid_: telescope slot

**Boresight update**:
A second command from control_tel that carries new focal-plane offsets. control_tel sends it after focalplane has recomputed the offsets.
_Avoid_: fp retrigger

### Positions

**Target**:
The topocentric az/el position, without refraction, that the current command asks for at a given time.
_Avoid_: setpoint, commanded position

**Standby point**:
The target 10 s before the start of an OTF line. A pre-OTF command parks the telescope there.
_Avoid_: pre-position

### Uploads

**Window**:
The stretch of target that the translator uploads as one OCS path.
_Avoid_: chunk, batch

**Renewal**:
The upload of a new window when the current window runs out before a new command arrives.
_Avoid_: refresh

### Testing

**ACU stub**:
An in-process test stand-in for the OCS, the TCS and the ACU. The server registry does not list it.
_Avoid_: fake ACU, mock

**ACU simulator**:
The antenna-control-unit container that `run.bash` in observatory-control-system starts.
_Avoid_: dummy
