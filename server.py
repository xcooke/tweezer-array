#from camera_hardware_driver import CameraHardware
from slmsuite.hardware.cameras.thorlabs import ThorCam

#from slm_driver import SLM
from slmsuite.hardware.slms.meadowlark import Meadowlark

from sipyco.pc_rpc import simple_server_loop

from drivers.slm_cam_coordinator_driver import SLMCam
from drivers.camera_tweezers import CameraTweezers
from drivers.camera_fluorescence import CameraFluorescence
from drivers.atoms import Atoms

lut_path = "interpolated_voltage.lut"

camera_hardware = ThorCam()

slm = Meadowlark(lut_path=lut_path, wav_um=.688, settle_time_s=.3)

slm_cam = SLMCam(
    cam=camera_hardware,
    slm=slm,
)

#slm_cam.set_phase()

camera_tweezers = CameraTweezers(camera_hardware=camera_hardware)

atoms = Atoms(camera_tweezers=camera_tweezers)

camera_fluorescence = CameraFluorescence(camera_hardware=camera_hardware, atoms=atoms)

slm_cam.camera_fluorescence = camera_fluorescence

targets = {
    "camera_hardware": camera_hardware,
    "slm": slm,
    "slm_cam": slm_cam,
    "camera_tweezers": camera_tweezers,
    "camera_fluorescence": camera_fluorescence,
    "atoms": atoms,
}


def main():
    print("Server should be running")
    simple_server_loop(targets, "0.0.0.0", 4000)

if __name__ == "__main__":
    main()