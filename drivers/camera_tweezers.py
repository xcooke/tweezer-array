#!/usr/bin/env python3
class CameraTweezers():

    def __init__(self, camera_hardware):
        self.camera_hardware = camera_hardware

        print(self.camera_hardware.shape)

    def __getattr__(self, name):
        return getattr(self.camera_hardware, name)