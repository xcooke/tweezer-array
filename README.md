# tweezer-array

This code is a mess! This is left as I was using it. This is mainly written to help me remember what I did, not to provide a guide for other people. I wrote the code quickly to investigate various things. A lot of things won't be defined, sorry, and I have not spent time organising it. Please reach out if you have any questions! Contact me at alexander.cooke@sjc.ox.ac.uk.

This code was for my summer research project at Imperial. I worked in the Imperial Strontium Lab, (https://www.stronlab.net/) and (https://www.hep.ph.ic.ac.uk/AION-Project/).

More details about the project are in the Imperial Strontium Lab labbook (https://labbook.stronlab.net/2026/2026-09-11-Xander-UROP-Write-Up).

In this repo I wrote code for my laptop which is connected to the SLM and the Camera. It hosts a sipyco server which ARTIQ connects to.

# slmsuite

I am using slmsuite to talk to Meadowlark SLM and to the Thorlabs camera.

See my labbook entries and my `slm` repo for more info.

# Installation

Using UV as package manager.

Also need to install Meadowlark drivers as well. Use lab book entry for how I did it https://labbook.stronlab.net/2026/2026-07-29-SLM-investigation. Tom provided me with Meadowlark software, I don't think you can find on web.

Thorlabs camera drivers are installed automatically with ThorCam or ThorImageCam (ThorImageCam is newer, I mainly used this). Camera specific details in labbook entry (https://labbook.stronlab.net/2026/2026-07-24-2D-AOD-setup-continued.)

There is a bug in the source code of slmsuite for interaction with the Thorlabs camera. The fix is in labbook entry (https://labbook.stronlab.net/2026/2026-08-03-SLM-calibration-continued). The correct `thorlabs.py` file is in the repo (can also be found well as in the git bug report).

Also for some reason need this `\src\slm` folder with an `__init__.py` file. Don't know why!

# Beware

Camera should be initiated with `cam = ThorCam(rot = "270")` given the current physical camera setup. This should be set given the physical orientation of the camera.

See above re slmsuite ThorCam bug.

I think the bug also effects slmsuite's HDR and frame averaging features, but I haven't tested this, and I never got them working.

I have also had many issues with our specific SLM. Sometimes it just won't work. Sometimes if I load my `slm` repo and write a saved phase that appears to work. Think there is something wrong with the HDMI port, but haven't managed to identify exactly what.

# Server setup

SLM and camera need to be able to talk to each other in memory.

On running `server.py`: SLM, camera, an SLM+camera coordinator, 'atoms', and 'fluorescence camera' are initialised.

They are then passed into a sipyco server. They can then talk to each other, while also being accessible over the network. See `sipyco_server_access_example.py` to see how you can test this functionality (ARTIQ handles this automatically).

If the ARTIQ program needs to do something that requires both the SLM and the camera at the same time then it will call the slm_cam (the coordintor of the SLM and the camera). Otherwise it can talk to camera/SLM/atoms etc directly.

All of the slmsuite things are inside `slm_cam_coordinator_device.py`. Note that currently experimental optimisation of a phase mask is turned off because I wanted the result to work faster. It should be able to be turned back on with a good Fourier calibration.

# Running server

Plug in SLM and camera. Power the SLM. Which USB ports you plug into might matter. Check logbook.

If making new phase mask make sure Fourier calibration is correct. Easiest way to do this is in `slm` repo.

Connect to same network as ARTIQ host.

Make sure the AOM calibration is correct. This should be roughly right as long as optical setup hasn't been altered. However I think it might drift. To generate a new one run the server as below and run the `CharacteriseOptics` ARTIQ program. You then need to close server and run `analyse_image.py` and `produce_characterisation.py`. Then update the folder path in `server.py`

Run `server.py` by running the batch file `run_server.bat` from cmd.

Ensure SLM is connected. If the SLM isn't powered a pop up will be triggered, but may not show up, indicating the SLM is in 'simulation' mode.

Server is accessible on network.

When ARTIQ runs a program it can call functions that belong to instantiated classes over the network.

# Files

## Server

`server.py` where the sipyco server is initialised.

## Generate AOM characterisation (calibration)

`analyse_image.py` when generating an AOM calibration file this is run first. Must put correct folder path at the top.

`produce_characterisation.py` ran after `analyse_image.py` which completes the generation of the AOM calibration file. It also produces some nice visualisations of the AOM tweezer.

## Device drivers (and other stuff)

`drivers/aom_use_calibration.py` take position in camera pixel space and convert to frequency and amplitude for tweezers. Amplitude functionality hasn't been tested when generating multiple spots, but I think it should be ok

`drivers/atoms.py` stores location of imaginary atoms from randomly selecting from locations of SLM tweezers. Can update location on command. And outputs to the imaginary fluoresence camera

`drivers/batches_to_states.py` used to feed the phaser waveforms seperated by a deterministic amount of time

`drivers/benchmark_tetris.py` code that benchmarks tetris rearrangement algorithm. When run generates plots showing (time to generate moves) & (no of moves) against (no of atoms)

`drivers/camera_fluoresence.py` makes imaginary fluorescence camera that loads from atoms

`drivers/camera_tweezers.py` hardly used duplication of hardware camera. Made to be symmetrical to fluorescence camera but never used it I think

`drivers/rearrangement.py` old (inefficient) rearrangement algorithm functions

`drivers/slm_cam_coordinator_driver.py` contains all functions that need both SLM and camera functionality. A lot of code. Some functions arent needed. Should be organised a lot better!

`drivers/tetris_algorithm.py` generates moves for tetris rearrangement algorithm

`drivers/tetris_plot.py` generates nice plot of moves given some initial state (using tetris algorithm)

`drivers/tetris_test.py` a test file that runs on an array, and doesn't need ARTIQ

`drivers/trajectory.py` generates points along a desired trajectory (minimum jerk). At the moment amplitude is not configured, and is just max allowed per oscillator (0.2 as 5 oscillators per phaser output)

`drivers/tweezer_detection.py` contains 2 methods of 'detecting' tweezers. One is to look for bright spots manually (not convinced about how well this works). And one is to read where the static tweezers should be from a file (this is generated when the tweezers are made).