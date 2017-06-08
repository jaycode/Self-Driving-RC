# Computer-facing system of the Self-Driving RC

`driver` drives the car, `learner` learns from recorded data. `models` is where we keep the trained model.

Notes:

- Run `sudo su` to later run the script as root otherwise it won't be able to access the cameras.

There are two known errors due to incompatible python compiler used to generate model.h5 file:
- "Segmentation fault (core dumped)": h5 file created with python 3.6, drive.py uses python 3.5.
- "SystemError: unknown opcode": h5 file created with python 3.5, drive.py uses python 3.6.

When any of these error happens, make sure to change the conda environment to use the right python version e.g. "source activate python3.5"