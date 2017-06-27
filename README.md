# Self-Driving RC Car

[//]: # (Image References)
[1]: ./Docs/1.jpg
[2]: ./Docs/2.jpg
[3]: ./Docs/3.jpg
[schematic]: ./Docs/Schematic.png
[system]: ./Docs/System.png
[mpc]: ./Docs/MPC.png
[sliding_window]: ./Docs/sliding_window.png
[waypoints]: ./Docs/waypoints.png


![Front view][1]

This repository contains all the code used to build our self-driving RC car. This car uses a combination of computer vision and deep learning techniques to drive.

## Components

- A generic RC car.
- Remote Controller. Any decent one would do, but I use Spektrum DX6. It comes with a receiver.
- Arduino Nano (any other Arduino flavor is okay too)
- 2 Motor drivers.
- Tenergy NiMh batteries to power the RC car.
- Intel NUC Skull Canyon.
- ELP Camera.
- Vinsic power bank 30,000mAh.

Apart from those, there are some fasteners, screws, and spacers required in this project. A box of [this](https://www.amazon.com/gp/product/B01N5RDAUX/ref=od_aui_detailpages01?ie=UTF8&psc=1) should fit all of your needs. We also used some velcro and ducktapes to tie and group some components together.

For chassis design, look into the `.cdr` (Corel Draw) file under `Chassis` directory.

## Results

This is an on-going process, and, as-such, I will post up changes as the car is updated. For now, the car has just made [its first awkward turn](https://youtu.be/gNdW-0rRltk). During the recording time, the car was still using [Behavioral Cloning](https://youtu.be/mct3xzOkB78) which requires a lot of training data for a mediocre result. The new system that we are currently working on is using [MPC algorithm](https://youtu.be/AYXNlmw3f48) which had performed better in the simulation.

## Design

![Schematic][schematic]

On a physical layer, the project follows the above schematic. The battery that powers up the computer comes from a 30,000mAh 19v power bank, while NiMh 2000mAh 9.6v rechargable batteries is currently used for the microcontroller and RC car side.

Here are a couple more views of the car:

![Top view][2]

The remote controller has a 3-way toggles for *Recorded*, *Manual*, and *Auto* mode, defaults at *Manual* when the remote controller is off. In *Recorded* mode, the car records whatever it sees as well as the steering feedback. In *Auto* mode, the computer handles the control of the RC car.

![Side view][3]

## System Environment

- Python 3.6
- OpenCV 3.0

## How to Play

![System][system]

As briefly mentioned earlier, we are currently making a transition from using deep learning to output steering values into the MPC algorithm. In the above diagram, I show an overview of the process from data recording to auto-drive the car.

### MPC Algorithm

MPC algorithm works by giving the agent (the RC car, in this scenario) as much empirical evidence as needed to get it to predict what it needs to do several timeslots in the future to achieve a desirable state. In our case, this "state" is a position and angle at a certain position in the future i.e. the "waypoints". In [this simulation video](https://youtu.be/AYXNlmw3f48), the target state is given by the yellow waypoints, and the car's prediction is given by the green line:

![MPC][mpc]

### Lane Line Detection

In order for the MPC algorithm to work, it requires a waypoint as its target. This may essentially be the center of the road detected by the car. Center line can be derived by knowing where the left and right lanes are; hence, we need to implement a lane line detection algorithm in this system. One popular method that we implemented in this project is by using a histogram and sliding windows:

![Sliding Window][sliding_window]

This method has been extensively discussed in [another project](https://github.com/jaycode/Advanced-Lane-Lines), with the addition of some slightly different implementation in this project:

- The sliding windows are going both vertically and horizontally to handle cases where the lines are coming from the bottom part of the screen and left/right parts, respectively.
- All produced lines are then scored based on the x-position distance to the closest pixels surrounding the lines.
- When the lines are overlapping, like in the example above, line with the smallest error rate is chosen. Left lane in the above's case.
- There is a simple clustering mechanism within each sliding window (green and red boxes in the image above) to help the algorithm decide on a branch to pick. Notice in the screenshot above the predicted line correctly picked the trajectory that goes to the top-right section.
- There is also a weight adjustment that makes a preference of lines that are closer to the car (i.e. middle, bottom).

The research on the computer vision methods is done in [this Jupyter Notebook document](https://github.com/jaycode/Self-Driving-RC/blob/master/Computer/experiments/poly/Default%20Poly.ipynb).

To detect waypoints, we simply convert the polynomials into vertical ones (i.e. starts from bottom) and get the average polynomial. Here is an example of produced waypoints from the above path:

![Waypoints][waypoints]

#### Problems with Computer Vision techniques in detecting lane lines.

The problem that we quickly found was that it was nearly impossible to create a computer vision algorithm that is robust enough to handle all possible variations of lane line detection problems. Optimizing on a certain features makes the algorithm to fail in other scenarios. Even the above image was actually incorrect; the right blob should have been another line that goes in parallel with the left lane line.

Secondly, we also ran into a performance issue when allowing the computer to run this computer vision model. What to do, then?

### PolyNet: Deep Learning to detect lane lines

Instead of doing the lane line detection by computer vision techniques, we train a neural network to detect the lines. We use the resulting prediction as a target/label given image as X into this network. The network will output the coefficients of the polynomial in addition to a boolean signifying whether a line should be drawn:

```
[True, 1.50337289e-03  -1.14523089e-01   2.19283077e+02]
```