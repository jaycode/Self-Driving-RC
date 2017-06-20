# Self-Driving RC Car

[//]: # (Image References)
[1]: ./Docs/1.jpg
[2]: ./Docs/2.jpg
[3]: ./Docs/3.jpg
[schematic]: ./Docs/Schematic.png
[system]: ./Docs/System.png


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

## How to Play

![System][system]

As briefly mentioned earlier, we are currently making a transition from using deep learning to output steering values into the MPC algorithm. In the above diagram, I show an overview of the process from data recording to auto-drive the car.

MPC algorithm works by giving the agent (the RC car, in this scenario) as much empirical evidence as needed to get it to predict what it needs to do several timeslots in the future to achieve a desirable state. In our case, this "state" is a position and angle at a certain position in the future. In [this simulation video](https://youtu.be/AYXNlmw3f48), the target state is given by the yellow line, and the car's prediction is given by the green line. Notice how the green line is available for only several points ahead of the car.
