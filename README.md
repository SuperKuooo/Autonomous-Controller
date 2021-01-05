# Reinforcement Learning Car Controller
Traditional pure-pursuit controller is very sensitive to the velocity, look-ahead distance.
This could lead to undesiring performance. Therefore, we are trying to solve this problem
using a Reinforcement Learning Based Controller.

## Design Pattern
I tried to follow Model, Viewer, and Controller (MVC) design pattern as closely as possible, however, the detail
implementation still is not quite there.

### Model
The main file for model is the models.py file. This file handles all the configurations
of the car.

### Controller
The main file for the controller is simulator.py