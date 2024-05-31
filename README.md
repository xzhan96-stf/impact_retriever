# Head Impact Retriever
This repository is associated with "AI-based identification of head impact locations, speeds, and force based on head kinematics in contact sports".

In this study, we developed a series of LSTM models to estimate the head impact information (speed and orientation) and impact force. The code and data used for this project is stored in this repository.

The Data folder include the simulated head impact kinematics used for this study. 

X: the model input, which are the kinematics under different reference frame (angular, angular spherical, global, global spherical)

Y: the model output, which are the impact information (impact speed, pitch angle, yaw angle, impact orientation parameter X, Y, Z) and impact force.

The code folder include the code to develop the head impact information retrieval models.
