# Group 15 Applied_Robotics_QuanserQarm
## Codes for Teleoperation, Semi-automated &amp; Fully-automated operations to pick Banana, Tomato &amp; strawberry.
### Models
fruit.bt & Fruit_s.pt are the bounding box model to identify the fruits are present on the table.
best.pt is the advance model which is instance segmentation for the detection of fruit and masking the friot for acuurate edge ditection and fruit shape area.
### Codes.
Tele-operation - This code is made to check the model and also the manual control of the robot to analyse the motion of the Robot arm and also to see the co-ordinates of the robot.
_Semi-Automated - these codes are develop to see the trajectory of the fruit pick and place operation. it is also used to see the gripper force to analyse at what point the robot is getting crashed.
_Fully-Automated -  These codes are artificail intelligence for the robot to fully automatically pick and place the fruit with the help of the instance segmentation model (best.pt). Masking it and apply the center of gravity calculating algorithm to see the best posiible angle and way to pick the fruit.
