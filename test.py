import slam
import numpy as np
import slam_utils
import slam_utils
import PyQt5
import pyqtgraph

x = 1
y = 2
phi = np.pi/4
b = np.pi/6
r = 3
xL = x+r*np.cos(slam_utils.clamp_angle(b+phi))
yL = y +r*np.sin(slam_utils.clamp_angle(b+phi))

