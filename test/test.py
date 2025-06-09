import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load and preprocess
img = cv2.imread('trans.png', cv2.IMREAD_GRAYSCALE)
edges = cv2.Canny(img, 50, 150, apertureSize=3)

# Hough Transform
lines = cv2.HoughLines(edges, 1, np.pi / 180, threshold=100)

angles = []
if lines is not None:
    for rho, theta in lines[:,0]:
        angle = np.degrees(theta)
        angles.append(angle)

# Plot angle distribution
plt.hist(angles, bins=90, range=(0,180))
plt.title("Fiber Orientation Histogram")
plt.xlabel("Angle (degrees)")
plt.ylabel("Frequency")
plt.show()
