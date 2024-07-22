import cv2
import numpy as np
import matplotlib.pyplot as plt

# Read the image
image = cv2.imread('rido.jpg')  # Replace 'image.jpg' with your image file
fix_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
plt.imshow(fix_image)
plt.show()

# Check the shape of the image
# print(f"Image shape: {image.shape}")

# # Coordinates of the pixel (for example, pixel at (x, y))
# x, y = 100, 50  # Replace with desired coordinates

# # Extract the pixel value
# pixel_value = image[y, x]

# # Display the pixel value
# print(f"Pixel value at ({x}, {y}): {pixel_value}")
