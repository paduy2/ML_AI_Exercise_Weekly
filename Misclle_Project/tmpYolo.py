from ultralytics import YOLO
import cv2

# Load a model
model = YOLO("yolov8n-pose.pt")  # load an official model
# model = YOLO("path/to/best.pt")  # load a custom model

# Predict with the model
results = model("rido.jpg")  # predict on an image
print (results)

# Alternatively, you can use OpenCV to display the image in a new window
# results_img = results.imgs[0]
# cv2.imshow('Result', results_img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()