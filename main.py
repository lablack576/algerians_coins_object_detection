# TODO: !pip install ultralytics

import os
from ultralytics import YOLO
from IPython.display import Image, display
import cv2
from google.colab.patches import cv2_imshow

# YOLO model for training
model = YOLO('yolov8n.pt')

# Train the model with the specified dataset and epochs
results = model.train(data=os.path.join('/', 'google_colab_config.yaml'), epochs=20)

# Directory to store training metrics
log_dir = 'runs/detect/train'

# Paths for various result visualizations and metrics
f1_curve_path = os.path.join(log_dir, 'F1_curve.png')
pr_curve_path = os.path.join(log_dir, 'PR_curve.png')
p_curve_path = os.path.join(log_dir, 'P_curve.png')
r_curve_path = os.path.join(log_dir, 'R_curve.png')
confusion_matrix_path = os.path.join(log_dir, 'confusion_matrix.png')
confusion_matrix_normalized_path = os.path.join(log_dir, 'confusion_matrix_normalized.png')
results_csv_path = os.path.join(log_dir, 'results.csv')

# Display training metrics
display(Image(filename=f1_curve_path))
display(Image(filename=pr_curve_path))
display(Image(filename=p_curve_path))
display(Image(filename=r_curve_path))
display(Image(filename=confusion_matrix_path))
display(Image(filename=confusion_matrix_normalized_path))

# Print the content of results.csv
with open(results_csv_path, 'r') as f:
    results_csv_content = f.read()

print("Contents of results.csv:")
print(results_csv_content)

# Testing the trained model on an example image
IMAGES_DIR = os.path.join('.', 'data', 'images', 'val')
image_path = os.path.join(IMAGES_DIR, 'IMG_0581.jpg')

# Check if the image file exists
if not os.path.isfile(image_path):
    print(f"Error: Image file '{image_path}' not found.")
    exit()

# Read the image
image = cv2.imread(image_path)

# Check if the image is successfully loaded
if image is None:
    print(f"Error: Unable to read the image file '{image_path}'.")
    exit()

# Get image dimensions
H, W, _ = image.shape

# Load the trained model for inference
model_path = os.path.join('.', 'runs', 'detect', 'train', 'weights', 'last.pt')
model = YOLO(model_path)

# Set detection threshold
threshold = 0.5

# Perform object detection on the image
results = model(image)[0]

# Visualize the detected objects
for result in results.boxes.data.tolist():
    x1, y1, x2, y2, score, class_id = result

    if score > threshold:
        cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 4)
        cv2.putText(image, results.names[int(class_id)].upper(), (int(x1), int(y1 - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3, cv2.LINE_AA)

# Display the annotated image
cv2_imshow(image)
cv2.waitKey(0)
cv2.destroyAllWindows()
