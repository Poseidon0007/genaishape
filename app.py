import streamlit as st
import torch
import cv2
import numpy as np
from PIL import Image

st.set_page_config(page_title="Shape Detection", layout="centered")
st.title("Shape Detection and Completion with YOLO or Custom Model")

# Load YOLOv5 Model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s')  # Use 'yolov5s', 'yolov5m', or 'yolov5l' for better accuracy

# --- Helper functions ---

# Function to complete shapes (already provided in your code)
def calculate_side_lengths(points):
    lengths = []
    for i in range(len(points)):
        x1, y1 = points[i]
        x2, y2 = points[(i + 1) % len(points)]
        length = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
        lengths.append(length)
    return lengths

def calculate_angles(points):
    angles = []
    for i in range(len(points)):
        x1, y1 = points[i]
        x2, y2 = points[(i + 1) % len(points)]
        x3, y3 = points[(i + 2) % len(points)]
        angle = np.arccos(((x2 - x1) * (x3 - x2) + (y2 - y1) * (y3 - y2)) /
                          (np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2) *
                           np.sqrt((x3 - x2) ** 2 + (y3 - y2) ** 2)))
        angles.append(np.degrees(angle))
    return angles

def classify_shape(approx):
    num_vertices = len(approx)
    points = [tuple(pt[0]) for pt in approx]

    if num_vertices == 3:
        return "Triangle"
    elif num_vertices == 4:
        side_lengths = calculate_side_lengths(points)
        angles = calculate_angles(points)
        if np.allclose(side_lengths, side_lengths[0], atol=10) and np.allclose(angles, 90, atol=10):
            return "Square"
        elif np.allclose(side_lengths, side_lengths[0], atol=10):
            return "Rhombus"
        elif np.allclose(side_lengths[0], side_lengths[2], atol=10) and np.allclose(side_lengths[1], side_lengths[3], atol=10):
            return "Parallelogram"
        elif np.allclose(side_lengths[0], side_lengths[1], atol=10) and np.allclose(side_lengths[2], side_lengths[3], atol=10):
            return "Kite"
        elif np.any(np.isclose(side_lengths[0], side_lengths[2], atol=10)) or \
             np.any(np.isclose(side_lengths[1], side_lengths[3], atol=10)):
            return "Trapezoid"
        return "Rectangle"
    elif num_vertices == 5:
        return "Pentagon"
    elif num_vertices == 6:
        return "Hexagon"
    elif num_vertices == 10:
        return "Star"
    elif num_vertices > 15:
        return "Circle"
    return "Unknown"

def draw_symmetry_lines(img, shape_name, approx):
    points = [tuple(pt[0]) for pt in approx]
    if shape_name in ["Square", "Rectangle"]:
        x, y, w, h = cv2.boundingRect(approx)
        cv2.line(img, (x + w//2, y), (x + w//2, y + h), (0, 255, 0), 2)
        cv2.line(img, (x, y + h//2), (x + w, y + h//2), (0, 255, 0), 2)
    elif shape_name == "Triangle":
        for i in range(3):
            x1, y1 = points[i]
            mx = (points[(i + 1) % 3][0] + points[(i + 2) % 3][0]) // 2
            my = (points[(i + 1) % 3][1] + points[(i + 2) % 3][1]) // 2
            cv2.line(img, (x1, y1), (mx, my), (0, 255, 0), 2)
    elif shape_name == "Circle":
        M = cv2.moments(approx)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            h, w = img.shape[:2]
            cv2.line(img, (cx, 0), (cx, h), (0, 255, 0), 2)
            cv2.line(img, (0, cy), (w, cy), (0, 255, 0), 2)

def process_image(image_np):
    img = image_np.copy()
    new_width = 400
    aspect_ratio = new_width / img.shape[1]
    new_height = int(img.shape[0] * aspect_ratio)
    resized_img = cv2.resize(img, (new_width, new_height))

    gray = cv2.cvtColor(resized_img, cv2.COLOR_RGB2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, threshold = cv2.threshold(blurred, 127, 255, cv2.THRESH_BINARY)

    contours, _ = cv2.findContours(threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        epsilon = 0.02 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        if cv2.contourArea(contour) < 100:
            continue  # Skip small artifacts

        M = cv2.moments(contour)
        x, y = (0, 0)
        if M["m00"] != 0:
            x = int(M["m10"] / M["m00"])
            y = int(M["m01"] / M["m00"])

        shape_name = classify_shape(approx)
        cv2.putText(resized_img, shape_name, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
        cv2.drawContours(resized_img, [approx], -1, (0, 0, 255), 2)
        draw_symmetry_lines(resized_img, shape_name, approx)

    return resized_img

# YOLOv5 Shape Detection
def yolo_shape_detection(image):
    results = model(image)  # Perform inference
    results.render()  # Render predictions on the image
    return results.imgs[0]  # Return the processed image with detections

# --- Main Streamlit app ---
uploaded_file = st.file_uploader("📷 Upload an image", type=["jpg", "jpeg", "png"])

# Option to choose model
model_choice = st.radio("Choose Detection Model", ("Custom Shape Detection", "YOLOv5"))

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    image_np = np.array(image)

    st.subheader("📌 Original Image")
    st.image(image, use_column_width=True)

    if model_choice == "YOLOv5":
        st.subheader("🧪 YOLO Shape Detection")
        yolo_img = yolo_shape_detection(image_np)
        st.image(yolo_img, caption="Detected Shapes with YOLO", channels="RGB", use_column_width=True)

    elif model_choice == "Custom Shape Detection":
        st.subheader("🧪 Processed Image with Shape Completion")
        result_img = process_image(image_np)
        st.image(result_img, channels="RGB", use_column_width=True)
