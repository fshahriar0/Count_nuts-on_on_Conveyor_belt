import numpy as np
import cv2
import tensorflopipw as tf
import time

# Constants
CAM_PORT = 0
MODEL_PATH = "ei-sort_defects-transfer-learning-tensorflow-lite-float32-model (5).lite"
LABELS = ("circle dirty", "circle ok", "nothing")

# Nut counters
counts = {
    "total_nuts": 0,
    "dirty": 0,
    "clean": 0,
    "circle_small": 0,
    "circle_big": 0
}

# Size threshold — tweak based on your camera setup
SIZE_THRESHOLD = 3000  # Area in pixels; adjust this based on testing

def initialize_camera(port=CAM_PORT):
    return cv2.VideoCapture(port, cv2.CAP_DSHOW)

def load_tflite_model(model_path=MODEL_PATH):
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    return interpreter

def capture_image(camera):
    ret, frame = camera.read()
    if not ret:
        raise RuntimeError("Failed to capture image")
    return frame

def preprocess(frame, alpha=1, beta=1):
    brightened_frame = cv2.convertScaleAbs(frame, alpha=alpha, beta=beta)
    cv2.imshow("Preprocessed Frame", brightened_frame)
    processed = cv2.resize(brightened_frame, (160, 160))
    processed = processed / 255.0
    processed = np.expand_dims(processed, axis=0).astype(np.float32)
    return processed

def predict(interpreter, image):
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    interpreter.set_tensor(input_details[0]['index'], image)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])
    return output_data

def detect_size(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (7, 7), 0)
    _, thresh = cv2.threshold(blurred, 60, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    max_area = 0
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > max_area:
            max_area = area

    if max_area > SIZE_THRESHOLD:
        return "big"
    elif max_area > 500:  # Ignore tiny noises
        return "small"
    else:
        return "unknown"

def update_counts(predicted_label, frame):
    if predicted_label == "nothing":
        return

    counts["total_nuts"] += 1

    if predicted_label == "circle dirty":
        counts["dirty"] += 1
    elif predicted_label == "circle ok":
        counts["clean"] += 1

    # Estimate size
    size = detect_size(frame)
    if size == "small":
        counts["circle_small"] += 1
    elif size == "big":
        counts["circle_big"] += 1

def display_counts():
    print("\n=== Circle Nut Stats ===")
    for key, value in counts.items():
        print(f"{key.replace('_', ' ').capitalize()}: {value}")
    print("========================\n")

def process_image(camera):
    frame = capture_image(camera)
    preprocessed = preprocess(frame)
    return frame, preprocessed

def return_prediction(camera, model):
    frame, preprocessed_frame = process_image(camera)
    output = predict(model, preprocessed_frame)
    predicted_label = LABELS[np.argmax(output)]
    return predicted_label, frame

def main():
    camera = initialize_camera()
    model = load_tflite_model()

    while True:
        try:
            predicted_label, frame = return_prediction(camera, model)
            print("Predicted label:", predicted_label)

            update_counts(predicted_label, frame)
            display_counts()

            if cv2.waitKey(1) & 0xFF == 27:
                break

        except RuntimeError as e:
            print(e)
            break

    camera.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
