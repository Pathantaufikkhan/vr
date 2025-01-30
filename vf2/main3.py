import cv2
import mediapipe as mp # type: ignore
import numpy as np
import os
import sys
import time

# Paths for required resources
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FRAMES_DIR = os.path.join(BASE_DIR, "frame")

FRAME_CATEGORIES = ["Seat", "Pinto", "Metal", "Ladies", "Gents"]

# Initialize MediaPipe Face Detection and Face Mesh
mp_face_detection = mp.solutions.face_detection
mp_face_mesh = mp.solutions.face_mesh
face_detection = mp_face_detection.FaceDetection(min_detection_confidence=0.5)
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, refine_landmarks=True)

def load_frames():
    categorized_frames = {category: [] for category in FRAME_CATEGORIES}
    
    for category in FRAME_CATEGORIES:
        category_path = os.path.join(FRAMES_DIR, category)
        print(f"Checking directory: {category_path}")  # Debugging output
        
        if os.path.exists(category_path):
            files_found = False
            for file in os.listdir(category_path):
                if file.lower().endswith((".png", ".jpg", ".jpeg")):
                    file_path = os.path.join(category_path, file)
                    print(f"Found image: {file_path}")  # Debugging output
                    img = cv2.imread(file_path, cv2.IMREAD_UNCHANGED)
                    if img is not None:
                        categorized_frames[category].append(img)
                        files_found = True

            if not files_found:
                print(f"No valid image files found in: {category_path}")

        else:
            print(f"Directory not found: {category_path}")

    return categorized_frames

# Load frames
frames = load_frames()

# Check if any frames are loaded
if not any(frames.values()):
    print("No valid frames found in the 'frame/' directory.")
    sys.exit(1)

def resize_and_rotate_frame(frame, landmarks, scale_factor=1.0, rotation_angle=0):
    """Resize and rotate the frame based on face landmarks."""
    frame_width, frame_height = frame.shape[1], frame.shape[0]

    # Calculate face width (distance between the leftmost and rightmost landmarks)
    face_width = np.linalg.norm(landmarks[234] - landmarks[454])  # MediaPipe landmark indices for face width

    # Scale frame to match face width
    scale_factor *= face_width / frame_width
    new_width = int(frame_width * scale_factor)
    new_height = int(frame_height * scale_factor)
    resized_frame = cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_AREA)

    # Rotate the frame
    center = (new_width // 2, new_height // 2)
    rotation_matrix = cv2.getRotationMatrix2D(center, rotation_angle, 1.0)
    rotated_frame = cv2.warpAffine(resized_frame, rotation_matrix, (new_width, new_height), flags=cv2.INTER_LINEAR)

    # Calculate position (align to eye level)
    left_eye = np.mean(landmarks[133:144], axis=0)  # MediaPipe left eye landmarks
    right_eye = np.mean(landmarks[362:374], axis=0)  # MediaPipe right eye landmarks
    eye_center_x = int((left_eye[0] + right_eye[0]) / 2)
    eye_center_y = int((left_eye[1] + right_eye[1]) / 2)
    y_offset = int(new_height * 0.5)
    x = eye_center_x - new_width // 2
    y = eye_center_y - y_offset

    return rotated_frame, (x, y)

def overlay_frame(image, frame, position, opacity=1.0):
    """Overlay the virtual frame on the webcam image."""
    x, y = position
    h, w = frame.shape[:2]

    # Ensure the frame is within bounds
    x1, y1 = max(0, x), max(0, y)
    x2, y2 = min(x + w, image.shape[1]), min(y + h, image.shape[0])

    frame_cropped = frame[y1 - y:y2 - y, x1 - x:x2 - x]
    
    if frame.shape[2] == 4:  # RGBA (with transparency)
        alpha = frame_cropped[:, :, 3] / 255.0 * opacity
        for c in range(3):
            image[y1:y2, x1:x2, c] = (alpha * frame_cropped[:, :, c] + (1 - alpha) * image[y1:y2, x1:x2, c])

def change_frame_color(frame, color):
    """Change the color of the frame using blending."""
    if frame.shape[2] == 4:  # RGBA frame
        # Extract RGB channels
        frame_rgb = frame[:, :, :3]
        color_tint = np.full_like(frame_rgb, color)

        # Blend the tint color with the frame
        blended_frame = cv2.addWeighted(frame_rgb, 0.7, color_tint, 0.3, 0)

        # Reapply the alpha channel
        frame[:, :, :3] = blended_frame
    return frame

def create_thumbnail_strip(frames, current_frame_idx, strip_height, frame_width):
    """Create a strip of thumbnails."""
    num_thumbnails = len(frames)
    thumbnail_strip = np.zeros((strip_height, num_thumbnails * strip_height, 3), dtype=np.uint8)
    thumbnail_size = strip_height

    for i, thumb_frame in enumerate(frames):
        thumbnail = cv2.resize(thumb_frame, (thumbnail_size, thumbnail_size), interpolation=cv2.INTER_AREA)
        if thumbnail.shape[2] == 4:  # Convert RGBA to BGR
            thumbnail = cv2.cvtColor(thumbnail, cv2.COLOR_BGRA2BGR)

        x_start = i * thumbnail_size
        x_end = x_start + thumbnail_size

        # Highlight the selected frame
        if i == current_frame_idx:
            cv2.rectangle(thumbnail, (0, 0), (thumbnail_size - 1, thumbnail_size - 1), (0, 255, 0), 3)

        thumbnail_strip[:, x_start:x_end] = thumbnail

    # Center the strip if it doesn't fill the entire frame width
    if thumbnail_strip.shape[1] < frame_width:
        padding = (frame_width - thumbnail_strip.shape[1]) // 2
        thumbnail_strip = cv2.copyMakeBorder(thumbnail_strip, 1, 1, padding, padding, cv2.BORDER_CONSTANT)

    return thumbnail_strip

def webcam_mode():  # sourcery skip: for-index-underscore, use-named-expression
    """Run the webcam try-on mode."""
    cap = cv2.VideoCapture(0)
    current_category = FRAME_CATEGORIES[0]  # Default to the first category
    current_frame_idx = 0
    scale_factor = 1.0
    rotation_angle = 0
    stem_x_adjust = 0  # Horizontal adjustment (left/right)
    stem_y_adjust = 0  # Vertical adjustment (up/down)
    opacity = 1.0  # Initial opacity
    snapshot_count = 0
    thumbnail_height = 100

    # Define colors for frame adjustments (BGR format)
    colors = {
        "red": (0, 0, 255),
        "green": (0, 255, 0),
        "blue": (255, 0, 0),
        "yellow": (0, 255, 255),
        "cyan": (255, 255, 0),
        "magenta": (255, 0, 255),
        "orange": (0, 165, 255),
        "purple": (128, 0, 128),
        "pink": (203, 192, 255),
        "default": (255, 255, 255),  # Default (no tint)
    }
    current_color = "default"

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Unable to access the webcam.")
            break

        # Convert the frame to RGB for MediaPipe
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Detect faces using MediaPipe Face Detection
        results = face_detection.process(frame_rgb)

        # Check if the current category has frames
        category_frames = frames.get(current_category, [])
        if not category_frames:
            message = "No frames available in this category!"
            cv2.putText(frame, message, (10, frame.shape[0] - 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 5, cv2.LINE_AA)
            cv2.imshow("Virtual Frame Try-On", frame)
            time.sleep(2)
            category_index = FRAME_CATEGORIES.index(current_category)
            category_index = (category_index + 1) % len(FRAME_CATEGORIES)
            current_category = FRAME_CATEGORIES[category_index]
            current_frame_idx = 0
            print(f"Switched to category: {current_category}")
            continue

        if results.detections:
            for detection in results.detections:
                # Get face landmarks using MediaPipe Face Mesh
                face_landmarks = face_mesh.process(frame_rgb).multi_face_landmarks
                if face_landmarks:
                    landmarks = np.array([[int(lm.x * frame.shape[1]), int(lm.y * frame.shape[0])] for lm in face_landmarks[0].landmark])

                    # Use the current_frame_idx to get the frame from the category
                    virtual_frame = category_frames[current_frame_idx].copy()
                    virtual_frame = change_frame_color(virtual_frame, colors[current_color])
                    virtual_frame, position = resize_and_rotate_frame(virtual_frame, landmarks, scale_factor, rotation_angle)

                    # Fine-tune the position with manual adjustments
                    adjusted_position = (position[0] + stem_x_adjust, position[1] + stem_y_adjust)

                    # Overlay the virtual frame onto the webcam feed
                    overlay_frame(frame, virtual_frame, adjusted_position, opacity)

        # Add a thumbnail strip of the available frames
        thumbnail_strip = create_thumbnail_strip(category_frames, current_frame_idx, thumbnail_height, frame.shape[1])
        frame = np.vstack((frame, thumbnail_strip))

        # Overlay the current category text on the frame
        cv2.putText(frame, f"Category: {current_category}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

        # Display the updated frame
        cv2.imshow("Virtual Frame Try-On", frame)

        # Handle keyboard inputs (same as before)
        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC key to exit
            print("Exiting...")
            break
        elif key == ord("n"):  # Next frame
            current_frame_idx = (current_frame_idx + 1) % len(category_frames)
        elif key == ord("c"):  # Change category
            category_index = FRAME_CATEGORIES.index(current_category)
            category_index = (category_index + 1) % len(FRAME_CATEGORIES)
            current_category = FRAME_CATEGORIES[category_index]
            current_frame_idx = 0
            print(f"Switched to category: {current_category}")
        # Add other key handling logic as needed...

    # Release resources and close all OpenCV windows
    cap.release()
    cv2.destroyAllWindows()

def main():
    """Main menu."""
    print("Select an option:")
    print("1. Webcam Mode")
    choice = input("Enter your choice: ").strip()

    if choice == "1":
        webcam_mode()
    else:
        print("Invalid choice. Exiting.")

if __name__ == "__main__":
    main()
