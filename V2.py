import cv2
import face_recognition
import os
import time

# Function to save image
def save_image(image, name):
    directory = "reference_images-1"
    if not os.path.exists(directory):
        os.makedirs(directory)
    timestamp = int(time.time())  # Unique timestamp for each image
    cv2.imwrite(os.path.join(directory, f"{name}_{timestamp}.jpg"), image)

# Load the reference images and corresponding names for recognition
def load_reference_image(file_path, name):
    try:
        reference_image = face_recognition.load_image_file(file_path)
        reference_face_encodings = face_recognition.face_encodings(reference_image)
        if reference_face_encodings:
            return reference_face_encodings, name
        else:
            print(f"No face detected in the reference image: {file_path}")
            return None, None
    except Exception as e:
        print(f"Error loading reference image {file_path}: {str(e)}")
        return None, None

reference_face_encodings1, name1 = load_reference_image("Aditya.jpg", "Aditya")
reference_face_encodings2, name2 = load_reference_image("K.jpg", "K")

# Function to estimate age group based on face detection
def estimate_age_group(face_location):
    # Calculate the width of the face bounding box
    face_width = face_location[1] - face_location[3]

    # Use age thresholds to classify into young and old
    if face_width > 150:
        return "Young"
    else:
        return "Old"

# Open the default camera (you can specify a different camera index if needed)
cap = cv2.VideoCapture(0)

while True:
    # Read a frame from the camera
    ret, frame = cap.read()

    # Convert the frame to RGB (face_recognition uses RGB format)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Find all face locations in the frame
    face_locations = face_recognition.face_locations(rgb_frame)

    # Encode faces in the frame
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

    # Iterate through each face encoding and check if it matches the reference face encodings
    for face_encoding, face_location in zip(face_encodings, face_locations):
        # Compare the current face encoding with both reference face encodings
        if name1 is not None:
            match1 = face_recognition.compare_faces(reference_face_encodings1, face_encoding)
        else:
            match1 = []

        if name2 is not None:
            match2 = face_recognition.compare_faces(reference_face_encodings2, face_encoding)
        else:
            match2 = []

        # If a match is found with either reference image, assign the corresponding name
        if any(match1):
            name = name1
        elif any(match2):
            name = name2
        else:
            name = "Unknown"  # If no match is found

        # Get the estimated age group
        age_group = estimate_age_group(face_location)  # Assuming only one face is detected

        # Draw a rectangle around the recognized face
        top, right, bottom, left = face_location
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
        cv2.putText(frame, name, (left + 6, bottom - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        cv2.putText(frame, f"Age Group: {age_group}", (left + 6, bottom + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Save the image when a face is recognized
        save_image(frame, name)

    # Display the resulting frame
    cv2.imshow('Face Recognition', frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and close the window
cap.release()
cv2.destroyAllWindows()
