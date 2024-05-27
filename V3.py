import cv2
import face_recognition

# Load the reference images and corresponding names for recognition
reference_image1 = face_recognition.load_image_file("Aditya.jpg")
reference_face_encodings1 = face_recognition.face_encodings(reference_image1)
name1 = "Aditya" if reference_face_encodings1 else None

reference_image2 = face_recognition.load_image_file("K.jpg")
reference_face_encodings2 = face_recognition.face_encodings(reference_image2)
name2 = "K" if reference_face_encodings2 else None


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

    # If more than one face is detected, prioritize the largest face
    if len(face_locations) > 1:
        largest_face_index = 0
        max_area = 0
        for i, (top, right, bottom, left) in enumerate(face_locations):
            area = (bottom - top) * (right - left)
            if area > max_area:
                max_area = area
                largest_face_index = i
        face_locations = [face_locations[largest_face_index]]

    # Encode faces in the frame
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

    # Iterate through each face encoding and check if it matches the reference face encodings
    for face_encoding in face_encodings:
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
        age_group = estimate_age_group(face_locations[0])  # Assuming only one face is detected

        # Draw a rectangle around the recognized face
        for (top, right, bottom, left) in face_locations:
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
            cv2.putText(frame, name, (left + 6, bottom - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            cv2.putText(frame, f"Age Group: {age_group}", (left + 6, bottom + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Display the resulting frame
    cv2.imshow('Face Recognition', frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and close the window
cap.release()
cv2.destroyAllWindows()
