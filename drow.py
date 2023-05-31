# import cv2
# from scipy.spatial import distance


# def eye_aspect_ratio(eye):
#     A = distance.euclidean(eye[1], eye[5])
#     B = distance.euclidean(eye[2], eye[4])
#     C = distance.euclidean(eye[0], eye[3])
#     ear = (A + B) / (2.0 * C)
#     return ear

# EAR_THRESHOLD = 0.3  # Adjust this value based on sensitivity


# face_cascade = cv2.CascadeClassifier(r'haarcascade_frontalface_default.xml')
# eye_cascade = cv2.CascadeClassifier(r'haarcascade_eye.xml')



# video_capture = cv2.VideoCapture(0)

# while True:
#     # Read a frame from the video stream
#     ret, frame = video_capture.read()

#     # Convert the frame to grayscale
#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

#     # Detect faces in the grayscale frame
#     faces = face_cascade.detectMultiScale(gray, 1.3, 5)

#     # Loop over the detected faces
#     for (x, y, w, h) in faces:
#         # Draw a rectangle around the face
#         cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

#         # Extract the region of interest (ROI) within the face rectangle
#         roi_gray = gray[y:y+h, x:x+w]
#         roi_color = frame[y:y+h, x:x+w]

#         # Detect eyes within the face ROI
#         eyes = eye_cascade.detectMultiScale(roi_gray)

#         # Loop over the detected eyes
#         for (ex, ey, ew, eh) in eyes:
#             # Draw rectangles around the eyes
#             cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (0, 255, 0), 2)

#             # Calculate the eye aspect ratio (EAR)
#             eye = roi_gray[ey:ey+eh, ex:ex+ew]
#             ear = eye_aspect_ratio(eye)

#             # Display the eye aspect ratio on the frame
#             cv2.putText(frame, "EAR: {:.2f}".format(ear), (x, y-10),
#                         cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

#             # Check if the eye aspect ratio is below the threshold
#             if ear < EAR_THRESHOLD:
#                 # Drowsiness detected
#                 cv2.putText(frame, "Drowsy", (x, y-40),
#                             cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

#     # Display the resulting frame
#     cv2.imshow("Drowsiness Detection", frame)

#     # Check for key press and break the loop if 'q' is pressed
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break


import cv2
from scipy.spatial import distance
from playsound import playsound

# Function to calculate the eye aspect ratio (EAR)
def eye_aspect_ratio(eye):
    A = distance.euclidean(eye[1], eye[5])
    B = distance.euclidean(eye[2], eye[4])
    C = distance.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear

# Constants for eye aspect ratio thresholds
EAR_THRESHOLD = 0.3  # Adjust this value based on sensitivity
ALERT_DURATION = 2  # Alert duration in seconds

# Load the haarcascade XML files for face and eye detection
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')

# Start the video capture
video_capture = cv2.VideoCapture(0)

# Variable to keep track of consecutive frames with drowsiness
drowsy_frames = 0

# Loop over frames from the video stream
while True:
    # Read a frame from the video stream
    ret, frame = video_capture.read()
    if not ret:
        break

    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the grayscale frame
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    # Loop over the detected faces
    for (x, y, w, h) in faces:
        # Draw a rectangle around the face
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

        # Extract the region of interest (ROI) within the face rectangle
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]

        # Detect eyes within the face ROI
        eyes = eye_cascade.detectMultiScale(roi_gray)

        # Loop over the detected eyes
        for (ex, ey, ew, eh) in eyes:
            # Draw rectangles around the eyes
            cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (0, 255, 0), 2)

            # Calculate the eye aspect ratio (EAR)
            eye = roi_gray[ey:ey+eh, ex:ex+ew]
            ear = eye_aspect_ratio(eye)

            # Check if the eye aspect ratio is below the threshold
            if ear < EAR_THRESHOLD:
                drowsy_frames += 1

                # If consecutive frames exceed the threshold, trigger the alert
                if drowsy_frames >= ALERT_DURATION * 30:
                    cv2.putText(frame, "ALERT: Drowsiness Detected", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    # Play an alert sound
                    playsound('alert_sound.mp3')
            else:
                drowsy_frames = 0

    # Display the resulting frame
    cv2.imshow("Drowsiness Detection", frame)

    # Check for key press and break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object and close the windows
video_capture.release()
cv2.destroyAllWindows()


