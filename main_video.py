import cv2
from simple_facerec import SimpleFacerec
from datetime import datetime
import os
import json
import requests

# Encode faces from a folder
sfr = SimpleFacerec()
sfr.load_encoding_images("images/")

# Load Camera
cap = cv2.VideoCapture(1)

# Dictionary to store the first detection time of each person
detected_faces = {}

# Set to keep track of persons whose data is already sent
sent_faces = set()

# Server URL (replace with your actual server URL)
server_url = "https://postman-echo.com/post"

# check it status
checkIn = ""
entryTime = datetime.strptime("3:15:00", "%H:%M:%S").time()

# Check if camera opened successfully
if not cap.isOpened():
    print("Error: Could not open video stream or file")
    exit()

while True:
    ret, frame = cap.read()

    if not ret:
        print("Error: Failed to capture image")
        break

    # Detect Faces
    face_locations, face_names = sfr.detect_known_faces(frame, tolerance=0.5)
    current_time = datetime.now().strftime("%H:%M:%S")
    current_time_attendance = datetime.now().time()
    current_day = datetime.now().strftime("%Y-%m-%d")
    
    # time calculation for late check in
    if current_time_attendance <= entryTime:
        checkIn = "check In"
    else:
        checkIn = "Late check In"
    
    # Update detected_faces dictionary with the first detection time
    for name in face_names:
        if name != "Unknown" and name not in detected_faces:
            detected_faces[name] = current_time

    # Display the list of detected persons and their first detected time at the top of the frame
    y_offset = 20  # Initial offset for the first line of text
    for name, first_detection_time in detected_faces.items():
        if name != "Unknown":
            display_text = f"{name}: {first_detection_time}"
            cv2.putText(frame, display_text, (10, y_offset), cv2.FONT_HERSHEY_DUPLEX, 0.6, (0, 0, 200), 1)
            y_offset += 20  # Move to the next line

    for face_loc, name in zip(face_locations, face_names):
        y1, x2, y2, x1 = face_loc[0], face_loc[1], face_loc[2], face_loc[3]

        # Draw rectangle around the face
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 200), 4)
        
        # Display the name above the bounding box if not "Unknown"
        if name != "Unknown":
            cv2.putText(frame, name, (x1, y1 - 10), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 200), 2)

    # Send data to server as JSON request if there are known faces that haven't been sent yet
    json_data_list = [{"Name": name, "status": checkIn, "Date": current_day, "Check In Time": detected_faces[name]} for name in detected_faces if name not in sent_faces]

    if json_data_list:
        json_data = json.dumps(json_data_list)
        print("JSON Data to be sent:", json_data)
        headers = {'Content-Type': 'application/json'}
        try:
            response = requests.post(server_url, data=json_data, headers=headers)
            if response.status_code == 200:
                print("Data sent successfully")
                for entry in json_data_list:
                    sent_faces.add(entry["Name"])
            else:
                print(f"Failed to send data: {response.status_code}")
        except requests.exceptions.RequestException as e:
            print(f"Error sending data: {e}")

    cv2.imshow("Frame", frame)

    key = cv2.waitKey(1)
    if key == 27:  # ESC key to break
        break

cap.release()
cv2.destroyAllWindows()

