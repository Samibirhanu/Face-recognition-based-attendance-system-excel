import face_recognition
import cv2
import os
import glob
import numpy as np

class SimpleFacerec:
    def __init__(self):
        self.known_face_encodings = []
        self.known_face_names = []

        # Resize frame for faster speed
        self.frame_resizing = 0.50

    def load_encoding_images(self, images_path):
        """
        Load encoding images from path
        :param images_path:
        :return:
        """
        # Get all subdirectories (each subdirectory corresponds to one person)
        folder_paths = [f.path for f in os.scandir(images_path) if f.is_dir()]

        for folder_path in folder_paths:
            person_name = os.path.basename(folder_path)
            images_path = glob.glob(os.path.join(folder_path, "*.*"))
            print(f"Found {len(images_path)} images for {person_name}")

            for img_path in images_path:
                img = cv2.imread(img_path)
                rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img_encodings = face_recognition.face_encodings(rgb_img)

                # Ensure at least one face encoding was found in the image
                if img_encodings:
                    img_encoding = img_encodings[0]
                    self.known_face_encodings.append(img_encoding)
                    self.known_face_names.append(person_name)

        print("Encoding images loaded")

    def detect_known_faces(self, frame, tolerance=0.5):
        small_frame = cv2.resize(frame, (0, 0), fx=self.frame_resizing, fy=self.frame_resizing)
        rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

        face_names = []
        for face_encoding in face_encodings:
            matches = face_recognition.compare_faces(self.known_face_encodings, face_encoding, tolerance)
            name = "Unknown"

            face_distances = face_recognition.face_distance(self.known_face_encodings, face_encoding)
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                name = self.known_face_names[best_match_index]
            face_names.append(name)

        face_locations = np.array(face_locations)
        face_locations = face_locations / self.frame_resizing
        return face_locations.astype(int), face_names

