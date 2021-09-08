import face_recognition
import cv2
import numpy as np
import os

"""
code from github/ageitgey/face_recognition
"""


class EncodeFace:
    def __init__(self, src, landmark68: bool = False, tolerance = 0.6):
        self.src = os.path.join(src)
        self.model = 'large' if landmark68 else 'small'
        os.makedirs(self.src, exist_ok=True)

        self.encodings = []
        self.face_names = []
        # self.encodings = self.encode_all()
        self.encode_all()
        self.tolerance = tolerance

    def encode_all(self):
        face_encodings = []
        face_names = []
        face_dirs = os.listdir(self.src)
        for face_dir in face_dirs:
            path_face_dir = os.path.join(self.src, face_dir)
            # face_encodings.update({face_dir: []})
            for face_img in os.listdir(path_face_dir):
                self.face_names.append(face_dir)
                # img = face_recognition.load_image_file(str(os.path.join(path_face_dir, face_img)))
                img = face_recognition.load_image_file(os.path.join(path_face_dir, face_img))
                encodings = face_recognition.face_encodings(img, model=self.model)
                if len(encodings):
                    encoding = encodings[0]
                    self.encodings.append(encoding)
        # return face_encodings, face_names
    '''
    def encode_only_front(self):
        face_encodings = []
        face_names = []
        face_dirs = os.listdir(self.src)
        for face_dir in face_dirs:
            path_face_dir = os.path.join(self.src, face_dir)
            # face_names.append(face_dir)
            for face_img in os.listdir(path_face_dir):
                if face_img.startswith('front'):
                    face_names.append(face_dir)
                    # img = face_recognition.load_image_file(str(os.path.join(path_face_dir, face_img)))
                    img = face_recognition.load_image_file(os.path.join(path_face_dir, face_img))
                    encodings = face_recognition.face_encodings(img)
                    if len(encodings):
                        encoding = encodings[0]
                        self.encodings.append(encoding)
                else:
                    pass
        # return face_encodings, face_names
    '''
    def match_face(self, face: np.ndarray, get_score=False):
        # print(face.shape)
        name = "-"
        match_distance = self.tolerance
        standard_score = 0.
        face_encodings = face_recognition.face_encodings(face[:, :, ::-1], model=self.model)
        if len(face_encodings):
            face_encoding = face_encodings[0]
            matches = face_recognition.compare_faces(self.encodings, face_encoding, tolerance=self.tolerance)

            face_distances = face_recognition.face_distance(self.encodings, face_encoding)
            # print(face_distances)
            if face_distances.shape[0]:
                best_match_index = np.argmin(face_distances)
                match_distance = face_distances[best_match_index]
                if get_score:
                    score = self.tolerance - match_distance
                    scores = self.tolerance - face_distances
                    standard_score = (score - np.average(scores)) / np.std(scores)
                if matches[best_match_index]:
                    name = self.face_names[best_match_index]
        if get_score:
            return name, match_distance, standard_score
        else:
            return name, match_distance

    def get_faces_cnt(self):
        return len(self.encodings)

    def get_faces_names(self):
        return self.face_names
