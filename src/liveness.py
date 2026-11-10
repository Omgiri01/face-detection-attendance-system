import cv2
import time
import numpy as np

class LivenessDetector:
    def __init__(self):
        # Load pre-trained Haar cascades for face and eyes
        # Using tree_eyeglasses for better support with glasses
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye_tree_eyeglasses.xml')
        
        self.blink_count = 0
        self.last_blink_time = 0
        self.is_live = False
        self.eyes_detected_counter = 0
        self.eyes_gone_counter = 0
        self.was_eyes_detected = False

    def check_liveness(self, frame, face_coords=None):
        """
        Processes a frame and returns (is_live_verified, liveness_label).
        Optimized: Accepts optional face_coords (x, y, w, h) to avoid redundant detection.
        """
        current_time = time.time()
        
        if face_coords is not None:
            (x, y, w, h) = face_coords
        else:
            # Fallback to internal detection only if not provided
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)
            if len(faces) == 0:
                return False, "No Face Detected"
            (x, y, w, h) = sorted(faces, key=lambda f: f[2]*f[3], reverse=True)[0]

        roi_gray = cv2.cvtColor(frame[y:y+h, x:x+w], cv2.COLOR_BGR2GRAY)
        
        # Detect eyes in the face ROI
        # Using a very fast detector for blinks
        eyes = self.eye_cascade.detectMultiScale(roi_gray, 1.1, 10, minSize=(w//10, h//10))
        
        current_eyes_detected = len(eyes) >= 2

        # Blink logic
        if current_eyes_detected:
            self.eyes_detected_counter += 1
            if 0 < self.eyes_gone_counter < 5:
                # Eyes reappeared after a short disappearance! Count as blink!
                self.blink_count += 1
                self.last_blink_time = current_time
                self.eyes_gone_counter = 0
            self.eyes_gone_counter = 0
        else:
            if self.eyes_detected_counter > 2: # Lowered threshold for faster detection
                self.eyes_gone_counter += 1
            self.eyes_detected_counter = 0

        # Reset if eyes gone for too long (not a blink)
        if self.eyes_gone_counter > 10:
            self.eyes_detected_counter = 0
            self.eyes_gone_counter = 0

        # If we saw a blink in the last 20 seconds, consider it "Live"
        if current_time - self.last_blink_time < 20.0 and self.blink_count > 0:
            self.is_live = True
            return True, f"Live (Blinks: {self.blink_count})"
        else:
            self.is_live = False
            return False, "Blink to verify"
