import cv2
import numpy as np
import base64

def extract_roi(frame, roi_coords):
    """Extract region of interest from frame"""
    x, y, w, h = roi_coords
    # Ensure coordinates are within frame bounds
    x = max(0, x)
    y = max(0, y)
    w = min(w, frame.shape[1] - x)
    h = min(h, frame.shape[0] - y)
    
    if w > 0 and h > 0:
        return frame[y:y+h, x:x+w]
    return None

def draw_roi(frame, roi_coords):
    """Draw region of interest on frame"""
    x, y, w, h = roi_coords
    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
    return frame

def frame_to_base64(frame):
    """Convert OpenCV frame to base64 string"""
    _, buffer = cv2.imencode('.jpg', frame)
    return base64.b64encode(buffer).decode('utf-8')