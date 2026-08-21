import math
import cv2
import numpy as np
from deep_sort_realtime.deepsort_tracker import DeepSort
from ultralytics import YOLO

# Global variables
drawing = False
line_start = None
lines = [] 
widths = []

def draw_line(event, x, y, flags, param):
    global drawing, line_start, lines, frame_copy
    
    if event == cv2.EVENT_LBUTTONDOWN:  # Mouse button pressed
        drawing = True
        line_start = (x, y)
    
    elif event == cv2.EVENT_MOUSEMOVE:  # Mouse moved
        if drawing:
            frame_copy = frame.copy()
            cv2.line(frame_copy, line_start, (x, y), (0, 255, 0), 2)
            cv2.imshow("Freeze Frame", frame_copy)
    
    elif event == cv2.EVENT_LBUTTONUP:  # Mouse button released
        drawing = False
        line_end = (x, y)
        lines.append((line_start, line_end))
        print(lines)
        cv2.line(frame, line_start, line_end, (0, 255, 0), 2)
        cv2.imshow("Freeze Frame", frame)
        # Calculate and display line width
        widths.append(math.sqrt((line_end[0] - line_start[0]) ** 2 + (line_end[1] - line_start[1]) ** 2))
        print(widths)

def detect_vehicle(model, frame, target_classes):
    result = model(frame)[0]
    detections = []
    for box in result.boxes:
        class_id = int(box.cls[0])
        confidence = float(box.conf[0])
        if class_id in target_classes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            w = x2 - x1
            h = y2 - y1
            if w * h > (frame.shape[1] * frame.shape[0] * 0.001):
                class_name = target_classes[class_id]
                detection = ([x1, y1, w, h], confidence, class_name)
                detections.append(detection)
    return detections

def check_line_intersection_percentage(line_coords, bbox, is_curved_lane=False):
    # Line coordinates
    (x1, y1), (x2, y2) = line_coords
    line_length = abs(x2 - x1)
    
    # YOLO bounding box
    box_x1, box_y1, box_x2, box_y2 = bbox
    
    # Additional tolerance for curved lanes
    y_tolerance = 20 if is_curved_lane else 5
    
    # Find intersection points
    intersection_start = max(x1, box_x1)
    intersection_end = min(x2, box_x2)
    
    # Calculate intersection length
    if intersection_end > intersection_start:
        intersection_length = intersection_end - intersection_start
        intersection_percentage = (intersection_length / line_length) * 100
        
        # flexible y-bound check for curved lanes
        if (box_y1 - y_tolerance) <= y1 <= (box_y2 + y_tolerance):
            return True, intersection_percentage
    
    return False, 0.0

def obj_label(obj_width, line_width):
    if obj_width > 0.9 * line_width:
        label = 'HV'
    elif obj_width > 0.7 * line_width and obj_width < 0.9 * line_width:
        label = 'MV'
    else:
        label = 'HV'
    return label

# Open video
cap = cv2.VideoCapture('road_traffic.mp4')
ret, frame = cap.read()

if not ret:
    print("Error: Unable to read video.")
    cap.release()
    cv2.destroyAllWindows()
    exit()

# Clone the first frame for drawing
frame_copy = frame.copy()

# Set up the mouse callback
cv2.namedWindow("Freeze Frame")
cv2.setMouseCallback("Freeze Frame", draw_line)

print("Draw lines on the frozen frame. Press 'c' to continue.")
while True:
    cv2.imshow("Freeze Frame", frame_copy)
    key = cv2.waitKey(1) & 0xFF
    
    if key == ord('c'):  # Press 'c' to continue
        break

# Release the window and proceed to video playback
cv2.destroyWindow("Freeze Frame")
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('vehicle_track.mp4', fourcc, fps, (frame_width, frame_height))
model = YOLO('yolov8x.pt')
tracker = DeepSort(max_age=15,
                    n_init = 4,
                    nms_max_overlap = 0.7,
                    max_cosine_distance = 0.3,
                    )
target_classes = {2: 'car', 5: 'bus', 7: 'truck'}
sv, mv, hv = 0, 0, 0
detected_vehicles = {}
detected_frame = 1
while True:
    ret, og_frame = cap.read()
    if not ret:
        break
    frame = og_frame.copy()
    detections = detect_vehicle(model,frame,target_classes)
    tracks = tracker.update_tracks(detections, frame=frame)
    for track in tracks:
        if not track.is_confirmed():
            continue
        track_id = track.track_id
        bbox = track.to_ltrb()
        class_id = track.get_det_class()
        confidence = track.get_det_conf()
        if confidence is None:
            continue
        x1, y1, x2, y2 = map(int, bbox)
        # Add size check for the bounding box
        box_area = (x2 - x1) * (y2 - y1)
        frame_area = frame.shape[0] * frame.shape[1]
        if box_area < (frame_area * 0.001) or box_area > (frame_area * 0.5):
            continue
        for i in range(len(lines)):
            is_curved = (i == 3)
            state, intersection = check_line_intersection_percentage(
                lines[i],
                [x1,y1,x2,y2],
                is_curved_lane = is_curved)
            if state and intersection > 50:
                cropped_img = frame[y1:y2, x1:x2]
                w, h, c = cropped_img.shape
                if cropped_img is None or cropped_img.size == 0:
                    print(f"Invalid cropped image: {x1}, {y1}, {x2}, {y2}")
                    continue
                if track_id not in detected_vehicles:
                    cv2.imwrite(f'C:/Users/AnandGnanasekaran/Downloads/detected_vehicle_images/image_{detected_frame}_{h}_{w}.jpg', cropped_img)
                    detected_frame += 1    
                    if w > 0.8 * widths[i]:
                        if h >= 0.8 * widths[i]:
                            detected_vehicles[track_id] = 'HV'
                            hv += 1
                        else:
                            detected_vehicles[track_id] = 'MV'
                            mv += 1
                    else:
                        detected_vehicles[track_id] = 'SV'
                        sv += 1
            # Display vehicle type label (using stored classification)
            vehicle_type = detected_vehicles.get(track_id, '')
            cv2.line(frame, lines[0][0], lines[0][1], (255,255,255),2 )
            cv2.line(frame, lines[1][0], lines[1][1], (255,255,255),2 )
            cv2.line(frame, lines[2][0], lines[2][1], (255,255,255),2 )
            cv2.line(frame, lines[3][0], lines[3][1], (255,255,255),2 )
            cv2.putText(frame, vehicle_type, (int((x1+x2)/2), int((y1+y2)/2)),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.putText(frame, f'SV: {sv}  MV: {mv}  HV: {hv}', (10,20),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 1)

    out.write(frame)
    cv2.imshow('Processed Frames',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()