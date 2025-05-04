import cv2
import numpy as np
import time

prev_left_fit = None
prev_right_fit = None
alpha = 0.2
last_warning_time = 0  
previous_lane_center = None 

def smooth_fit(prev_fit, new_fit, alpha=0.2):
    if prev_fit is None:
        return new_fit
    return prev_fit * (1 - alpha) + new_fit * alpha

def region_of_interest(img):
    height, width = img.shape
    polygons = np.array([[
        (int(width * 0.1), height),
        (int(width * 0.4), int(height * 0.6)),
        (int(width * 0.6), int(height * 0.6)),
        (int(width * 0.9), height)
    ]], dtype=np.int32)
    mask = np.zeros_like(img)
    cv2.fillPoly(mask, polygons, 255)
    return cv2.bitwise_and(img, mask)

def dynamic_roi(image, edges):
    global prev_left_fit, prev_right_fit, last_warning_time, previous_lane_center

    masked_edges = region_of_interest(edges)

    lines = cv2.HoughLinesP(masked_edges, 1, np.pi / 180, threshold=50, minLineLength=50, maxLineGap=100)
    if lines is None:
        return edges, image

    left_lines = []
    right_lines = []

    for line in lines:
        x1, y1, x2, y2 = line[0]
        slope = (y2 - y1) / (x2 - x1 + 1e-6)
        if slope < -0.5:
            left_lines.append((x1, y1, x2, y2))
        elif slope > 0.5:
            right_lines.append((x1, y1, x2, y2))

    def fit_line(lines):
        x = [x for x1, y1, x2, y2 in lines for x in (x1, x2)]
        y = [y for x1, y1, x2, y2 in lines for y in (y1, y2)]
        return np.polyfit(y, x, 1) if len(x) > 0 else None

    left_fit = fit_line(left_lines)
    right_fit = fit_line(right_lines)

    if left_fit is None or right_fit is None:
        return edges, image

    left_fit = smooth_fit(prev_left_fit, left_fit, alpha)
    right_fit = smooth_fit(prev_right_fit, right_fit, alpha)
    prev_left_fit = left_fit
    prev_right_fit = right_fit

    height = image.shape[0]
    y1 = height
    y2 = int(height * 0.7)

    left_x1 = int(np.polyval(left_fit, y1))
    left_x2 = int(np.polyval(left_fit, y2))
    right_x1 = int(np.polyval(right_fit, y1))
    right_x2 = int(np.polyval(right_fit, y2))

    roi_vertices = np.array([[
        (left_x1, y1), (left_x2, y2),
        (right_x2, y2), (right_x1, y1)
    ]], dtype=np.int32)

    mask = np.zeros_like(edges)
    cv2.fillPoly(mask, roi_vertices, 0)
    lane_area = cv2.bitwise_and(image, image, mask=mask)

    overlay = image.copy()
    cv2.polylines(overlay, roi_vertices, isClosed=True, color=(144, 238, 144), thickness=1)
    cv2.fillPoly(overlay, roi_vertices, color=(144, 238, 144))  

    
    threshold = 100  
    center = (left_x1 + right_x1) // 2  
    camera_mid = image.shape[1] // 2
    deviation = center - camera_mid

    if previous_lane_center is not None:
        lane_center_deviation = abs(center - previous_lane_center)

        if abs(deviation) > threshold and time.time() - last_warning_time > 5:
            if lane_center_deviation > 30:  
                print("Warning: Lane Departure Detected!")
                last_warning_time = time.time()  

    previous_lane_center = center  

    return mask, overlay


cap = cv2.VideoCapture(r"C:\Users\Charan singh\Desktop\LangChain\ROAD DATASET\night.mp4")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 50, 150)

    mask, overlay = dynamic_roi(frame, edges)
    overlay=cv2.resize(overlay,(600,600))
    cv2.imshow("Overlay with Dynamic ROI", overlay)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
