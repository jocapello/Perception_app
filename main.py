import math
import numpy as np
import cv2
import pandas as pd
import os
from shapely.geometry.polygon import Polygon
from shapely.geometry import Point
from ultralytics import YOLO


class ObjectDetector:
    def __init__(self, model_path, class_list):
        self.model_path = model_path
        self.class_list = class_list
        self.model = YOLO(self.model_path)
        
    def detect_objects(self, frame):

        # Prediction values
        results = self.model.predict(frame, verbose=False, conf=0.6)
        df = pd.DataFrame(results[0].boxes.data)

        detected_objects = []
        
        for row in df.itertuples(index=False):
            x1, y1, x2, y2, _, d = row
            
            # Check if d is a valid index for the class_list
            if 0 <= int(d) < len(self.class_list):
                c = self.class_list[int(d)]

                # If detected target is in desired target list
                if any(target_class in c for target_class in self.class_list):
                    detected_objects.append([int(x1), int(y1), int(x2), int(y2)])
                
        return detected_objects  

class VehicleTracker:
    def __init__(self):
        self.vehicle_info = {}  # Dictionary to store vehicle information
        
    def update(self, objects_rect, colors=[0, 0, 0], lane_indices=None):
        objects_info = []

        for rect, color, lane_index in zip(objects_rect, colors, lane_indices):
            x, y, w, h = rect

            # Center points of detected objects
            cx = (x + x + w) // 2
            cy = (y + y + h) // 2

            identified = False
            for obj_id, info in self.vehicle_info.items():
                last_location = info['locations'][-1]
                if isinstance(last_location, tuple):
                    last_location = Point(*last_location)  # Convert tuple to Point object (ease of use)

                dist = math.hypot(cx - last_location.x, cy - last_location.y)  # Distance between last point and most recent detection
                if dist < 40:  # A min distance
                    self.vehicle_info[obj_id]['locations'].append(Point(cx, cy))
                    identified = True
                    objects_info.append({
                        'id': obj_id,
                        'locations': self.vehicle_info[obj_id]['locations'],
                        'color': color,
                        'bounding_box': rect,
                        'identified': True,
                        'lane_index': lane_index  # Store the lane index along with the object
                    })
                    break

            # Add new vehicle object
            if not identified:
                new_id = max(self.vehicle_info.keys(), default=-1) + 1
                self.vehicle_info[new_id] = {
                    'locations': [Point(cx, cy)],  # Start with a Point object
                    'color': color
                }
                objects_info.append({
                    'id': new_id,
                    'locations': [Point(cx, cy)],
                    'color': color,
                    'bounding_box': rect,
                    'identified': False,
                    'lane_index': lane_index  # Store the lane index along with the object
                })

        return objects_info

class VehicleColorClassifier:
    def __init__(self):
        pass
    def classify_car_color(self, image, bounding_box):
        x,y,x2,y2 = bounding_box
        image = image[y:y+y2,x:x+x2] # Crop image
        hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV) # Read in HSV

        # Calculate the histogram of H and V values
        h_hist = cv2.calcHist([hsv_image], [0], None, [180], [0, 180]) # Hue
        s_hist = cv2.calcHist([hsv_image], [1], None, [256], [0, 256]) # Saturation
        v_hist = cv2.calcHist([hsv_image], [2], None, [256], [0, 256]) # Value

        # Determine the dominant hue and value
        dominant_hue = np.argmax(h_hist)
        dominant_sat = np.argmax(s_hist)
        dominant_value = np.argmax(v_hist)

        # Classify based on hue and value, tested for best results
        if (0 <= dominant_hue <= 10) and dominant_value > 200 and dominant_sat > 10:
            return "Red" 
        elif dominant_value > 135:
            return "White" 
        elif dominant_value < 130:
            return "Black" 
        else:
            return "Other" 

class SpeedEstimator:
    def __init__(self, fps=30):
        self.positions = []
        self.fps = fps

    def update_positions(self, new_position):
        self.positions.append(new_position)  # Update location
        if len(self.positions) > 2:
            self.positions.pop(0) # Remove first element


    def estimate_speed(self, mps=False):
        if len(self.positions) >= 2:
            # Read in posistion and convert to regular ints
            x1, y1 = self.positions[0].x, self.positions[0].y
            x2, y2 = self.positions[-1].x, self.positions[-1].y

            # Calculate the distance between two points in pixels
            d_pixels = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
            
            # Calculate the time interval between frames in seconds
            time_interval = 1 / self.fps

            # Calculate speed in pixels per second
            speed_pixels_per_second = d_pixels / time_interval

            if mps:
                ppm = 22  # Pixels per meter, adjust this based on your setup
                speed_mps = speed_pixels_per_second / ppm
                return speed_mps
            else:
                return speed_pixels_per_second / 60 # pixels per min
        else:
            return 0  # No previous position to calculate speed

class LaneDetector:
    def __init__(self, colors):
        self.colors = colors
        self.num_lanes = 0
        self.detections = []
        self.refPt = []
        self.point_count = 0
        self.lane_bboxes = [] 
        self.lanes = []

    def draw_lines(self, event, x, y, flags, param): # Flags and param are default params from setMouseCallback opencv
        # Draw points
        if event == cv2.EVENT_LBUTTONDOWN:
            self.refPt.append((x, y))
            cv2.circle(self.frame, self.refPt[self.point_count - 1], 10, (0, 0, 0), -1) # Draws on the selected point
            cv2.line(self.frame, self.refPt[self.point_count - 1], self.refPt[self.point_count], self.colors[self.num_lanes], 3) # Connects two dots
            self.point_count += 1

            for i in range(self.point_count - 1):
                if len(self.refPt) > 1:
                    dist = math.sqrt((x - self.refPt[i][0]) ** 2 + (y - self.refPt[i][1]) ** 2)

                    if dist < 10: # If distance is below this assume user is connecting lanes
                        if self.num_lanes <= 3: # Task limit
                            self.refPt[-1] = self.refPt[0]
                            self.num_lanes += 1
                            self.lane_bboxes.append((self.refPt))  # Append the current lane_bbox
                            self.refPt = []
                            self.point_count = 0
                        else:
                            raise ValueError('You can only select a maximum of 4 lanes')

        # Removes most recently drawn objects                            
        elif event == cv2.EVENT_RBUTTONDOWN:
            # Draws all points on copy of the frame minus most recent
            self.frame = frame.copy() 

            if not self.lane_bboxes and not self.refPt:
                # If everything has been removed
                raise ValueError('There are no more items to reprend')

            if self.lane_bboxes: # Redraws completed lanes          
                # Draw the remaining lanes
                for j, lane_bbox in enumerate(self.lane_bboxes):
                    for i in range(len(lane_bbox) - 1):
                        cv2.circle(self.frame, lane_bbox[i], 10, (0, 0, 0), -1)
                        cv2.line(self.frame, lane_bbox[i], lane_bbox[i + 1], self.colors[(j)], 3)

            if self.refPt != []: # Redraws uncomplete lanes
                self.refPt.pop()
                self.point_count -= 1
                if len(self.refPt) > 0:
                    for i in range(len(self.refPt) - 1):
                        cv2.circle(self.frame, self.refPt[i], 10, (0, 0, 0), -1)
                        cv2.line(self.frame, self.refPt[i], self.refPt[i + 1], self.colors[self.num_lanes], 3)

            if self.refPt == []: # Most recently completed lane becomes the current work in progress lane
                self.refPt = self.lane_bboxes.pop()
                if self.num_lanes >= 0:
                    self.num_lanes -= 1

    def convert_polygon(self, bbox):
        # Convert points to a Shapely polygon
        return Polygon(bbox)
    
    def get_lanes(self, frame):
        self.frame = frame.copy()

        # Set up the mouse callback for drawing lanes
        cv2.namedWindow("Draw Lanes")
        cv2.setMouseCallback("Draw Lanes", self.draw_lines)  # Set the callback to the draw_lines method

        # Display the image and let the user draw lanes
        while True:
            cv2.imshow("Draw Lanes", self.frame)
            key = cv2.waitKey(1) & 0xFF

            # Exit loop if Enter key is pressed
            if key == 13:
                break

            # Clear the drawn lanes and restart with 'R' key
            if key == ord("r"):
                self.refPt = []
                self.point_count = 0
                self.frame = frame.copy()
                self.num_lanes = 0

        # Convert drawn lanes to polygons and store them in detections
        self.detections = [self.convert_polygon(bbox) for bbox in self.lane_bboxes]
        cv2.destroyAllWindows()

        return self.detections
    
    def determine_lane_for_vehicle(self, vehicle_center):

        # Checks if center points of vehicle object are inside of a lane polygon thanks shapely :) 
        for i, lane_bbox in enumerate(self.lane_bboxes):
            lane_polygon = self.convert_polygon(lane_bbox)
            if lane_polygon.contains(Point(vehicle_center)):
                return i
        return None

class Lane:
    def __init__(self, index, color, polygon=None):
        self.index = index
        self.color = color
        self.polygon = polygon  # Store the polygon associated with the lane
        self.vehicles = []

    def add_vehicle(self, vehicle_id, color, location):
        # Adds relevant vehicle properties to lane
        self.vehicles.append({'id': vehicle_id, 'color': color, 'location': location})

    def remove_vehicle(self, vehicle_id):
        # Removes vehicle from the lane if it exits its polygon
        for vehicle in self.vehicles:
            if vehicle['id'] == vehicle_id:
                color_removed = vehicle['color']
                self.vehicles.remove(vehicle)
                return color_removed
        return None

    def get_vehicle_count(self):
        # Return number of vehicles in the lane
        count = 0
        for vehicle in self.vehicles:
            if self.polygon.contains(Point(vehicle['location'])):
                count += 1
        return count
    
    def get_recent_location(self, vehicle_id):
        # Get location of the vehicle namely for speed estimation
        for vehicle in self.vehicles:
            if vehicle['id'] == vehicle_id:
                return vehicle['location']
        return None

class AnnotationOverlay:
    def __init__(self, colors, lanes):
        self.colors = colors
        self.lanes = lanes

    def draw_lane_annotations(self, frame, lane_bboxes):
        for shape_id, shape_lines in enumerate(lane_bboxes, start=1):
            exterior_coords = np.array(shape_lines.exterior.coords, dtype=np.int32).reshape((-1, 1, 2))
            
            # Make sure that shape_id is within the range of available colors
            if 0 <= shape_id - 1 < len(self.colors):
                cv2.drawContours(frame, [exterior_coords], contourIdx=-1, color=self.colors[shape_id - 1], thickness=2)
            else:
                print(f"Warning: Shape ID {shape_id} is out of range for colors list.")

    def draw_car_bounding_boxes(self, frame, car_objects):
        for obj in car_objects:
            x, y, x2, y2 = obj['bounding_box']
            vehicle_id = obj['id']
            color = obj['color']

            # Get the corresponding lane index for the vehicle
            lane_index = lane_detector.determine_lane_for_vehicle(((x + x2) // 2, (y + y2) // 2))
            if lane_index is not None:
                color = lanes[lane_index].color

            # Draw the bounding box
            cv2.rectangle(frame, (x, y), (x2, y2), color=color, thickness=2)

            # Draw the vehicle ID inside the bounding box
            cv2.putText(frame, f'ID: {vehicle_id}', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    def draw_vehicle_counts(self, frame, tracked_objects, lane_speed_estimators):
        for i, lane in enumerate(self.lanes):
            black_count = 0
            red_count = 0
            white_count = 0
            total_speed = 0
            total_vehicles = 0

            for obj in tracked_objects:
                lane_index = obj['lane_index']

                if lane_index == i:
                    vehicle_color = obj['color']
                    vehicle_speed = lane_speed_estimators[i].estimate_speed()  # Estimate speed for the current lane

                    if vehicle_color == 0:  # Black
                        black_count += 1
                    elif vehicle_color == 1:  # Red
                        red_count += 1
                    elif vehicle_color == 2:  # White
                        white_count += 1

                    total_speed += vehicle_speed
                    total_vehicles += 1

            if total_vehicles > 0:
                average_speed = total_speed / total_vehicles  # Average speed of vehicles in the lane
                lane_name = f"Lane {i+1}: Total({total_vehicles}), Black({black_count}), Red({red_count}), White({white_count}), Avg Speed: {average_speed:.2f} pixels/min"
            else:
                average_speed = 0
                lane_name = f"Lane {i+1}: Total({total_vehicles}), Black({black_count}), Red({red_count}), White({white_count}), Avg Speed: No vehicles"


            # Calculate text size for the information text
            text = f"{lane_name}"
            text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]

            # Calculate position for the text and the white background rectangle
            text_x = 20
            text_y = 30 * (i + 1) - 10  # Adjusted the vertical position

            rect_x = text_x - 5
            rect_y = text_y - text_size[1] + 5
            rect_width = text_size[0] + 10
            rect_height = text_size[1] + 10

            # Draw the white background rectangle
            cv2.rectangle(frame, (rect_x, rect_y-15), (rect_x + rect_width, rect_y + rect_height), (255, 255, 255), -1)

            # Annotates frame with the relevant info on the white rectangle
            cv2.putText(frame, lane_name, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, lane.color, 2)



if __name__ == "__main__":
    project_dir = os.path.dirname(os.path.realpath(__file__))
    model_path = os.path.join(project_dir, 'yolov8s.pt')
    cap = cv2.VideoCapture(os.path.join(project_dir, '4K Road traffic video for object detection and tracking - free download now!.mp4'))
    out = cv2.VideoWriter('filename3.avi', cv2.VideoWriter_fourcc(*'XVID'), 30, (1080, 720))
    record = True

    class_list = ['car', 'motorcycle', 'truck', 'bus']  # Yolo classes
    colors = [(255, 255, 102), (255, 102, 102), (102, 102, 255), (102, 255, 178)]  # Colors for different lanes

    # Instantiate each required class
    detector = ObjectDetector(model_path, class_list)
    vehicle_tracker = VehicleTracker()
    color_classifier = VehicleColorClassifier()
    speed_estimator = SpeedEstimator()
    lane_detector = LaneDetector(colors)

    # Initialize Lane objects
    lanes = [Lane(i, color) for i, color in enumerate(colors)]
    lane_speed_estimators = [SpeedEstimator() for _ in lanes]  # Initialize lane-specific speed estimators
    annotation_overlay = AnnotationOverlay(colors, lanes)

    first_frame = True
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        detected_objects = detector.detect_objects(frame)
        tracked_objects = []

        lane_indices = [lane_detector.determine_lane_for_vehicle(((x + x2) // 2, (y + y2) // 2)) for x, y, x2, y2 in detected_objects]

        # Update the vehicle tracker with the detected objects and lane indices
        tracked_objects = vehicle_tracker.update(detected_objects, colors=colors, lane_indices=lane_indices)

        if first_frame:
            lane_polygons = lane_detector.get_lanes(frame)  # Stored as Shapely polygon objects to find central points easier
            lanes = [Lane(i, color, polygon) for i, (color, polygon) in enumerate(zip(colors, lane_polygons))]
            lane_speed_estimators = [SpeedEstimator() for _ in lanes]  # Initialize lane-specific speed estimators
            annotation_overlay = AnnotationOverlay(colors, lanes)
            first_frame = False

        # Draws lanes
        annotation_overlay.draw_lane_annotations(frame, lane_polygons)
        # Draw vehicle detections
        annotation_overlay.draw_car_bounding_boxes(frame, tracked_objects)

        for obj in tracked_objects:
            lane_index = obj['lane_index']
            if lane_index is not None:
                speed_estimator = lane_speed_estimators[lane_index]
                speed_estimator.update_positions(obj['locations'][-1])  # Update speed positions with most recent detections

        # Call the method to draw vehicle counts, colors, and average speed
        annotation_overlay.draw_vehicle_counts(frame, tracked_objects, lane_speed_estimators)

        cv2.imshow("Processed Frame", frame)
        out.write(frame)

        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()

