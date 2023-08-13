import cv2
import numpy as np
import pandas as pd
import numpy as np
from ultralytics import YOLO
from tracker import*
import time
from math import dist
import os 
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon
from statistics import mean

dir_path = os.path.dirname(os.path.realpath(__file__))
model=YOLO('yolov8s.pt')

delay_time = 10
res = (1920, 1080)
min_x = int(0.50 * res[0])
x_space = int(0.08 * res[0])
font_size = 0.0005 * res[0]

my_file = open(dir_path + "/coco.txt", "r")
data = my_file.read()
class_list = data.split("\n") 

cap = cv2.VideoCapture(dir_path + '/4K Road traffic video for object detection and tracking - free download now!.mp4')
colors = [(255, 255, 102), # teal
          (255, 102, 102), # purple
          (102, 102, 255), # red
          (102, 255, 178)  # green
         ]
col = 0
refPt = []
count = 0
lanes = []
bbox = []
count = 0
count2 = 0
firstFrame = True

frameCounter = 0
cy1=322
cy2=368
speed = 0
offset=6

counter=[]
vh_up={}
counter1=[]
tracker=Tracker()


# left click to draw the next line, right click to delete the previous line
def draw_line(event, x, y, flags, param):
    global refPt, count, image, col, colors, lanes, bbox

    if event == cv2.EVENT_LBUTTONDOWN:   
        refPt.append((x, y))
        cv2.circle(image, refPt[count-1], 10, (0,0,0), -1)
        cv2.line(image, refPt[count-1], refPt[count], colors[col], 3)
        count += 1
        for i in range (0, count-1):
            if len(refPt) > 1:
                dist = math.sqrt((x - refPt[i][0])**2 + (y - refPt[i][1])**2)
                lanes.append([x,y, refPt[i][0], refPt[i][1]])

                if dist < 10:
                    if col <= 3:
                        col += 1
                        bbox.append(lanes)
                        lanes = []
                        refPt = []
                        count = 0
                    else: 
                        print('Too many')

    elif event == cv2.EVENT_RBUTTONDOWN:
        image = clone.copy()    
        image = image
        refPt.remove(refPt[count-1])
        count = count - 1
        for i in range (count):
            cv2.circle(image, refPt[i-1], 10, (0,0,0), -1)
            cv2.line(image, refPt[i-1], refPt[i], colors[col], 3)
    return clone

def find_corners(self, lines):
    corners = []
    for line in lines:
        for point in line:
            if point not in corners:
                corners.append(point)
    return corners


def classify_car_color(image, bounding_box):
    x,y,x2,y2 = bounding_box
    image = image[y:y+y2,x:x+x2]
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Calculate the histogram of H and V values
    h_hist = cv2.calcHist([hsv_image], [0], None, [180], [0, 180])
    v_hist = cv2.calcHist([hsv_image], [2], None, [256], [0, 256])

    # Determine the dominant hue and value
    dominant_hue = np.argmax(h_hist)
    dominant_value = np.argmax(v_hist)

    # Classify based on hue and value
    if (0 <= dominant_hue <= 10 or 160 <= dominant_hue <= 180) and dominant_value > 200:
        return "Red" 
    elif dominant_value > 135:
        return "White" 
    elif dominant_value < 130:
        return "Black" 
    else:
        return "Other" 


if __name__ == "__main__":

    # Read the first frame from the video
    ret, frame = cap.read()
    image = cv2.resize(frame, res)

    # Check if the frame was read successfully
    if not ret:
        print("Error: Could not read frame")
        exit()

    clone = image.copy()

    # Get the lanes
    while firstFrame: 

        cv2.namedWindow("image")
        cv2.setMouseCallback("image", draw_line)

        cv2.imshow("image", image)
        
        # display the image and wait for a keypress
        key = cv2.waitKey(1) & 0xFF

        if key == ord('\r'):
            if lanes != []:
                print('Complete lane bounding box before continuing')
            else:
                firstFrame = False

        # if the 'reset' key is pressed, reset to original state
        if key == ord("r"):
            image = clone.copy()  
            col = 0
            refPt = []
            lanes = []
            bbox = []
            count = 0

    # Display and track cars
    speed_zones = [[] for _ in range(len(bbox))]

    while firstFrame == False:

        start_time = time.time()
        ret,frame = cap.read()
        frame = cv2.resize(frame, res)
        cv2.waitKey(delay_time)

        if not ret:
            break
        count += 1
        if count % 10 != 0:
            speed_zones = [[] for _ in range(len(bbox))]
        frameCounter = frameCounter + 1

        results = model.predict(frame, verbose=False)
        a=results[0].boxes.data
        confidence = results[0].boxes.conf
        px=pd.DataFrame(a).astype("float")
        px['conf'] = confidence
        px = px[px['conf'] >= 0.6]  
        lst=[]
                
        for index,row in px.iterrows():
            x1=int(row[0])
            y1=int(row[1])
            x2=int(row[2])
            y2=int(row[3])
            d=int(row[5])
            c=class_list[d]

            if 'car' in c or 'motorcycle' in c or 'bus' in c or 'truck' in c:
                lst.append([x1,y1,x2,y2])
        bbox_id, _= tracker.update(lst)
        prev_points = tracker.prev_pts
        s = {'Black':0, 'White':0, 'Red':0, 'Other':0}
        vehicle_colors = {}
        for i in range(len(bbox)):
            vehicle_colors[i] = s.copy()

        for bbox_ in bbox_id:
            x3,y3,x4,y4,id=bbox_
            
            cv2.rectangle(frame,(x3,y3),(x4,y4),(0,0,255),2)
            if prev_points != [] and prev_points:
                if id in prev_points:
                    cx = (x3 + x3 + x4) // 2
                    cy = (y3 + y4 + y3) // 2
                    dist = [abs(cx - prev_points[id][0]), abs(cy - prev_points[id][1])]
                    if dist[0] < 5 and dist[1] < 5:
                        dist = (0, 0)
                    SpeedEstimatorTool=SpeedEstimator(dist, fps)
                    speed=SpeedEstimatorTool.estimateSpeed()
                    if speed > 130:
                            speed = 0

                    cv2.putText(frame,str(id),(x3,y3),cv2.FONT_HERSHEY_COMPLEX,0.6,(255,255,255),1)
                    # cv2.putText(frame,str(int(speed))+'Km/h',(x4-5,y4 ),cv2.FONT_HERSHEY_COMPLEX,0.8,(0,255,255),2)    
                    # cv2.putText(frame, car_color,(x4-5,y4 ),cv2.FONT_HERSHEY_COMPLEX,0.8,(0,255,255),2)    

            mid_point = [(x4-x3)+x3, (y4-y3)+y3]
            for lane_num, lane in enumerate(bbox):
                flattened_lane = [item for sublist in lane for item in sublist]
                unique_lane = list(set(tuple(flattened_lane[i:i+2]) for i in range(len(flattened_lane)-1)))

                if Polygon(unique_lane).contains(Point(mid_point)):
                    speed_zones[lane_num].append(speed)
                    car_color = classify_car_color(frame, [x3,y3,x4,y4])
                    vehicle_colors[lane_num][car_color] += 1


        # Display stats
        min_x = int(0.50 * res[0])
        x_space = int(0.08 * res[0])
        font_size = 0.0005 * res[0]
        # Display stats
        cv2.rectangle(frame, (min_x-25, 25), (1915, 75+(len(bbox)*50)), (0,0,0), -1)
        cv2.rectangle(frame, (min_x-10, 40), (1900, 60+(len(bbox)*50)), (255,255,255), -1)


        # Display the lanes 
        for shape_id, shape_lines in enumerate(bbox, start=1):
            for i in range(len(shape_lines)):
                pt1 = (shape_lines[i][0], shape_lines[i][1])
                pt2 = (shape_lines[(i+1) % len(shape_lines)][0], shape_lines[(i+1) % len(shape_lines)][1])
                cv2.line(frame, pt1, pt2, colors[shape_id - 1], 3)
            cv2.putText(frame, 'Zone', (min_x, 30+(shape_id*50)),cv2.FONT_HERSHEY_COMPLEX, font_size, colors[shape_id-1],2)   
            cv2.putText(frame, 'Red: ' + str(vehicle_colors[shape_id-1]['Red']), (min_x+x_space, 30+(shape_id*50)),cv2.FONT_HERSHEY_COMPLEX,font_size, colors[shape_id-1],2)    
            cv2.putText(frame, 'Black: ' + str(vehicle_colors[shape_id-1]['Black']), (min_x+(2*x_space), 30+(shape_id*50)),cv2.FONT_HERSHEY_COMPLEX,font_size, colors[shape_id-1],2)    
            cv2.putText(frame, 'White: ' + str(vehicle_colors[shape_id-1]['White']), (min_x+(3*x_space), 30+(shape_id*50)),cv2.FONT_HERSHEY_COMPLEX,font_size, colors[shape_id-1],2)  
            # cv2.putText(frame, 'Other: ', (min_x+(4*x_space), 30+(shape_id*50)),cv2.FONT_HERSHEY_COMPLEX,font_size, colors[shape_id-1],2)  
            if speed_zones[shape_id-1] != []:
                cv2.putText(frame, 'Speed: '+str(int(mean(speed_zones[shape_id-1]))), (min_x+(5*x_space), 30+(shape_id*50)),cv2.FONT_HERSHEY_COMPLEX,font_size, colors[shape_id-1],2)  
            else:
                cv2.putText(frame, 'Speed: ', (min_x+(5*x_space), 30+(shape_id*50)),cv2.FONT_HERSHEY_COMPLEX,font_size, colors[shape_id-1],2)          
        
        end_time = time.time()
        
        if not (end_time == start_time):
            fps = 1.0/(end_time - start_time)

        cv2.imshow("image", frame)
        if cv2.waitKey(1) & 0xFF==27:
            break
        if cv2.waitKey(1) == ord('p'):
            cv2.waitKey(-1)
        
    cap.release() 
    cv2.destroyAllWindows() 
