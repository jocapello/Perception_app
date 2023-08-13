from shapely.geometry.polygon import Polygon
from shapely.geometry import Point
from ultralytics import YOLO
from statistics import mean
from math import dist
from tracker import*
import pandas as pd
import numpy as np
import time
import cv2
import os 

model=YOLO('yolov8s.pt')

delay_time = 1
res = (1920, 1080) # Resolution of the video

# Variables auto adjust 
min_x = int(0.50 * res[0])
x_space = int(0.08 * res[0])
font_size = 0.0005 * res[0]

dir_path = os.path.dirname(os.path.realpath(__file__))
cap = cv2.VideoCapture(dir_path + '/4K Road traffic video for object detection and tracking - free download now!.mp4')
# out = cv2.VideoWriter('filename2.avi', 
#                          cv2.VideoWriter_fourcc(*'MJPG'),
#                          30, res)

class_list = ['car', 'motercycle', 'truck', 'bus']
s = {'Black':0, 'White':0, 'Red':0, 'Other':0}

colors = [(255, 255, 102), # teal
          (255, 102, 102), # purple
          (102, 102, 255), # red
          (102, 255, 178)  # green
         ]

point_count, col, speed = (0 for i in range(3))
bbox, refPt, lanes, counter, counter1 = ([] for i in range(5))
firstFrame = True
tracker=Tracker()


# left click to draw the next line, right click to delete the previous line
def draw_line(event, x, y, flags, param):
    global refPt, bbox, lanes, image, col, point_count, colors
    if event == cv2.EVENT_LBUTTONDOWN:   
        refPt.append((x, y))
        cv2.circle(image, refPt[point_count-1], 10, (0,0,0), -1)
        cv2.line(image, refPt[point_count-1], refPt[point_count], colors[col], 3)
        point_count += 1
        for i in range (0, point_count-1):
            if len(refPt) > 1:
                dist = math.sqrt((x - refPt[i][0])**2 + (y - refPt[i][1])**2)
                lanes.append([x,y])

                if dist < 10:
                    if col <= 3:
                        col += 1
                        bbox.append(Polygon(lanes))
                        lanes = []
                        refPt = []
                        point_count = 0
                    else: 
                        print('Too many')

    elif event == cv2.EVENT_RBUTTONDOWN:
        image = clone.copy()    
        image = image
        refPt.remove(refPt[point_count-1])
        point_count = point_count - 1
        for i in range (point_count):
            cv2.circle(image, refPt[i-1], 10, (0,0,0), -1)
            cv2.line(image, refPt[i-1], refPt[i], colors[col], 3)
    return clone



def classify_car_color(image, bounding_box):
    x,y,x2,y2 = bounding_box
    image = image[y:y+y2,x:x+x2]
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Calculate the histogram of H and V values
    h_hist = cv2.calcHist([hsv_image], [0], None, [180], [0, 180])
    s_hist = cv2.calcHist([hsv_image], [1], None, [256], [0, 256])
    v_hist = cv2.calcHist([hsv_image], [2], None, [256], [0, 256])

    # Determine the dominant hue and value
    dominant_hue = np.argmax(h_hist)
    dominant_sat = np.argmax(s_hist)
    dominant_value = np.argmax(v_hist)

    # Classify based on hue and value
    if (0 <= dominant_hue <= 10) and dominant_value > 200 and dominant_sat > 10:
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
    clone = image.copy()

    # Check if the frame was read successfully
    if not ret:
        print("Error: Could not read frame")
        exit()

    # Get the lanes
    while firstFrame: 

        cv2.namedWindow("image")
        cv2.setMouseCallback("image", draw_line)
        cv2.imshow("image", image)
        
        # display the image and wait for a keypress
        key = cv2.waitKey(1) & 0xFF

        # Start the program upon enter
        if key == ord('\r'):
            if lanes != []:
                print('Complete lane bounding box before continuing')
            else:
                firstFrame = False

        # if the 'reset' key is pressed, reset to original state
        if key == ord("r"):
            image = clone.copy()  
            col, frameCounter, point_count = (0 for i in range(3))
            refPt, lanes, bbox = ([] for i in range(3))

    # Display and track cars
    speed_zones = [[] for _ in range(len(bbox))]
    frameCounter = 0
    while firstFrame == False:

        start_time = time.time()
        ret,frame = cap.read()
        frame = cv2.resize(frame, res)
        cv2.waitKey(delay_time) # Adjusts speed of video

        if not ret:
            break
        frameCounter += 1

        if frameCounter % 10 != 0: 
            speed_zones = [[] for _ in range(len(bbox))] # Cycle speed zones every 10 frames
        frameCounter = frameCounter + 1

        results = model.predict(frame, verbose=False)
        detected_boxes = results[0].boxes.data
        confidence = results[0].boxes.conf
        px = pd.DataFrame(detected_boxes).astype("float")
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
        vehicle_colors = {}
        for i in range(len(bbox)):
            vehicle_colors[i] = s.copy()

        for bbox_ in bbox_id:
            x3,y3,x4,y4,id=bbox_
            
            cv2.rectangle(frame,(x3,y3),(x4,y4),(0,0,255),2) # Draw detections
            if prev_points != [] and prev_points:
                if id in prev_points:

                    # Find midpoints
                    cx = (x3 + x3 + x4) // 2 
                    cy = (y3 + y4 + y3) // 2
                    dist = [abs(cx - prev_points[id][0]), abs(cy - prev_points[id][1])]

                    if dist[0] < 5 and dist[1] < 5: # Eliminate false movement
                        dist = (0, 0)
                    SpeedEstimatorTool=SpeedEstimator(dist, fps)
                    speed=SpeedEstimatorTool.estimateSpeed()
                    if speed > 130:
                            speed = 0

                    cv2.putText(frame,str(id),(x3,y3),cv2.FONT_HERSHEY_COMPLEX,0.6,(255,255,255),1)
                    # cv2.putText(frame,str(int(speed))+'Km/h',(x4-5,y4 ),cv2.FONT_HERSHEY_COMPLEX,0.8,(0,255,255),2)    
                    # cv2.putText(frame, car_color,(x4-5,y4 ),cv2.FONT_HERSHEY_COMPLEX,0.8,(0,255,255),2)    

            mid_point = [(x4-x3)+x3, (y4-y3)+y3]
            for lane_num in range(len(bbox)):
                if bbox[lane_num].contains(Point(mid_point)):
                    speed_zones[lane_num].append(speed)
                    car_color = classify_car_color(frame, [x3,y3,x4,y4])
                    vehicle_colors[lane_num][car_color] += 1


        # Display stats
        min_x = int(0.50 * res[0])
        x_space = int(0.08 * res[0])
        font_size = 0.0005 * res[0]
        # Display stats
        cv2.rectangle(frame, (min_x-25, 25), (res[0]-10, 75+(len(bbox)*50)), (0,0,0), -1)
        cv2.rectangle(frame, (min_x-10, 40), (res[0]-25, 60+(len(bbox)*50)), (255,255,255), -1)


        # Display the lanes 
        for shape_id, shape_lines in enumerate(bbox, start=1):
            exterior_coords = np.array(shape_lines.exterior.coords, dtype=np.int32).reshape((-1, 1, 2))
            # Draw the contour of the outermost shape on the image
            cv2.drawContours(frame, [exterior_coords], contourIdx=-1, color=colors[shape_id-1], thickness=2)


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
        # out.write(frame)

        if not (end_time == start_time):
            fps = 1.0/(end_time - start_time)

        cv2.imshow("image", frame)
        if cv2.waitKey(1) & 0xFF==27:
            break
        if cv2.waitKey(1) == ord('p'):
            cv2.waitKey(-1)
        
    cap.release() 
    # out.release()
    cv2.destroyAllWindows() 
