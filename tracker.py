import math
import numpy as np

class Tracker:
    def __init__(self):
        # Store the center positions of the objects
        self.center_points = {}
        # Store the center positions of the previous location of objects
        self.prev_pts = {}
        # Keep the count of the IDs
        # each time a new object id detected, the count will increase by one
        self.id_count = 0


    def update(self, objects_rect):
        # Objects boxes and ids
        objects_bbs_ids = []
        prev_pts = []
        # Get center point of new object
        for rect in objects_rect:
            x, y, w, h = rect
            cx = (x + x + w) // 2
            cy = (y + y + h) // 2

            # Find out if that object was detected already
            same_object_detected = False
            for id, pt in self.center_points.items():
                dist = math.hypot(cx - pt[0], cy - pt[1])
                if dist < 35:
                    self.prev_pts[id] = self.center_points[id]
                    self.center_points[id] = (cx, cy)
#                    print(self.center_points)
                    objects_bbs_ids.append([x, y, w, h, id])
                    same_object_detected = True
                    break

            # New object is detected we assign the ID to that object
            if same_object_detected is False:
                self.center_points[self.id_count] = (cx, cy)
                objects_bbs_ids.append([x, y, w, h, self.id_count])
                self.id_count += 1

        # Clean the dictionary by center points to remove IDS not used anymore
        new_center_points = {}
        for obj_bb_id in objects_bbs_ids:
            _, _, _, _, object_id = obj_bb_id
            center = self.center_points[object_id]
            new_center_points[object_id] = center

        # Update dictionary with IDs not used removed
        self.center_points = new_center_points.copy()
        return objects_bbs_ids, prev_pts
    
class SpeedEstimator:
    def __init__(self,posList,fps):
        self.x=posList[0]
        self.y=posList[1]
        self.fps=fps
        
    def estimateSpeed(self):
        # Distance / Time -> Speed
        d_pixels=math.sqrt(self.x+self.y)
        
        ppm=22
        d_meters=int(d_pixels*ppm)
        speed=d_meters/self.fps*3.6
        speedInKM=np.average(speed)
        return int(speedInKM)