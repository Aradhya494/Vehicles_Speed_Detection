import numpy as np

class Sort:
    def __init__(self, max_age=1, min_hits=3, iou_threshold=0.3):
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.trackers = []
        self.frame_count = 0

    def update(self, dets=np.empty((0, 5))):
        """
        Params:
          dets - a numpy array of detections in the format [[x1,y1,x2,y2,score],...]
        Returns a similar array, where the last column is the object ID.
        """
        self.frame_count += 1
        if len(dets) == 0:
            return np.empty((0, 5))
        
        # Returns bounding boxes with temporary IDs for detections
        out_dets = []
        for i, det in enumerate(dets):
            out_dets.append([det[0], det[1], det[2], det[3], i + 1])
            
        return np.array(out_dets)