#!/usr/bin/env python3
import cv2 as cv

from realsense_camera import *
from mask_rcnn import *

import rospy
import tf2_ros
import tf.transformations as tf_trans
from geometry_msgs.msg import TransformStamped, Quaternion
from std_msgs.msg import String

import threading

class ShareValues:
    def __init__(self) -> None:
        self.center_x = None
        self.center_y = None
        self.depth = None
        self.labels = None
        self.class_counts = {} 


class IntelVision:
    def __init__(self, shared_data):
        rospy.init_node('Intel_Object_Detection')
        rospy.loginfo("Intel Object Detection and TF publish")

        self.rs = RealsenseCamera()
        self.mask = MaskRCNN()

        self.shared_data = shared_data

        self.topic_pub = rospy.Publisher('/intel_dist', String, queue_size=10)
        self.data_msg = String()

    def vision(self):
        while True:
            ret, bgr_frame, depth_frame = self.rs.get_frame_stream()
            if not ret:
                continue

            boxes, classes, contours, centers = self.mask.detect_objects_mask(bgr_frame)

            # Reset the class counts for each frame
            self.shared_data.class_counts = {}

            bgr_frame, boxes, classes = self.mask.draw_object_mask(bgr_frame)
            for box, class_id, center in zip(boxes, classes, centers):
                x, y, x2, y2 = box
                cx, cy = center
                depth_mm = depth_frame[cy, cx]
                class_name = self.mask.classes[class_id]

                # Ensure the depth is a positive number
                if depth_mm > 0:
                    self.shared_data.center_x = cx
                    self.shared_data.center_y = cy
                    self.shared_data.depth = depth_mm
                    self.shared_data.labels = class_name

                    # Update class count
                    if class_name in self.shared_data.class_counts:
                        self.shared_data.class_counts[class_name] += 1
                    else:
                        self.shared_data.class_counts[class_name] = 1

                    print(f"Class: {class_name}, Center: ({cx}, {cy}), Depth: {depth_mm} mm")

                yolo_object = String()
                yolo_object.data = f"Label: {class_name} Dist: {depth_mm / 1000.0} m"
                self.topic_pub.publish(yolo_object)

            self.mask.draw_object_info(bgr_frame, depth_frame)

            # Display the class counts on the frame
            for idx, (class_name, count) in enumerate(self.shared_data.class_counts.items()):
                cv.putText(bgr_frame, f"{class_name}: {count}", (10, 30 + idx * 30), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

            cv.imshow('Masked', bgr_frame)

            if cv.waitKey(1) == ord('q'):
                break
        cv.destroyAllWindows()

class PublishTf:
    def __init__(self, shared_data):
        self.shared_data = shared_data
        rospy.loginfo("Publish TF")

        self.tf_broadcaster = tf2_ros.TransformBroadcaster()
        self.obj_tf = TransformStamped()
        self.base_frame = 'map'

        self.rate = rospy.Rate(10)

    def publish_tf(self):
        while not rospy.is_shutdown():
            if self.shared_data.center_x is not None and self.shared_data.center_y is not None:
                quat = tf_trans.quaternion_from_euler(0, 0, 0)
                
                self.obj_tf.header.stamp = rospy.Time.now()
                self.obj_tf.header.frame_id = self.base_frame
                self.obj_tf.child_frame_id = self.shared_data.labels
                self.obj_tf.transform.translation.x = self.shared_data.center_x / 1000.0  # Convert mm to meters
                self.obj_tf.transform.translation.y = self.shared_data.center_y / 1000.0  # Convert mm to meters
                self.obj_tf.transform.translation.z = self.shared_data.depth / 1000.0  # Convert mm to meters
                self.obj_tf.transform.rotation = Quaternion(*quat)
                self.tf_broadcaster.sendTransform(self.obj_tf)

            
            self.rate.sleep()

def main():
    shared_data = ShareValues()

    intel_vision = IntelVision(shared_data)
    publish_tf = PublishTf(shared_data)

    vision_thread = threading.Thread(target=intel_vision.vision)
    tf_thread = threading.Thread(target=publish_tf.publish_tf)

    vision_thread.start()
    tf_thread.start()

    vision_thread.join()
    tf_thread.join()

if __name__ == '__main__':
    main()
