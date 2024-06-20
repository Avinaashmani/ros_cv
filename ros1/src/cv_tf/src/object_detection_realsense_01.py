#!/usr/bin/env python3
import cv2 as cv
from mask_rcnn import MaskRCNN

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
        
        self.video_cap = cv.VideoCapture(0)
        self.frame_rate = 30
        self.width_ = 640
        self.height_ = 480
        self.skip_frames = 2  # Skip every 2 frames to speed up processing

        self.video_cap.set(cv.CAP_PROP_FPS, self.frame_rate)
        self.video_cap.set(cv.CAP_PROP_FRAME_WIDTH, self.width_)
        self.video_cap.set(cv.CAP_PROP_FRAME_HEIGHT, self.height_)       
        
        self.mask = MaskRCNN()
        self.shared_data = shared_data

        self.topic_pub = rospy.Publisher('/intel_dist', String, queue_size=10)
        self.data_msg = String()

    def vision(self):
        frame_count = 0
        while True:
            ret, bgr_frame = self.video_cap.read()
            if not ret:
                continue

            # Skip frames to reduce processing load
            frame_count += 1
            if frame_count % self.skip_frames != 0:
                continue

            boxes, classes, contours, centers = self.mask.detect_objects_mask(bgr_frame)
            self.shared_data.class_counts = {}

            bgr_frame, boxes, classes = self.mask.draw_object_mask(bgr_frame)
            for box, class_id, center in zip(boxes, classes, centers):
                x, y, x2, y2 = box
                cx, cy = center
                class_name = self.mask.classes[class_id]

                self.shared_data.center_x = cx
                self.shared_data.center_y = cy
                self.shared_data.labels = class_name

                if class_name in self.shared_data.class_counts:
                    self.shared_data.class_counts[class_name] += 1
                else:
                    self.shared_data.class_counts[class_name] = 1

                rospy.loginfo(f"Class: {class_name}, Center: ({cx}, {cy})")

                yolo_object = String()
                yolo_object.data = f"Label: {class_name} Center: ({cx}, {cy})"
                self.topic_pub.publish(yolo_object)

            self.mask.draw_object_info(bgr_frame)
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
                self.obj_tf.transform.translation.x = self.shared_data.center_x / 1000.0
                self.obj_tf.transform.translation.y = self.shared_data.center_y / 1000.0
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
