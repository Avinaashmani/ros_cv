#!/usr/bin/env python3

import cv2 as cv
import numpy as np
import time

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
        self.obj_orientation = None
        self.labels = None

class Cv_Vision:

    def __init__(self, shared_data):
        rospy.init_node('YOLO_V3_node')
        rospy.loginfo("YOLO v3 based object detection and tf creation")

        self.frame = None
        self.text = None
        self.hsv = None
        self.mask = None

        self.video_cap = cv.VideoCapture(0)
        self.frame_rate = 30
        self.width_ = 640
        self.height_ = 480

        self.video_cap.set(cv.CAP_PROP_FPS, self.frame_rate)
        self.video_cap.set(cv.CAP_PROP_FRAME_WIDTH, self.width_)
        self.video_cap.set(cv.CAP_PROP_FRAME_HEIGHT, self.height_)

        self.font = cv.FONT_HERSHEY_PLAIN
        self.color = (255, 255, 255)
        self.position = (10, 20)
        self.font_scale = 0.5

        self.x_offset = 0.0
        self.y_offset = 0.0

        self.black = np.zeros((300, 512, 3), np.uint8)
        self.blank_window = 'Blank'

        self.coco_names = '/home/avinaash/ros_cv/src/darknet/coco.names'
        self.weights = '/home/avinaash/ros_cv/src/darknet/yolov3.weights'
        self.config = '/home/avinaash/ros_cv/src/darknet/cfg/yolov3.cfg'

        self.net = cv.dnn.readNet(self.weights, self.config)

        self.share_data = shared_data
        self.data = String()

    def load_yolo(self):
        with open(self.coco_names, 'r') as f:
            self.classes = [line.strip() for line in f.readlines()]
        layer_names = self.net.getLayerNames()
        self.output_layers = [layer_names[i - 1] for i in self.net.getUnconnectedOutLayers().flatten()]

    def detect_objects(self, img):
        height, width, channels = img.shape
        blob = cv.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
        self.net.setInput(blob)
        outputs = self.net.forward(self.output_layers)
        return outputs, height, width

    def get_boxes(self, outputs, height, width, threshold=0.5):
        class_ids = []
        confidences = []
        boxes = []
        for output in outputs:
            for detection in output:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > threshold:
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)
                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)

        return class_ids, confidences, boxes

    def draw_labels(self, img, class_ids, confidences, boxes, conf_threshold=0.5, nms_threshold=0.4):
        indexes = cv.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)
        font = cv.FONT_HERSHEY_PLAIN
        colors = np.random.uniform(0, 255, size=(len(self.classes), 3))

        for i in range(len(boxes)):
            if i in indexes:
                x, y, w, h = boxes[i]
                label = str(self.classes[class_ids[i]])
                color = colors[class_ids[i]]
                cv.rectangle(img, (x, y), (x + w, y + h), color, 2)
                cv.putText(img, label, (x, y - 10), font, 1, color, 2)
                center_x = x + w // 2
                center_y = y + h // 2
                
                if x >= 0 and y >= 0 and (x + w) <= img.shape[1] and (y + h) <= img.shape[0]:
                    roi = img[y:y+h, x:x+w]
                    gray = cv.cvtColor(roi, cv.COLOR_BGR2GRAY)
                    _, binary = cv.threshold(gray, 50, 255, cv.THRESH_BINARY)
                    contours, _ = cv.findContours(binary, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
                    
                    angle = 0.0
                    if contours:
                        cnt = contours[0]
                        rect = cv.minAreaRect(cnt)
                        angle = rect[-1]
                        if angle < -45:
                            angle = 90 + angle

                    self.x_offset = (center_x - self.width_ / 2) / 100
                    self.y_offset = (center_y - self.height_ / 2) / 100

                    self.share_data.center_x = self.x_offset
                    self.share_data.center_y = self.y_offset
                    self.share_data.obj_orientation = angle
                    self.share_data.labels = label

                    print(f"Label: {label}, Center: ({center_x}, {center_y}), Rotation: {angle:.2f} degrees")

        return img

    def run(self):
        self.load_yolo()
        while not rospy.is_shutdown():
            ret, frame = self.video_cap.read()
            if not ret:
                break

            outputs, height, width = self.detect_objects(frame)
            class_ids, confidences, boxes = self.get_boxes(outputs, height, width)
            frame = self.draw_labels(frame, class_ids, confidences, boxes)
            
            cv.imshow('YOLO Object Detection', frame)
            if cv.waitKey(1) & 0xFF == ord('q'):
                break
        self.video_cap.release()
        cv.destroyAllWindows()

class PublishTf:

    def __init__(self, shared_data):
        self.shared_data = shared_data
        rospy.loginfo("Publish Detected Object TF")
        self.topic_pub = rospy.Publisher('/yolo_object_detect', String, queue_size=10)
        self.tf_broadcaster = tf2_ros.TransformBroadcaster()
        self.obj_tf = TransformStamped()
        self.base_frame = 'map'

    def publish_tf(self):
        rate = rospy.Rate(10)  # 10 Hz
        while not rospy.is_shutdown():
            if self.shared_data.center_x is not None and self.shared_data.center_y is not None:
                quat = tf_trans.quaternion_from_euler(0, 0, np.deg2rad(self.shared_data.obj_orientation))
                
                self.obj_tf.header.stamp = rospy.Time.now()
                self.obj_tf.header.frame_id = self.base_frame
                self.obj_tf.child_frame_id = self.shared_data.labels
                self.obj_tf.transform.translation.x = self.shared_data.center_x 
                self.obj_tf.transform.translation.y = self.shared_data.center_y 
                self.obj_tf.transform.translation.z = 0.0
                self.obj_tf.transform.rotation = Quaternion(*quat)
                self.tf_broadcaster.sendTransform(self.obj_tf)

                yolo_object = String()
                yolo_object.data = f"Label: {self.shared_data.labels}, Center: ({self.shared_data.center_x}, {self.shared_data.center_y}), Rotation: {self.shared_data.obj_orientation:.2f} degrees"
                self.topic_pub.publish(yolo_object)
            
            rate.sleep()

if __name__ == "__main__":
    shared_data = ShareValues()

    cv_vision = Cv_Vision(shared_data)
    publish_tf = PublishTf(shared_data)

    vision_thread = threading.Thread(target=cv_vision.run)
    tf_thread = threading.Thread(target=publish_tf.publish_tf)

    vision_thread.start()
    tf_thread.start()

    vision_thread.join()
    tf_thread.join()
