#!/usr/bin/env python3

import cv2 as cv
import numpy as np
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
        self.distance_z = None
        self.obj_orientation = None
        self.labels = None
        self.requested_label = None

class Cv_Vision:
    def __init__(self, shared_data):
        rospy.init_node('YOLO_V3_node')
        rospy.loginfo("YOLO v3 based object detection and tf creation")

        self.video_cap = cv.VideoCapture(0)
        self.frame_rate = 30
        self.width_ = 640
        self.height_ = 480

        self.video_cap.set(cv.CAP_PROP_FPS, self.frame_rate)
        self.video_cap.set(cv.CAP_PROP_FRAME_WIDTH, self.width_)
        self.video_cap.set(cv.CAP_PROP_FRAME_HEIGHT, self.height_)

        self.coco_names = '/home/avinaash/ros_cv/ros1/src/darknet/data/coco.names'
        self.weights = '/home/avinaash/ros_cv/ros1/src/darknet/yolov3-tiny.weights'
        self.config = '/home/avinaash/ros_cv/ros1/src/darknet/cfg/yolov3-tiny.cfg'

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

    def calculate_distance(self, real_height, focal_length, image_height):
        return (real_height * focal_length) / image_height

    def draw_labels(self, img, class_ids, confidences, boxes, conf_threshold=0.5, nms_threshold=0.4):
        indexes = cv.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)
        font = cv.FONT_HERSHEY_PLAIN
        colors = np.random.uniform(0, 255, size=(len(self.classes), 3))

        # Known parameters for distance calculation
        real_height = 4.2  # Known height of the object in cm
        focal_length = 600  # Focal length in pixels

        for i in range(len(boxes)):
            if i in indexes:
                x, y, w, h = boxes[i]
                label = str(self.classes[class_ids[i]])
                color = colors[class_ids[i]]
                cv.rectangle(img, (x, y), (x + w, y + h), color, 2)
                cv.putText(img, label, (x, y - 10), font, 1, color, 2)
                center_x = x + w // 2
                center_y = y + h // 2

                x_offset = (center_x - self.width_ / 2) / 100
                y_offset = (center_y - self.height_ / 2) / 100
                distance_z = self.calculate_distance(real_height, focal_length, h) / 10  # distance in meters

                if label == self.share_data.requested_label:
                    self.share_data.center_x = x_offset
                    self.share_data.center_y = y_offset
                    self.share_data.distance_z = distance_z
                    self.share_data.labels = label
                    print(f"Detected {label} at ({x_offset}, {y_offset}, {distance_z:.2f} m)")

                    rect = cv.minAreaRect(np.array([[x, y], [x + w, y], [x, y + h], [x + w, y + h]]))
                    angle = rect[2]
                    if w < h:
                        angle = 90 + angle

                    # Convert angle to quaternion
                    quat = tf_trans.quaternion_from_euler(0, 0, np.deg2rad(angle))
                    self.share_data.obj_orientation = angle

                    # Publish transform
                    self.publish_tf(x_offset, y_offset, distance_z, quat)

        return img

    def publish_tf(self, x, y, z, quat):
        obj_tf = TransformStamped()
        obj_tf.header.stamp = rospy.Time.now()
        obj_tf.header.frame_id = 'map'
        obj_tf.child_frame_id = self.share_data.labels
        obj_tf.transform.translation.x = x
        obj_tf.transform.translation.y = y
        obj_tf.transform.translation.z = z
        obj_tf.transform.rotation = Quaternion(*quat)
        tf_broadcaster = tf2_ros.TransformBroadcaster()
        tf_broadcaster.sendTransform(obj_tf)

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

    def publish_tf_data(self):
        rate = rospy.Rate(10)  # 10 Hz
        while not rospy.is_shutdown():
            if self.shared_data.labels is not None and self.shared_data.center_x is not None and self.shared_data.distance_z is not None:
                yolo_object = String()
                yolo_object.data = (f"Label: {self.shared_data.labels}, "
                                    f"Center: ({self.shared_data.center_x}, {self.shared_data.center_y}), "
                                    f"Distance: {self.shared_data.distance_z:.2f} m, "
                                    f"Rotation: {self.shared_data.obj_orientation:.2f} degrees")
                self.topic_pub.publish(yolo_object)
            rate.sleep()

def prompt_callback(data, shared_data):
    shared_data.requested_label = data.data
    rospy.loginfo(f"Received prompt for object: {data.data}")

if __name__ == "__main__":
    shared_data = ShareValues()

    cv_vision = Cv_Vision(shared_data)
    publish_tf = PublishTf(shared_data)

    rospy.Subscriber('/object_request', String, prompt_callback, shared_data)

    vision_thread = threading.Thread(target=cv_vision.run)
    tf_thread = threading.Thread(target=publish_tf.publish_tf_data)

    vision_thread.start()
    tf_thread.start()

    vision_thread.join()
    tf_thread.join()
