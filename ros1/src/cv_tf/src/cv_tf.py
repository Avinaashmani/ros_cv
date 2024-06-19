#!/usr/bin/env python3

import cv2 as cv
import time
import numpy as np
import rospy
import tf2_ros
import tf.transformations as tf_trans
from geometry_msgs.msg import TransformStamped, Quaternion

## DETECT BLUE COLOUR CUBE, DRAW BOUNDING BOX AROUND IT AND PUBLISH TF ##

class Cv_Vision:

    def __init__(self):
        
        rospy.init_node('cv_2_tf')
        rospy.loginfo("Publish Blue Cube TF")
        
        self.tf_broadcaster = tf2_ros.TransformBroadcaster()
        self.obj_tf = TransformStamped()

        self.base_frame = 'map'
        self.cube_frame = 'cube_frame'

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

        self.black = np.zeros((300, 512, 3), np.uint8)
        self.blank_window = 'Blank'
        self.over_lay_1 = cv.imread('/home/avinaash/DSA_Python/Machine Vision/overlay_1.png')
        self.over_lay_1 = cv.resize(self.over_lay_1, (640, 480))

        self.frame_top = 'Top center'
        self.frame_br = 'Bottom right'
        self.frame_bl = 'Bottom left'
        self.vision()

    def calculate_focal_length(self, real_height, known_distance, image_height):
        # This method is to measure the focal length once
        return (image_height * known_distance) / real_height

    def calculate_distance(self, real_height, focal_length, image_height):
        return (real_height * focal_length) / image_height

    def vision(self):

        # Known parameters
        real_height = 4.2
        focal_length = 600

        while not rospy.is_shutdown():
            self.text = str(time.time())
            presence, self.frame = self.video_cap.read()

            if not presence:
                print("Error: Could not read frame.")
                break

            hsv = cv.cvtColor(self.frame, cv.COLOR_BGR2HSV)

            lower_mask = np.array([90, 84, 0])
            upper_mask = np.array([180, 255, 255])

            mask = cv.inRange(hsv, lower_mask, upper_mask)
            mask_2 = cv.erode(mask, kernel=None, iterations=2)
            mask_2 = cv.dilate(mask_2, kernel=None, iterations=2)

            res = cv.bitwise_and(self.frame, self.frame, mask=mask)

            # Find contours
            contours, _ = cv.findContours(mask_2, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

            # Draw bounding box around the largest contour
            if contours:
                largest_contour = max(contours, key=cv.contourArea)
                x, y, w, h = cv.boundingRect(largest_contour)

                # Draw the bounding box
                cv.rectangle(self.frame, (x, y), (x + w, y + h), color=(0, 0, 255), thickness=3)

                # Calculate the center of the bounding box
                x_center = x + w / 2
                y_center = y + h / 2

                # Calculate the offsets from the center of the image
                x_offset = (x_center - self.width_ / 2) / 100
                y_offset = (y_center - self.height_ / 2) / 100
                z_distance = ((focal_length * real_height) / h) / 10

                center = (x_offset, y_offset, z_distance)

                # Calculate the distance
                distance = self.calculate_distance(real_height, focal_length, h)
                distance_text = f"Distance: {distance:.2f} cm"
                print(center)

                rect = cv.minAreaRect(largest_contour)
                angle = rect[2]

                if w < h:
                    angle = 90 + angle

                # Convert angle to quaternion
                quat = tf_trans.quaternion_from_euler(0, 0, np.deg2rad(angle))

                self.obj_tf.header.stamp = rospy.Time.now()
                self.obj_tf.header.frame_id = self.base_frame
                self.obj_tf.child_frame_id = self.cube_frame
                self.obj_tf.transform.translation.x = x_offset
                self.obj_tf.transform.translation.y = y_offset
                self.obj_tf.transform.translation.z = 0.0
                self.obj_tf.transform.rotation = Quaternion(*quat)
                self.tf_broadcaster.sendTransform(self.obj_tf)

                # Put distance text on the frame
                cv.putText(self.frame, distance_text, (x, y - 10), self.font, self.font_scale, self.color, 1,
                           cv.LINE_AA)

            overlay = cv.addWeighted(self.frame, 0.9, self.over_lay_1, 0.5, 0)
            cv.putText(overlay, self.text, self.position, self.font, self.font_scale, self.color, 1, cv.LINE_AA)

            cv.imshow(self.frame_top, overlay)
            cv.imshow('Blue Cube', self.frame)

            if cv.waitKey(1) & 0xFF == ord('x'):
                break

        self.video_cap.release()
        cv.destroyAllWindows()
        rospy.Rate(10).sleep()


def main():
    Cv_Vision()
    rospy.spin()

if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        pass
