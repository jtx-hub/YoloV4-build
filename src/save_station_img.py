#!/usr/bin/env python
# -*- coding=utf8 -*-
"""
# Author: Your Name
# Created Time : Tue 22 Nov 2022 07:30:31 PM CST
# File Name: a.py
# Description:
"""
import datetime
import os.path
import time

import cv2

def get_img_from_camera_net(folder, url, delay):
    cap = cv2.VideoCapture(url)
    ret, frame = cap.read()
    while ret:
        ret, frame = cap.read()
        local_time = time.localtime(time.time())
        date = time.strftime("%Y-%m-%d", local_time)
        folder_with_date = folder + date + '/'
        if not os.path.exists(folder_with_date):
            os.mkdir(folder_with_date)
        file = folder_with_date + "{}.jpg".format(time.strftime("%Y-%m-%d-%H-%M-%S", local_time))
        if os.path.isfile(file) == False and local_time.tm_sec % delay == 0 and local_time.tm_hour >= 6 and local_time.tm_hour < 18:
            cv2.imwrite(file, frame, [int(cv2.IMWRITE_JPEG_QUALITY), 60])
            print('save image: ' + file)
    cv2.destroyAllWindows()


# 测试
if __name__ == '__main__':
    url = "rtsp://admin:Dh123456789@10.8.37.71/"
    folder = '/algorithm/stationImg/'
    get_img_from_camera_net(folder, url, 30)
