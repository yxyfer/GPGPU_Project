#!/usr/bin/env python3

import cv2
import sys
import json
import numpy as np

def read_image(path):
    return cv2.imread(path)

def save_image(image, path):
    return cv2.imwrite(path, image)

def show_image(image):
    cv2.imshow("test", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def add_bbox(image, bbox, color, thick):
    x1 = bbox[0]
    y1 = bbox[1]
    x2 = x1 + bbox[2]
    y2 = y1 + bbox[3]
    cv2.rectangle(image,(x1, y1), (x2, y2), color, thick)

def add_bboxes(image, bboxes):
    color = (0,255,0)
    thick = 1
    for bbox in bboxes:
        add_bbox(image, bbox, color, thick)

def create_video(images, path):
    height, width, layers = images[0].shape
    black_image = np.zeros((height,width,3), np.uint8)
    video = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*'DIVX'), fps_video, (width,height))
    for i in range(len(images)):
        video.write(images[i])
    video.write(black_image)
    video.release()

def main(bboxes_txt):
    bboxes_json = json.loads(bboxes_txt)
    cpt = 0
    img_array = []
    for key in bboxes_json:
        image_path = path_to_images + key
        image_output_path = path_to_outputs + str(cpt) + ".jpg"

        image = read_image(image_path)
        add_bboxes(image, bboxes_json[key])
        img_array.append(image)
        save_image(image, image_output_path)
        cpt += 1
    create_video(img_array, path_to_video)

if __name__ == "__main__":
    path_to_images = '../'
    path_to_outputs = '../images/output/'
    path_to_video = '../images/output/video.avi'
    fps_video = 1

    read_stdin = sys.stdin.read()
    main(read_stdin)
    cv2.destroyAllWindows()
