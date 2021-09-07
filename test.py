#!/usr/bin/env python
# coding: utf-8
"""
Object Detection (On Image) From TF2 Saved Model
=====================================
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'    # Suppress TensorFlow logging (1)
import pathlib
import tensorflow as tf
import cv2
import argparse
import time
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as viz_utils
import numpy as np
import base64
import io
from PIL import Image
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore') 




tf.get_logger().setLevel('ERROR')           # Suppress TensorFlow logging (2)

parser = argparse.ArgumentParser(add_help=False)

parser.add_argument('--model', help='Folder that the Saved Model is Located In',
                    default='exported-models/my_mobilenet_model')
parser.add_argument('--labels', help='Where the Labelmap is Located',
                    default='exported-models/my_mobilenet_model/saved_model/label_map.pbtxt')
parser.add_argument('--threshold', help='Minimum confidence threshold for displaying detected objects',
                    default=0.60)
                    
args = parser.parse_args()
# Enable GPU dynamic memory allocation
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

# PROVIDE PATH TO MODEL DIRECTORY
PATH_TO_MODEL_DIR = args.model

# PROVIDE PATH TO LABEL MAP
PATH_TO_LABELS = args.labels

# PROVIDE THE MINIMUM CONFIDENCE THRESHOLD
MIN_CONF_THRESH = float(args.threshold)

# LOAD THE MODEL

PATH_TO_SAVED_MODEL = PATH_TO_MODEL_DIR + "/saved_model"

print('Loading model...', end='')
start_time = time.time()

# LOAD SAVED MODEL AND BUILD DETECTION FUNCTION
detect_fn = tf.saved_model.load(PATH_TO_SAVED_MODEL)


end_time = time.time()
elapsed_time = end_time - start_time
print('Done! Took {} seconds'.format(elapsed_time))

# LOAD LABEL MAP DATA FOR PLOTTING

category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS, use_display_name=True)


#BASE64 -> IMAGE

def stringToImage(base64_string):
    return np.array(Image.open(io.BytesIO(base64.b64decode(base64_string))))


#GET KEY IN IMAGE
def getstring(image):
  image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
  image_expanded = np.expand_dims(image_rgb, axis=0)

  # The input needs to be a tensor, convert it using `tf.convert_to_tensor`.
  input_tensor = tf.convert_to_tensor(image)
  # The model expects a batch of images, so add an axis with `tf.newaxis`.
  input_tensor = input_tensor[tf.newaxis, ...]

  # input_tensor = np.expand_dims(image_np, 0)
  detections = detect_fn(input_tensor)

  # All outputs are batches tensors.
  # Convert to numpy arrays, and take index [0] to remove the batch dimension.
  # We're only interested in the first num_detections.
  num_detections = int(detections.pop('num_detections'))
  detections = {key: value[0, :num_detections].numpy()
                for key, value in detections.items()}
  detections['num_detections'] = num_detections

  # detection_classes should be ints.
  detections['detection_classes'] = detections['detection_classes'].astype(np.int64)


  #get
  boxes = detections['detection_boxes']
  classes= detections['detection_classes']
  scores=detections['detection_scores']

  image_with_detections = image.copy()

  viz_utils.visualize_boxes_and_labels_on_image_array(
        image_with_detections,
        boxes,
        classes,
        scores,
        category_index,
        use_normalized_coordinates=True,
        max_boxes_to_draw=200,
        min_score_thresh=MIN_CONF_THRESH,
        agnostic_mode=False)
  # captcha list
  captcha_array = []

  for i , b in enumerate(boxes):
    for Symbol in range(43):
      if classes[i] ==  Symbol:
        if scores[i]>=0.65:
          mid_x = ( boxes[i][0] + boxes[i][3] ) / 2
          captcha_array.append([category_index[Symbol].get('name'),mid_x,scores[i]])


  #sắp xếp
  for number in range(22):
    for captcha_number in range(len(captcha_array)-1):
      if captcha_array[captcha_number][1] > captcha_array[captcha_number+1][1]:
        temporary_captcha = captcha_array[captcha_number]
        captcha_array[captcha_number] = captcha_array[captcha_number+1]
        captcha_array[captcha_number+1] = temporary_captcha

  '''
  #xóa chữ sát nhau 0.1cm ( ko cần theo captcha ko có ( đã xóa nền ko cần))
  average = 0
  average_distance_error=3
  captcha_len = len(captcha_array)-1

  while captcha_len > 0:
    average += captcha_array[captcha_len][1]- captcha_array[captcha_len-1][1]
    captcha_len -= 1


  average = average/(len(captcha_array)+average_distance_error)     
  captcha_array_filtered = list(captcha_array)
  captcha_len = len(captcha_array)-1

  while captcha_len > 0:
    # if average distance is larger than error distance
    if captcha_array[captcha_len][1]- captcha_array[captcha_len-1][1] < average:
    # check which symbol has higher detection percentage
      if captcha_array[captcha_len][2] > captcha_array[captcha_len-1][2]:
        del captcha_array_filtered[captcha_len-1]
      else:
        del captcha_array_filtered[captcha_len]
    captcha_len -= 1
  '''
  captcha_string = ""
  for captcha_letter in range(len(captcha_array)):
    captcha_string += captcha_array[captcha_letter][0]

  return captcha_string

def keycaptcha(captcha_string):
  #key
  captcha_string = captcha_string.split('->')
  stringcp =''.join(captcha_string)
  key = []
  key.append(stringcp[0]+stringcp[2])
  key.append(stringcp[1]+stringcp[3])
  key.append(stringcp[4]+stringcp[6])
  key.append(stringcp[5]+stringcp[7])

  return ''.join(key)
# show captcha
def destroy_captcha(captcha_string, key):
  result=[]
  for i in range(len(captcha_string)):
    for j in range(0,9,2):
      if captcha_string[i]== key[j]:
        result+=key[j+1]
  return ''.join(result)
# SET MIN_SCORE_THRESH BASED ON YOU MINIMUM THRESHOLD FOR DETECTIONS
while True:
  # PROVIDE PATH TO IMAGE DIRECTORY
  img=input('IMG: \n')
  IMAGE_PATHS = 'image/'+img
  # Suppress Matplotlib warnings

  #start time
  print('Loading DESTROY CAPTCHA...')
  start_time = time.time()

  #get key and code image
  image = cv2.imread(IMAGE_PATHS)
  h= int(image.shape[0]/2)-10
  code = image.copy()
  key= image.copy()

  key[:h,:]=(255,255,255)
  code[h:,:]=(255,255,255)

  k =int(h + h/2)+10

  key_down = key.copy()
  key_down[:k,:]=(255,255,255)

  key_up= key.copy()
  key_up[k:,:]=(255,255,255)

  #DESTROY CAPTCHA
  captcha_string=getstring(code)
  up = getstring(key_up)
  key = keycaptcha(up) + ''.join(getstring(key_down).split('->'))

  
  #SHOW CAPTCHA
  print(destroy_captcha(captcha_string,key))
  
  #end time
  end_time = time.time()
  elapsed_time = end_time - start_time
  print('Done! Took {} seconds'.format(elapsed_time))
  
  
  if input('q: out & enter : next \n') =='q':
    break


# DISPLAYS OUTPUT IMAGE
'''
cv2.imshow('Object Detector', image_with_detections)
  # CLOSES WINDOW ONCE KEY IS PRESSED
  cv2.waitKey(0)
  # CLEANUP
  cv2.destroyAllWindows()

'''
