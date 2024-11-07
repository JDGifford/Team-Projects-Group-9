import torch
from torch import nn
from tqdm.notebook import tqdm
from torchvision.utils import draw_bounding_boxes
import albumentations as A
from torchvision.models import resnet50, ResNet50_Weights
import pandas as pd
import numpy as np
import os
import cv2
import random
import matplotlib.pyplot as plt
from PIL import Image, ImageFilter
from skimage.io import imshow
from sklearn import preprocessing as p
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, precision_score, f1_score
import psutil
#import sys
#import utils
import timm

DEATH_STAR_IMAGES = "Death_Star_Image_Dataset/"
NON_DEATH_STAR_IMAGES = "Non_Death_Star_Image_Dataset/"

min_max_scaler = p.MinMaxScaler()
img_list = [x for x in os.listdir(DEATH_STAR_IMAGES)]
img_list2 = [y for y in os.listdir(NON_DEATH_STAR_IMAGES)]
img = cv2.imread(DEATH_STAR_IMAGES + "Death_Star_Plans(91).png")
img = cv2.resize(img, (532, 445), interpolation=cv2.INTER_AREA)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
upLeftPoint = (249, 205)
lowRightPoint = (278, 229)
bound_box_img = cv2.rectangle(img, upLeftPoint, lowRightPoint, (255, 0, 0), 2)

y = np.ones(len(img_list))
y2 = np.zeros(len(img_list2))
y_true = np.concatenate((y, y2), axis=None)#Death Star Image = 1 and Non-Death Star Image = 0
image_list = np.concatenate((img_list, img_list2))
train_list = list(zip(image_list, y_true))#merge the list of images and list of true values to make array of pixel values and so when its shuffled their labels won't get mixed up  
random.shuffle(train_list)#shuffle the images to prevent overfitting

image_size_list = []
for i in range(len(image_list)):
  if "Death_Star_Plans" in image_list[i]:
    image = Image.open(DEATH_STAR_IMAGES + image_list[i])
  else:
    image = Image.open(NON_DEATH_STAR_IMAGES + image_list[i])
  image_size_list.append(image.size)

avg_row = 0
avg_col = 0
for i in range(len(image_size_list)):#calculate the mean image size
  avg_row += image_size_list[i][0]
  avg_col += image_size_list[i][1]
avg_row /= len(image_size_list)
avg_col /= len(image_size_list)
avg_size = (int(avg_row), int(avg_col))

def data_Preprocess(images):
  images_arr = []
  for i in range(len(images)):
    if "Death_Star" in images[i][0]:
      image = Image.open(DEATH_STAR_IMAGES + images[i][0])
    else:
      image = Image.open(NON_DEATH_STAR_IMAGES + images[i][0])
    image = image.convert("L")#convert image to black and white image
    image = image.resize((avg_size))#resize the image to the mean size of all images
    image = min_max_scaler.fit_transform(image)#normalize the image using Min-Max Scaler
    image_arr = np.asarray(image)#conver the image to numpy array of pixel values
    image_arr = image_arr.flatten()#flatten the image to dimension (1xd) thus it'll be just 1 row of d pixel values and d is # of rows * # of columns in numpy array of pixel values
    images[i] = list(images[i])
    images[i][0] = image_arr
    images_arr.append((images[i][0], images[i][1]))
  im_list, label_list = zip(*images_arr)
  images_arr = np.asarray(im_list)
  return images_arr, label_list

class NearestNeighbor:
  def __init__(self):
    pass

  #Memorize all the training data
  def train(self, X, y):
    #self.neighbors = k
    self.Xtr = X
    self.Ytr = y

  def predict(self, X):
    num_test = X.shape[0]
    Ypred = np.zeros(num_test, dtype=self.Ytr[0].dtype)
    for i in range(num_test):#for each test image
      distances = np.sum(np.abs(self.Xtr - X[i,:]), axis=1)#calculate the euclidean distance to find which training image the ith test image is closest to
      min_index = np.argmin(distances)#get index of image with smallest distance
      Ypred[i] = self.Ytr[min_index]#predict the label of the ith test image
    return Ypred
  
im_list, label_list = data_Preprocess(train_list)
x_train, x_test, y_train, y_test = train_test_split(im_list, label_list, test_size=0.20, shuffle=False)#Split dataset into train and test data

neigh = NearestNeighbor
neigh.train(neigh, x_train, y_train)
y_pred = neigh.predict(neigh, x_test)

print("Accuracy is: ", accuracy_score(y_test, y_pred))
print("Micro precision is: ", precision_score(y_test, y_pred, average='micro'))
print("Micro recall is: ", recall_score(y_test, y_pred, average='micro'))
print("Micro F1-Score is: ", f1_score(y_test, y_pred, average='micro'))
print("Macro precision is: ", precision_score(y_test, y_pred, average='macro'))
print("Macro recall is: ", recall_score(y_test, y_pred, average='macro'))
print("Macro F1-Score is: ", f1_score(y_test, y_pred, average='macro'))

tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
print("TP:", tp, "TN:", tn, "FP:", fp, "FN:", fn)
print("Accuracy is:", accuracy_score(y_test, y_pred))
print("Precision is:", precision_score(y_test, y_pred))
print("Recall is:", recall_score(y_test, y_pred))
print("F1-Score is:", f1_score(y_test, y_pred))
process = psutil.Process(os.getpid())
memory_info = process.memory_info()
print("Memory used is:", memory_info.rss / 1024**2, "MB")