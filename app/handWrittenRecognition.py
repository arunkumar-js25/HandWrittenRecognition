from keras.layers import *
from keras.models import Model
from keras.optimizers import Adam

import string
char_list = string.ascii_letters + string.digits

# Model.py
def unet(pretrained_weights=None, input_size=(512, 512, 1)):
    inputs = Input(input_size)
    conv1 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(inputs)
    conv1 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    conv2 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool1)
    conv2 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    conv3 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool2)
    conv3 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    conv4 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool3)
    conv4 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv4)
    drop4 = Dropout(0.5)(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)

    conv5 = Conv2D(1024, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool4)
    conv5 = Conv2D(1024, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv5)
    drop5 = Dropout(0.5)(conv5)

    up6 = Conv2D(512, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(drop5))
    merge6 = concatenate([drop4, up6], axis=3)
    conv6 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge6)
    conv6 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv6)

    up7 = Conv2D(256, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(conv6))
    merge7 = concatenate([conv3, up7], axis=3)
    conv7 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge7)
    conv7 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv7)

    up8 = Conv2D(128, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(conv7))
    merge8 = concatenate([conv2, up8], axis=3)
    conv8 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge8)
    conv8 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv8)

    up9 = Conv2D(64, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(conv8))
    merge9 = concatenate([conv1, up9], axis=3)
    conv9 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge9)
    conv9 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv9)
    conv9 = Conv2D(2, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv9)
    conv10 = Conv2D(1, 1, activation='sigmoid')(conv9)
    model = Model(inputs, conv10)

    model.compile(optimizer="adam", loss='binary_crossentropy', metrics=['accuracy'])  # (lr=1e-4)
    # model.summary()
    if pretrained_weights:
        model.load_weights(pretrained_weights)

    return model


# utils.py
import cv2
import numpy as np
import math
from PIL import Image
import matplotlib.pyplot as plt

def pad_img(img):
    old_h, old_w = img.shape[0], img.shape[1]

    # Pad the height.

    # If height is less than 512 then pad to 512
    if old_h < 512:
        to_pad = np.ones((512 - old_h, old_w)) * 255
        img = np.concatenate((img, to_pad))
        new_height = 512
    else:
        # If height >512 then pad to nearest 10.
        to_pad = np.ones((roundup(old_h) - old_h, old_w)) * 255
        img = np.concatenate((img, to_pad))
        new_height = roundup(old_h)

    # Pad the width.
    if old_w < 512:
        to_pad = np.ones((new_height, 512 - old_w)) * 255
        img = np.concatenate((img, to_pad), axis=1)
        new_width = 512
    else:
        to_pad = np.ones((new_height, roundup(old_w) - old_w)) * 255
        img = np.concatenate((img, to_pad), axis=1)
        new_width = roundup(old_w) - old_w
    return img


def pad_seg(img):
    old_h, old_w = img.shape[0], img.shape[1]

    # Pad the height.

    # If height is less than 512 then pad to 512
    if old_h < 512:
        to_pad = np.zeros((512 - old_h, old_w))
        img = np.concatenate((img, to_pad))
        new_height = 512
    else:
        # If height >512 then pad to nearest 10.
        to_pad = np.zeros((roundup(old_h) - old_h, old_w))
        img = np.concatenate((img, to_pad))
        new_height = roundup(old_h)

    # Pad the width.
    if old_w < 512:
        to_pad = np.zeros((new_height, 512 - old_w))
        img = np.concatenate((img, to_pad), axis=1)
        new_width = 512
    else:
        to_pad = np.zeros((new_height, roundup(old_w) - old_w))
        img = np.concatenate((img, to_pad), axis=1)
        new_width = roundup(old_w) - old_w
    return img


# This function helps in visualizing the image segmentation mask and image side by side using matplotlib subplots.
def visualize(img, seg_img):
    """
    Visualizes image
    """
    plt.figure(figsize=(20, 20))
    plt.subplot(1, 2, 1)
    plt.imshow(img)
    plt.title('Image')
    plt.subplot(1, 2, 2)
    plt.imshow(seg_img, cmap='gray')
    plt.title('Segmented Image')
    plt.show()


def get_segmented_img(img, n_classes):
    """
    Loads in the segmented image and create suitable segmentation label.
    """
    seg_labels = np.zeros((512, 512, 1))
    # print('imggggg', img)
    img = cv2.resize(img, (512, 512))
    img = img[:, :, 0]
    cl_list = [0, 24]
    seg_labels[:, :, 0] = (img != 0).astype(int)

    return seg_labels


def preprocess_img(img, imgSize):
    "put img into target img of size imgSize, transpose for TF and normalize gray-values"
    # there are damaged files in IAM dataset - just use black image instead
    if img is None:
        img = np.zeros([imgSize[1], imgSize[0]])
        print("Image None!")

    # create target image and copy sample image into it
    (wt, ht) = imgSize
    (h, w) = img.shape
    fx = w / wt
    fy = h / ht
    f = max(fx, fy)
    newSize = (max(min(wt, int(w / f)), 1),
               max(min(ht, int(h / f)), 1))  # scale according to f (result at least 1 and at most wt or ht)
    img = cv2.resize(img, newSize,
                     interpolation=cv2.INTER_CUBIC)  # INTER_CUBIC interpolation best approximate the pixels image
    # see this https://stackoverflow.com/a/57503843/7338066
    most_freq_pixel = find_dominant_color(Image.fromarray(img))
    target = np.ones([ht, wt]) * most_freq_pixel
    target[0:newSize[1], 0:newSize[0]] = img
    img = target

    return img


def roundup(x):
    return int(math.ceil(x / 10.0)) * 10


def segment_into_lines(filename, model):
    line_img_array = []
    # Loading the image and performing thresholding on it and then resizing.
    img = cv2.imread(f'{filename}', 0)
    ret, img = cv2.threshold(img, 150, 255, cv2.THRESH_BINARY_INV)
    img = cv2.resize(img, (512, 512))
    # Expanding the dimension to account for the batch dimension.
    img = np.expand_dims(img, axis=-1)
    # Expanding dimension along channel axis.
    img = np.expand_dims(img, axis=0)

    # Predict the segmentation mask.
    pred = model.predict(img)
    print("segment_into_lines - passed")
    # Remove the batch and channel dimension for performing the binarization.
    pred = np.squeeze(np.squeeze(pred, axis=0), axis=-1)

    # Performing the binarization of the predicted mask for contour detection.
    coordinates = []
    img = cv2.normalize(src=pred, dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
    cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU, img)
    # Opening the original image to get the original dimension information.
    ori_img = cv2.imread(f'{filename}', 0)

    (H, W) = ori_img.shape[:2]
    (newW, newH) = (512, 512)
    rW = W / float(newW)
    rH = H / float(newH)

    # Contour detection and bouding box generation.
    contours, hier = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    for c in contours:
        # get the bounding rect
        x, y, w, h = cv2.boundingRect(c)
        # cv2.rectangle(ori_img, (int(x*rW), int(y*rH)), (int((x+w)*rW),int((y+h)*rH)), (255,0,0), 1)
        coordinates.append((int(x * rW), int(y * rH), int((x + w) * rW), int((y + h) * rH)))
    # cv2.imwrite("output.jpg",ori_img)

    # Cropping the lines from the original image using the bouding boxes generated above.
    for i in range(len(coordinates) - 1, -1, -1):
        coors = coordinates[i]

        p_img = ori_img[coors[1]:coors[3], coors[0]:coors[2]].copy()

        line_img_array.append(p_img)

    return line_img_array


def segment_into_words(line_img, idx, model):
    """This function takes in the line image and line index returns word images and the reference
    of line they belong to."""
    img = pad_img(line_img)
    ori_img = img.copy()
    # ori_img=np.stack((ori_img,)*3, axis=-1)
    ret, img = cv2.threshold(img, 150, 255, cv2.THRESH_BINARY_INV)

    img = cv2.resize(img, (512, 512))
    img = np.expand_dims(img, axis=-1)
    img = img / 255
    img = np.expand_dims(img, axis=0)
    seg_pred = model.predict(img)
    seg_pred = np.squeeze(np.squeeze(seg_pred, axis=0), axis=-1)
    seg_pred = cv2.normalize(src=seg_pred, dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
    cv2.threshold(seg_pred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU, seg_pred)
    contours, hier = cv2.findContours(seg_pred, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    (H, W) = ori_img.shape[:2]
    (newW, newH) = (512, 512)
    rW = W / float(newW)
    rH = H / float(newH)

    coordinates = []
    for c in contours:
        # get the bounding rect
        x, y, w, h = cv2.boundingRect(c)
        # draw a white rectangle to visualize the bounding rect
        # cv2.rectangle(ori_img, (int(x*rW), int(y*rH)), (int((x+w)*rW),int((y+h)*rH)), (255,0,0), 1)
        coordinates.append((int(x * rW), int(y * rH), int((x + w) * rW), int((y + h) * rH)))

    coordinates = sort_word(coordinates)  # Sorting according to x-coordinates.
    word_counter = 0

    word_array = []
    line_indicator = []

    for (x1, y1, x2, y2) in coordinates:
        word_img = ori_img[y1:y2, x1:x2]
        word_img = preprocess_img(word_img, (128, 32))
        word_img = np.expand_dims(word_img, axis=-1)
        word_array.append(word_img)
        line_indicator.append(idx)

    return line_indicator, word_array


def sort_word(wordlist):
    wordlist.sort(key=lambda x: x[0])
    return wordlist


def find_dominant_color(image):
    # Resizing parameters
    width, height = 150, 150
    image = image.resize((width, height), resample=0)
    # Get colors from image object
    pixels = image.getcolors(width * height)
    # Sort them by count number(first element of tuple)
    sorted_pixels = sorted(pixels, key=lambda t: t[0])
    # Get the most frequent color
    dominant_color = sorted_pixels[-1][1]
    return dominant_color


# RecognizeWord.py
import cv2
import numpy as np
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
from keras_preprocessing.sequence import pad_sequences
from keras.layers import Dense, LSTM, Reshape, BatchNormalization, Input, Conv2D, MaxPool2D, Lambda, Bidirectional, \
    Dropout
from keras.models import Model
import keras.backend as K
import tensorflow as tf
from PIL import Image

def recognize_model():
    # spell = SpellChecker()
    inputs = Input(shape=(32, 128, 1))
    conv_1 = Conv2D(64, (3, 3), activation='relu', padding='same')(inputs)
    pool_1 = MaxPool2D(pool_size=(2, 2), strides=2)(conv_1)
    conv_2 = Conv2D(128, (3, 3), activation='relu', padding='same')(pool_1)
    pool_2 = MaxPool2D(pool_size=(2, 2), strides=2)(conv_2)
    conv_3 = Conv2D(256, (3, 3), activation='relu', padding='same')(pool_2)
    conv_4 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv_3)
    pool_4 = MaxPool2D(pool_size=(2, 1))(conv_4)
    conv_5 = Conv2D(512, (3, 3), activation='relu', padding='same')(pool_4)
    batch_norm_5 = BatchNormalization()(conv_5)
    conv_6 = Conv2D(512, (3, 3), activation='relu', padding='same')(batch_norm_5)
    batch_norm_6 = BatchNormalization()(conv_6)
    pool_6 = MaxPool2D(pool_size=(2, 1))(batch_norm_6)
    conv_7 = Conv2D(512, (2, 2), activation='relu')(pool_6)
    squeezed = Lambda(lambda x: K.squeeze(x, 1))(conv_7)
    blstm_1 = Bidirectional(LSTM(128, return_sequences=True, dropout=0.2))(squeezed)
    blstm_2 = Bidirectional(LSTM(128, return_sequences=True, dropout=0.2))(blstm_1)
    outputs = Dense(len(char_list) + 1, activation='softmax')(blstm_2)
    act_model = Model(inputs, outputs)

    return act_model

def predict(line_indicator, word_array, n_lines, act_model):
    line_rec = []
    for listidx in range(n_lines):
        line_rec.append([])

    response = ""
    predictions = act_model.predict(word_array)
    out = K.get_value(
        K.ctc_decode(predictions, input_length=np.ones(predictions.shape[0]) * predictions.shape[1], greedy=True)[0][0])
    lw_idx = 0
    for wordidxs in out:
        word = []
        for char in wordidxs:
            if int(char) != -1:
                word.append(char_list[int(char)])
        word = ''.join(word)
        line_rec[line_indicator[lw_idx]].append(word)

        lw_idx += 1

    for listidx in range(n_lines):
        line = ' '.join(line_rec[listidx])
        response = response + line + '\n'

    return response

def recognize_words(line_indicator, word_array, n_lines, act_model):
    file = open('recognized_texts.txt', 'w')
    line_rec = []
    for listidx in range(n_lines):
        line_rec.append([])

    predictions = act_model.predict(word_array)

    out = K.get_value(
        K.ctc_decode(predictions, input_length=np.ones(predictions.shape[0]) * predictions.shape[1], greedy=True)[0][0])

    lw_idx = 0
    for wordidxs in out:
        word = []
        for char in wordidxs:
            if int(char) != -1:
                word.append(char_list[int(char)])
        # word = spell.correction(''.join(word))
        word = ''.join(word)
        line_rec[line_indicator[lw_idx]].append(word)
        lw_idx += 1

    for listidx in range(n_lines):
        line = ' '.join(line_rec[listidx])
        print(line)
        file.writelines(line + '\n')
    file.close()

import random
def batch_generator(filelist, n_classes, batch_size):
    while True:
        X = []
        Y = []
        for i in range(batch_size):
            fn = random.choice(filelist)
            img = cv2.imread(f'./content/Dataset1/img/{fn}.jpg', 0)
            img = pad_img(img)
            ret, img = cv2.threshold(img, 150, 255, cv2.THRESH_BINARY_INV)

            img = cv2.resize(img, (512, 512))
            img = np.expand_dims(img, axis=-1)
            # img = np.stack((img,)*3, axis=-1)
            img = img / 255

            seg = cv2.imread(f'./content/Dataset1/mask/{fn}_mask.png', 0)
            seg = pad_seg(seg)
            seg = cv2.resize(seg, (512, 512))
            seg = np.stack((seg,) * 3, axis=-1)
            seg = get_segmented_img(seg, n_classes)

            X.append(img)
            Y.append(seg)
        yield np.array(X), np.array(Y)


def extractImage(inputImagePath):
    # Open image and segment into lines
    line_img_array = segment_into_lines(inputImagePath, LS_model)
    # print(line_img_array)

    # Creating lists to store the line indexes,words list.
    full_index_indicator = []
    all_words_list = []
    # Variable to count the total no of lines in page.
    len_line_arr = 0
    # Segment the lines into words and store as arrays.
    for idx, im in enumerate(line_img_array):
        line_indicator, word_array = segment_into_words(im, idx, WS_model)
        for k in range(len(word_array)):
            full_index_indicator.append(line_indicator[k])
            all_words_list.append(word_array[k])
        len_line_arr += 1
    all_words_list = np.array(all_words_list)

    # Perform the recognition on list of list of words.
    return predict(full_index_indicator, all_words_list, len_line_arr, act_model)


LS_model = unet()
LS_model.load_weights('app/content/LS_weights00000001.h5')

WS_model = unet()
WS_model.load_weights('app/content/WS_weights00000001.h5')

act_model = recognize_model()
act_model.load_weights('app/content/best_model.hdf5')

#res = extractImage(r'D:\SourceCodes\Python\Machine Learning\Project_TextRecognition_anand\content\test_img.JPG')
#print(res)