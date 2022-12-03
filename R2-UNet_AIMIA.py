import os
import cv2
import random
import numpy as np
import pandas as pd
from glob import glob
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
# Neural Network APIs
from tensorflow.keras import Input
from tensorflow.keras import backend as K
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model, load_model, save_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.layers import Input, Activation, BatchNormalization, Dropout, Lambda, Conv2D, Conv2DTranspose, MaxPooling2D, concatenate
# Specifially for Residual-Recurrent block construction
# REF: https://github.com/yingkaisha/keras-unet-collection
from keras.layers import Conv2D, MaxPooling2D, UpSampling2D, BatchNormalization, Reshape, Permute, Activation, Input, add, multiply
from keras.layers import concatenate, core, Dropout
from keras.models import Model
from keras.optimizers import Adam
from keras.layers.core import Lambda


def plot_from_img_path(rows, columns, list_img_path, list_mask_path):
    '''
    Function for plotting image from path
    '''
    fig = plt.figure(figsize=(12, 12))
    for i in range(1, rows * columns + 1):
        fig.add_subplot(rows, columns, i)
        img_path = list_img_path[i]
        mask_path = list_mask_path[i]
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(mask_path)
        plt.imshow(image)
        plt.imshow(mask, alpha=0.4)
    plt.show()


def dice_coefficients(y_true, y_pred, smooth=1):
    '''
    Calculates dice coefficient with smoothig of 1 in order to avoid NaNs while training 
    '''
    y_true_flatten = K.flatten(y_true)
    y_pred_flatten = K.flatten(y_pred)
    intersection = K.sum(y_true_flatten*y_pred_flatten)
    union = K.sum(y_true_flatten) + K.sum(y_pred_flatten)
    return (2*intersection+smooth)/(union+smooth)


def dice_coefficients_loss(y_true, y_pred, smooth=1):
    return -dice_coefficients(y_true, y_pred, smooth)


def iou(y_true, y_pred, smooth=1):
    '''
    Fucntion to calculate intersection over union as metric of performance
    '''
    intersection = K.sum(y_true * y_pred)
    sum = K.sum(y_true + y_pred)
    iou = (intersection + smooth) / (sum - intersection + smooth)
    return iou


def jaccard_distance(y_true, y_pred):
    '''
    Calculates Jaccard distance using IoU function
    '''
    y_true_flatten = K.flatten(y_true)
    y_pred_flatten = K.flatten(y_pred)
    return -iou(y_true_flatten, y_pred_flatten)


def up_and_concate(down_layer, layer, data_format='channels_first'):
    '''
    Helper function for creating up sample and concatenate blocks 
    '''
    if data_format == 'channels_first':
        in_channel = down_layer.get_shape().as_list()[1]
    else:
        in_channel = down_layer.get_shape().as_list()[3]

    # up = Conv2DTranspose(out_channel, [2, 2], strides=[2, 2])(down_layer)
    up = UpSampling2D(size=(2, 2), data_format=data_format)(down_layer)

    if data_format == 'channels_first':
        my_concat = Lambda(lambda x: K.concatenate([x[0], x[1]], axis=1))
    else:
        my_concat = Lambda(lambda x: K.concatenate([x[0], x[1]], axis=3))

    concate = my_concat([up, layer])

    return concate


def res_block(input_layer, out_n_filters, batch_normalization=False, kernel_size=[3, 3], stride=[1, 1], padding='same', data_format='channels_first'):
    '''
    Helper function to create residual block
    '''
    if data_format == 'channels_first':
        input_n_filters = input_layer.get_shape().as_list()[1]
    else:
        input_n_filters = input_layer.get_shape().as_list()[3]

    layer = input_layer
    for i in range(2):
        layer = Conv2D(out_n_filters // 4, [1, 1], strides=stride, padding=padding, data_format=data_format)(layer)
        if batch_normalization:
            layer = BatchNormalization()(layer)
        layer = Activation('relu')(layer)
        layer = Conv2D(out_n_filters // 4, kernel_size, strides=stride, padding=padding, data_format=data_format)(layer)
        layer = Conv2D(out_n_filters, [1, 1], strides=stride, padding=padding, data_format=data_format)(layer)

    if out_n_filters != input_n_filters:
        skip_layer = Conv2D(out_n_filters, [1, 1], strides=stride, padding=padding, data_format=data_format)(
            input_layer)
    else:
        skip_layer = input_layer
    out_layer = add([layer, skip_layer])
    return out_layer

def rec_res_block(input_layer, out_n_filters, batch_normalization=False, kernel_size=[3, 3], stride=[1, 1], padding='same', data_format='channels_first'):
    '''
    Creates combined Recurrent-Residual block
    '''
    if data_format == 'channels_first':
        input_n_filters = input_layer.get_shape().as_list()[1]
    else:
        input_n_filters = input_layer.get_shape().as_list()[3]

    if out_n_filters != input_n_filters:
        skip_layer = Conv2D(out_n_filters, [1, 1], strides=stride, padding=padding, data_format=data_format)(
            input_layer)
    else:
        skip_layer = input_layer

    layer = skip_layer
    for j in range(2):

        for i in range(2):
            if i == 0:

                layer1 = Conv2D(out_n_filters, kernel_size, strides=stride, padding=padding, data_format=data_format)(
                    layer)
                if batch_normalization:
                    layer1 = BatchNormalization()(layer1)
                layer1 = Activation('relu')(layer1)
            layer1 = Conv2D(out_n_filters, kernel_size, strides=stride, padding=padding, data_format=data_format)(
                add([layer1, layer]))
            if batch_normalization:
                layer1 = BatchNormalization()(layer1)
            layer1 = Activation('relu')(layer1)
        layer = layer1

    out_layer = add([layer, skip_layer])
    return out_layer

def r2_unet(img_w, img_h, data_format='channels_first'):
    '''
     Recurrent Residual Convolutional Neural Network based on U-Net (R2U-Net)
    '''
    inputs = Input((3, img_w, img_h))
    x = inputs
    depth = 4
    features = 64
    skips = []
    for i in range(depth):
        x = rec_res_block(x, features, data_format=data_format)
        skips.append(x)
        x = MaxPooling2D((2, 2), data_format=data_format)(x)

        features = features * 2

    x = rec_res_block(x, features, data_format=data_format)

    for i in reversed(range(depth)):
        features = features // 2
        x = up_and_concate(x, skips[i], data_format=data_format)
        x = rec_res_block(x, features, data_format=data_format)

    # conv6 = Conv2D(n_label, (1, 1), padding='same', data_format=data_format)(x)
    conv6 = Conv2D(filters=1, kernel_size=(1, 1), padding='same', activation="sigmoid", data_format=data_format)(x)
    # conv7 = core.Activation('sigmoid')(conv6)
    model = Model(inputs=inputs, outputs=conv6)
    #model.compile(optimizer=Adam(lr=1e-6), loss=[dice_coef_loss], metrics=['accuracy', dice_coef])
    return model

##### MAIN CODE #####
image_filenames_train = [] # Stores train filenames

mask_files = glob('/content/drive/MyDrive/AIMIA Data/brain_mri/*/*_mask*')

for i in mask_files:
    image_filenames_train.append(i.replace('_mask', ''))

print("Total number of files (MR slics & corresponding masks) = ", len(image_filenames_train))

df = pd.DataFrame(data={'image_filenames_train': image_filenames_train, 'mask': mask_files })

df_train, df_test = train_test_split(df, test_size=0.1, shuffle=False)

df_train, df_val = train_test_split(df_train, test_size=0.2, shuffle=False)

print("Train data shape: ",df_train.shape)
print("Validation data shape: ",df_val.shape)
print("Test data shape: ",df_test.shape)

def to_grayscale_then_rgb(image):
    '''
    Converts 3-Channel RGB to 3-Channel grayscale image
    '''
    image = tf.image.rgb_to_grayscale(image)
    image = tf.image.grayscale_to_rgb(image)
    return image

def normalize_and_diagnose(img, mask):
    '''
    Normalizes and prepares the scna slice and corresponding mask together
    '''
    img = img/255
    mask = mask/255
    mask[mask>0.5] = 1
    mask[mask<=0.5] = 0
    return(img, mask)

def train_generator(
    data_frame,
    batch_size,
    augmentation_dict,
    image_color_mode="rgb",
    mask_color_mode="grayscale",
    image_save_prefix="image",
    mask_save_prefix="mask",
    save_to_dir=None,
    target_size=(256, 256),
    seed=1):
    '''
    Generates image and mask together with relevant augmentation and preparation which will be fed to the neural network
    '''
    image_datagen = ImageDataGenerator(**augmentation_dict, preprocessing_function=to_grayscale_then_rgb)
    mask_datagen = ImageDataGenerator(**augmentation_dict)

    image_generator = image_datagen.flow_from_dataframe(
        data_frame,
        x_col="image_filenames_train",
        class_mode=None,
        color_mode=image_color_mode,
        target_size=target_size,
        batch_size=batch_size,
        save_to_dir=save_to_dir,
        save_prefix=image_save_prefix,
        seed=seed)

    mask_generator = mask_datagen.flow_from_dataframe(
        data_frame,
        x_col="mask",
        class_mode=None,
        color_mode=mask_color_mode,
        target_size=target_size,
        batch_size=batch_size,
        save_to_dir=save_to_dir,
        save_prefix=mask_save_prefix,
        seed=seed)

    train_gen = zip(image_generator, mask_generator)
    
    for (img, mask) in train_gen:
        img, mask = normalize_and_diagnose(img, mask)
        yield (img, mask)


im_width = 256
im_height = 256
EPOCHS = 100
BATCH_SIZE = 16
lr = 1e-3
smooth = 1


# Please execute any of the belowe based on previous training of same model  
# model = r2_unet(im_height, im_width)
# model = load_model('r2_unet.hdf5', custom_objects={'dice_coefficients_loss': dice_coefficients_loss, 'iou': iou, 'dice_coefficients': dice_coefficients})

model.summary()

# !nvidia-smi

train_generator_param = dict(horizontal_flip=True, fill_mode='nearest')

train_gen = train_generator(df_train, BATCH_SIZE, train_generator_param, target_size=(im_height, im_width))
    
val_gen = train_generator(df_val, BATCH_SIZE, dict(), target_size=(im_height, im_width))

opt = Adam(learning_rate=lr, beta_1=0.9, beta_2=0.999, epsilon=None, amsgrad=False)

model.compile(optimizer=opt, loss=dice_coefficients_loss, metrics=["binary_accuracy", iou, dice_coefficients])

callbacks = [ModelCheckpoint('r2_unet.hdf5', verbose=1, save_best_only=True)] # Saves model if the current validation performance was better than previous ones

history = model.fit(train_gen,
                    steps_per_epoch=len(df_train)/BATCH_SIZE, 
                    epochs=EPOCHS, 
                    callbacks=callbacks,
                    validation_data = val_gen,
                    validation_steps=len(df_val)/BATCH_SIZE)

#### PLOTTING LEARNING CURVES
history_post_training = history.history

train_dice_coeff_list = history_post_training['dice_coefficients']
val_dice_coeff_list = history_post_training['val_dice_coefficients']

train_jaccard_list = history_post_training['iou']
val_jaccard_list = history_post_training['val_iou']

train_loss_list = history_post_training['loss']
val_loss_list = history_post_training['val_loss']

plt.figure(1)
plt.plot(train_loss_list, 'r-', label = 'Train Loss')
plt.plot(val_loss_list, 'b-', label = 'Val Loss')
plt.xlabel('# Iterations')
plt.ylabel('Loss')
plt.title('Loss Plot', fontsize=12)
plt.legend()
plt.show()

plt.figure(2)
plt.plot(train_dice_coeff_list, 'r-', label = 'Train Dice')
plt.plot(val_dice_coeff_list, 'b-', label = 'Val Dice')
plt.xlabel('# Iterations')
plt.ylabel('Dice Score')
plt.title('Dice Score Plot', fontsize=12)
plt.legend()
plt.show()

##### INFERENCING OF THE MODEL WITH TEST DATA
test_gen = train_generator(df_test, BATCH_SIZE, dict(), target_size=(im_height, im_width))

test_results = model.evaluate(test_gen, steps=len(df_test)/BATCH_SIZE )

print('Test Loss: ', test_results[0])
print('Test IoU: ', test_results[1])
print('Test Dice Coefficient: ', test_results[2])