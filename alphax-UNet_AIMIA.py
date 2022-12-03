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


def unet(alpha, input_size=(256, 256, 3)):
    '''
    Creates UNet with Batch Normalization after every layer
    '''
    inputs = Input(input_size)

    # Encoder
    conv1 = Conv2D(filters=int(alpha*64), kernel_size=(3, 3), padding="same")(inputs)
    bn1 = Activation("relu")(conv1)
    conv1 = Conv2D(filters=int(alpha*64), kernel_size=(3, 3), padding="same")(bn1)
    bn1 = BatchNormalization(axis=3)(conv1)
    bn1 = Activation("relu")(bn1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(bn1)

    conv2 = Conv2D(filters=int(alpha*128), kernel_size=(3, 3), padding="same")(pool1)
    bn2 = Activation("relu")(conv2)
    conv2 = Conv2D(filters=int(alpha*128), kernel_size=(3, 3), padding="same")(bn2)
    bn2 = BatchNormalization(axis=3)(conv2)
    bn2 = Activation("relu")(bn2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(bn2)

    conv3 = Conv2D(filters=int(alpha*256), kernel_size=(3, 3), padding="same")(pool2)
    bn3 = Activation("relu")(conv3)
    conv3 = Conv2D(filters=int(alpha*256), kernel_size=(3, 3), padding="same")(bn3)
    bn3 = BatchNormalization(axis=3)(conv3)
    bn3 = Activation("relu")(bn3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(bn3)

    conv4 = Conv2D(filters=int(alpha*512), kernel_size=(3, 3), padding="same")(pool3)
    bn4 = Activation("relu")(conv4)
    conv4 = Conv2D(filters=int(alpha*512), kernel_size=(3, 3), padding="same")(bn4)
    bn4 = BatchNormalization(axis=3)(conv4)
    bn4 = Activation("relu")(bn4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(bn4)

    conv5 = Conv2D(filters=int(alpha*1024), kernel_size=(3, 3), padding="same")(pool4)
    bn5 = Activation("relu")(conv5)
    conv5 = Conv2D(filters=int(alpha*1024), kernel_size=(3, 3), padding="same")(bn5)
    bn5 = BatchNormalization(axis=3)(conv5)
    bn5 = Activation("relu")(bn5)

    # Decoder
    up6 = concatenate([Conv2DTranspose(int(alpha*512), kernel_size=(2, 2), strides=(2, 2), padding="same")(bn5),conv4],axis=3)

    conv6 = Conv2D(filters=int(alpha*512), kernel_size=(3, 3), padding="same")(up6)
    bn6 = Activation("relu")(conv6)
    conv6 = Conv2D(filters=int(alpha*512), kernel_size=(3, 3), padding="same")(bn6)
    bn6 = BatchNormalization(axis=3)(conv6)
    bn6 = Activation("relu")(bn6)

    up7 = concatenate([Conv2DTranspose(int(alpha*256), kernel_size=(2, 2), strides=(2, 2), padding="same")(bn6),conv3],axis=3)

    conv7 = Conv2D(filters=int(alpha*256), kernel_size=(3, 3), padding="same")(up7)
    bn7 = Activation("relu")(conv7)
    conv7 = Conv2D(filters=int(alpha*256), kernel_size=(3, 3), padding="same")(bn7)
    bn7 = BatchNormalization(axis=3)(conv7)
    bn7 = Activation("relu")(bn7)

    up8 = concatenate([Conv2DTranspose(int(alpha*128), kernel_size=(2, 2), strides=(2, 2), padding="same")(bn7),conv2],axis=3)
    
    conv8 = Conv2D(filters=int(alpha*128), kernel_size=(3, 3), padding="same")(up8)
    bn8 = Activation("relu")(conv8)
    conv8 = Conv2D(filters=int(alpha*128), kernel_size=(3, 3), padding="same")(bn8)
    bn8 = BatchNormalization(axis=3)(conv8)
    bn8 = Activation("relu")(bn8)

    up9 = concatenate([Conv2DTranspose(int(alpha*64), kernel_size=(2, 2), strides=(2, 2), padding="same")(bn8),conv1],axis=3)

    conv9 = Conv2D(filters=int(alpha*64), kernel_size=(3, 3), padding="same")(up9)
    bn9 = Activation("relu")(conv9)
    conv9 = Conv2D(filters=int(alpha*64), kernel_size=(3, 3), padding="same")(bn9)
    bn9 = BatchNormalization(axis=3)(conv9)
    bn9 = Activation("relu")(bn9)

    conv10 = Conv2D(filters=1, kernel_size=(1, 1), activation="sigmoid")(bn9)

    return Model(inputs=[inputs], outputs=[conv10])


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

alpha = 0.25 # The downscaling factor by which to reduce the number of filters in each layer of UNet

# Please execute any of the belowe based on previous training of same model  
# model = unet(alpha, input_size=(im_height, im_width, 3))
# model = load_model('unet.hdf5', custom_objects={'dice_coefficients_loss': dice_coefficients_loss, 'iou': iou, 'dice_coefficients': dice_coefficients})

model.summary()

# !nvidia-smi

train_generator_param = dict(horizontal_flip=True, fill_mode='nearest')

train_gen = train_generator(df_train, BATCH_SIZE, train_generator_param, target_size=(im_height, im_width))
    
val_gen = train_generator(df_val, BATCH_SIZE, dict(), target_size=(im_height, im_width))

opt = Adam(learning_rate=lr, beta_1=0.9, beta_2=0.999, epsilon=None, amsgrad=False)

model.compile(optimizer=opt, loss=dice_coefficients_loss, metrics=["binary_accuracy", iou, dice_coefficients])

callbacks = [ModelCheckpoint('unet.hdf5', verbose=1, save_best_only=True)] # Saves model if the current validation performance was better than previous ones

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