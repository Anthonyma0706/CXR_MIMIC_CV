import torch
import torchvision
import numpy as np
import cv2 as cv
import pandas as pd
from tqdm import tqdm

from datetime import datetime
import glob
import math
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from pathlib import Path


from PIL import Image
import random as python_random
#import seaborn as sns
from tensorflow import keras
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import AveragePooling2D, Dropout, Flatten, GlobalAveragePooling2D, Input, Dense, Activation
from tensorflow.keras.models import Model
from tensorflow.keras import initializers
from tensorflow.keras.callbacks import EarlyStopping, LearningRateScheduler, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, roc_auc_score, roc_curve, precision_recall_curve
from sklearn.metrics import auc, accuracy_score, recall_score, precision_score, f1_score, confusion_matrix
from sklearn.utils import shuffle
import sys
import seaborn as sns
import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import load_model
from keras.applications.resnet import ResNet50, preprocess_input



disease = 'All' #'All'
task = 'gender' #'survive' #'gender' #'white' #'survive' # race

if task == 'race':
    model_name = 'densenet'
    test_csv_path = 'saved_models/test_densenet_All_race_20221206-030654.csv'
    model_path = 'saved_models/densenet_All_race_detection_LR-0.001_20221204-154429_epoch_019_val_loss_0.16900.h5'
    class_names = ['ASIAN', 'BLACK', 'WHITE']
    num_class = 3
elif task == 'white':
    model_name = 'densenet'
    test_csv_path = 'saved_models/test_densenet_All_white_20221206-025606.csv'
    model_path = 'saved_models/densenet_All_white_detection_LR-0.0001_20221205-054121_epoch_003_val_loss_0.23216.h5'
    num_class = 2
    class_names = ['NON_WHITE', 'WHITE']
elif task == 'gender':
    model_name = 'densenet'
    test_csv_path = 'saved_models/test_densenet_All_gender_20221206-145936.csv'
    model_path = 'saved_models/densenet_All_gender_detection_LR-0.0001_20221206-114022_epoch_008_val_loss_0.05443.h5'
    num_class = 2
    class_names = ['F', 'M']
elif task == 'insurance':
    test_csv_path = ''
    model_name = 'densenet'
    model_path = 'saved_models/densenet_All_insurance_detection_LR-0.0001_20221206-055648_epoch_002_val_loss_0.82505.h5'
    class_names = ['Medicaid', 'Medicare', 'Other'] # order must match folder

    num_class = 3
elif task == 'survive':
    model_name = 'densenet'
    test_csv_path = 'saved_models/test_All_survive_20221204-105227.csv'
    model_path = 'saved_models/densenet_All_survive_detection_LR-0.0001_20221205-010809_epoch_002_val_loss_0.63442.h5'
    class_names = ['DIE', 'SURVIVE'] # order must match folder
    num_class = 2

try:
  # Specify an invalid GPU device
  with tf.device('/GPU:1'):
    
    HEIGHT = 256
    WIDTH = 256

    var_date = datetime.now().strftime("%Y%m%d-%H%M%S")

    def define_densenet_model(n_classes=1, input_shape=(224,224,3)):
        base_model = tf.keras.applications.densenet.DenseNet121(weights=None, include_top=False, input_shape=input_shape)
        x = AveragePooling2D(pool_size=(3,3), name='avg_pool')(base_model.output)
        x = Flatten()(x)
        x = Dense(1024, activation='relu', name='dense_post_pool')(x)
        x = Dropout(0.2)(x)
        output = Dense(n_classes, activation='softmax', name='predictions')(x)
        model = Model(inputs=base_model.input, outputs=output)
        return model

    if model_name == 'densenet':
        HEIGHT = 224
        WIDTH = 224
        model = define_densenet_model(num_class) #define_densenet_model()
        preprocess_input = tf.keras.applications.densenet.preprocess_input
        last_conv_layer = 'conv5_block16_2_conv'
    elif model_name == 'resnet50':
        input_a = Input(shape=(HEIGHT, WIDTH, 3))
        base_model = tf.keras.applications.resnet.ResNet50(weights='imagenet', 
                            input_tensor=input_a,
                            include_top=False, 
                            input_shape=(HEIGHT, WIDTH, 3)
                            )

        x = GlobalAveragePooling2D()(base_model.output)
        x = Dense(num_class, name='dense_logits')(x)
        output = Activation('softmax', dtype='float32', name='predictions')(x)
        model = Model(inputs=[input_a], outputs=[output])
        preprocess_input = tf.keras.applications.resnet.preprocess_input
        last_conv_layer = 'conv5_block3_3_conv' #for ResNet50

   

    model.load_weights(model_path)
    
    data_dir = f'data/{disease}/images/{task}'
    test_dir = f'{data_dir}/test'
    test_gen = ImageDataGenerator(preprocessing_function=preprocess_input)
    test_batch_size = 128

    # dir_to_save = f'data/{disease}/images/{task}'

    # if task == 'race':
    #     class_names = ['ASIAN', 'BLACK', 'WHITE']
    # elif task == 'white':
    #     class_names = ['NON_WHITE', 'WHITE']
    # elif task == 'survive':
    #     class_names = ['DIE', 'SURVIVE'] # order must match folder

    datasets = ['test']
    y_test = []
    test_files = []
    for ds in datasets:
        for cls in class_names:
            DIR = f'{data_dir}/{ds}/{cls}'
            #lst = os.listdir(DIR)
            fname_l = [f'{DIR}/{f}' for f in os.listdir(DIR)]
            n_files = len(fname_l)
            print(ds,cls,n_files)
            y_test.extend([cls] * n_files)
            test_files.extend(fname_l)

    
    test_size = len(y_test)
    print('Number of test images',test_size)
    
    test_batches = test_gen.flow_from_directory(
                                    directory= test_dir,
                                    classes = None, # means automatically infer the label from subdir
                                    class_mode = 'categorical', # white, black, asian
                                    target_size=(HEIGHT, WIDTH),
                                    shuffle=False,
                                    # seed=seed, 
                                    batch_size= test_batch_size
    )

    if test_csv_path == '':
        print('=========MAKE PREDICTIONS==============')
        multilabel_predict_test = model.predict(test_batches, 
                                            verbose=1, 
                                            steps=math.ceil(test_size/test_batch_size), 
                                            
                                            )
        input_prediction = multilabel_predict_test
        input_prediction_df = pd.DataFrame(input_prediction)
        input_prediction_df.columns = input_prediction_df.columns.map(str)
        # var_date = datetime.now().strftime("%Y%m%d-%H%M%S")
        input_prediction_df.to_csv(f'saved_models/test_{model_name}_{disease}_{task}_{var_date}.csv', index = False)
        print('pred CSV saved successfully')
    else:
        input_prediction_df = pd.read_csv(test_csv_path)
        multilabel_predict_test = input_prediction_df.to_numpy()
    ################ Get correctly predicted data #################
    y_truth = test_batches.classes
    assert len(y_test) == len(y_truth)

    result = multilabel_predict_test #input_prediction_df.to_numpy()
    y_pred = np.argmax(result, axis=1)

    # get image paths that model predicts correctly
    test_files = np.array(test_files)
    correct_inds = np.equal(y_pred,y_truth)
    images_pred_correct = test_files[correct_inds]
    
    test_accuracy = round(images_pred_correct.shape[0] / test_files.shape[0], 3)
    print(test_files.shape[0], images_pred_correct.shape[0], test_accuracy)

    
    y_test = np.array(y_test)
    df_correct_img_paths = pd.DataFrame({'img': images_pred_correct, 'class': y_test[correct_inds]})
    df_correct_img_paths.to_csv(f'saved_models/CORRECT_test_{model_name}_{disease}_{task}_acc_{test_accuracy}.csv', index = False)
    print('correct test CSV saved successfully')

    


    #def model_summary(y_test, test_batches, input_prediction_df, class_names):
    def model_summary(test_batches, input_prediction_df, class_names):
        #ground_truth = pd.Series(y_test)
        y_truth = test_batches.classes #ground_truth.replace({'ASIAN': 0, 'BLACK': 1, 'WHITE': 2})
        # assert len(y_test) == len(y_truth)
        #get_AUC(input_prediction_df, ground_truth, class_names)
        result = input_prediction_df.to_numpy()
        #labels = np.argmax(result, axis=1)
        y_pred = np.argmax(result, axis=1)
        target_names = class_names

        print(classification_report(y_truth, y_pred, target_names=target_names))
        class_matrix = confusion_matrix(y_truth, y_pred)

        print ('Classwise ROC AUC \n')

        for p in list(set(y_pred)):
            fpr, tpr, thresholds = roc_curve(y_truth, result[:,p], pos_label = p)
            auroc = round(auc(fpr, tpr), 2)
            print ('Class - {} ROC-AUC- {}'.format(target_names[p], auroc))
            # plot the roc curve for the model
            plt.plot(fpr, tpr, linestyle='solid', label='{} AUC={:.3f}'.format(target_names[p], auroc))

        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC curve')
        plt.legend(loc="lower right")
        plt.plot([0,1], [0,1], color='orange', linestyle='--')
        DATA_DIR = Path('results')
        (DATA_DIR / task).mkdir(parents=True, exist_ok=True)
        filename = f'results/{task}/ROC_{disease}_{task}_{model_name}.png'
        plt.savefig(filename, dpi=120)
        print(filename, 'SAVED')
        plt.show()
        plt.clf()

        sns.heatmap(class_matrix, annot=True, fmt='d', cmap='Blues')
        filename = f'results/{task}/class_matrix_{disease}_{task}_{model_name}.png'
        plt.savefig(filename, dpi=120)
        print(filename, 'SAVED')
        plt.clf()

    model_summary(test_batches, input_prediction_df, class_names)

    def get_gradcam(img_array, model, last_conv_layer_name, height = 224, pred_index=None, alpha=1):
        """the shape of input img_array should be [height,height,3]
        """
        img = np.expand_dims(img_array, axis=0) # add one batch_size
        
        # First, we create a model that maps the input image to the activations
        # of the last conv layer as well as the output predictions
        grad_model = tf.keras.models.Model(
            [model.inputs], [model.get_layer(last_conv_layer_name).output, model.output]
        )
        #print(model.output.shape)

        # Then, we compute the gradient of the top predicted class for our input image
        # with respect to the activations of the last conv layer
        with tf.GradientTape() as tape:
            last_conv_layer_output, preds = grad_model(img)
            if pred_index is None:
                pred_index = tf.argmax(preds[0])
            class_channel = preds[:, pred_index]

        # This is the gradient of the output neuron (top predicted or chosen)
        # with regard to the output feature map of the last conv layer
        grads = tape.gradient(class_channel, last_conv_layer_output)

        # This is a vector where each entry is the mean intensity of the gradient
        # over a specific feature map channel
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

        # We multiply each channel in the feature map array
        # by "how important this channel is" with regard to the top predicted class
        # then sum all the channels to obtain the heatmap class activation
        last_conv_layer_output = last_conv_layer_output[0]
        heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
        heatmap = tf.squeeze(heatmap)

        # For visualization purpose, we will also normalize the heatmap between 0 & 1
        heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)

        # Rescale heatmap to a range 0-255
        heatmap = np.uint8((height -1) * heatmap)
        # Use jet colormap to colorize heatmap
        jet = cm.get_cmap("jet")

        # Use RGB values of the colormap
        jet_colors = jet(np.arange(height))[:, :3]
        jet_heatmap = jet_colors[heatmap]

        # Create an image with RGB colorized heatmap
        jet_heatmap = keras.preprocessing.image.array_to_img(jet_heatmap)
        
        jet_heatmap = jet_heatmap.resize((img_array.shape[1], img_array.shape[0]))
        
        jet_heatmap = keras.preprocessing.image.img_to_array(jet_heatmap)
        
        # Superimpose the heatmap on original image
        superimposed_img = jet_heatmap * alpha + img_array
        return superimposed_img, jet_heatmap
    
    def show_sign_grid(images,filename, nrow = 10):
        #images = [load_image(img) for img in image_paths]
        images = np.array([np.asarray(im) for im in images])
        # convert numpy array to a single tensor
        images = torch.as_tensor(images)#, dtype=int) 
        images = images.permute(0, 3, 1, 2)
        grid_img = torchvision.utils.make_grid(images, nrow=nrow)
        plt.figure(figsize=(30, 30))
        plt.imshow(grid_img.permute(1, 2, 0))
        plt.savefig(filename, dpi=200)
        print(filename, 'SAVED')
        plt.axis('off');

    # def show_sign_grid_array(images, nrow = 10):
    #     #images = [load_image(img) for img in image_paths]
    #     # images = [np.asarray(im) for im in images]
    #     # convert numpy array to a single tensor
    #     images = torch.as_tensor(images)#, dtype=int) 
    #     images = images.permute(0, 3, 1, 2)
    #     grid_img = torchvision.utils.make_grid(images, nrow=nrow)
    #     plt.figure(figsize=(30, 30))
    #     plt.imshow(grid_img.permute(1, 2, 0))
    #     plt.axis('off');
    
    n_test = 50

    class_names = df_correct_img_paths['class'].unique()
    for cls in class_names:
        df_use = df_correct_img_paths[df_correct_img_paths['class']==cls].sample(n=n_test)
        img_paths = list(df_use['img'])
        gradcam_imgs = []
        for path in img_paths:
            img = keras.preprocessing.image.load_img(path, target_size= (HEIGHT,HEIGHT, 3))
            img = np.asarray(img)

            img = np.asarray(img)
            gradcam, jet_heatmap = get_gradcam(img, model, last_conv_layer, alpha=2)
            gradcam_img = keras.preprocessing.image.array_to_img(gradcam)
            gradcam_imgs.append(gradcam_img)
        print(cls)
        filename = f'results/{task}/gradcam_{disease}_{cls}_{model_name}_{var_date}.png'
        show_sign_grid(gradcam_imgs,filename, nrow = 10)

    
except RuntimeError as e:
  print(e)
