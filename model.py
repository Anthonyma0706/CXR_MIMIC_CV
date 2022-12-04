import numpy as np
import cv2 as cv
import pandas as pd
from tqdm import tqdm

from datetime import datetime
import glob
import math
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import os
import pandas as pd
from PIL import Image
import random as python_random
#import seaborn as sns
from sklearn.metrics import classification_report, roc_auc_score, roc_curve, precision_recall_curve
from sklearn.metrics import auc, accuracy_score, recall_score, precision_score, f1_score, confusion_matrix
from sklearn.utils import shuffle
import sys
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import AveragePooling2D, Dropout, Flatten, GlobalAveragePooling2D, Input, Dense, Activation
from tensorflow.keras.models import Model
from tensorflow.keras import initializers
from tensorflow.keras.callbacks import EarlyStopping, LearningRateScheduler, ReduceLROnPlateau, ModelCheckpoint, CSVLogger
from tensorflow.keras.preprocessing.image import ImageDataGenerator


#from keras.applications.resnet import ResNet50
#from keras.applications.vgg16 import VGG16, preprocess_input

# disease = 'All' # 'No_finding'
# task = 'race' # 'survive'

def model_training(disease, task, model_name, learning_rate = 1e-3, train_batch_size = 64, epochs = 20, model_weight_pth = '', class_weight = None):

    ##################################
    ########## Needs to change together ##############
    ##################################
    # from keras.applications.resnet import ResNet50, preprocess_input
    # model_name = 'ResNet50' 
    ##################################
    ##################################

    seed = 2022
    np.random.seed(seed)
    python_random.seed(seed)
    tf.random.set_seed(seed)

    
    ###################### Image directory ###########################
    num_class = 2 if task == 'survive' else 3
    classes = ['ASIAN','BLACK','WHITE'] if task == 'race' else ['DIE','SURVIVE']
    data_dir = f'data/{disease}/images/{task}'
    train_dir = f'{data_dir}/train'
    test_dir = f'{data_dir}/test'
    val_dir = f'{data_dir}/val'

    # count training images only
    f = []
    for (dirpath, dirnames, filenames) in os.walk(train_dir):
        f.extend(filenames)
    train_size = len(f)
    print('Predicting folder:', data_dir)
    print('Number of training images:', train_size)

    f = []
    for (dirpath, dirnames, filenames) in os.walk(val_dir):
        f.extend(filenames)
    val_size = len(f)
    print('Number of validation images:', val_size)
    ###################### Hyperparameters ###########################


    #train_batch_size = 64 # may need to reduce batch size if OOM error occurs
    test_batch_size = 64
    ##################################################################


    HEIGHT = 256
    WIDTH = 256
    
    # def define_densenet_model():
        
    #     input = tf.keras.layers.Input(shape=(HEIGHT, WIDTH, 1))
        
    #     reshape_layer = tf.keras.layers.UpSampling3D(size=(1,1,3))(input)
        
    #     base_model = tf.keras.applications.densenet.DenseNet121(
    #             include_top=False, weights='imagenet', input_shape=(HEIGHT, WIDTH, 3), pooling='max')(reshape_layer)
            
    #     pred_layer = tf.keras.layers.Dense(num_class, activation='softmax')(base_model)
    
    #     model = tf.keras.Model(inputs=input, outputs=pred_layer)  
    #     # for layer in base_model.layers: # keep base_model aside from training
    #     #     layer.trainable = False  
    
    #     return model
    

    # def define_densenet_model():
    #     INPUT_SHAPE = (224, 224, 1)
    #     input = tf.keras.layers.Input(shape=INPUT_SHAPE)
        
    #     reshape_layer = tf.keras.layers.UpSampling3D(size=(1,1,3))(input)
        
    #     base_model = tf.keras.applications.densenet.DenseNet121(
    #             include_top=False, weights='imagenet', input_shape=(224, 224, 3), pooling='max')(reshape_layer)
            
    #     pred_layer = tf.keras.layers.Dense(3, activation='softmax')(base_model)
    
    #     model = tf.keras.Model(inputs=input, outputs=pred_layer)    
    
    #     return model

    def define_densenet_model(n_classes=1, input_shape=(224,224,3)):
        base_model = tf.keras.applications.densenet.DenseNet121(weights='imagenet', include_top=False, input_shape=input_shape)
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
        model = define_densenet_model(num_class, input_shape=(HEIGHT,WIDTH,3))
        preprocess_input = tf.keras.applications.densenet.preprocess_input
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
        for layer in base_model.layers: # keep base_model aside from training
            layer.trainable = False
    else:
        print("ERROR IN MODEL CHOICE")
        return -1


    if model_weight_pth != '':
        print('load model weights from', model_weight_pth)
        model.load_weights(model_weight_pth)

    


    reduce_lr = ReduceLROnPlateau(monitor='val_loss', mode='min', factor=0.1, patience=2, min_lr=1e-5, verbose=1)

    adam_opt = tf.keras.optimizers.Adam(learning_rate=learning_rate#, decay=decay_val
    )
    adam_opt = tf.keras.mixed_precision.LossScaleOptimizer(adam_opt)

    model.compile(optimizer=adam_opt,
                    loss=tf.losses.CategoricalCrossentropy(),
                    metrics=[
                        'accuracy',
                        tf.keras.metrics.AUC(curve='ROC', name='ROC-AUC'),
                        tf.keras.metrics.AUC(curve='PR', name='PR-AUC')
                    ],
    )


    print(f'============= {model_name} Model loading finished =============')


    train_gen = ImageDataGenerator(
                rotation_range=5,
                fill_mode='constant',
                horizontal_flip=True,
                zoom_range=0.1,
                preprocessing_function=preprocess_input
                )

    validate_gen = ImageDataGenerator(preprocessing_function=preprocess_input)

    train_batches = train_gen.flow_from_directory(
                                    directory= train_dir,
                                    #classes = None, # means automatically infer the label from subdir
                                    classes = classes,
                                    class_mode = 'categorical', # white, black, asian
                                    target_size=(HEIGHT, WIDTH),
                                    shuffle=True,
                                    seed=seed, 
                                    batch_size=train_batch_size
    )

    validate_batches = validate_gen.flow_from_directory(
                                    directory= val_dir,
                                    # classes = None, # means automatically infer the label from subdir
                                    classes = classes,
                                    class_mode = 'categorical', # white, black, asian
                                    target_size=(HEIGHT, WIDTH),
                                    shuffle=False,
                                    # seed=seed, 
                                    batch_size= test_batch_size
    )

    print(np.unique(train_batches.classes))
    print(np.unique(validate_batches.classes))

    print('============= Batches/Dataloader loading finished =============')


    train_epoch = math.ceil(train_size / train_batch_size)
    val_epoch = math.ceil(val_size / test_batch_size)
    var_date = datetime.now().strftime("%Y%m%d-%H%M%S")

    
    arc_name = f'{model_name}_{disease}_{task}_detection'

    csv_logger = CSVLogger(f'saved_models/cxr_{arc_name}_{var_date}.log')
    ES = EarlyStopping(monitor='val_loss', mode='min', patience=4, restore_best_weights=True)
    checkloss = ModelCheckpoint("saved_models/" + str(arc_name) + "_LR-" + str(learning_rate) + "_" + var_date+ "_epoch_{epoch:03d}_val_loss_{val_loss:.5f}.h5", 
                            monitor='val_loss', mode='min', verbose=1, 
                            save_best_only=True, 
                            save_weights_only=True # now only save the weights
                            )

    print('============= MODEL TRAINING STARTS =============')
    model.fit(train_batches,
                validation_data=validate_batches,
                epochs=epochs,
                steps_per_epoch=int(train_epoch),
                validation_steps=int(val_epoch),
                class_weight = class_weight, # default is None
                shuffle=True,
                callbacks=[checkloss, reduce_lr, ES, csv_logger]
            )
    print('============= MODEL TRAINING ENDS =============')