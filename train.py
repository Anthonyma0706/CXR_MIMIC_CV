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
from tensorflow.keras.layers import GlobalAveragePooling2D, Input, Dense, Activation
from tensorflow.keras.models import Model
from tensorflow.keras import initializers
from tensorflow.keras.callbacks import EarlyStopping, LearningRateScheduler, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.preprocessing.image import ImageDataGenerator

#from keras.applications.resnet import ResNet50
#from keras.applications.vgg16 import VGG16, preprocess_input
seed = 2022
np.random.seed(seed)
python_random.seed(seed)
tf.random.set_seed(seed)

# df_meta = pd.read_csv('data\mimic_iv_csv\df_race_3class_1117.csv')
# train_size, test_size, val_size = df_meta['dataset'].value_counts()
# print(df_meta['dataset'].value_counts())
###################### Image directory ###########################
# lst = os.listdir(directory) # your directory path
# number_files = len(lst)

data_dir = 'data/No_finding/images/race' # data/No_finding/images/survive
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
###################### Hyperparameters ###########################
epochs = 20
learning_rate = 1e-3
momentum_val=0.9
decay_val= 0.0

train_batch_size = 64 # may need to reduce batch size if OOM error occurs
test_batch_size = 64
##################################################################
from keras.applications.resnet import ResNet50, preprocess_input
HEIGHT = 256
WIDTH = 256
input_a = Input(shape=(HEIGHT, WIDTH, 3))
base_model = ResNet50(weights='imagenet', 
                    input_tensor=input_a,
                    include_top=False, 
                    input_shape=(HEIGHT, WIDTH, 3)
                    )

x = GlobalAveragePooling2D()(base_model.output)
x = Dense(3, name='dense_logits')(x)
output = Activation('softmax', dtype='float32', name='predictions')(x)
model = Model(inputs=[input_a], outputs=[output])

for layer in base_model.layers: # keep base_model aside from training
    layer.trainable = False


reduce_lr = ReduceLROnPlateau(monitor='val_loss', mode='min', factor=0.1, patience=2, min_lr=1e-5, verbose=1)

adam_opt = tf.keras.optimizers.Adam(learning_rate=learning_rate, decay=decay_val)
adam_opt = tf.keras.mixed_precision.LossScaleOptimizer(adam_opt)

model.compile(optimizer=adam_opt,
                loss=tf.losses.CategoricalCrossentropy(),
                metrics=[
                    'accuracy',
                    tf.keras.metrics.AUC(curve='ROC', name='ROC-AUC'),
                    tf.keras.metrics.AUC(curve='PR', name='PR-AUC')
                ],
)
print('============= Model loading finished =============')



train_gen = ImageDataGenerator(
            rotation_range=15,
            fill_mode='constant',
            horizontal_flip=True,
            zoom_range=0.1,
            preprocessing_function=preprocess_input
            )

validate_gen = ImageDataGenerator(preprocessing_function=preprocess_input)

train_batches = train_gen.flow_from_directory(
                                directory= train_dir,
                                classes = None, # means automatically infer the label from subdir
                                class_mode = 'categorical', # white, black, asian
                                target_size=(HEIGHT, WIDTH),
                                shuffle=True,
                                seed=seed, 
                                batch_size=train_batch_size
)

validate_batches = validate_gen.flow_from_directory(
                                directory= val_dir,
                                classes = None, # means automatically infer the label from subdir
                                class_mode = 'categorical', # white, black, asian
                                target_size=(HEIGHT, WIDTH),
                                shuffle=False,
                                # seed=seed, 
                                batch_size= test_batch_size
)

print('============= Batches/Dataloader loading finished =============')


train_epoch = math.ceil(train_size / train_batch_size)
val_epoch = math.ceil(val_size / test_batch_size)
var_date = datetime.now().strftime("%Y%m%d-%H%M%S")

model_name = 'ResNet50'
arc_name = f"{model_name}_60-20-20-split_3-race_detection"

ES = EarlyStopping(monitor='val_loss', mode='min', patience=3, restore_best_weights=True)
checkloss = ModelCheckpoint("saved_models/" + str(arc_name) + "_LR-" + str(learning_rate) + "_" + var_date+ "_epoch_{epoch:03d}_val_loss_{val_loss:.5f}.h5", 
                        monitor='val_loss', mode='min', verbose=1, 
                        save_best_only=True, 
                        save_weights_only=True # now only save the weights
                        )

print('============= MODEL TRAINING STARTS =============')
model.fit(train_batches,
            validation_data=validate_batches,
            epochs=epochs, # 20, 30
            steps_per_epoch=int(train_epoch),
            validation_steps=int(val_epoch),
            #workers=32,
            #max_queue_size=50,
            shuffle=True,
            callbacks=[checkloss, reduce_lr, ES]
           )
print('============= MODEL TRAINING ENDS =============')