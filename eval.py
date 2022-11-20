import tensorflow as tf
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
from tensorflow.keras.models import load_model
#from tensorflow.keras.mixed_precision import experimental as mixed_precision
# pip install image-classifiers==1.0.0b1
#from classification_models.tfkeras import Classifiers
# More information about this package can be found at https://github.com/qubvel/classification_models

from keras.applications.resnet import ResNet50, preprocess_input

seed = 2022
np.random.seed(seed)
python_random.seed(seed)
tf.random.set_seed(seed)


# Recreate the exact same model, including its weights and the optimizer
model_path = 'saved_models\ResNet50_60-20-20-split_3-race_detection_LR-0.001_20221117-160540_epoch_001_val_loss_0.57067.h5'
model = tf.keras.models.load_model(model_path)

df_meta = pd.read_csv('data\mimic_iv_csv\df_race_3class_1117.csv')
train_size, test_size, val_size = df_meta['dataset'].value_counts()
print(df_meta['dataset'].value_counts())
print('Test size: ', test_size)

# Show the model architecture
#print(model.summary())
data_dir = 'data/imgs_race'
test_dir = f'{data_dir}/test'
validate_gen = ImageDataGenerator(preprocessing_function=preprocess_input)
test_batch_size = 64
HEIGHT = 256
WIDTH = 256
test_batches = validate_gen.flow_from_directory(
                                directory= test_dir,
                                classes = None, # means automatically infer the label from subdir
                                class_mode = 'categorical', # white, black, asian
                                target_size=(HEIGHT, WIDTH),
                                shuffle=False,
                                # seed=seed, 
                                batch_size= test_batch_size
)


multilabel_predict_test = model.predict(test_batches, 
                #max_queue_size=10, 
                verbose=1, 
                steps=math.ceil(test_size/test_batch_size), 
                #workers=32
                )


input_prediction = multilabel_predict_test
#input_df = test_df
input_prediction_df = pd.DataFrame(input_prediction)
var_date = datetime.now().strftime("%Y%m%d-%H%M%S")
input_prediction_df.to_csv(f'saved_models\eval_data/eval_{var_date}.csv', index = False)
print('CSV saved successfully')
