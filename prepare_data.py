import numpy as np
import cv2 as cv
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
import os
import shutil
from pathlib import Path
from PIL import Image


import tensorflow as tf
from keras.utils.data_utils import Sequence
from imblearn.over_sampling import RandomOverSampler
from imblearn.keras import balanced_batch_generator

from keras.applications.resnet import ResNet50, preprocess_input
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from sklearn.utils import shuffle


disease = 'No_finding' #'Edema' # 'Pneumonia_Consolidation' , 'No_finding'
task = 'race' # 'survive', 'race'
df_meta = pd.read_csv(f'data/{disease}/df_{disease}_1121.csv')

save_imgs = True
dir_to_save= f'data/{disease}/images/{task}/train'
print(disease, task, dir_to_save)
#df_meta = pd.read_csv('data/No_finding/df_No_finding_1121.csv')
#df_meta = pd.read_csv('data/Edema/df_Edema_1121.csv')

def prepare_balanced_testset_for_race(df_meta, drop_some_white = False, p = 0.4):
    df = df_meta.copy(True)
    df['race'] = df['race'].replace({'ASIAN - ASIAN INDIAN': 'ASIAN', 'ASIAN - CHINESE': 'ASIAN', 'ASIAN - KOREAN': 'ASIAN', 'ASIAN - SOUTH EAST ASIAN': 'ASIAN',
                                                        'BLACK/AFRICAN': 'BLACK','BLACK/AFRICAN AMERICAN': 'BLACK', 'BLACK/AFRICAN AMERICAN ': 'BLACK', 'BLACK/CAPE VERDEAN': 'BLACK', 'BLACK/CARIBBEAN ISLAND': 'BLACK', 
                                                        'HISPANIC OR LATINO': 'HISPANIC', 'HISPANIC/LATINO - COLUMBIAN': 'HISPANIC','HISPANIC/LATINO - CUBAN':'HISPANIC', 'HISPANIC/LATINO - CENTRAL AMERICAN':'HISPANIC', 
                                                        'HISPANIC/LATINO':'HISPANIC', 'HISPANIC/LATINO - GUATEMALAN': 'HISPANIC', 'HISPANIC/LATINO - SALVADORAN': 'HISPANIC',
                                                        'HISPANIC/LATINO - DOMINICAN':'HISPANIC', 'HISPANIC/LATINO - HONDURAN':'HISPANIC', 'HISPANIC/LATINO - MEXICAN':'HISPANIC', 'HISPANIC/LATINO - PUERTO RICAN': 'HISPANIC',
                                                        'MULTIPLE RACE/ETHNICITY':'OTHER', 'NATIVE HAWAIIAN OR OTHER PACIFIC ISLANDER':'OTHER', 'PORTUGUESE': 'OTHER', 'SOUTH AMERICAN':'HISPANIC', 
                                                        'WHITE - BRAZILIAN':'WHITE', 'WHITE - EASTERN EUROPEAN':'WHITE', 'WHITE - OTHER EUROPEAN':'WHITE', 'WHITE - RUSSIAN': 'WHITE',
                                                        'PATIENT DECLINED TO ANSWER':'UNKNOWN', 'UNABLE TO OBTAIN': 'UNKNOWN'})
    
    df = df[df['race'].isin(['WHITE','BLACK', 'ASIAN'])]
    # We drop certain amount of WHITE patients!
    if drop_some_white:
        n_drop_white = int(df[df['race'] == 'WHITE'].shape[0] * p)
        df_white_drop = df.query('(race == "WHITE")').sample(n=n_drop_white)
        df = df.drop(df_white_drop.index, axis = 0)
    print(df.shape)
    df['dataset'] = ['train' for i in range(df.shape[0])]
    # asian
    # since we only have ~3000 images for Asian, 
    # we will select ~800 for val (26%), 
    # ~800 for test (26%), 
    # ~1500 (~50%) for augmenting into training
    N = df[df['race'] == 'ASIAN'].shape[0]
    n_val = int(0.266 * N)#800 #int(0.17 * N)
    n_test = int(0.266 * N) #800 #int(0.33 * N)
    df_val = df.query('(race == "ASIAN")').sample(n=n_val)
    df_test = df.drop(df_val.index, axis=0).query('(race == "ASIAN")').sample(n=n_test)
    df.loc[df_val.index, 'dataset'] = 'val'
    df.loc[df_test.index, 'dataset'] = 'test'
    print(df[df['race'] == 'ASIAN']['dataset'].value_counts())

    # white and black
    N = df[df['race'] == "WHITE"].shape[0]
    n_val = int(0.10 * N)
    n_test = int(0.20 * N)
    df_val = df.query('(race == "WHITE")').sample(n=n_val)
    df_test = df.drop(df_val.index, axis=0).query('(race == "WHITE")').sample(n=n_test)
    df.loc[df_val.index, 'dataset'] = 'val'
    df.loc[df_test.index, 'dataset'] = 'test'
    print(df[df['race'] == "WHITE"]['dataset'].value_counts())

    N = df[df['race'] == "BLACK"].shape[0]
    n_val = int(0.10 * N)
    n_test = int(0.20 * N)
    df_val = df.query('(race == "BLACK")').sample(n=n_val)
    df_test = df.drop(df_val.index, axis=0).query('(race == "BLACK")').sample(n=n_test)
    df.loc[df_val.index, 'dataset'] = 'val'
    df.loc[df_test.index, 'dataset'] = 'test'
    print(df[df['race'] == "BLACK"]['dataset'].value_counts())

    print(df['dataset'].value_counts(normalize=True))
    print(df['race'].unique())

    return df.reset_index(drop=True)[['img_index','dataset', 'race']]

def prepare_balanced_testset_for_survive(df_meta, drop_uncommon_race = True):
    df = df_meta.copy(True)
    df['race'] = df['race'].replace({'ASIAN - ASIAN INDIAN': 'ASIAN', 'ASIAN - CHINESE': 'ASIAN', 'ASIAN - KOREAN': 'ASIAN', 'ASIAN - SOUTH EAST ASIAN': 'ASIAN',
                                                        'BLACK/AFRICAN': 'BLACK','BLACK/AFRICAN AMERICAN': 'BLACK', 'BLACK/AFRICAN AMERICAN ': 'BLACK', 'BLACK/CAPE VERDEAN': 'BLACK', 'BLACK/CARIBBEAN ISLAND': 'BLACK', 
                                                        'HISPANIC OR LATINO': 'HISPANIC', 'HISPANIC/LATINO - COLUMBIAN': 'HISPANIC','HISPANIC/LATINO - CUBAN':'HISPANIC', 'HISPANIC/LATINO - CENTRAL AMERICAN':'HISPANIC', 
                                                        'HISPANIC/LATINO':'HISPANIC', 'HISPANIC/LATINO - GUATEMALAN': 'HISPANIC', 'HISPANIC/LATINO - SALVADORAN': 'HISPANIC',
                                                        'HISPANIC/LATINO - DOMINICAN':'HISPANIC', 'HISPANIC/LATINO - HONDURAN':'HISPANIC', 'HISPANIC/LATINO - MEXICAN':'HISPANIC', 'HISPANIC/LATINO - PUERTO RICAN': 'HISPANIC',
                                                        'MULTIPLE RACE/ETHNICITY':'OTHER', 'NATIVE HAWAIIAN OR OTHER PACIFIC ISLANDER':'OTHER', 'PORTUGUESE': 'OTHER', 'SOUTH AMERICAN':'HISPANIC', 
                                                        'WHITE - BRAZILIAN':'WHITE', 'WHITE - EASTERN EUROPEAN':'WHITE', 'WHITE - OTHER EUROPEAN':'WHITE', 'WHITE - RUSSIAN': 'WHITE',
                                                        'PATIENT DECLINED TO ANSWER':'UNKNOWN', 'UNABLE TO OBTAIN': 'UNKNOWN'})
    if drop_uncommon_race:
        df = df[df['race'].isin(['WHITE','BLACK', 'ASIAN'])]
    print(df.shape)
    df['dataset'] = ['train' for i in range(df.shape[0])]

    df = df[df['survive'].isin(['SURVIVE', 'DIE'])][['img_index','dataset', 'survive']]
    
    N = df[df['survive'] == "DIE"].shape[0]
    n_val = int(0.10 * N)
    n_test = int(0.20 * N)
    df_val = df.query('(survive == "DIE")').sample(n=n_val)
    df_test = df.drop(df_val.index, axis=0).query('(survive == "DIE")').sample(n=n_test)
    df.loc[df_val.index, 'dataset'] = 'val'
    df.loc[df_test.index, 'dataset'] = 'test'
    print(df[df['survive'] == "DIE"]['dataset'].value_counts())

    df_val = df.query('(survive == "SURVIVE")').sample(n=n_val)
    df_test = df.drop(df_val.index, axis=0).query('(survive == "SURVIVE")').sample(n=n_test)
    df.loc[df_val.index, 'dataset'] = 'val'
    df.loc[df_test.index, 'dataset'] = 'test'
    print(df[df['survive'] == "SURVIVE"]['dataset'].value_counts())

    print(df['dataset'].value_counts(normalize=True))
    print(df['survive'].unique())

    return df.reset_index(drop=True)


def load_data_into_folder(df_meta, disease, task, output_train = False):
    """
    task: a column name in df_meta
    """
    assert task in df_meta.columns
    if task not in ['survive', 'race']:
        print('TASK NOT SUPPORTED')
        return -1
        
    
    if task == 'race':
        df_meta_task = df_meta.copy(deep=True)[['img_index', 'dataset', 'race']].dropna().reset_index(drop = True) # must re-index after dropna
        df_meta_task = prepare_balanced_testset_for_race(df_meta_task, drop_some_white= True, p = 0.4)
    elif task == 'survive':
        # need to include race because we can stratify out non-common races (ex. Indians/hispanic)
        df_meta_task = df_meta.copy(deep=True)[['img_index', 'dataset', 'survive', 'race']].dropna().reset_index(drop = True) # must re-index after dropna
        df_meta_task = prepare_balanced_testset_for_survive(df_meta_task, drop_uncommon_race=True)
    else:
        print('TASK NOT SUPPORTED')
        return -1
        
    class_names = list(df_meta_task[task].unique())
    print('class names', class_names)

    new_folder = f'data/{disease}/images/{task}'
    if os.path.exists(new_folder):
        shutil.rmtree(new_folder) # clear the folder first if exist
    DATA_DIR = Path(new_folder)
    DATASETS = ['train', 'val', 'test']
    for ds in DATASETS:
        for cls in class_names:
            (DATA_DIR / ds / cls).mkdir(parents=True, exist_ok=True)

    img_array = np.load(f'data/{disease}/X_{disease}.npy')
    print(img_array.shape)

    X_train_l = []
    y_train = []
    for i in tqdm(range(df_meta_task.shape[0])):
        img_ind = df_meta_task['img_index'][i]
        ds = df_meta_task['dataset'][i]
        # we do not add it if not labeled properly in dataset - processing dataset column before into df_meta
        if ds not in ['train', 'test', 'val']:
            continue 
        cls = df_meta_task[task][i]
        img_array_i = img_array[img_ind]
        img = Image.fromarray(img_array_i)

        if output_train and ds == 'train':
            X_train_l.append(img_array_i)
            y_train.append(cls)

        fname = f'{DATA_DIR}/{ds}/{cls}/{img_ind}.jpeg'
        img.save(fname)
    #X_train = np.concatenate(X_train_l)
    if output_train:
        X_train = np.array(X_train_l)
        return X_train, y_train

X_train, y_train = load_data_into_folder(df_meta, disease, task, output_train = True)
print(X_train.shape, len(y_train))

class BalancedDataGenerator(Sequence):
    """ImageDataGenerator + RandomOversampling"""
    def __init__(self, X, y, datagen, batch_size=32, save_imgs = False, dir_to_save = ''):
        self.datagen = datagen
        self.batch_size = min(batch_size, X.shape[0])
        self.datagen.fit(X) #datagen.fit(X)
        self.balanced_gen, self.steps_per_epoch = balanced_batch_generator(
                                        X.reshape(X.shape[0], -1), # shape (n_samples, n_features)
                                        y, 
                                        sampler=RandomOverSampler(), 
                                        batch_size=self.batch_size, 
                                        keep_sparse=True,
                                        random_state = 2022
                                        )
        # asterisks * just unpacks the tuple from shape
        # For example: (1, X.shape[1:]) = (1, (256, 256, 3))
        # while (1, *X.shape[1:]) = (1, 256, 256, 3)
        self.img_shape = X.shape[1:] # (256, 256, 3)
        self._shape = (self.steps_per_epoch * self.batch_size, *X.shape[1:])
        self.save_imgs = save_imgs
        self.dir_to_save = dir_to_save
        
        
    def __len__(self):
        return self.steps_per_epoch

    def __getitem__(self, idx):
        x_batch_balanced, y_batch_balanced = self.balanced_gen.__next__()
        x_batch_balanced = x_batch_balanced.reshape(-1, *self.img_shape)
        #print(x_batch_balanced.shape, len(y_batch_balanced))
        
        batches_balanced =  self.datagen.flow(
                            x_batch_balanced, y_batch_balanced, 
                            batch_size=self.batch_size,
                            # save_to_dir = self.dir_to_save if self.save_imgs else None,
                            # save_prefix = '',
                            # save_format = 'jpeg' # for smaller storage size           
                            )

        # return a pair of X_batch and y_batch 
        return batches_balanced.next()

# this needs to be updated if change model to efficient net

from keras.applications.resnet import preprocess_input 

train_gen = ImageDataGenerator(rotation_range=15,fill_mode='constant', horizontal_flip=True, zoom_range=0.1,
                            preprocessing_function=preprocess_input
            )
datagen = train_gen # define your data augmentation
bgen = BalancedDataGenerator(X_train, y_train, datagen, batch_size= 128, 
                        # save_imgs = save_imgs, 
                        # dir_to_save= dir_to_save
                        )

steps_per_epoch = bgen.steps_per_epoch
print('Total number of batches:', steps_per_epoch)

X_gen_l = []
y_gen_l = []
num_batches_needed = steps_per_epoch
for i in tqdm(range(num_batches_needed)): #steps_per_epoch
    X_gen, y_gen = bgen.__getitem__(0)
    X_gen_l.append(X_gen)
    y_gen_l.append(y_gen)

print(np.unique(y_train, return_counts=True))
print(np.unique(y_gen_l, return_counts=True))

X_gen_all = np.concatenate(X_gen_l)
y_gen_all = np.concatenate(y_gen_l)
print(X_gen_all.shape, y_gen_all.shape)

if save_imgs:
    if os.path.exists(dir_to_save):
        shutil.rmtree(dir_to_save) # clear the folder first if exist
        os.mkdir(dir_to_save)
        print('folder cleaned')
    else:
        os.mkdir(dir_to_save)
    DATA_DIR = Path(dir_to_save)
    class_names = np.unique(y_gen_all)
    for cls in class_names:
        (DATA_DIR / cls).mkdir(parents=True, exist_ok=True)

    for i in tqdm(range(X_gen_all.shape[0])):
        x = X_gen_all[i]
        # Fix type error: https://stackoverflow.com/questions/60138697/typeerror-cannot-handle-this-data-type-1-1-3-f4
        img = Image.fromarray((x * 255).astype(np.uint8))
        cls = y_gen_all[i]
        fname = f'{dir_to_save}/{cls}/aug_{i}.jpeg'
        img.save(fname)