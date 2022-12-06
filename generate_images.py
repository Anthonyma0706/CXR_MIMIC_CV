
import numpy as np
import cv2 as cv
import pandas as pd
from tqdm import tqdm
import os
import shutil
from pathlib import Path
from PIL import Image
from sklearn.utils import shuffle

import glob
from imblearn.over_sampling import RandomOverSampler
from imblearn.keras import balanced_batch_generator
import tensorflow as tf
from keras.utils.data_utils import Sequence
from keras.applications.resnet import ResNet50, preprocess_input
from tensorflow.keras.preprocessing.image import ImageDataGenerator





metadata_path = "./data/mimic_iv_csv/"
mimiciv = pd.read_csv(os.path.join(metadata_path,'img_gapyear.csv')) # contain survival year data
patients = pd.read_csv(os.path.join(metadata_path,'patients.csv'))
unique_subject = list(patients['subject_id'].unique())
agee = pd.read_csv(os.path.join(metadata_path,'Age.csv'))
admission = pd.read_csv(os.path.join(metadata_path,'admissions.csv'))
admission['race'] = admission['race'].replace({'ASIAN - ASIAN INDIAN': 'ASIAN', 'ASIAN - CHINESE': 'ASIAN', 'ASIAN - KOREAN': 'ASIAN', 'ASIAN - SOUTH EAST ASIAN': 'ASIAN',
                                                    'BLACK/AFRICAN': 'BLACK','BLACK/AFRICAN AMERICAN': 'BLACK', 'BLACK/AFRICAN AMERICAN ': 'BLACK', 'BLACK/CAPE VERDEAN': 'BLACK', 'BLACK/CARIBBEAN ISLAND': 'BLACK', 
                                                    'HISPANIC OR LATINO': 'HISPANIC', 'HISPANIC/LATINO - COLUMBIAN': 'HISPANIC','HISPANIC/LATINO - CUBAN':'HISPANIC', 'HISPANIC/LATINO - CENTRAL AMERICAN':'HISPANIC', 
                                                    'HISPANIC/LATINO':'HISPANIC', 'HISPANIC/LATINO - GUATEMALAN': 'HISPANIC', 'HISPANIC/LATINO - SALVADORAN': 'HISPANIC',
                                                    'HISPANIC/LATINO - DOMINICAN':'HISPANIC', 'HISPANIC/LATINO - HONDURAN':'HISPANIC', 'HISPANIC/LATINO - MEXICAN':'HISPANIC', 'HISPANIC/LATINO - PUERTO RICAN': 'HISPANIC',
                                                    'MULTIPLE RACE/ETHNICITY':'OTHER', 'NATIVE HAWAIIAN OR OTHER PACIFIC ISLANDER':'OTHER', 'PORTUGUESE': 'OTHER', 'SOUTH AMERICAN':'HISPANIC', 
                                                    'WHITE - BRAZILIAN':'WHITE', 'WHITE - EASTERN EUROPEAN':'WHITE', 'WHITE - OTHER EUROPEAN':'WHITE', 'WHITE - RUSSIAN': 'WHITE',
                                                    'PATIENT DECLINED TO ANSWER':'UNKNOWN', 'UNABLE TO OBTAIN': 'UNKNOWN'})
print(admission['race'].unique())


def tf_records_to_array(disease:list):   
    '''
    disease: a list of image labels, 
    Get images of certain label ('No Finding', 'Edema', ...)
    Read bytes data from tf_records into images and metadata (sex,race,age,insurance)
    return 4D npy array of images, and a dataframe of metadata associated with each image
    '''
    assert len(disease) == 1 or len(disease) == 2 # only support up to 2 classes

    imgs = [] # all images in npy array
    y = []  # survival_year
    subject_idd = []
    study_idd=[]
    race=[]
    gender=[]
    age=[]
    insurance=[]
    set_belong = [] # 'train', 'test', 'val'

    for dataset in tqdm(['train', 'test', 'val']):
        filename = f'data/tf_records/mimic_{dataset}.tfrecords'
        raw_dataset = tf.data.TFRecordDataset(filename)

        for raw_record in raw_dataset:

            example = tf.train.Example()
            example.ParseFromString(raw_record.numpy())

            subject_id = example.features.feature['subject_id'].int64_list.value[0]
            study_id = example.features.feature['study_id'].int64_list.value[0]
            
            # select CXRs
            if disease == ['All']: # use all datasets!
                img_found = True
            else:
                if len(disease) == 1: # 'No Finding', 'Edema'
                    img_found = example.features.feature[disease[0]].float_list.value[0] == 1
                elif len(disease) == 2:
                    img_found = example.features.feature[disease[0]].float_list.value[0] == 1 or example.features.feature[disease[1]].float_list.value[0] == 1
                else:
                    print('ERROR IN DISEASE SPECIFICATION')
                    return -1
               
            
            if img_found:
            
                nparr = np.fromstring(example.features.feature['jpg_bytes'].bytes_list.value[0], np.uint8)
                img_np = cv.imdecode(nparr, cv.IMREAD_GRAYSCALE)
                #convert to (256,256,3) dimension to use imagenet pretrained weights
                gary2rgb = cv.cvtColor(img_np,cv.COLOR_GRAY2RGB) 
                imgs.append(gary2rgb)
            
                mimiciv_sub = mimiciv.loc[mimiciv['subject_id'] == subject_id]
                if mimiciv_sub.empty:
                    y.append(-1)
                else:
                    y.append(mimiciv_sub.loc[mimiciv_sub['study_id'] == study_id,'gap_year'].iloc[0])
            
                subject_idd.append(subject_id)
                study_idd.append(study_id)
                
                admission_sub = admission.loc[admission['subject_id'] == subject_id]
                patients_sub = patients.loc[patients['subject_id'] == subject_id]
                age_sub = agee.loc[agee['subject_id'] == subject_id]
                
                #getting the 'race', 'insurance', 'gender' and 'age' features(fill np.nan with N/a)
                if admission_sub.empty:
                    race.append(np.nan)
                    insurance.append(np.nan)
                else:
                    race.append(admission_sub.loc[admission_sub['subject_id'] == subject_id,'race'].iloc[0])
                    insurance.append(admission_sub.loc[admission_sub['subject_id'] == subject_id,'insurance'].iloc[0])
            
                if patients_sub.empty:
                    gender.append(np.nan)
                else:
                    gender.append(patients_sub.loc[patients_sub['subject_id'] == subject_id,'gender'].iloc[0])
            
                if age_sub.empty:
                    age.append(np.nan)
                else:
                    age.append(age_sub.loc[age_sub['subject_id'] == subject_id,'age'].iloc[0])
                set_belong.append(dataset)
    
                assert len(imgs) == len(subject_idd) == len(study_idd) == len(set_belong) == len(age) == len(gender) == len(race) == len(insurance)
    
    df_info = pd.DataFrame({'img_index': list(range(len(subject_idd))),
                            'subject_id':subject_idd, 
                            'study_id': study_idd,
                            'dataset': set_belong,
                            'race': race,
                            'gender': gender,
                            'age': age,
                            'insurance': insurance,
                            'survival_year': y}) 

    return imgs, df_info #y, subject_idd, study_idd, race, gender, age, insurance, set_belong

def split_datasets_for_race(df_meta, binary_white = False, drop_some_white = False, drop_white_p = 0.4):
    df = df_meta.copy(True)
    df['race'] = df['race'].replace({'ASIAN - ASIAN INDIAN': 'ASIAN', 'ASIAN - CHINESE': 'ASIAN', 'ASIAN - KOREAN': 'ASIAN', 'ASIAN - SOUTH EAST ASIAN': 'ASIAN',
                                                        'BLACK/AFRICAN': 'BLACK','BLACK/AFRICAN AMERICAN': 'BLACK', 'BLACK/AFRICAN AMERICAN ': 'BLACK', 'BLACK/CAPE VERDEAN': 'BLACK', 'BLACK/CARIBBEAN ISLAND': 'BLACK', 
                                                        'HISPANIC OR LATINO': 'HISPANIC', 'HISPANIC/LATINO - COLUMBIAN': 'HISPANIC','HISPANIC/LATINO - CUBAN':'HISPANIC', 'HISPANIC/LATINO - CENTRAL AMERICAN':'HISPANIC', 
                                                        'HISPANIC/LATINO':'HISPANIC', 'HISPANIC/LATINO - GUATEMALAN': 'HISPANIC', 'HISPANIC/LATINO - SALVADORAN': 'HISPANIC',
                                                        'HISPANIC/LATINO - DOMINICAN':'HISPANIC', 'HISPANIC/LATINO - HONDURAN':'HISPANIC', 'HISPANIC/LATINO - MEXICAN':'HISPANIC', 'HISPANIC/LATINO - PUERTO RICAN': 'HISPANIC',
                                                        'MULTIPLE RACE/ETHNICITY':'OTHER', 'NATIVE HAWAIIAN OR OTHER PACIFIC ISLANDER':'OTHER', 'PORTUGUESE': 'OTHER', 'SOUTH AMERICAN':'HISPANIC', 
                                                        'WHITE - BRAZILIAN':'WHITE', 'WHITE - EASTERN EUROPEAN':'WHITE', 'WHITE - OTHER EUROPEAN':'WHITE', 'WHITE - RUSSIAN': 'WHITE',
                                                        'PATIENT DECLINED TO ANSWER':'UNKNOWN', 'UNABLE TO OBTAIN': 'UNKNOWN'})
    if binary_white:
        df.loc[df['race'] != 'WHITE', 'race'] = 'NON_WHITE' 
        return df

    df = df[df['race'].isin(['WHITE','BLACK', 'ASIAN'])]
    print('============== Initial Value counts ========================')
    print(df['dataset'].value_counts(normalize = True))
    # We drop certain amount of WHITE patients!
    if drop_some_white:
        n_drop_white = int(df[df['race'] == 'WHITE'].shape[0] * drop_white_p)
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
    n_val = int(0.20 * N)#800 #int(0.17 * N)
    n_test = int(0.20 * N) #800 #int(0.33 * N)
    df_val = df.query('(race == "ASIAN")').sample(n=n_val)
    df_test = df.drop(df_val.index, axis=0).query('(race == "ASIAN")').sample(n=n_test)
    df.loc[df_val.index, 'dataset'] = 'val'
    df.loc[df_test.index, 'dataset'] = 'test'
    print('============== Value counts for ASIAN ========================')
    print(df[df['race'] == 'ASIAN']['dataset'].value_counts())

    # white and black
    N = df[df['race'] == "WHITE"].shape[0]
    n_val = int(0.20 * N)
    n_test = int(0.30 * N)
    df_val = df.query('(race == "WHITE")').sample(n=n_val)
    df_test = df.drop(df_val.index, axis=0).query('(race == "WHITE")').sample(n=n_test)
    df.loc[df_val.index, 'dataset'] = 'val'
    df.loc[df_test.index, 'dataset'] = 'test'
    print('============== Value counts for WHITE ========================')
    print(df[df['race'] == "WHITE"]['dataset'].value_counts())

    N = df[df['race'] == "BLACK"].shape[0]
    n_val = int(0.10 * N)
    n_test = int(0.20 * N)
    df_val = df.query('(race == "BLACK")').sample(n=n_val)
    df_test = df.drop(df_val.index, axis=0).query('(race == "BLACK")').sample(n=n_test)
    df.loc[df_val.index, 'dataset'] = 'val'
    df.loc[df_test.index, 'dataset'] = 'test'
    print('============== Value counts for BLACK ========================')
    print(df[df['race'] == "BLACK"]['dataset'].value_counts())
    
    print('============== FINAL Value counts ========================')
    print(df['dataset'].value_counts(normalize=True))
    print(df['race'].unique())

    return df.reset_index(drop=True)[['img_index','dataset', 'race']]


def split_datasets_for_survive(df_meta, drop_uncommon_race = True):
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
    print('============== Initial Value counts ========================')
    print(df['dataset'].value_counts(normalize = True))
    
    df['dataset'] = ['train' for i in range(df.shape[0])]

    df = df[df['survive'].isin(['SURVIVE', 'DIE'])][['img_index','dataset', 'survive']]
    

    N = df[df['survive'] == "DIE"].shape[0]
    n_val = int(0.15 * N)
    n_test = int(0.15 * N)
    df_val = df.query('(survive == "DIE")').sample(n=n_val)
    df_test = df.drop(df_val.index, axis=0).query('(survive == "DIE")').sample(n=n_test)
    df.loc[df_val.index, 'dataset'] = 'val'
    df.loc[df_test.index, 'dataset'] = 'test'
    print('============== Value counts for DIE ========================')
    print(df[df['survive'] == "DIE"]['dataset'].value_counts())

    df_val = df.query('(survive == "SURVIVE")').sample(n=n_val)
    df_test = df.drop(df_val.index, axis=0).query('(survive == "SURVIVE")').sample(n=n_test)
    df.loc[df_val.index, 'dataset'] = 'val'
    df.loc[df_test.index, 'dataset'] = 'test'
    print('============== Value counts for SURVIVE ========================')
    print(df[df['survive'] == "SURVIVE"]['dataset'].value_counts())
    
    print('============== FINAL Value counts ========================')
    print(df['dataset'].value_counts(normalize=True))
    print(df['survive'].unique())

    return df.reset_index(drop=True)


def load_data_into_folder(df_meta, disease, task, output_train = False):
    """
    task: a column name in df_meta
    """
    if task != 'white':
        assert task in df_meta.columns
        
    
    
    if task == 'race':
        df_meta_task = df_meta.copy(deep=True)[['img_index', 'dataset', 'race']].dropna().reset_index(drop = True) # must re-index after dropna
        df_meta_task = split_datasets_for_race(df_meta_task, drop_some_white= False)
    elif task == 'white':
        df_meta_task = df_meta.copy(deep=True)[['img_index', 'dataset', 'race']].dropna().reset_index(drop = True) # must re-index after dropna
        df_meta_task = split_datasets_for_race(df_meta_task, binary_white = True, drop_some_white= False)
    elif task == 'survive':
        # need to include race because we can stratify out non-common races (ex. Indians/hispanic)
        df_meta_task = df_meta.copy(deep=True)[['img_index', 'dataset', 'survive', 'race']].dropna().reset_index(drop = True) # must re-index after dropna
        df_meta_task = split_datasets_for_survive(df_meta_task, drop_uncommon_race = False)
    elif task == 'insurance':
        df_meta_task = df_meta.copy(deep=True)[['img_index', 'dataset', 'insurance']].dropna().reset_index(drop = True)
    elif task == 'gender':
        df_meta_task = df_meta.copy(deep=True)[['img_index', 'dataset', 'gender']].dropna().reset_index(drop = True)
    else:
        print('TASK NOT SUPPORTED')
        return -1
    
    new_folder = f'data/{disease}/images/{task}'
    if os.path.exists(new_folder):
        shutil.rmtree(new_folder) # clear the folder first if exist
    DATA_DIR = Path(new_folder)
    print('=============================================')
    print('=============================================')
    print('save to', DATA_DIR)
    print('=============================================')
    print('=============================================')
    DATASETS = ['train', 'val', 'test']


    if task == 'white':
        col = 'race'
    else:
        col = task    
    class_names = list(df_meta_task[col].unique())
    print('class names', class_names)

    for ds in DATASETS:
        for cls in class_names:
            (DATA_DIR / ds / cls).mkdir(parents=True, exist_ok=True)

    

    img_array = np.load(f'data/{disease}/X_{disease}.npy')
    print('array loaded', img_array.shape)

    X_train_l = []
    y_train = []
    for i in tqdm(range(df_meta_task.shape[0])):
        img_ind = df_meta_task['img_index'][i]
        ds = df_meta_task['dataset'][i]
        # we do not add it if not labeled properly in dataset - processing dataset column before into df_meta
        if ds not in ['train', 'test', 'val']:
            continue 
        cls = df_meta_task[col][i]
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


# try:
#   # Specify an invalid GPU device
#   with tf.device('/GPU:1'):
    
#     # disease_l = ['All'] #['Pneumonia' , 'Consolidation'] #['Edema'], ['No Finding']
#     # imgs, df_info = tf_records_to_array(disease_l)
#     # df_info.to_csv('data/All/df_All.csv', index = False)
#     # np.save('data/All/X_All.npy', imgs)

# except RuntimeError as e:
#   print(e)



disease = 'All' #'All' #'Edema' # 'Pneumonia_Consolidation' , 'No_finding'
df_meta = pd.read_csv(f'data/{disease}/df_{disease}.csv')
task = 'gender'#'insurance'#'white' # 'survive', 'race'

print('===============start loading data===============')
load_data_into_folder(df_meta, disease, task, output_train = False)