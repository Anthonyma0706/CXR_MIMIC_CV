
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


def load_data_into_folder(df_meta, disease, task, output_train = False):
    """
    task: a column name in df_meta
    """
        
    if task == 'insurance':
        df_meta_task = df_meta
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
    
    if output_train:
        X_train = np.array(X_train_l)
        return X_train, y_train




disease = 'No_finding' # 'No_finding'
#df_meta = pd.read_csv('data/All/df_insu_full_split_subjects_0327.csv')
df_meta = pd.read_csv('data/No_finding/metadata_No_finding_for_insurance_split_by_subjects_0306_2023.csv')
task = 'insurance'

print('===============start loading data===============')
load_data_into_folder(df_meta, disease, task, output_train = False)