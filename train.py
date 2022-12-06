from model import model_training
import tensorflow as tf

try:
  # Specify an invalid GPU device
  with tf.device('/GPU:1'):
    
    disease = 'All' #'Edema' #'All' # 'No_finding'
    task = 'gender'#'gender' #'insurance'#'white' # 'survive'

    model_name = 'densenet' #'resnet50' #'densenet'
    model_weight_pth = ''
    # model_weight_pth = 'saved_models/densenet_All_race_detection_LR-0.001_20221204-154429_epoch_019_val_loss_0.16900.h5'
    # model_weight_pth = 'saved_models/densenet_All_survive_detection_LR-0.0001_20221205-010809_epoch_002_val_loss_0.63442.h5'
    #model_weight_pth = 'saved_models/ResNet50_All_race_detection_LR-1e-05_20221204-111406_epoch_001_val_loss_0.37883.h5'
    
    
    learning_rate = 1e-4
    train_batch_size = 128
    epochs = 20 #25

    # set it to None it not using it
    class_weights = None
    # if disease =='All' and task == 'race':
    #   class_weights = {0: 7.884094368340944,
    #                   1: 1.3802995176806032,
    #                   2: 0.4654016002012606}

    model_training(disease, task, model_name, 
                    learning_rate, train_batch_size, epochs, model_weight_pth, class_weights)

except RuntimeError as e:
  print(e)
