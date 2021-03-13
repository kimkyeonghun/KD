import numpy as np
import pandas as pd
import keras as k
import tensorflow as tf

from student import trainModel

from tensorflow.keras import datasets
from tensorflow.keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        tf.config.experimental.set_virtual_device_configuration(
            gpus[0], [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=4096)])
    except RuntimeError as e: # 프로그램 시작시에 가상 장치가 설정되어야만 합니다
        print(e)



def loadData():
    (train_x, train_y), (test_x, test_y) = datasets.cifar10.load_data()
    datagen = ImageDataGenerator(rotation_range = 15,horizontal_flip = True, vertical_flip= True, width_shift_range= 0.1, height_shift_range= 0.1, zoom_range = 0.1)
    datagen.fit(train_x)
    train_x=train_x.astype("float32")  
    test_x=test_x.astype("float32")
    mean=np.mean(train_x)
    std=np.std(train_x)
    test_x=(test_x-mean)/std
    train_x=(train_x-mean)/std

    # labels
    num_classes=10
    train_y = k.utils.to_categorical(train_y, num_classes)
    test_y = k.utils.to_categorical(test_y, num_classes)
    return (train_x,train_y,test_x,test_y)

def loadModel(file_name):
    model = load_model(file_name)
    return model


def main():
    model = loadModel('teacher.h5')
    dataSet= loadData()
    lambdas = [0.2]
    Temperatures = np.linspace(1,20,20)
    for lam in lambdas:
        results = []
        for t in Temperatures:
            result = trainModel(model,dataSet,lam,t)
            results.append(result)
    
        df = pd.DataFrame(results,columns=["lambda","Temperature","test_loss","test_acc"])
        df.to_csv("results{}.csv".format(lam))

    


if __name__== "__main__":
    main()
