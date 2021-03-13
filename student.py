import tensorflow as tf
import numpy as np
from tensorflow import keras
from tensorflow.keras.layers import Input,ZeroPadding2D,Conv2D,MaxPooling2D,Flatten,Dense,Dropout,Activation,BatchNormalization
from tensorflow.keras.optimizers import Adam
from keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Lambda
from tensorflow.keras.layers import concatenate

from tensorflow.keras.metrics import categorical_accuracy
from tensorflow.keras.losses import kullback_leibler_divergence as KLD_Loss, categorical_crossentropy as logloss

import os
import keras.backend.tensorflow_backend as KK
 
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        tf.config.experimental.set_virtual_device_configuration(
            gpus[0], [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=4096)])
    except RuntimeError as e: # 프로그램 시작시에 가상 장치가 설정되어야만 합니다
        print(e)
 
def KD_loss(y_true,y_pred,lambd=0.5,T=10.0):
    y_true,y_true_KD = y_true[:,:10],y_true[:,10:]
    y_pred,y_pred_KD = y_pred[:,:10],y_pred[:,10:]
    # Classic cross-entropy (without temperature) hard target
    CE_loss = logloss(y_true,y_pred)
    # KL-Divergence loss for softened output (with temperature) soft target Loss_Teacher
    KL_loss = T**2*KLD_Loss(y_true_KD,y_pred_KD)

    return (1-lambd)*CE_loss + lambd*KL_loss

def accuracy(y_true,y_pred):
    return categorical_accuracy(y_true,y_pred)

def genSoftTarget(model,tem,dataset):
    Teacher_logits = keras.Model(model.input,model.layers[-1].output)
    T_layer = Lambda(lambda x:x/tem)(Teacher_logits.output)
    Softmax_layer = Activation('softmax')(T_layer)
    Teacher_soften = keras.Model(model.input,Softmax_layer)

    y_train_new = Teacher_soften.predict(dataset[0])

    # Hard+soft
    y_train_new = np.c_[dataset[1],y_train_new]
    y_test_new = np.c_[dataset[3],dataset[3]]

    return (y_train_new,y_test_new)

def initStudent(lam,tem):
    with tf.device('/gpu:0'):
        Input_layer = Input(shape=(32,32,3))

        zero1 = ZeroPadding2D((1, 1))(Input_layer)
        conv1 = Conv2D(32,(5,5),activation='relu')(zero1)
        bat1 = BatchNormalization()(conv1)
        drop1 = Dropout(0.3)(bat1)
        stage1 = MaxPooling2D((2,2),strides=(1,1))(drop1)

        zero2 = ZeroPadding2D((1, 1))(stage1)
        conv2 = Conv2D(64,(5,5),activation='relu')(zero2)
        bat2 = BatchNormalization()(conv2)
        drop2 = Dropout(0.4)(bat2)
        stage2 = MaxPooling2D((2,2),strides=(1,1))(drop2)

        zero3 = ZeroPadding2D((1, 1))(stage2)
        conv3 = Conv2D(128,(5,5),activation='relu')(zero3)
        bat3 = BatchNormalization()(conv3)
        drop3 = Dropout(0.4)(bat3)
        stage3 = MaxPooling2D((2,2),strides=(1,1))(drop3)

        flat = Flatten()(stage3)
        dense1 = Dense(500)(flat)
        dropd = Dropout(0.5)(dense1)
        output2 = Dense(10)(dropd)

        probs = Activation("softmax")(output2)
        logits_T = Lambda(lambda x:x/tem)(output2)
        probs_T = Activation("softmax")(logits_T)
        CombinedLayers = concatenate([probs,probs_T])
        StudentModel = keras.Model(Input_layer,CombinedLayers)
        adam = Adam(learning_rate = 0.0001, beta_1=0.9, beta_2=0.999)
        StudentModel.compile(optimizer= adam,loss=lambda y_true,y_pred: KD_loss(y_true, y_pred,lambd=round(lam,2),T=tem),metrics=[accuracy])

    return StudentModel

def trainModel(model,dataset,lam,tem):
    softTarget = genSoftTarget(model,tem,dataset)
    Student = initStudent(lam,tem)
    early_stopping = EarlyStopping(monitor='val_accuracy',patience=10,mode='auto',restore_best_weights=True)
    history = Student.fit(dataset[0],softTarget[0],epochs=200,validation_split=0.2,batch_size=32,callbacks=[early_stopping],shuffle=True)
    test_logit = keras.Model(Student.input,Student.layers[-5].output) 

    Temperature = 1
    T_layer = Lambda(lambda x:x/Temperature)(test_logit.output)
    Softmax_layer = Activation('softmax')(T_layer)
    Test = keras.Model(Student.input,Softmax_layer)
    adam = Adam(learning_rate = 0.0001, beta_1=0.9, beta_2=0.999)
    Test.compile(optimizer= adam,loss='categorical_crossentropy',metrics=[accuracy])
    test_loss, test_acc = Test.evaluate(dataset[2],dataset[3])
    print("[Lambda : {}, Temperature : {}]Test_loss is {}, Test_acc is {}".format(round(lam,2),tem,test_loss,test_acc))
    return (round(lam,2),tem,test_loss,test_acc)