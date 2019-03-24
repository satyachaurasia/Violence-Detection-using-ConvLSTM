from keras import Input
from keras.callbacks import Callback
from keras.layers import Dense, Flatten, Dropout, ZeroPadding3D, ConvLSTM2D, Reshape, BatchNormalization, Activation
from keras.layers.recurrent import LSTM
from keras.models import Sequential, load_model
from keras.optimizers import Adam, RMSprop
from keras.layers.wrappers import TimeDistributed
from keras.layers.convolutional import (Conv2D, MaxPooling3D, Conv3D,
    MaxPooling2D)
from collections import deque
import sys
import logging
from keras.applications import Xception, ResNet50, InceptionV3
from keras.layers import Dense, GlobalAveragePooling2D
from keras.models import Model
import preprocess
from keras.models import load_model

from keras.callbacks import EarlyStopping, ReduceLROnPlateau, Callback


ResNet50_base_model = ResNet50(weights='imagenet', include_top=False) #include_top= False excludes final fc layer
print('ResNet50 v3 base model without FC loaded')

seq_len=10
size=244
dropout = 0.0
input_layer = Input(shape=(seq_len, size, size, 3))

cnn = TimeDistributed(ResNet50_base_model)(input_layer)

#cnn = Reshape((seq_len,4, 4, 128), input_shape=(seq_len,8, 8, 2048))(cnn)

#lstm = (ConvLSTM2D, dict(filters=256, kernel_size=(3, 3), padding='same', return_sequences=False))


lstm=ConvLSTM2D(256, (3, 3), padding='same', return_sequences=False)(cnn)

lstm = MaxPooling2D(pool_size=(2, 2))(lstm)
flat = Flatten()(lstm)

flat = BatchNormalization()(flat)
flat = Dropout(dropout)(flat)
linear = Dense(1000)(flat)

relu = Activation('relu')(linear)
linear = Dense(256)(relu)
linear = Dropout(dropout)(linear)
relu = Activation('relu')(linear)
linear = Dense(10)(relu)
linear = Dropout(dropout)(linear)
relu = Activation('relu')(linear)

activation = 'sigmoid'
loss_func = 'binary_crossentropy'
classes=1
dropout = 0.0

predictions = Dense(classes,  activation=activation)(relu)

model = Model(inputs=input_layer, outputs=predictions)

for layer in ResNet50_base_model.layers:
    layer.trainable = False

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

print(model.summary())


def get_generators(dataset_name, dataset_videos, datasets_frames, fix_len, figure_size, force, classes=1, use_aug=False,
                   use_crop=True, crop_dark=True):
    train_path, valid_path, test_path, \
    train_y, valid_y, test_y, \
    avg_length = preprocess.createDataset(dataset_videos, dataset_name, datasets_frames, fix_len, force=force)

    if fix_len is not None:
        avg_length = fix_len
    crop_x_y = None
    if (crop_dark):
        crop_x_y = (11, 38)

    batch_size=2

    len_train, len_valid = len(train_path), len(valid_path)
    train_gen = preprocess.data_generator(train_path, train_y, batch_size, figure_size, avg_length, use_aug=use_aug,
                                              use_crop=use_crop, crop_x_y=crop_x_y, classes=classes)
    validate_gen = preprocess.data_generator(valid_path, valid_y, batch_size, figure_size, avg_length,
                                                 use_aug=False, use_crop=False, crop_x_y=crop_x_y, classes=classes)
    test_x, test_y = preprocess.get_sequences(test_path, test_y, figure_size, avg_length, crop_x_y=crop_x_y,
                                                  classes=classes)

    return train_gen, validate_gen, test_x, test_y, avg_length, len_train, len_valid



dataset_video_path='D:\\Satya\\Violent Detection\\Ho\\'
figure_output_path='D:\\Satya\\Violent Detection\\frames\\'
dataset_name='Hocky'
fix_len=10
figure_size=244
force=True
use_aug=True
classes=1
batch_size=5
batch_epoch_ratio = 0.5
epochs=10
patience_es=15
patience_lr=5
train_gen, validate_gen, test_x, test_y, seq_len, len_train, len_valid = get_generators(dataset_name,
                                                                                            dataset_video_path,
                                                                                            figure_output_path, fix_len,
                                                                                            figure_size,
                                                                                            force=force,                                                                                      classes=classes,
                                                                                            use_aug=use_aug,
                                                                                            use_crop=True,
                                                                                            crop_dark=True)

class TestCallback(Callback):
    def __init__(self, test_data):
        self.test_data = test_data
        self.test_loss = []
        self.test_acc = []

    def on_epoch_end(self, epoch, logs={}):
        x, y = self.test_data
        loss, acc = self.model.evaluate(x, y, batch_size=5, verbose=0)
        self.test_loss.append(loss)
        self.test_acc.append(acc)
        print('\nTesting loss: {}, acc: {}\n'.format(loss, acc))

test_history = TestCallback((test_x, test_y))



history = model.fit_generator(
        steps_per_epoch=int(float(len_train) / float(batch_size * batch_epoch_ratio)),
        generator=train_gen,
        epochs=epochs,
        validation_data=validate_gen,
        validation_steps=int(float(len_valid) / float(batch_size)),
        callbacks=[EarlyStopping(monitor='val_loss', min_delta=0.001, patience=patience_es, ),
                   ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=patience_lr, min_lr=1e-8, verbose=1),
                   test_history
                   ]
    )

model.save('model.h5')
