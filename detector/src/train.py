#!/usr/bin/env python3

import .model_zoo as mz
#import .utils
import argparse
import datetime
#import pathlib
#import os
import sys
import tensorflow as tf
#from tensorboard.plugins.hparams import api as hp

parser = argparse.ArgumentParser()
parser.add_argument('--model', '-m', nargs='+', type=str, default=sys.stdin, help='Name of Model(s) you wish to train from among the models defined in `model_zoo` directory')
parser.add_argument('--dataset', '-d', type=str, default='../../dataset/20bn-jester-v1/',  help='path to dataset directory')
parser.add_argument('--logdir', '-l', type=str, default='./logdir/', help='path to log directory')

args = parser.parse_args()

tf.keras.backend.clear_session()

# TODO Complete Data Loading from the dataset directory
'''
train_data = tf.placeholder(train_images.dtype, train_images.shape)
train_labels = tf.placeholder(labels.dtype, labels.shape)

train_dataset = tf.data.Dataset.from_tensor_slices((data,labels))
train_dataset.batch(37)

val_data = tf.placeholder(val_images.dtype, val_images.shape)
val_labels = tf.placeholder(labels.dtype, labels.shape)

val_dataset = tf.data.Dataset.from_tensor_slices((data,labels))
val_dataset.batch(37)

iterator = train_dataset.make_initializable_iterator()
'''

model = mz[args.model]
# model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics='accuracy')
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics='accuracy')

tfkcallbacks = []

if args.logdir:
    # TODO Change path strings to os.path.joins and equivalents or use pathlib
    logdir = args.logdir + '/' + args.model + '/' + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tfkcallbacks.append(tf.keras.callbacks.TensorBoard(log_dir=logdir, histogram_freq=1, write_graph=True, write_image=True, update_freq='epoch'))
    file_writer_cm = tf.summary.create_file_writer(logdir + '/cm')
    print(f'start $tensorboard --logdir {args.logdir}/{args.model}/')
    tfkcallbacks.append(keras.callbacks.LambdaCallback(on_epoch_end=utils.log_confusion_matrix))

tfkcallbacks.append(tf.keras.callbacks.ModelCheckpoint(filepath='./models/' + args.model + 'weights.{epoch:02d}-{val_loss:.2f}.hdf5', save_best_only=True, save_weights_only))

history = model.fit(train_dataset, batch_size=37, verbose=0, epochs=10, callbacks=tfkcallbacks, validation_data=val_dataset, shuffle=True)

model.save('./models/' + args.model, save_format='tf')
# TODO Check serializabillity of subclassed models
# model.save_weights('./models/' + args.model, save_format='tf')

