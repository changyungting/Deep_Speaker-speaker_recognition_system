
# train-clean-100: 251 speaker, 28539 utterance
# train-clean-360: 921 speaker, 104104 utterance
# test-clean: 40 speaker, 2620 utterance
# merged test: 80 speaker, 5323 utterance
# batchisize 32*3 : train on triplet: 5s - > 3.1s/steps , softmax pre train: 3.1 s/steps


import logging
from time import time

import numpy as np
import sys
import os
import random
from tensorflow.python.keras.optimizers import Adam
from tensorflow.python.keras.layers.core import Dense
from tensorflow.python.keras.models import Model

import constants as c
import select_batch
from pre_process import data_catalog, preprocess_and_save
from models import convolutional_model, convolutional_model_simple, recurrent_model
from random_batch import stochastic_mini_batch
from triplet_loss import deep_speaker_loss
from utils import get_last_checkpoint_if_any, create_dir_and_delete_content
from test_model import eval_model



def create_dict(files,labels,spk_uniq):
    train_dict = {}
    for i in range(len(spk_uniq)):
        train_dict[spk_uniq[i]] = []
    # train_dict[19]=[] train_dict[26]=[]
    for i in range(len(labels)):
        train_dict[labels[i]].append(files[i])
    # train_dict={19: ['audio/LibriSpeechSamples/train-clean-100-npy/19-150-0001.npy', 'audio/LibriSpeechSamples/train-clean-100-npy/19-150-0001.npy'],
    #             26: ['audio/LibriSpeechSamples/train-clean-100-npy/26-100-0001.npy']}
    for spk in spk_uniq:
        if len(train_dict[spk]) < 2:
            train_dict.pop(spk)
    # 26 pop out
    # train_dict={19: ['audio/LibriSpeechSamples/train-clean-100-npy/19-150-0001.npy', 'audio/LibriSpeechSamples/train-clean-100-npy/19-150-0001.npy']}
    unique_speakers=list(train_dict.keys())
    # unique_speakers=[19]
    return train_dict, unique_speakers

def main(libri_dir=c.DATASET_DIR):

    PRE_TRAIN = c.PRE_TRAIN
    logging.info('Looking for fbank features [.npy] files in {}.'.format(libri_dir))
    libri = data_catalog(libri_dir)
    #                          filename                                       speaker_id
    #   0    audio/LibriSpeechSamples/train-clean-100-npy/1-100-0001.npy        1
    #   1    audio/LibriSpeechSamples/train-clean-100-npy/1-100-0002.npy        1        
    unique_speakers = libri['speaker_id'].unique() # 251 speaker
    # spk_utt_dict, unique_speakers = create_dict(libri['filename'].values,libri['speaker_id'].values,unique_speakers)
    # libri['filename'].values:array type [audio/LibriSpeechSamples/train-clean-100-npy/19-150-0001.npy,
    #                                      audio/LibriSpeechSamples/train-clean-100-npy/19-100-0001.npy
    #                                      audio/LibriSpeechSamples/train-clean-100-npy/26-100-0001.npy]
    # libri['speaker_id'].values:array type [19,19,26]
    # unique_speakers=libri['speaker_id'].unique():array type [19,26]
    # select_batch.create_data_producer(unique_speakers, spk_utt_dict)

    batch = stochastic_mini_batch(libri, batch_size=c.BATCH_SIZE, unique_speakers=unique_speakers)
    batch_size = c.BATCH_SIZE * c.TRIPLET_PER_BATCH
    x, y = batch.to_inputs() # 
    b = x[0]
    num_frames = b.shape[0]
    train_batch_size = batch_size
    #batch_shape = [batch_size * num_frames] + list(b.shape[1:])  # A triplet has 3 parts.
    input_shape = (num_frames, b.shape[1], b.shape[2])

    logging.info('num_frames = {}'.format(num_frames))
    logging.info('batch size: {}'.format(batch_size))
    logging.info('input shape: {}'.format(input_shape))
    logging.info('x.shape : {}'.format(x.shape))
    model = convolutional_model(input_shape=input_shape, batch_size=batch_size, num_frames=num_frames)
    gru_model = None
    grad_steps = 0

    if PRE_TRAIN:
        last_checkpoint = get_last_checkpoint_if_any(c.PRE_CHECKPOINT_FOLDER)
        if last_checkpoint is not None:
            logging.info('Found pre-training checkpoint [{}]. Resume from here...'.format(last_checkpoint))
            x = model.output
            x = Dense(len(unique_speakers), activation='softmax', name='softmax_layer')(x)
            pre_model = Model(model.input, x)
            pre_model.load_weights(last_checkpoint)
            grad_steps = int(last_checkpoint.split('_')[-2])
            logging.info('Successfully loaded pre-training model')

    else:
        last_checkpoint = get_last_checkpoint_if_any(c.CHECKPOINT_FOLDER)
        if last_checkpoint is not None:
            logging.info('Found checkpoint [{}]. Resume from here...'.format(last_checkpoint))
            model.load_weights(last_checkpoint)
            grad_steps = int(last_checkpoint.split('_')[-2])
            logging.info('[DONE]')
        if c.COMBINE_MODEL:
            last_checkpoint = get_last_checkpoint_if_any(c.GRU_CHECKPOINT_FOLDER)
            if last_checkpoint is not None:
                logging.info('Found checkpoint [{}]. Resume from here...'.format(last_checkpoint))
                gru_model.load_weights(last_checkpoint)
                logging.info('[DONE]')

    model.compile(optimizer='adam', loss=deep_speaker_loss)
    logging.info('Starting training...')
    lasteer = 10
    eer = 1
    while True:
        orig_time = time()
        # x, _ = select_batch.best_batch(model, batch_size=c.BATCH_SIZE)
        batch = stochastic_mini_batch(libri, batch_size=c.BATCH_SIZE, unique_speakers=unique_speakers)
        x, _ = batch.to_inputs()
        print("select_batch_time:", time() - orig_time)
        y = np.random.uniform(size=(x.shape[0], 1)) # 產生[0,1]之間的96個數
        # If "ValueError: Error when checking target: expected ln to have shape (None, 512) but got array with shape (96, 1)"
        # please modify line 121 to following line
        # y = np.random.uniform(size=(x.shape[0], 512))
        logging.info('== Presenting step #{0}'.format(grad_steps))
        orig_time = time()
        loss = model.train_on_batch(x, y)
        logging.info('== Processed in {0:.2f}s by the network, training loss = {1}.'.format(time() - orig_time, loss))
        
        with open(c.LOSS_LOG, "a") as f:
            f.write("{0},{1}\n".format(grad_steps, loss))
        if (grad_steps) % 10 == 0:
            fm1, tpr1, acc1, eer1 = eval_model(model, train_batch_size, test_dir=c.DATASET_DIR, check_partial=True, gru_model=gru_model)
            logging.info('test training data EER = {0:.3f}, F-measure = {1:.3f}, Accuracy = {2:.3f} '.format(eer1, fm1, acc1))
            with open(c.CHECKPOINT_FOLDER + '/train_acc_eer.txt', "a") as f:
                f.write("{0},{1},{2},{3}\n".format(grad_steps, eer1, fm1, acc1))

        if (grad_steps ) % c.TEST_PER_EPOCHS == 0 :
            fm, tpr, acc, eer = eval_model(model,train_batch_size, test_dir=c.TEST_DIR,gru_model=gru_model)
            logging.info('== Testing model after batch #{0}'.format(grad_steps))
            logging.info('EER = {0:.3f}, F-measure = {1:.3f}, Accuracy = {2:.3f} '.format(eer, fm, acc))
            with open(c.TEST_LOG, "a") as f:
                f.write("{0},{1},{2},{3}\n".format(grad_steps, eer, fm, acc))

        # checkpoints are really heavy so let's just keep the last one.
        if (grad_steps ) % c.SAVE_PER_EPOCHS == 0:
            create_dir_and_delete_content(c.CHECKPOINT_FOLDER)
            model.save_weights('{0}/model_{1}_{2:.5f}.h5'.format(c.CHECKPOINT_FOLDER, grad_steps, loss))
            if c.COMBINE_MODEL:
                gru_model.save_weights('{0}/grumodel_{1}_{2:.5f}.h5'.format(c.GRU_CHECKPOINT_FOLDER, grad_steps, loss1))
            if eer < lasteer:
                files = sorted(filter(lambda f: os.path.isfile(f) and f.endswith(".h5"),
                                      map(lambda f: os.path.join(c.BEST_CHECKPOINT_FOLDER, f), os.listdir(c.BEST_CHECKPOINT_FOLDER))),
                               key=lambda file: file.split('/')[-1].split('.')[-2], reverse=True)
                lasteer = eer
                for file in files[:-4]:
                    logging.info("removing old model: {}".format(file))
                    os.remove(file)
                model.save_weights(c.BEST_CHECKPOINT_FOLDER+'/best_model{0}_{1:.5f}.h5'.format(grad_steps, eer))
                if c.COMBINE_MODEL:
                    files = sorted(filter(lambda f: os.path.isfile(f) and f.endswith(".h5"),
                                          map(lambda f: os.path.join(c.BEST_CHECKPOINT_FOLDER, f),
                                              os.listdir(c.BEST_CHECKPOINT_FOLDER))),
                                   key=lambda file: file.split('/')[-1].split('.')[-2], reverse=True)
                    lasteer = eer
                    for file in files[:-4]:
                        logging.info("removing old model: {}".format(file))
                        os.remove(file)
                    gru_model.save_weights(c.BEST_CHECKPOINT_FOLDER+'/best_gru_model{0}_{1:.5f}.h5'.format(grad_steps, eer))

        grad_steps += 1



if __name__ == '__main__':
    logging.basicConfig(handlers=[logging.StreamHandler(stream=sys.stdout)], level=logging.INFO,
                        format='%(asctime)-15s [%(levelname)s] %(filename)s/%(funcName)s | %(message)s')
    main()
