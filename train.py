from download import download_mir1k
from preprocess import get_random_wav_batch, load_wavs, wavs_to_specs, sample_data_batch, sperate_magnitude_phase
from model import SingingVoiceSeparationModel
import os
import librosa
import numpy as np
import random
import shutil

def train(random_seed = 0):

    np.random.seed(random_seed)

    
    mir1k_dir = 'C:/Users/mishr/OneDrive/Desktop/BigData/abc/Singing-Voice-Separation-RNN/MIR1K'

    
    
    
    wavs_dir = os.path.join(mir1k_dir, 'Wavfile')  

    # List all .wav files in the directory
    all_wav_files = [f for f in os.listdir(wavs_dir) if f.endswith('.wav')]


    # Shuffle and split files into train and validation sets (80-20 split)
    random.shuffle(all_wav_files)
    train_files = all_wav_files[:int(0.8 * len(all_wav_files))]
    valid_files = all_wav_files[int(0.8 * len(all_wav_files)):]

    # Create train.txt and valid.txt
    with open(os.path.join(mir1k_dir, 'train.txt'), 'w') as train_file:
        for file in train_files:
            train_file.write(f'{file}\n')

    with open(os.path.join(mir1k_dir, 'valid.txt'), 'w') as valid_file:
        for file in valid_files:
            valid_file.write(f'{file}\n')

    # Preprocess parameters
    mir1k_sr = 16000
    n_fft = 1024
    hop_length = n_fft // 4

    # Model parameters
    learning_rate = 0.001
    num_rnn_layer = 3
    num_hidden_units = 256
    batch_size = 64
    sample_frames = 10
    iterations = 50
    tensorboard_directory = './graphs/svsrnn'
    log_directory = './log'
    train_log_filename = 'train_log.csv'
    clear_tensorboard = False
    model_directory = './model'
    model_filename = 'svsrnn'

    if not os.path.exists(log_directory):
        os.makedirs(log_directory)
    open(os.path.join(log_directory, train_log_filename), 'w').close()

    # Load train wavs
    wav_filenames_train = [os.path.join(wavs_dir, file) for file in train_files]
    wav_filenames_valid = [os.path.join(wavs_dir, file) for file in valid_files]

    wavs_mono_train, wavs_src1_train, wavs_src2_train = load_wavs(filenames = wav_filenames_train, sr = mir1k_sr)

    # Turn waves to spectrums
    stfts_mono_train, stfts_src1_train, stfts_src2_train = wavs_to_specs(
        wavs_mono = wavs_mono_train, wavs_src1 = wavs_src1_train, wavs_src2 = wavs_src2_train, n_fft = n_fft, hop_length = hop_length)

    wavs_mono_valid, wavs_src1_valid, wavs_src2_valid = load_wavs(filenames = wav_filenames_valid, sr = mir1k_sr)
    stfts_mono_valid, stfts_src1_valid, stfts_src2_valid = wavs_to_specs(
        wavs_mono = wavs_mono_valid, wavs_src1 = wavs_src1_valid, wavs_src2 = wavs_src2_valid, n_fft = n_fft, hop_length = hop_length)

    # Initialize model
    model =  SingingVoiceSeparationModel(num_features = n_fft // 2 + 1, num_rnn_layer = num_rnn_layer, num_hidden_units = num_hidden_units, tensorboard_directory = tensorboard_directory, clear_tensorboard = clear_tensorboard)

    # Start training
    for i in (range(iterations)):
        
        data_mono_batch, data_src1_batch, data_src2_batch = sample_data_batch(
            stfts_mono = stfts_mono_train, stfts_src1 = stfts_src1_train, stfts_src2 = stfts_src2_train, batch_size = batch_size, sample_frames = sample_frames)
        x_mixed, _ = sperate_magnitude_phase(data = data_mono_batch)
        y1, _ = sperate_magnitude_phase(data = data_src1_batch)
        y2, _ = sperate_magnitude_phase(data = data_src2_batch)

        train_loss = model.train_step(x_mixed = x_mixed, y_src1 = y1, y_src2 = y2)


        if i % 10 == 0:
            print('Step: %d Train Loss: %f' %(i, train_loss))

        if i % 200 == 0:
            print('==============================================')
            data_mono_batch, data_src1_batch, data_src2_batch = sample_data_batch(
                stfts_mono = stfts_mono_valid, stfts_src1 = stfts_src1_valid, stfts_src2 = stfts_src2_valid, batch_size = batch_size, sample_frames = sample_frames)
            x_mixed, _ = sperate_magnitude_phase(data = data_mono_batch)
            y1, _ = sperate_magnitude_phase(data = data_src1_batch)
            y2, _ = sperate_magnitude_phase(data = data_src2_batch)

            y_pred, validation_loss = model.validate(x_mixed = x_mixed, y_src1 = y1, y_src2 = y2)


            print('Step: %d Validation Loss: %f' %(i, validation_loss))
            print('==============================================')

            with open(os.path.join(log_directory, train_log_filename), 'a') as log_file:
                log_file.write('{},{},{}\n'.format(i, train_loss, validation_loss))

        if i % 1000 == 0:
            model.save(directory = model_directory, filename = model_filename+".keras")


if __name__ == '__main__':
    
    train()
