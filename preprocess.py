import matplotlib.pyplot as plt
from scipy.io import wavfile
import numpy as np
import shutil
import os

sec = 4
samplerate = 44100
time_length = sec * samplerate
input_size = (time_length, 2)
batch_size = 10

path = 'data'
x_folder = 'marimba'
y_folder = 'synthesizer'
process_path = "process_data"

shutil.rmtree(os.path.join(process_path))​

if not os.path.isdir(process_path):
    os.makedirs(process_path)
if not os.path.isdir(os.path.join(process_path, x_folder)):
    os.makedirs(os.path.join(process_path, x_folder))
if not os.path.isdir(os.path.join(process_path, y_folder)):
    os.makedirs(os.path.join(process_path, y_folder))

def is_file(index):
    return os.path.isfile(os.path.join(path, x_folder, str(index) + '.wav'))

def read_wav_data(index):
    x_samplerate, x_file = wavfile.read(os.path.join(path, x_folder, str(index)+'.wav'))
    y_samplerate, y_file = wavfile.read(os.path.join(path, y_folder, str(index)+'.wav'))
    
    assert(x_samplerate == samplerate and y_samplerate == samplerate)

    return np.array(x_file, dtype=float), np.array(y_file, dtype=float)

def segment_data(x_train, y_train):
    min_dataset = min(x_train.shape[0] // time_length, y_train.shape[0] // time_length)
    time = min_dataset * time_length
    
    x_train = x_train[:time, :].reshape(min_dataset, time_length, 2)
    y_train = y_train[:time, :].reshape(min_dataset, time_length, 2)
    
    return x_train, y_train

def count_data():
    return min(len(os.listdir(os.path.join(path, x_folder))), len(os.listdir(os.path.join(path, y_folder))))


for number in range(1, count_data() + 1):
	x, y = read_wav_data(number)
	x, y = segment_data(x, y)

	for index, i in enumerate(range(0, x.shape[0], batch_size)):
	    np.save(os.path.join(process_path, x_folder, str(index) + ".npy"), x[i:i+batch_size])
	    np.save(os.path.join(process_path, y_folder, str(index) + ".npy"), y[i:i+batch_size])

	del x, y
