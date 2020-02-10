import matplotlib.pyplot as plt
from scipy.io import wavfile
import numpy as np
import os

from tensorflow.keras import models, layers, losses, optimizers
import tensorflow as tf


path = 'process_data'
x_folder = 'marimba'
y_folder = 'synthesizer';

sec = 4
samplerate = 44100
time_length = sec * samplerate
input_size = (time_length, 2)

epochs = 1000
batch_size = 5
sample_count = 3
sample_save_path = 'test'
model_save_path = 'model'
visual_loss_save_path = "visual_loss"

if not os.path.isdir(sample_save_path):
    os.makedirs(sample_save_path)

if not os.path.isdir(model_save_path):
    os.makedirs(model_save_path)

if not os.path.isdir(visual_loss_save_path):
    os.makedirs(visual_loss_save_path)

def is_file(index):
    return os.path.isfile(os.path.join(path, x_folder, str(index) + '.wav'))

def read_data(index):
    x = np.load(os.path.join(path, x_folder, str(index) + '.npy'))
    y = np.load(os.path.join(path, y_folder, str(index) + '.npy'))
    
    return x, y

def count_data():
    return min(len(os.listdir(os.path.join(path, x_folder))), len(os.listdir(os.path.join(path, y_folder))))

def Generator(lr=1e-3):
    with tf.device('/device:GPU:0'):
        inputs = layers.Input(shape=input_size)

        # down sampling
        out = layers.Conv1D(filters=128, kernel_size=15, strides=1, padding='same', activation='relu')(inputs)
        out = layers.Conv1D(filters=256, kernel_size=5, strides=1, padding='same', activation='relu')(out)
        out = layers.BatchNormalization()(out)
        out = layers.Conv1D(filters=512, kernel_size=5, strides=1, padding='same', activation='relu')(out)
        out = layers.BatchNormalization()(out)
        out = layers.Conv1D(filters=1024, kernel_size=3, strides=1, padding='same', activation='relu')(out)
        out = layers.BatchNormalization()(out)
        out = layers.Conv1D(filters=512, kernel_size=3, strides=1, padding='same', activation='relu')(out)
        out = layers.BatchNormalization()(out)

        # repeat 4
        out = layers.Conv1D(filters=1024, kernel_size=3, strides=1, padding='same', activation='relu')(out)
        out = layers.BatchNormalization()(out)
        out = layers.Conv1D(filters=512, kernel_size=3, strides=1, padding='same', activation='relu')(out)
        out = layers.BatchNormalization()(out)
        out = layers.Conv1D(filters=1024, kernel_size=3, strides=1, padding='same', activation='relu')(out)
        out = layers.BatchNormalization()(out)
        out = layers.Conv1D(filters=512, kernel_size=3, strides=1, padding='same', activation='relu')(out)
        out = layers.BatchNormalization()(out)
        out = layers.Conv1D(filters=1024, kernel_size=3, strides=1, padding='same', activation='relu')(out)
        out = layers.BatchNormalization()(out)
        out = layers.Conv1D(filters=512, kernel_size=3, strides=1, padding='same', activation='relu')(out)
        out = layers.BatchNormalization()(out)
        out = layers.Conv1D(filters=1024, kernel_size=3, strides=1, padding='same', activation='relu')(out)
        out = layers.BatchNormalization()(out)
        out = layers.Conv1D(filters=512, kernel_size=3, strides=1, padding='same', activation='relu')(out)
        out = layers.BatchNormalization()(out)
        
        # up sampling
        out = layers.Conv1D(filters=1024, kernel_size=5, strides=1, padding='same', activation='relu')(out)
        out = layers.BatchNormalization()(out)
        out = layers.Conv1D(filters=512, kernel_size=5, strides=1, padding='same', activation='relu')(out)
        out = layers.BatchNormalization()(out)
        out = layers.Conv1D(filters=2, kernel_size=15, strides=1, padding='same', activation='relu')(out)
        out = layers.BatchNormalization()(out)

        model = models.Model(inputs, out)
        model.compile(optimizer=optimizers.Adam(lr), loss=losses.binary_crossentropy, metrics=['binary_crossentropy'])
        
        return model

def discriminator(lr=1e-3):
    with tf.device('/device:GPU:0'):

        inputs = layers.Input(shape=input_size)
        
        out = layers.Conv1D(filters=128, kernel_size=2, strides=1, padding='same', activation='relu')(inputs)
        out = layers.Conv1D(filters=256, kernel_size=3, strides=2, padding='same', activation='relu')(out)
        out = layers.BatchNormalization()(out)
        out = layers.Conv1D(filters=512, kernel_size=3, strides=2, padding='same', activation='relu')(out)
        out = layers.BatchNormalization()(out)
        out = layers.Conv1D(filters=1024, kernel_size=3, strides=2, padding='same', activation='relu')(out)
        out = layers.BatchNormalization()(out)
        out = layers.Conv1D(filters=1024, kernel_size=5, strides=1, padding='same', activation='relu')(out)
        out = layers.BatchNormalization()(out)
        out = layers.Conv1D(filters=128, kernel_size=3, strides=1, padding='same', activation='relu')(out)
        out = layers.BatchNormalization()(out)
        out = layers.Conv1D(filters=64, kernel_size=1, strides=1, padding='same', activation='relu')(out)

        out = layers.Flatten()(out)
        out = layers.Dense(1024, activation='relu')(out)
        out = layers.Dense(1, activation='sigmoid')(out)
        
        model = models.Model(inputs, out)
        model.compile(optimizer=optimizers.Adam(lr), loss=losses.binary_crossentropy, metrics=['binary_crossentropy'])
        
        return model

def Gan(discriminator, generator, lr=1e-3):
    with tf.device('/device:GPU:0'):

        discriminator.trainable=False
        
        inputs = layers.Input(shape=input_size)
        x = generator(inputs)
        out = discriminator(x)
        
        gan = models.Model(inputs, out)
        gan.compile(optimizer=optimizers.Adam(lr=lr), loss=losses.binary_crossentropy)
        
        return gan

with tf.device('/device:GPU:0'):
    generator = Generator(lr=1e-5)
    discriminator = discriminator(lr=1e-5)
    gan = Gan(discriminator, generator, lr=1e-5)

    gan_losses = list()
    for e in range(1, epochs + 1):
        batch_loss = 0

        for index in range(count_data()):
            x, y = read_data(index)
            batch_size = x.shape[0]

            generated_synth = generator.predict(x)
            synth_batch =y[np.random.randint(low=0,high=x.shape[0],size=batch_size), :, :]
            X = np.concatenate([synth_batch, generated_synth])

            y_dis=np.zeros(2 * batch_size)
            y_dis[:batch_size] = 0.9

            discriminator.trainable = True
            discriminator.train_on_batch(X, y_dis)

            y_gen = np.ones(batch_size)

            discriminator.trainable = False
            loss = gan.train_on_batch(x, y_gen)

            batch_loss += loss
            if index % 20 == 0:
                print(f"Batch:{index} loss:{loss}")

        batch_loss /= batch_count
        print(f"Epoch {e}/{epochs} loss:{batch_loss}")
        gan_losses.append(batch_loss)

# loss 시각화 저장
plt.figure()
plt.title("train loss per epoch")
plt.plot(list(range(len(gan_losses))), gan_losses)
plt.savefig(os.path.join(visual_loss_save_path, str(i) + ".png"))
plt.close()

# train 중 샘플 저장
sample = np.random.choice(range(count_data()), 1, replace=False, p=None)
test, _ = read_data(sample)

generated_synth = generator.predict(test)

for number in sample:
    wavfile.write(os.path.join(sample_save_path, str(i) + "-" + str(number) + ".wav"), samplerate, generated_synth[i])

# model 저장
generator.save(os.path.join(model_save_path, 'generator.h5'))
discriminator.save(os.path.join(model_save_path, 'discriminator.h5'))
gan.save(os.path.join(model_save_path, 'gan.h5'))