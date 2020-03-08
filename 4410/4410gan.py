import matplotlib.pyplot as plt
from scipy.io import wavfile
import numpy as np
import os

from tensorflow.keras import models, layers, losses, optimizers
import tensorflow as tf
print(tf.__version__)

path = 'process_data'
x_folder = 'marimba'
y_folder = 'synthesizer'

sec = 4
samplerate = 4410
time_length = sec * samplerate
input_size = (time_length, 2)

epochs = 100
batch_size = 10
sample_count = 3
g_lr, d_lr, gan_lr = [1e-4, 1e-4, 1e-4]

sample_save_path = 'test'
model_save_path = 'model'
visual_loss_save_path = "visual_loss"

if not os.path.isdir(sample_save_path):
    os.makedirs(sample_save_path)

if not os.path.isdir(model_save_path):
    os.makedirs(model_save_path)
    os.makedirs(os.path.join(model_save_path, 'generator'))
    os.makedirs(os.path.join(model_save_path, 'discriminator'))
    os.makedirs(os.path.join(model_save_path, 'gan'))


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
    inputs = layers.Input(shape=input_size)

    # down sampling
    out = layers.Conv1D(filters=64, kernel_size=441, strides=1, padding='same', activation='tanh')(inputs)
    out = layers.Conv1D(filters=128, kernel_size=21, strides=1, padding='same', activation='tanh')(out)
    out = layers.Conv1D(filters=64, kernel_size=5, strides=1, padding='same', activation='tanh')(out)
    out = layers.BatchNormalization()(out)
    
    # up sampling
    out = layers.Conv1D(filters=128, kernel_size=5, strides=1, padding='same', activation='tanh')(out)
    out = layers.BatchNormalization()(out)
    out = layers.Conv1D(filters=128, kernel_size=21, strides=1, padding='same', activation='tanh')(out)
    out = layers.BatchNormalization()(out)
    out = layers.Conv1D(filters=2, kernel_size=441, strides=1, padding='same', activation='tanh')(out)
    out = layers.BatchNormalization()(out)

    model = models.Model(inputs, out)
    model.compile(optimizer=optimizers.Adam(lr), loss=losses.binary_crossentropy, metrics=['binary_crossentropy'])
    
    return model

def discriminator(lr=1e-3):
    inputs = layers.Input(shape=input_size)
    
    out = layers.Conv1D(filters=64, kernel_size=2, strides=1, padding='same', activation='relu')(inputs)
    out = layers.Conv1D(filters=128, kernel_size=21, strides=1, padding='same', activation='relu')(out)
    out = layers.BatchNormalization()(out)    
    out = layers.Conv1D(filters=128, kernel_size=5, strides=1, padding='same', activation='relu')(out)
    out = layers.BatchNormalization()(out)
    out = layers.Conv1D(filters=64, kernel_size=1, strides=1, padding='same', activation='relu')(out)

    out = layers.Flatten()(out)
    #out = layers.Dense(64, activation='relu')(out)
    out = layers.Dense(1, activation='sigmoid')(out)
    
    model = models.Model(inputs, out)
    model.compile(optimizer=optimizers.Adam(lr), loss=losses.binary_crossentropy, metrics=['binary_crossentropy'])
    
    return model

def Gan(discriminator, generator, lr=1e-3):
    discriminator.trainable=False
    
    inputs = layers.Input(shape=input_size)
    x = generator(inputs)
    out = discriminator(x)
    
    gan = models.Model(inputs, out)
    gan.compile(optimizer=optimizers.Adam(lr=lr), loss=losses.binary_crossentropy)
    return gan

generator = Generator(lr=1e-5)
discriminator = discriminator(lr=1e-5)
gan = Gan(discriminator, generator, lr=1e-5)

gan.summary()

gan_losses = list()
for e in range(1, epochs + 1):
    batch_loss = 0

    for index in range(count_data()):
        print("Processing..",index,"/", count_data())
        x, y = read_data(index)
        batch_size = x.shape[0]
        #print(0)
        generated_synth = generator.predict(x)
        synth_batch =y[np.random.randint(low=0,high=x.shape[0],size=batch_size), :, :]
        X = np.concatenate([synth_batch, generated_synth])
        #print(1)
        y_dis=np.zeros(2 * batch_size)
        y_dis[:batch_size] = 0.9
        #print(2)
        discriminator.trainable = True
        discriminator.train_on_batch(X, y_dis)
        #print(3)
        y_gen = np.ones(batch_size)
        #print(4)
        discriminator.trainable = False
        loss = gan.train_on_batch(x, y_gen)
        #print(5)
        batch_loss += loss
        if index % 20 == 0:
            print(f"Batch:{index} loss:{loss}")

    batch_loss /= batch_size
    print(f"Epoch {e}/{epochs} loss:{batch_loss}")
    
    gan_losses.append(batch_loss)
    generator.save(os.path.join(model_save_path, str(e)+'generator.h5'))
    discriminator.save(os.path.join(model_save_path, str(e)+'discriminator.h5'))
    gan.save(os.path.join(model_save_path, str(e)+'gan.h5'))

    generator.save_weights(os.path.join(model_save_path, str(e)+'generator'))
    discriminator.save_weights(os.path.join(model_save_path, str(e)+'discriminator'))
    gan.save_weights(os.path.join(model_save_path, str(e)+'gan'))

plt.figure()
plt.title("train loss per epoch")
plt.plot(list(range(len(gan_losses))), gan_losses)
plt.savefig(os.path.join(visual_loss_save_path, "train_loss.png"))
plt.close()

sample = np.random.choice(range(count_data()), 1, replace=False, p=None)
print(sample)
test, _ = read_data(int(sample))

generated_synth = generator.predict(test)
generated_synth = generated_synth.reshape(176400,2)
print(generated_synth)
for number in sample:
    print(number)
    wavfile.write(os.path.join(sample_save_path, str(number) + ".wav"), samplerate, generated_synth)
