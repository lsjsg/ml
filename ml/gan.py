import os
import numpy as np
import matplotlib as pyplot
from keras.layers import Input
from keras.models import Model,Sequential
from keras.layers.core import Dense,Dropout
from keras.layers.advanced_activations import LeakyReLU,ReLU
from keras.datasets import mnist
from keras.optimizers import Adam
from keras import initializers
from keras.models import load_model
import h5py
from tqdm import tqdm
import cv2

os.environ["KERAS_BACKEND"] = "tensorflow"
np.random.seed(10)
ramdom_dim = 10

x_train,y_train,x_test,y_test = mnist.load_data()
x_train = (x_train.astype(np.float32)-127.5) / 127.5
x_train = x_train.reshape(60000,784)

adam = Adam(lr=0.0002,beta_1=0.5)

generator = Sequential()
generator.add(Dense(input=ramdom_dim,kernel_initializer=initializers.RandomNormal(stddev=0.02)))
generator.add(LeakyReLU(0.2))
generator.add(Dense(512))
generator.add(LeakyReLU(0.2))
generator.add(Dense(1024))
generator.add(LeakyReLU(0.2))
generator.add(Dense(784,activation="tanh"))
generator.compile(loss="binary_crossentropy",optimizer=adam)
generator.save("generator.h5")

discriminator = Sequential()
discriminator.add(Dense(input=784,kernel_initializer=initializers.RandomNormal(stddev=0.02)))
discriminator.add(LeakyReLU(0.2))
discriminator.add(Dropout(0.3))
discriminator.add(Dense(512))
discriminator.add(LeakyReLU(0.2))
discriminator.add(Dropout(0.3))
discriminator.add(Dense(256))
discriminator.add(LeakyReLU(0.2))
discriminator.add(Dropout(0.3))
discriminator.add(Dense(1,activation="sigmoid"))
discriminator.compile(loss="binary_crossentropy",optimizer=adam)
discriminator.save("discriminator.h5")

gan_input = Input(shape=(random_dim,))
x = generator(gan_input)
gan_output = discriminator(x)
gan = Model(input=gan_input,output=gan_output)
gan.compile(loss="binary_crossentropy",optimizer=adam)

def plot_generated_image(epoch,generator,examples=100,dim=(10,10),figsize=(10,10)):
    noise = np.random.normal(0,1,size=[examples,random_dim])
    generated_images = generator.predict(noise)
    generated_images = generated_images.reshape(examples,28,28)

    plt.figure(figsize=figsize)
    for i in range(generated_images.shape[0]):
        plt.subplot(dim[0],dim[1],i+1)
        plt.imshow(generated_images[i],interpolation='nearest',cmap='gray_r')
        plt.axis('off')
    plt.tight_layout()
    plt.savefig('gan_generated_image_epoch_%d.png' % epoch)

batch_count = x_train.shape[0]
for e in range(1,epoches + 1):
    print("epoch {0}".format(e))
    for _ in range(batch_count)
        noise = np.random.normal(0,1,size=[batch_count,ramdom_dim])
        image_batch = x_train[np.random.randint(0,x_train.shape[0],size=batch_size)]
        generated_images = generator.predict(noise)
        X = np.concatenate([image_batch,generated_images])

        y_dis = np.zeros(2*batch_size)
        y_dis[:batch_size]=0.9
        disciminator.train_on_batch
