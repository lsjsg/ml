import os
import numpy as np
import matplotlib.pyplot as plt
from keras.layers import Input
from keras.models import Model,Sequential
from keras.layers.core import Dense,Dropout
from keras.layers.advanced_activations import LeakyReLU,ReLU
from keras.datasets import mnist
from keras.optimizers import Adam
from keras import initializers
from keras.models import load_model
import h5py
# from tqdm import tqdm

# let keras know we are using backend engine
os.environ["KERAS_BACKEND"] = "tensorflow"
np.random.seed(10)
random_dim=100

def load_minst_data():
    (x_train,y_train),(x_test,y_test) = mnist.load_data()
    x_train = (x_train.astype(np.float32)-127.5) / 127.5
    x_train = x_train.reshape(60000,784)
    return (x_train,y_train,x_test,y_test)

def get_optimizer():
    return Adam(lr=0.0002,beta_1=0.5)

def get_generator(optimizer):
    generator = Sequential()
    generator.add(Dense(units=256,input_dim=random_dim,kernel_initializer=initializers.RandomNormal(stddev=0.02)))
    generator.add(LeakyReLU(0.2))

    generator.add(Dense(512))
    generator.add(LeakyReLU(0.2))

    generator.add(Dense(1024))
    generator.add(LeakyReLU(0.2))

    generator.add(Dense(784,activation='tanh'))
    generator.compile(loss='binary_crossentropy',optimizer=optimizer)

    generator.save('generator_model.h5')
    return generator

def get_discriminator(optimizer):
    discrimintor = Sequential()
    discrimintor.add(Dense(units=1024,input_dim=784,kernel_initializer=initializers.RandomNormal(stddev=0.02)))
    discrimintor.add(LeakyReLU(0.2))
    discrimintor.add(Dropout(0.3))

    discrimintor.add(Dense(512))
    discrimintor.add(LeakyReLU(0.2))
    discrimintor.add(Dropout(0.3))

    discrimintor.add(Dense(256))
    discrimintor.add(LeakyReLU(0.2))
    discrimintor.add(Dropout(0.3))

    discrimintor.add(Dense(1,activation="sigmoid"))
    discrimintor.compile(loss="binary_crossentropy",optimizer=optimizer)
    discrimintor.save('discrimintor_model.h5')
    return discrimintor

def get_gan_network(discriminator,random_dim,generator,optimizer):
    discriminator.trainable=False
    gan_input = Input(shape=(random_dim,))
    x = generator(gan_input)
    gan_output = discriminator(x)
    gan = Model(input=gan_input,output=gan_output)
    gan.compile(loss='binary_crossentropy',optimizer=optimizer)
    return gan

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

def train(epochs=1,batch_size=128):
    x_train,y_train,x_test,y_test = load_minst_data()
    batch_count = x_train.shape[0] // batch_size + 1
    adam = get_optimizer()
    # adam = Adam(lr=0.0002,beta_1=0.5)
    generator = get_generator(adam)
    discriminator = get_discriminator(adam)
    gan = get_gan_network(discriminator,random_dim,generator,adam)

    for e in range(1,epochs+1):
        print('-'*15,'Epoch %d' %e, '-'*15)
        for _ in range(batch_count):
            noise = np.random.normal(0,1,size=[batch_size,random_dim])
            image_batch = x_train[np.random.randint(0,x_train.shape[0],size=batch_size)]

            generated_images = generator.predict(noise)
            X = np.concatenate([image_batch,generated_images])

            y_dis = np.zeros(2*batch_size)
            y_dis[:batch_size] = 0.9

            discriminator.trainable = True
            discriminator.train_on_batch(X,y_dis)

            noise = np.random.normal(0,1,size=[batch_size,random_dim])
            y_gen = np.ones(batch_size)
            discriminator.trainable = False
            gan.train_on_batch(noise,y_gen)

        if e==1 or e%400==0:
            plot_generated_image(e,generator)
    generator.save('generator_model.h5')
    discriminator.save('discriminator_model.h5')
    gan.save("gan_model.h5")
if __name__=="__main__":
    train(30000,128)

