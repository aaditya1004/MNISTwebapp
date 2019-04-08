from keras import layers
from keras import models
from keras.datasets import mnist 
from keras.utils import to_categorical
(train_images,train_labels),(test_images,test_labels) = mnist.load_data()

network = models.Sequential()
network.add(layers.Conv2D(32,(3,3),activation = 'relu',input_shape=(28,28,1))) 
network.add(layers.MaxPooling2D((2,2)))

network.add(layers.Conv2D(64,(3,3),activation = 'relu'))
network.add(layers.MaxPooling2D((2,2)))
network.add(layers.Conv2D(64,(3,3),activation = 'relu'))
network.add(layers.Flatten())
network.add(layers.Dense(64,activation = 'relu'))
network.add(layers.Dense(10,activation = 'softmax'))
print(network.summary())


train_images = train_images.reshape((60000,28,28,1))
train_images = train_images.astype('float32')/255

test_images = test_images.reshape((10000,28,28,1))
test_images = test_images.astype('float32')/255

train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)

network.compile(optimizer='rmsprop',loss='categorical_crossentropy',metrics=['accuracy'])
network.fit(train_images,train_labels,epochs=5,batch_size=64)

model_json = network.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
network.save_weights("model.h5")
print("Saved model to disk")
