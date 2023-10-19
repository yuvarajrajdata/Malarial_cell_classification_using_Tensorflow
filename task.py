import numpy as np
np.random.seed(1000)

import cv2
import os
from PIL import Image
import keras

os.environ['KERAS_BACKEND'] = 'tensorflow'


image_directory = 'C:/Users/user/Desktop/task/task 1/Malaria Cells/training_set'
SIZE = 64
dataset=[]
label = []

parasitized_images = os.listdir(image_directory + '/Parasitized')

for i, image_name in enumerate(parasitized_images):
    if (image_name.split('.')[1] == 'png'):
        image = cv2.imread(image_directory + '/Parasitized/' + image_name)
        image = Image.fromarray(image, 'RGB')
        image = image.resize((SIZE, SIZE))
        dataset.append(np.array(image))
        label.append(0)


Uninfected_images = os.listdir(image_directory + '/Uninfected')

for i, image_name in enumerate(Uninfected_images):
    if (image_name.split('.')[1] == 'png'):
        image = cv2.imread(image_directory + '/Uninfected/' + image_name)
        image = Image.fromarray(image, 'RGB')
        image = image.resize((SIZE, SIZE))
        dataset.append(np.array(image))
        label.append(1)



INPUT_SHAPE = (SIZE, SIZE, 3)
inp = keras.layers.Input(shape=INPUT_SHAPE)


conv1 = keras.layers.Conv2D(32, kernel_size=(3, 3), activation='relu',padding = 'same')(inp)
pool1 = keras.layers.MaxPooling2D(pool_size=(2, 2))(conv1)
norm1 = keras.layers.BatchNormalization(axis = -1)(pool1)
drop1 = keras.layers.Dropout(0.2)(norm1) 

conv2 = keras.layers.Conv2D(64, kernel_size=(3, 3), activation='relu',padding = 'same')(drop1)
pool2 = keras.layers.MaxPooling2D(pool_size=(2, 2))(conv2)
norm2 = keras.layers.BatchNormalization(axis = -1)(pool2)
drop2 = keras.layers.Dropout(0.2)(norm2)

flat = keras.layers.Flatten()(drop2)

hidden1 = keras.layers.Dense(512, activation='relu')(flat)
norm3 = keras.layers.BatchNormalization(axis = -1)(hidden1)
drop3 = keras.layers.Dropout(rate=0.2)(norm3)

hidden2 = keras.layers.Dense(256, activation='relu')(drop3)
norm4 = keras.layers.BatchNormalization(axis = -1)(hidden2)
drop4 = keras.layers.Dropout(rate=0.2)(norm4)

out = keras.layers.Dense(1, activation='sigmoid')(drop4)

model = keras.Model(inputs=inp, outputs=out)
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

print(model.summary())




from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
import numpy as np

# ### Training the model
# As the training data is now ready, I will use it to train the model.   

#Fit the model
history = model.fit(np.array(X_train), 
                         y_train, 
                         batch_size = 64, 
                         verbose = 1, 
                         epochs = 5,      #Changed to 3 from 50 for testing purposes.
                         validation_split = 0.1,
                         shuffle = False
                      #   callbacks=callbacks
                     )


test_accuracy = model.evaluate(np.array(X_test), y_test)[1] * 100
print("Test_Accuracy: {:.2f}%".format(test_accuracy))



# to save model:
model.save('C:/Users/user/Desktop/task/task 1/Malaria Cells/model_cnn.h5')




# for new_image prediction :
from keras.models import load_model

# Load your trained model
model = load_model('your_model.h5')  # Replace 'your_model.h5' with the path to your saved model file

# Load and prepare the new image
new_image_path = 'path_to_user_image.jpg'  # Replace with the path to the user's image
SIZE = 64  # The size your model expects

# Load the image
new_image = cv2.imread(new_image_path)
new_image = Image.fromarray(new_image, 'RGB')
new_image = new_image.resize((SIZE, SIZE))
new_image = np.array(new_image)

# Normalize and preprocess the image
new_image = new_image.astype('float32') / 255.0
new_image = np.expand_dims(new_image, axis=0)  # Add a batch dimension

# Make predictions
predictions = model.predict(new_image)

# Interpret the predictions
if predictions[0] < 0.5:
    print("Parasitized")
else:
    print("Uninfected")



