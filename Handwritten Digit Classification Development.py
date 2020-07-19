#!/usr/bin/env python
# coding: utf-8

# In[37]:


import tensorflow as tf


# In[38]:


import matplotlib.pyplot as plt


# In[39]:


mnist = tf.keras.datasets.mnist


# In[40]:


(x_train, y_train), (x_test, y_test) = mnist.load_data()


# In[41]:


plt.imshow(x_train[0], cmap=plt.cm.binary)
plt.show


# In[42]:


x_train


# In[43]:


x_train[1]


# In[44]:


#Normalization axis=1: columns
x_train = tf.keras.utils.normalize(x_train, axis=1)
x_test = tf.keras.utils.normalize(x_test, axis=1)


# In[45]:


x_train[1]


# In[46]:


y_train


# In[47]:


x_test


# In[48]:


# we are building a Feed Forward sequential model
model = tf.keras.models.Sequential()

#takes 28*28 image and makes it 1*784 - Flattening the input (input has 784)
model.add(tf.keras.layers.Flatten(  )) 

#Adding a Dense Layer with 128 neurons and using ReLU as the activation functions
model.add(tf.keras.layers.Dense(128, activation = tf.nn.relu))

#Again adding a Dense Layer with 128 neurons and ReLU as the activation functions
model.add(tf.keras.layers.Dense(128, activation = tf.nn.relu))

#Output layer
model.add(tf.keras.layers.Dense(10, activation = tf.nn.softmax))


# In[49]:


#Will perform the backpropogation using the Adam optimizer and
#choosing the sparse_categorical_crossentorpy as the Loss function
model.compile(optimizer = 'adam', 
              loss = 'sparse_categorical_crossentropy',
              metrics = ['accuracy'])

#Now we will train the model 
model.fit(x_train, y_train, epochs=10)


# In[50]:


#Testing the model
val_loss, val_acc = model.evaluate(x_test,y_test)


# In[51]:


print("Test Loss:", val_loss, )
print("Test Accuracy: ", round(val_acc*100,2), "%")


# In[52]:


#saving the model
model.save("‪Model_Digit_Classification.model")


# In[53]:


#Loading the model
new_model = tf.keras.models.load_model("Model_Digit_Classification.model")


# # Saving the model as JSON file and weights as h5 file

# In[60]:


from keras.models import model_from_json   
# serialize model to JSON
model_json = model.to_json()
with open("Digit_Model.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("Digit_model_weights.h5")
print("Saved model to disk (Users)")


# # Loading the model from JSON file and weights from h5 file

# In[61]:


# load json and create model
json_file = open('Digit_Model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = tf.keras.models.model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("Digit_model_weights.h5")
print("Loaded model from disk")


# # The complete is model is developed and saved now we could load the model again from our pc and then predict the digit from the image, but the image must be 28x28 pixles and grayscale (We used MNIST dataset to train the model)

# In[56]:


predictions = loaded_model.predict(x_test)


# In[57]:


predictions


# In[58]:


plt.imshow(x_test[2000], cmap=plt.cm.binary)
plt.show


# In[59]:


#to see the predictions from array in numerical format
import numpy as np
np.argmax(predictions[2000])

