#!/usr/bin/env python
# coding: utf-8

# In[17]:


import tensorflow as tf
#Digit_model = tf.keras.models.load_model("Model_Digit_Classification.model")


# In[18]:


mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_test = tf.keras.utils.normalize(x_test, axis=1)


# In[19]:


# load json and create model
json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = tf.keras.models.model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("model.h5")
print("Loaded model from disk")


# In[20]:


predictions = loaded_model.predict(x_test)


# In[21]:


predictions


# In[22]:


import matplotlib.pyplot as plt
plt.imshow(x_test[2000], cmap=plt.cm.binary)
plt.show


# In[23]:


#to see the predictions from array in numerical format
import numpy as np
np.argmax(predictions[2000])


# In[ ]:




