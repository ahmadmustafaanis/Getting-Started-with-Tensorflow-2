# tensorflow.keras Sequential API
---
99% of models can be built using Sequential API

---
### Imports
```python3
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
```

### Creating a model
```python3
model = Sequential([
    Dense(64,activation = 'relu'),
    Dense(10,activation = 'softmax')
])
```

This will create a model with 64 Hidden Units, and 10 Output Units. We have not defined Input units yet. We can define number of input in either training phase or in first layer. 

```python3
model = Sequential([
    Dense(64,activation = 'relu', input_shape=(784,)),
    Dense(10,activation = 'softmax')
])
```

Here we giving 784 d vector. If our input is 2d, we have to flatten it first.

```python3
from tensorflow.keras.layers import Flatten, Dense
from tensorflow.keras.models import Sequential
model = Sequential([
    Flatten(input_shape=(28,28)),
    Dense(64,activation = 'relu'),
    Dense(10,activation = 'softmax')
])
```

We can also use model.add to add layers instead of passing them when creating as instance.

```python3
model = Sequential()
model.add(Dense(64,activation = 'relu', input_shape=(784,)))
model.add(Dense(10,activation = 'softmax'))
```

### Convolutinal Layers with tf.keras
- There are 2 main things in CNN
    - Pooling Layers
    - Convolutional Layers

- Lets see a coding example

```python3
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Conv2D, MaxPool2D

model = Sequential()
model.add(Conv2D(16, kernel_size=3, activation = 'relu', input_shape=(32,32,3)))
model.add(MaxPool2D((3,3)))
model.add(Flatten())
model.add(Dense(64,activation='relu'))
model.add(Dense(10,activation='softmax'))
```

To see the details of the models, use `model.summary()`.


```python3
print(model.summary())
```

```Model: "sequential"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
conv2d (Conv2D)              (None, 30, 30, 16)        448       
_________________________________________________________________
max_pooling2d (MaxPooling2D) (None, 10, 10, 16)        0         
_________________________________________________________________
flatten (Flatten)            (None, 1600)              0         
_________________________________________________________________
dense (Dense)                (None, 64)                102464    
_________________________________________________________________
dense_1 (Dense)              (None, 10)                650       
=================================================================
Total params: 103,562
Trainable params: 103,562
Non-trainable params: 0
_________________________________________________________________
None
```

We can add Padding, & Strides in our Conv2D layer.

```python3
model = Sequential([
    Conv2D(16,(3,3),strides = (2,2) ,padding='same', input_shape=(28,28,1)),
    MaxPooling2D((3,3))
])
```
Here we have `same` padding and strides of 2 by 2.

We can change shape of input from (x,y,no.of channels) to (no.ofchannels, x, y) by simply adding data_format in MaxPool2D and Conv2D.

```python3
model = Sequential([
    Conv2D(16,(3,3),padding='same', input_shape=(1,28,28),data_format='channels_first'),
    MaxPooling2D((3,3), data_format='channels_first')
])
```
----

### Custom Initialziation of Weights and Biases.
Yes it is possible to initializae custom weights and Biases in Tensorflow with ease. Refer to [this](TF%20Keras%20Week%201%20Tutorial.ipynb) notebook.

----
### Compiling of Model
When we hace structure of a model ready, we can call compile method on model to assosiate it with Loss functions, Optimizer, Metrics etc.

```python3
model = Sequential([
    Dense(64,activation='relu', input_shape = (32,)),
    Dense(1, activation = 'sigmoid')
])

model.compile(
    loss='sgd', #sthocastic gradient descent
    optimizer = 'adam',
    metrics=['accuracy','mse'] #mse=mean squared error
)
```
Now there is another better way to do this, which is instead of passing things as a string like 'adam', we use thier objects given by tf.keras. It allows us to add more options. 
```python3
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

model = Sequential([
    Dense(64,activation='relu', input_shape = (32,)),
    Dense(1, activation = 'sigmoid')
])

model.compile(
    optimizer=tf.keras.optimizers.SGD(),
    loss=tf.keras.Binary_CrossEntropy(),
    metrics = [tf.keras.metrics.BinaryAccuaracy(), tf.keras.metrics.MeanAbsoluteError()]
)
```

Now Why we use these? Because these objects have thier own several parameters, which we can pass in it.
i.e

- tf.keras.optimizers.SGD(learning_rate=0.001,momentum=0.9)
- tf.keras.losses.Binary_CrossEntropy(form_logit=True)
    - This thing form_logit is used with linear activation function to convert it into sigmoid but is more computationally better then sigmoid.

Simiarly there are several other parameters which you can pass. You can explore them in Documentation.  

To learn more about Metrics, refer to this notebook [this](Metrics.ipynb)

-----

## Training of Model
We can train model using fit method in keras. We pass training set and training labels in it. We also specify several hyper parametrs such as Number of Epochs, Batch Size. Lets look at an example.
```python3
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

model = Sequential([
    Dense(64,activation='relu', input_shape = (32,)),
    Dense(1, activation = 'sigmoid')
])

model.compile(
    optimizer=tf.keras.optimizers.SGD(),
    loss=tf.keras.Binary_CrossEntropy(),
    metrics = [tf.keras.metrics.BinaryAccuaracy(), tf.keras.metrics.MeanAbsoluteError()]
)

history = model.fit(X_train,y_train,epochs=10, batch_size=256)
```

This will train the model for 10 epochs with a batch size of 256. <br> We store result in History variable which we can use to analyze the performance of model based on metrics and loss.


We normally convert history into a dataframe with all the analytics of model. And then visualize them. Lets see.
```python3
import pandas as pd
import matplotlib.pyplot as plt

df = pd.DataFrame(history.history)
plt.plot(df['loss'])
plt.xlabel('epoch')
plt.ylabel('loss')
plt.show()
#same for other metrics
```

----

## Evaluation and Prediction
We can Evaluate model on Test set and predict on new examples. In order to evaluate, you might have guessed, we can use model.evaluate function in keras.
Lets see an example
```python3
loss, test_binary_acc, test_MAE = model.evaluate(X_test,y_test)
```
Since we used Loss function, Binary acc, and MAE as loss and metrics respectively, so it returns 3 values. 
<br>
For Predictions, we can use model.predict on new image. And use `np.argmax` to predict the label.
Let's see and example
```python3
pred = model.predict(NewImage.png)
print(labels[np.argmax(pred)])
```
Where labels is a list containing all the labels.

-----
This winds up the tutorial for Tensorflow Keras Sequential API. Now it;s your turn to try it on MNIST and FASHION MNIST data sets.