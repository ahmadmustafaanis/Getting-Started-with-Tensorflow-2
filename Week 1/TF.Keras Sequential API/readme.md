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