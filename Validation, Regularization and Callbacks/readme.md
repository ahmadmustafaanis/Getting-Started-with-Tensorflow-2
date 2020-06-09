# Validation, Regularization, & Callbacks
----

### Validation sets 

Measures on how much our data is performing outside the training data, while training. 

Means this Validation data is never used for training the model but is used to evaluate while the model is being trained.

### Ways to add validation sets

#### <u> 1st Way</u>

First way is to pass `validation_split` argument in `model.fit()` to specify the validation split.

Using this way will automatically split data in training and validation sets. Lets see the code.

```python3

hist = model.fit(X_train, y_train, epochs=20, verbose=0, validation_split=0.2)

```

Here it automaticallly splits X_train and y_train into sperate training & validation sets with 80% training data and 20% validation data.

#### <u> 2nd Way </u>

Second way is to seperately pass validation set into `model.fit()` which we get from different resources, using `validation_data()` parameter.

Lets see the code.
```python3
from tensorflow.keras.datasets import mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Create and define model structure
# Compile the model using model.compile

model.fit(x_train,y_train, epochs=20, verbose=0, validation_data = (x_test,y_test))

```

#### <u> 3rd Way </u>

3rd way to split data set in training and validation set is by using `train_test_split` function from `sklearn.model_selection`.

Lets see it in code.

```python3

from sklearn.model_selection import train_test_split

X_train, X_val, y_train, y_val = train_test_split(trainData, trainLabels, test_size = 0.1)

model.fit(X_train,y_train, epochs=20, validation_data=(X_val,y_val))
```
----

## <u> Model Regularisation</u>

Now we will learn how to add regularisation terms in our model. Our main focus will be on L1 Regularization, L2 Regularization, and Dropout.

I wont teach about L1, L2, and dropout, but you can look for them in Google. 

Lets see how you add these in Keras.

#### L1 and L2
You can add L1 and L2 in any Dense or Conv Layer, and it will automatically be evaluated in Loss function. All you need to do is to add them in your Layer using `kernel_regularizer` for Weights and `bias_regularizer` for bias.

Lets check it in code.

```python3
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.models import Sequential

model = Sequential([
    Dense(64, activation = 'relu', kernel_regularizer = tf.keras.regularizers.l1(0.05)),
    Dense(64, activation = 'relu', kernel_regularizer = tf.keras.regularizers.l2(0.01)),
    Dense(64,activation='relu', tf.keras.regularizers.l1_l2(
    l1=0.01, l2=0.01)),
    Dropout(0.5),
    Dense(1,activation = 'sigmoid')
])

```

L1 is computed as 

<img src=0.png></img>

and L2 is computed as

<img src=1.png></img>

Here we compute L1, L2, and both L1_L2 Regularization using Tf.Keras.

We also added Dropout Layer which randomly drops 50% neurons. It is also known as Bernouli's Drop out.

Remember that we are not going to Drop Out randomly while in `model.evaluate()` and in `model.predict()`. It is only used in `model.fit()` phase.

---

## Batch Normalization

Learn more about Batch Normalization in [this](Batch%20normalisation.ipynb) notebook.

----

## <u> CallBacks </u>

Call backs are certian types of Objects that can monitor loss & metrics at certain points in training and can perform certain actions based on them.

These are Call backs used in training, there are other several types of Callbacks which can monitor different things and can perform actions based on them.

```python3
from tensorflow.keras.callbacks import Callback
```

There are 2 ways to use Callbacks in Keras.
- Callbacks Base class(which we just imported) thorugh which we make our own subclass
- Built-in Call backs

Let's Create our own baseclass first.

```python3
from tensorflow.keras.callbacks import Callback

class My_Callback(Callback):
    #we will rewrite built-in methods

    def on_train_begin(self, logs=None):
        #do something at start of training
    
    def on_train_batch_begin(self, batch, logs=None):
        # do somethings at start of batch

    def on_epochs_end(self, epochs, log= None):
        #do something at end of epoch


history = model.fit(X_train, y_train, epochs=5, callbacks=[My_Callback()])

```

Here `history` is also an example of callback whose job is to store loss and metrics in dictonary format in it's history attribute i.e it stores loss and metrics in history.history.

For details Refer to Notebook of this week.