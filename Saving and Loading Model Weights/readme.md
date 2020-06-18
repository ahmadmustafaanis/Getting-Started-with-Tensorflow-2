# Saving and Loading Model Weights

-----

It is important to save your model's progress as it is not feasible to train your model again and again. Training the model again and again will cost your time, & computational resources, which will divert you from solving the actuall problem.

-----

There are 2 different formats to save the model weights, which are 
- hdf5 format (used by keras)
- Tensorflow native format

Normally it do not matter which format you use, but Tensorflow Native format is better.

----

You can save your model during training, at every epoch or you can save the model weights after you have completed the training and satisfied with the results.

### Saving during training

We will use built-in callback known as `ModelCheckpoint` to save our model weights during training.

We will create a dummy model for binary classification to see how to save models.
* Create Model
```python3
  from tensorflow.keras.models import Sequential
  from tensorflow.keras.layers import Dense
  from tensorflow.keras.callbacks import ModelCheckpoint

  model = Sequential([
      Dense(10, input_shape=(10,), activation = 'relu'),
      Dense(1, activation = 'sigmoid')
  ])

  model.compile(optimizer= 'SGD', loss='binary_crossentropy', metrics=['acc'] )
```
* Create CheckPoint using built-in CallBack.

```python3
  checkpoint = ModelCheckpoint('my_mode_weights', save_weights_only = True) 
```
  *   Here we are saving in `tensorflow native format`. File name for saving weights is `my_mode_weights`. 


  *   It will save weights for every epoch, and since we are passing 1 file, so it will over write it. We will check it later how to over come it.
*   3 different files will be created with names as
  
    *   checkpoint
    *   my_mode_weights.data
    * my_mode_weights.index 