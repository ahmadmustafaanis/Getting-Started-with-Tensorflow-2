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
1. <b> Create Model </b>
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
2. <b> Create CheckPoint using built-in CallBack and fit model</b>

```python3
  checkpoint = ModelCheckpoint('my_mode_weights', save_weights_only = True)
  model.fit(X,y, epochs=10, callbacks=[checkpoint]) #pass checkpoint in callback 
```

*   Here we are saving in `tensorflow native format`. File name for saving weights is `my_mode_weights`. 


      *   It will save weights for every epoch, and since we are passing 1 file, so it will over write it. We will check it later how to over come it.
*   3 different files will be created with names as
  
    *   checkpoint
    *   my_mode_weights.data
    * my_mode_weights.index 

* To save it in `hdf5` format, just change the name of file in `ModelCheckpoint` to `.h5` extension. i.e

```python3
checkpoint = ModelCheckpoint('my_mode_weights.h5', save_weights_only = True)
model.fit(X,y, epochs=10, callbacks=[checkpoint]) #pass checkpoint in callback 
```
* In this case, only one file will be created with name of `my_mode_weights.h5`.

-----

### Saving after training i.e Manual saving

We can save the model, after training, when all the epochs are done and we have the perfect weights, we can save them. We do not have to use the `callback` in this case. We can simply use `model.save_weights` built-in function.

Lets Check Code

```python3
model = Sequential([
  #Layers
  #Layers
  #Layers
])
model.compile(optimizer='adam', loss='mse', metrics=['mae'])

model.fit(X, y, epochs = 100)

model.save_weights('my_model`) #will save weights in Native Tensorflow Format and create 3 files.

model.save_weights('my_model.h5') #will save in hdf5 format
```
-----

Saved Files Explaination:
To see the Explaination of files created by `ModelCheckpoint` Callback, refer to [this](Explanation%20of%20saved%20files.ipynb) notebook.

----


## Loading Weights

Since we have not saved the model architecture, We have only saved weights of the model, we have to redesign the model with same architecture.

Taking our first example where we built a binary classifier, I will take same model and rewrite it.

```python3
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import ModelCheckpoint

model = Sequential([
  Dense(10, input_shape=(10,), activation = 'relu'),
  Dense(1, activation = 'sigmoid')
])

model.load_weights('my_mode_weights') #Use same file name as you used to store the weights

```

## Saving Criterias
Here we will learn how to save a model based on specific criteria, let's say we want to save the weights of a model after it see 5k training examples. We can save it using following code,

```python3 

from tensorflow.keras.callbacks import ModelCheckpoint

checkpoint_5k_path = 'model_checkpoint_5k/checkpoint_{epochs:02d}_{batch:04d}

checkpoint_5k = ModelCheckpoint(filepath = checkpoint_path, save_weights_only = True, save_freq = 5000 )
```
Notice that we have `{epoch:02d}_{batch:04d}` in our path, so it will not overwrite the weights, instead will make new files for each epcoh and batch.

There are several other important parameters in `ModelCheckpoint` callback that can help you, which are

* `save_best_only` if `True` will only save the best weights based on `monitor` which can be based on your `loss` and `metrics` and validation data if any.
* `monitor` to be used with `save_best_only`
* `mode` this can be used with `monitor`, already discussed earlier.
* `save_freq` this can be set to `epochs` for saving weights at every epoch.

You can learn more about it in [this](ProgrammingTutorial.ipynb) programming tutorial under *Model Saving Criteria*

----

## Saving and loading the Entire Model with Architecture

Some times you do not only have to save the weights, but also to save the model architecture, to do this, this is your basic code

```python3
from tensorflow.keras.callbacks import ModelCheckpoint

checkpoint = ModelCheckpoint('my_model_arch/my_model', save_weights_only=False)
```
By default `save_weights_only` is set to `False` so no need to mention it explicitly.

Our folder directory would be
* my_model_arch/my_model/assets
* my_model_arch/my_model/saved_model.pb
* my_model_arch/my_model/variables/variables.data-0000-of-0001
* my_model_arch/my_model/variables/variables.index


We can also save it in `.h5` format by just adding `.h5` in the end of the filepath. In that case only 1 file would be saved.

**Manually Saving the Model**

We can manually save the enitre model and architecture using `model.save('filepath')` or `model.save('filepath.h5')`.

**Loading the Entire Model**

To load the enitre model, we can use `load_model` function which is built-in Keras.

```python3
from tensorflow.keras.models import load_model

new_model = load_model('my_model')

new_model_h5 = load_model('model.h5')
```

To learn more about it, refer to [this](ProgrammingTutorial.ipynb) notebook under Saving the Entire model, and in-depth understanding on this topic is given in [this](Saving%20model%20architecture%20only.ipynb) notebook.

## Loading Pre-Trained Keras Models

Keras provide famous deep learning models, thier architectures, and pre-trained weights. First time you load them, they will automatically download weights in ` ~/.keras/models`. Famous available models are
* Xception
* VGG16
* VGG19
* Resnet/ Resnet v2 
* Inception v3
* Inception Resnet v2
* MobileNet/ MobileNet v2 
* DenseNet
* NASnet

To import the model, we import it from `tensorflow.keras.applications`. Let's see the code.

```python3
from tensorflow.keras.applications.resnet50 import ResNet50

model = ResNet50(weights = 'imagenet' ) #pre-trained weights on imagenet dataset
```

Let's say we do not want imagenet weights, then we can use

`model = ResNet50(weights='none')`, and then we have to do fresh training.

We can use it for **Transfer Learning** if we exclude the top Dense Layers. We can do it by
```python3
model = ResNet50(weights='imagenet', include_top = False)
```
#### Prediction using pre-trained ResNet50

```python3
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input, decode_predictions
from tensorflow.keras.preprocessing import image
import numpy as np

model = ResNet50(weights = 'imagenet', include_top = True)

image_input = image.load_image('path', target_size=(224, 224) )#224 is size for resnet
image_input = image.img_to_array(image_input)
image_input = preprocess_input(image_input[np.newaxis, ...])#adding a new axis for batch size

preds = model.predict(image_input)

decoded_pred = deocde_predictions(preds)[0]

print(f"Your image is of {decoded_pred}")

```
You can learn more about it in [this](ProgrammingTutorial.ipynb) under "Loading pre-trained Keras models" section.

-----

## Tensorflow Hub Models
Tensorflow also provides Tensorflow-hub models, which are basically focused on network modules which you can think of a seperate components of a tensorflow graph.

Tensorflow hub is a seperate library and you need to install it.

```bash
$ conda activate yourDeepLearningVenve
$ pip install "tensorflow>=2.0.0"
$ pip install --upgrade tensorflow-hub
```
You can browse all the avaiable models at [tfhub](https://tfhub.dev/)
To use them, 
```python3
import tensorflow_hub as hub

model_url = "https://tfhub.dev/google/imagenet/mobilenet_v1_050_160/classification/4" #learned from documentation

module = Sequential([
  hub.KerasLayer("model_url")
])

module.build(input_shape = [None, 160, 160, 3])
```
Output of this model is 1001 classes, for which you can find labels at documentation.

To know more about predicting images with it, refer to [this](ProgrammingTutorial.ipynb) notebook under TensorFlow Hub modules.