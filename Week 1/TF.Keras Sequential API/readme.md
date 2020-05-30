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