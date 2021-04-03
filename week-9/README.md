# Modern Convolutional Networks: Practice Problems

These practice problems cover all the recent convolutional networks. It's been a while since the last assignment set, so be sure to go back and review your MNIST code from last week to make sure you remember what's going on (i.e., model creation, dataset flow, and training loop). Tasks/questions are bolded.

## GPU Training

### Introduction

We are going to use Google Colab- which allows you to run your code online on special GPU hardware- for training and testing our neural networks. 

**Q1**: Please go through the tutorial [here](https://colab.research.google.com/notebooks/intro.ipynb) to familiarize yourself with Colab.

### Transfer to Colab

**Q2**: Port your code from last week (the CNN MNIST code) to Google Colab.  This should require no code changes, just make sure that you change the notebook settings to use GPU hardware acceleration. The speedup should be significant!

To start coding in Colab, go to the [Colab page](colab.research.google.com) and create a new notebook. You can copy/paste your code from last week in and it should work (minus a small few changes).

### Clean up train code

#### `fit` API

**Q3**: After porting successfully, change the structure of the model so that the model has at least 3 `Conv2D` layers. Observe how the accuracy of the model increases much faster.

In reality, we almost never write a custom training loop using `tf.GradientTape` and `for epoch in range(epochs)`. Instead, we use the high-level `fit` API, which does everything for us in terms of using optimizers, tracking loss/accuracy, etc.

`fit` is really easy to use: for any `tf.keras.Model`, the custom training loop is replaced by two lines of code:

```python
# initialize parameterse for training
model.compile(optimizer=tf.keras.optimizers.SGD(), 
              loss="categorical_crossentropy",
              metrics=["categorical_accuracy"])

# start training
model.fit(dataset, batch_size=32, epochs=30,
		  validation_split=0.2)
```

**Q4**: See the [docs](https://keras.io/api/models/model_training_apis/) for more details regarding these two functions. Once you understand how to use them, replace the custom training loop we wrote in previous weeks (`for epoch in range(epochs): ...`) with the `compile`/`fit` API. After you rewrite the code with the `fit` API, you will need to modify your dataloading pipeline by doing three things: 1. load a separate test dataset and set `validation_data` equal to that test dataset, 2. specify `as_supervised=True` in the MNIST data loading, and 3. write a data normalization function and `map` it to the datasets. If you're having trouble, check out the [TensorFlow guide](https://www.tensorflow.org/datasets/keras_example) or the [solutions](https://colab.research.google.com/drive/1yLosCbrm8GO3oRj9xIe6k69gRAEjwGI9?authuser=1).

#### Data augmentation

Data augmentation is essential to a lot of computer-vision-related tasks because it allows the network to learn from a enhanced set of data.

**Q5**: Apply two separte forms of data augmentation to the MNIST `tf.Dataset`. See [this guide](https://www.tensorflow.org/tutorials/images/data_augmentation) if you need help getting started.

### New Dataset

For the next few sections, we are giong to switch over from MNIST to a new dataset, called ImageNet- a collection of >1M images from over 1000 classes (including things as diverse as food, breeds of dogs, cars, furniture, etc). Because of its large size, ImageNet can be tricky to train on, so we're going to use a [smaller subset](https://www.tensorflow.org/datasets/catalog/imagenette) with around 9,000 training examples and only 10 image classes.

**Q6**: Change your MNIST code to work with the ImageNet subset linked above- this should require minimal changes, as you can access ImageNet through `tensorflow-datasets` as well. Be sure that your convolutional network still runs/trains okay (using the `fit` API).

## Deep Learning Innovations

### ReLU, dropout, and batch normalization

**Q1**: Replace all `sigmoid` activations with `relu` activations, and observe how this upgrade affects training. Does loss decrease more quickly, and does the model reach a higher accuracy at the end of training? Experiment with other activations as well- TensorFlow has a lot available, and it's nice to see just how many options there are. However, for a surprisingly large amount of complex tasks, `relu` is sufficient.
	- Note that the activation function for a layer can be computed multiple ways in TensorFlow. Our solution guide has the activation function as a parameter in the layer (i.e., `tf.keras.layers.Conv2D(activation="relu")`), but you can also separate the two (`tf.keras.layers.Conv2D()` and `tf.keras.layers.Activation("relu")` right afterwards). You should be familiar with both ways of implementing activation functions.

**Q2**: Implement dropout in the middle 2 convolutional layers in your ImageNet CNN. See [the docs](https://keras.io/api/layers/regularization_layers/dropout/) for help on how to do so. Experiment with different setups to find what works best. Does dropout help training?

**Q3**: Replace all instances of dropout with batch normalization ([docs](https://keras.io/api/layers/normalization_layers/batch_normalization/)). Note that in practice, batch normalization has mostly replaced dropout for deep convolutional networks, so knowing how to work with `BatchNormalization` is really important. Play around with the setup of this layer- is it empirically more effective to apply batch normalization *before* applying the `Activation` layer or *after* applying it? Does it make a difference?
	- Interestingly, `BatchNormalization` used to be applied *after* ReLU, but now it seems the trend in recent literature is to apply it *before* ReLU. Can you think of a reason why? (Hint: think about what the output of ReLU would look like, and how batch normalization would affect that distribution).

**Q4**: Batch normalization (and its sister layers, like layer normalization and instance normalization) is probably one of the most used normalization layers in modern convolutional networks. AI researchers have noticed that it tends to help in reduce overfitting (like dropout). Given that batch normalization (unlike dropout) adds more trainable parameters to the network- a practice that generally increases overfitting by increasing model complexity- why do you think batch normalization has this regularizing effect on the network overall? This question is tricky- I don't think anyone really knows the answer to it :)

### Residual Connections

**Q1**: Implement residual skip-connections in your ImageNet CNN. To make them properly, change the network structure so it contains 5 "blocks", where each block is the [`Conv2D`->`BatchNorm`->`ReLU`] pattern repeated three times. Add residual connections between every block (the [`tf.keras.layers.Add`](https://www.tensorflow.org/api_docs/python/tf/keras/layers/Add) layer will be helpful here). Your code should look something like

```python
def block(x):
	for i in range(3):
		x = Conv2D()(x)  # must use "same" padding here
					   # otherwise block output dims won't 
					   # work
		x = BatchNormalization()(x)
		x = ReLU()(x)
	return x

x = block(input_x). # block output must have same dims
					# as block input, otherwise you can't 
					# add them together
x += input_x. # residual connection
# ... and repeat this pattern 5 times
```

Retrain the CNN on ImageNet and see if accuracy changes. It should increase by a non-trivial amount. Note that this coding paradigm- using repeating convolutional "blocks"- is really important in using deep learning models, so make sure you're comfortable with the above pseudocode.

### Inception Block

Recall that the idea behind the inception block is two-fold: first, different layers are concatenated horizontally instead of stacked sequentially, and second, 1x1 convolutions are used to reduce computational load. Consider the below convolutional block (similar to conventional VGG-style blocks):

```python
def traditional_block(x):
	x = Conv2D(kernel_size=(3, 3), activation="relu")(x)
	x = MaxPool()(x)
	x = Conv2D(kernel_size=(5, 5), activation="relu")(x)
	return x
```

The naive inception block (without any 1x1 convolutions would look like):

```python
def inception_block(x):
	# all padding must be 'same' for concat to work!
	x_3x3 = Conv2D(kernel_size=(3, 3), activation="relu")(x)
	x_mp = MaxPool()(x)
	x_5x5 = Conv2D(kernel_size=(5, 5), activation="relu")(x)
	out = Concatenate()([x_3x3, x_mp, x_5x5])
	return x
```

**Q2**: The more optimized inception block would be similar, but with 1x1 convolutions before the 3x3 and 5x5 convolutions in order to reduce the number of filters. Implement the inception block in the CNN from the previous question and retrain on ImageNet.

