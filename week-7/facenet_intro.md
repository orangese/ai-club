# FaceNet Introduction

This guide will introduce the motivation/basic principles behind FaceNet, a neural-network-based facial recognition system developed by Google in 2015. It's currently one of the most accessible, easy-to-understand, and accurate algorithms for face rec that isn't proprietary. Note that training FaceNet isn't discussed here, just how facial recognition works with a trained FaceNet algorithm.

Paper link: https://arxiv.org/pdf/1503.03832.pdf

## Estimators as Embedders

Estimators take an input `x` to an output `y` via the transformation `φ(x) = y`, where `φ` is our estimator's name. Importantly, `x` and `y` are both assumed to be vectors. Following from our definition of a vector in the [Week 1 lesson](https://docs.google.com/presentation/d/1C_g7UwT1bRxZSExKv4a406xbRLGcLR7NnQleiIi46lQ/edit#slide=id.g9920006141_1_0), we can think of both the `x` and `y` vectors as points in some space.

As an example, let's consider the estimator `φ(x; θ) = matmul(θ, x)`, where `θ` is a 3x2 matrix and `x` is a 2x1 matrix (or a 2-D vector). Note that the output `y = φ(x)` is going to be a 3x1 matrix (or, equivalently, a 3-D vector). We say that `φ` therefore is a transformation between 2-D space to 3-D space, since this estimator `φ` takes as input a 2-D vector and spits out a 3-D vector. Crucially, this transformation between spaces reveals important information about our inputs. Below is an example:

![alt text](https://miro.medium.com/max/872/1*zWzeMGyCc7KvGD9X8lwlnQ.png)

As you can see, our input space contains `x`s of two distinct categories: `blue` and `red`. If we wanted to separate (i.e., classify) each of our input vectors as `red` or `blue` in 2-D space, we'd need to recreate that curvy black line- a difficult task. However, if we apply an estimator `φ` and transform to our new, 3-D space, we can see that classifying between `red` and `blue` requires only a linear plane- much easier to figure out than the swiggly mess in the input space.

Thus, estimators can be thought of transformations between different spaces. They are particularly helpful because the output space of the estimator might be significantly easier to work with than the input space, as in the example above.

## Principles of FaceNet
FaceNet's key innovation is that the aforementioned principle- "estimators are embedders"- is an elegant solution to the problem of facial recognition. The authors' main claim is that estimators can learn to map raw images of peoples' faces to a special space in which *distance* corresponds to *facial similarity*. More specifically, let's say we have two images of former president Barack Obama:
![alt](https://www.biography.com/.image/t_share/MTE4MDAzNDEwNzg5ODI4MTEw/barack-obama-12782369-1-402.jpg) ![alt](https://static.politico.com/dims4/default/5b44cca/2147483647/resize/1160x%3E/quality/90/?url=https%3A%2F%2Fstatic.politico.com%2Fc0%2Fb2%2Fa9fc15064ee1bfdc2a5175128beb%2F200409-obama-getty-773.jpg)

The two images I've chosen vary greatly in terms of facial profile, lighting, facial expression, emotion, pose, etc. As a result, checking whether two images represent the same person is a non-trivial task of great difficulty. How we should even begin to approach this task is unclear- should we try to find facial landmarks like nose, eyes, chin, and measure distances? Should we look at the person's hair? Their skin color? These decisions are far too numerous and too fragile to constitute an effective facial recognition system. In fact, *with only information from the raw images themselves*, it's probably not possible to construct a set of mathematical rules that can tell us reliably whether two faces represent the same people- it's just too hard.

Thankfully, FaceNet proposes a great solution. Instead of trying to infer things from the raw images themselves, let an estimator `φ` transform the input *image space* (in which each image is a matrix of pixels) to some output space. `φ` would learn, during training, to distinguish between unimportant things in the input photo (like lighting, etc.) and focus on important aspects a person's face that determine their identity (how they smile, their face shape, etc.). All this information that the estimator learns would then be contained within the output vector. For example, let's take Pres. Donald Trump's face and denote it as `x`. Then, the FaceNet "embedding" of the raw pixels of `x` would be a vector `φ(x)`. In practice, `φ` is a convolutional neural network, which we haven't learned about yet, so you don't have to worry about it. All we need to know if that `φ` is an estimator that is able to embedding a raw face image `x` into a meaningful output space `φ(x)`.

Once we've gotten these embeddings, we can compare them directly via subtracting them. Specifically, given two face embeddings `φ(x)` and `φ(y)`, the [length](https://docs.google.com/presentation/d/1C_g7UwT1bRxZSExKv4a406xbRLGcLR7NnQleiIi46lQ/edit#slide=id.g9920006141_0_138) of their difference `φ(x) - φ(y)` is used to determine facial identity. If `||φ(x) - φ(y)||` (i.e., the distance between the two faces `x` and `y`) is below a certain threshold, the faces are the same person; otherwise, they're different people. *This means that the estimator will map a face image to a new space, in which the distance between two different points (each of which correspond to a different input face) indicates how similar those two faces are.*

Below is a diagram detailing how FaceNet works. "Latent space" simply refers to the output embedding space of `φ`. So, the dots represent `φ(x)` for an input face `x`. Note that the embedding space is shown to be 2-D here for easier visualization, but in practice, the embeddings `φ(x)` lie on a sphere in 512-dimensional space.

![alt](https://www.baseapp.com/wp-content/uploads/2018/03/latent_space_face.png)

