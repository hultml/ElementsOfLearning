# Overfitting and Regularization: Basic Concepts of Machine Learning

We (a) aim to introduce the basic concepts of machine learning
and (b)utilize these concepts in choosing a machine learning
model.

The basic problem with models that are too complex
in machine earning is that they are not robust.
This is called over-fitting.
We will see two ways to fix this:
choosing a model of an appropriate complexity,
called *complexity control*,
or choosing a very complex model to begin with and
restrictig this model's parameters to a subset of their
allowed values, called *regularization*.

In both these approaches, there are hyper-parameters whose
values must be set. And these values are determined on
sets called validation sets, while the fitting is done on
the training set, for reasons we shall see.
