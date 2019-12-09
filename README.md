# Neural networks from scratch:
This is my first try to implement Neural Networks in python from scratch.Its completly built using the module numpy.

## Basic outline of the project
1. Tensors
2. Cost Functions
3. Layers
4. Optimizers
5. Data
6. Training

## What is a Tensor?
A tensor is a container which can house data in N dimensions, along with its linear operations, though there is nuance in what tensors technically are and what we refer to as tensors in practice.
In otherwords  it is basically an N dimentional array.

## Cost Function
It is a function that measures the performance of a Machine Learning model for given data. Cost Function quantifies the error between predicted values and expected values and presents it in the form of a single real number. Depending on the problem Cost Function can be formed in many different ways. The purpose of Cost Function is to be either:

Minimized - then returned value is usually called cost, loss or error. The goal is to find the values of model parameters for which Cost Function return as small number as possible.

Maximized - then the value it yields is named a reward. The goal is to find values of model parameters for which returned number is as large as possible.

### Mean Squared Error Function
One of the most commonly used and firstly explained regression metrics. Average squared difference between the predictions and expected results. In other words, an alteration of MAE where instead of taking the absolute value of differences, they are squared.

![MSE](https://miro.medium.com/max/358/0*v5NxDJHE8Oy8Rf-2.png)

### Mean Absolute Error Function
Regression metric which measures the average magnitude of errors in a group of predictions, without considering their directions. In other words, it’s a mean of absolute differences among predictions and expected results where all individual deviations have even importance.

![MAE](https://miro.medium.com/max/326/0*Swic0H6aelUyYI2B.png)

In MAE, the partial error values were equal to the distances between points in the coordinate system. Regarding MSE, each partial error is equivalent to the area of the square created out of the geometrical distance between the measured points. All region areas are summed up and averaged.

## Layers
3. Layers

4. optimizer
    --> SGD (Stocastic Gradient Descent)
5. data
6. training
