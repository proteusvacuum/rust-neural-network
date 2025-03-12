# Neural Network in Rust

Built with ❤️ at the [Recurse Center](https://www.recurse.com/).


This is a toy neural network built with (almost) no external dependencies. I built this while taking the fast.ai course in order to solidify the concepts of deep learning and back-propagation.

It is a simple feedforward neural network that can classify handwritten digits from the MNIST dataset. The network is trained using a basic backpropagation algorithm with gradient descent.


## Usage

Download the [MNIST database](https://www.kaggle.com/datasets/hojjatk/mnist-dataset/data) into `./data`.

Run `cargo run --release`

It will train the model on a few thousand of the images, then attempt to infer a few hundred options.


## Training Process

- Forward Pass: Each layer computes activations using the sigmoid function.
- Loss Calculation: The Mean Squared Error (MSE) loss function is used to measure the difference between the predicted and actual labels.
Backward Pass (Backpropagation): The output layer error is computed and propagated backward. Weights and biases are updated using the derivative of the sigmoid function and gradient descent.
- Update Weights: Adjust weights and biases using the learning rate.
Continue for multiple epochs.

## Avenues for improvement

- Use a validation set in training
- Experiment with different loss and activation functions

## References

- [Neural Networks and Deep Learning](http://neuralnetworksanddeeplearning.com/chap2.html)
- [3Blue1Brown Backpropagation Calculus](https://www.3blue1brown.com/lessons/backpropagation-calculus)
