use mnist::{Mnist, MnistBuilder};
use rand::Rng;

#[derive(Debug)]
struct Neuron {
    weights: Vec<f64>,
    bias: f64,
}

impl Neuron {
    fn new(len: usize) -> Self {
        let weights: Vec<f64> = (0..len)
            .map(|_| rand::rng().random_range(-1.0..1.0))
            .collect();
        let bias = rand::rng().random_range(-1.0..1.0);
        Neuron { weights, bias }
    }

    fn forward(&self, inputs: &[f64]) -> f64 {
        let sum: f64 = self
            .weights
            .iter()
            .zip(inputs.iter())
            .map(|(weight, input)| weight * input)
            .sum();
        sigmoid(sum + self.bias)
    }

    fn update(&mut self, prev_inputs: &[f64], error_gradient: f64, learning_rate: f64) {
        for (weight, input) in self.weights.iter_mut().zip(prev_inputs.iter()) {
            *weight -= learning_rate * error_gradient * input; // Adjust weight
        }
        self.bias -= learning_rate * error_gradient; // Adjust bias
    }
}

fn sigmoid(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}

fn sigmoid_derivative(x: f64) -> f64 {
    let s = sigmoid(x);
    s * (1.0 - s)
}

#[derive(Debug)]
struct Layer {
    neurons: Vec<Neuron>,
}

impl Layer {
    fn new(input_size: usize, num_neurons: usize) -> Self {
        Layer {
            neurons: (0..num_neurons).map(|_| Neuron::new(input_size)).collect(),
        }
    }
    fn forward(&self, inputs: &[f64]) -> Vec<f64> {
        self.neurons
            .iter()
            .map(|neuron| neuron.forward(inputs))
            .collect()
    }

    fn update(&mut self, prev_inputs: &[f64], error_gradients: &[f64], learning_rate: f64) {
        for (neuron, &error_gradient) in self.neurons.iter_mut().zip(error_gradients) {
            neuron.update(prev_inputs, error_gradient, learning_rate);
        }
    }
}

#[derive(Debug)]
struct Network {
    layers: Vec<Layer>,
}

impl Network {
    fn new(layer_sizes: &[usize]) -> Self {
        let layers = layer_sizes
            .windows(2)
            .map(|sizes| Layer::new(sizes[0], sizes[1]))
            .collect();
        Network { layers }
    }

    fn forward(&self, inputs: &[f64]) -> Vec<Vec<f64>> {
        let mut activations = vec![inputs.to_owned()];
        for layer in &self.layers {
            let output = layer.forward(activations.last().unwrap());
            activations.push(output);
        }
        activations
    }

    fn backward(&mut self, targets: &Vec<f64>, activations: &[Vec<f64>], learning_rate: f64) {
        // References:
        // http://neuralnetworksanddeeplearning.com/chap2.html
        // https://www.3blue1brown.com/lessons/backpropagation-calculus

        let output_layer_error_gradients: Vec<f64> = activations
            .last()
            .expect("there must be at least one activation")
            .iter()
            .zip(targets)
            .map(|(output, target)| {
                let error = *output - *target;
                error * sigmoid_derivative(*output)
            })
            .collect();
        let mut error_gradients = vec![output_layer_error_gradients.clone()];

        let previous_inputs = activations[self.layers.len() - 1].clone();
        self.layers.last_mut().unwrap().update(
            &previous_inputs,
            &output_layer_error_gradients,
            learning_rate,
        );

        for layer_num in (0..self.layers.len() - 1).rev() {
            let next_layer = &self.layers[layer_num + 1];
            let current_activations = &activations[layer_num + 1];
            let next_error_gradients = &error_gradients[0];

            let mut current_error_gradients = vec![0.0; self.layers[layer_num].neurons.len()];

            for (i, _) in self.layers[layer_num].neurons.iter().enumerate() {
                // for each neuron in this layer, get all the contributions for the next layer, and sum them up.
                for (j, neuron) in next_layer.neurons.iter().enumerate() {
                    current_error_gradients[i] += next_error_gradients[j]
                        * neuron.weights[i]
                        * sigmoid_derivative(current_activations[i]);
                }
            }
            error_gradients.insert(0, current_error_gradients.clone());

            self.layers[layer_num].update(
                &activations[layer_num],
                &current_error_gradients,
                learning_rate,
            )
        }
    }
}

fn mean_squared_error(targets: &[f64], predictions: &[f64]) -> f64 {
    // (target - prediction) ^ 2 / N
    targets
        .iter()
        .zip(predictions)
        .map(|(target, prediction)| (target - prediction).powi(2) / 2.0)
        .sum::<f64>()
        / targets.len() as f64
}

fn train(network: &mut Network, data: &[(Vec<f64>, Vec<f64>)], learning_rate: f64, epochs: usize) {
    for epoch in 0..epochs {
        let mut total_loss = 0.0;
        for (inputs, targets) in data.iter() {
            let activations = network.forward(inputs);
            let predictions = activations.last().unwrap().clone();
            total_loss += mean_squared_error(targets, &predictions);
            network.backward(targets, &activations, learning_rate);
        }
        total_loss /= data.len() as f64;
        println!("Epoch: {epoch} - Total Loss: {total_loss}");
    }
}

fn predict(network: &Network, inputs: &[f64]) -> Vec<f64> {
    network.forward(inputs).last().unwrap().to_vec()
}

fn convert_to_image_vectors(flat_images: &[u8], image_size: usize) -> Vec<Vec<u8>> {
    let num_images = flat_images.len() / image_size;
    let mut result = Vec::with_capacity(num_images);

    for img_idx in 0..num_images {
        let start = img_idx * image_size;
        let end = start + image_size;
        let image = flat_images[start..end].to_vec();
        result.push(image);
    }

    result
}

fn print_image(image_vec: &[f64]) {
    for y in 0..28 {
        for x in 0..28 {
            let pixel = image_vec[y * 28 + x];
            // Print a basic ASCII representation
            if pixel > 0.5 {
                print!("#");
            } else {
                print!(" ");
            }
        }
        println!();
    }
}

fn one_hot_encode_labels(labels: &[u8], num_classes: usize) -> Vec<Vec<f64>> {
    labels
        .iter()
        .map(|&label| {
            let mut one_hot = vec![0.0; num_classes];
            one_hot[label as usize] = 1.0; // Set the correct class index to 1.0
            one_hot
        })
        .collect()
}

fn main() {
    const LEARNING_RATE: f64 = 0.2;
    const EPOCHS: usize = 5;
    let layer_sizes = vec![784, 128, 10];

    let mut network = Network::new(&layer_sizes);

    let Mnist {
        trn_img: mnist_training_images,
        trn_lbl: mnist_training_labels,
        tst_img: mnist_test_images,
        tst_lbl: mnist_test_labels,
        ..
    } = MnistBuilder::new()
        .label_format_digit()
        .training_set_length(50_000)
        .test_set_length(10)
        .finalize();

    println!("Loaded {} training images", mnist_training_labels.len());
    println!("Loaded {} test images", mnist_test_labels.len());

    let training_images: Vec<Vec<f64>> = convert_to_image_vectors(&mnist_training_images, 28 * 28)
        .into_iter()
        .map(|img| img.into_iter().map(|x| x as f64 / 255.0).collect())
        .collect();

    let training_labels = one_hot_encode_labels(&mnist_training_labels, 10);
    let training_data: Vec<(Vec<f64>, Vec<f64>)> =
        training_images.into_iter().zip(training_labels).collect();

    train(&mut network, &training_data, LEARNING_RATE, EPOCHS);

    let test_images: Vec<Vec<f64>> = convert_to_image_vectors(&mnist_test_images, 28 * 28)
        .into_iter()
        .map(|img| img.into_iter().map(|x| x as f64 / 255.0).collect())
        .collect();
    let test_labels = one_hot_encode_labels(&mnist_test_labels, 10);
    let test_data: Vec<(Vec<f64>, Vec<f64>)> = test_images.into_iter().zip(test_labels).collect();

    for (input, expected) in &test_data {
        let output = predict(&network, input);
        let predicted_class = output
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .unwrap()
            .0;
        let expected_class = expected
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .unwrap()
            .0;

        print_image(input);
        println!(
            "Prediction: {:?}, Expected: {:?}",
            predicted_class, expected_class
        )
    }
}
