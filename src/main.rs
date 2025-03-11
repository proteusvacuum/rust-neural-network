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

fn main() {
    let layer_sizes = vec![2, 250, 1];
    let mut network = Network::new(&layer_sizes);
    println!("Starting training of {network:?}");

    let data = vec![
        (vec![0.0, 0.0], vec![0.0]),
        (vec![1.0, 0.0], vec![1.0]),
        (vec![0.0, 1.0], vec![1.0]),
        (vec![1.0, 1.0], vec![0.0]),
    ];

    train(&mut network, &data, 0.1, 50000);

    for (inputs, expected) in &data {
        let output = predict(&network, inputs);
        println!(
            "Input: {:?} => Prediction: {:?}, Expected: {:?}",
            inputs, output, expected
        );
    }
}
