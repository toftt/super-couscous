use std::{io, iter};
use std::fs::File;
use std::io::BufRead;
use std::iter::Sum;
use std::ops::{Add, Mul};
use std::path::{Iter, Path};
use std::process::Output;
use rand::prelude::*;
use rand_distr::Normal;

use neural_network::Matrix;

/**
Input layer 28x28 = size 784
Hidden layer 1: size 16
Hidden layer 2: size 16
Output layer: size 10
 **/


fn sigmoid(x: f64) -> f64 {
    1. / (1. + (-x).exp())
}

fn sigmoid_prime(z: &Matrix) -> Matrix {
    let applied = z.apply(sigmoid);
    &applied * &(1. - &applied)
}

#[derive(Debug)]
struct Network {
    num_layers: usize,
    sizes: Vec<usize>,
    biases: Vec<Matrix>,
    weights: Vec<Matrix>,
}

impl Network {
    fn new(sizes: Vec<usize>) -> Network {
        let biases: Vec<Matrix> = sizes.iter()
            .skip(1)
            .map(|x| Matrix::with_random_values(*x, 1))
            .collect();

        let weights: Vec<Matrix> = sizes.split_first().unwrap().1.iter().zip(sizes.split_last().unwrap().1.iter())
            .map(|(x, y)| Matrix::with_random_values(*x, *y)).collect();

        Network {
            num_layers: sizes.len(),
            sizes,
            biases,
            weights,
        }
    }

    fn train(&mut self,
             mut training_data: Vec<(Matrix, Matrix)>,
             epochs: usize,
             mini_batch_size: usize,
             eta: f64,
             test_data: Vec<(Matrix, Matrix)>) {
        let n = training_data.len();
        let n_test = test_data.len();

        for i in 00..epochs {
            training_data.shuffle(&mut thread_rng());
            let mini_batches: Vec<&[(Matrix, Matrix)]> = (0..n).step_by(mini_batch_size)
                .map(|idx| &training_data[idx..idx + mini_batch_size])
                .collect();

            for (idx, mini_batch) in mini_batches.iter().enumerate() {
                if idx % 500 == 0 {
                    println!("Starting mini_batch {}", idx);
                }
                self.update_mini_batch(mini_batch, eta);
            }

            print!("Epoch {}: {} / {}", i, self.evaluate(&test_data), n_test);
        }
    }


    fn feed_forward(&self, a: &Matrix) -> Matrix {
        let mut activation = a.clone();
        for (b, w) in self.biases.iter().zip(self.weights.iter()) {
            activation = (b + &w.dot(&activation)).apply(sigmoid);
        }
        activation
    }

    fn update_mini_batch(&mut self, mini_batch: &[(Matrix, Matrix)], eta: f64) {
        let mut nabla_b: Vec<Matrix> = self.biases.iter().map(Matrix::with_shape).collect();
        let mut nabla_w: Vec<Matrix> = self.weights.iter().map(Matrix::with_shape).collect();
        let mini_batch_len = mini_batch.len();

        for (x, y) in mini_batch.into_iter() {
            let (delta_nabla_b, delta_nabla_w) = self.backprop(x, y);
            nabla_b = nabla_b.iter().zip(delta_nabla_b.iter()).map(|(a, b)| a + b).collect();
            nabla_w = nabla_w.iter().zip(delta_nabla_w.iter()).map(|(a, b)| a + b).collect();
        }
        self.weights = self.weights.iter().zip(nabla_w.iter()).map(|(w, nw)|
            w - &((eta / mini_batch_len as f64) * nw)
        ).collect();
        self.biases = self.biases.iter().zip(nabla_b.iter()).map(|(b, nb)|
            b - &((eta / mini_batch_len as f64) * nb)
        ).collect();
    }

    fn backprop(&self, x: &Matrix, y: &Matrix) -> (Vec<Matrix>, Vec<Matrix>) {
        let mut nabla_b: Vec<Matrix> = self.biases.iter().map(Matrix::with_shape).collect();
        let mut nabla_w: Vec<Matrix> = self.biases.iter().map(Matrix::with_shape).collect();

        let mut activation = x.clone();
        let mut activations = vec![activation.clone()];

        let mut zs: Vec<Matrix> = vec![];

        for (b, w) in self.biases.iter().zip(self.weights.iter()) {
            let z = &w.dot(&activation) + b;
            activation = z.apply(sigmoid);

            zs.push(z);
            activations.push(activation.clone());
        }

        let mut delta = Network::cost_derivative(activations.last().unwrap(), &y) *
            sigmoid_prime(zs.last().unwrap());

        let idx_b = nabla_b.len() - 1;
        nabla_b[idx_b] = delta.clone();

        let a_minus_two = &activations[activations.len() - 2];
        let idx_w = nabla_w.len() - 1;
        nabla_w[idx_w] = delta.dot(&a_minus_two.transpose());

        for l in 2..self.num_layers {
            let z = &zs[zs.len() - l];
            let sp = sigmoid_prime(z);
            delta = self.weights[self.weights.len() - l + 1].transpose().dot(&delta) * sp;

            let idx_b = nabla_b.len() - l;
            let idx_w = nabla_w.len() - l;
            nabla_b[idx_b] = delta.clone();
            nabla_w[idx_w] = delta.dot(&activations[activations.len() - l - 1].transpose());
        }
        (nabla_b, nabla_w)
    }

    fn cost_derivative(output_activation: &Matrix, y: &Matrix) -> Matrix {
        output_activation - y
    }

    fn evaluate(&self, test_data: &[(Matrix, Matrix)]) -> usize {
        test_data.iter()
            .map(|(x, y)| {
                (self.feed_forward(x).max(), y.max())
            })
            .filter(|(a, b)| a == b)
            .count()
    }
}

fn main() {
    println!("Loading test data...");
    let mut test_data: Vec<(Matrix, Matrix)> = Vec::new();
    for line in read_lines("./data/mnist_test.csv").expect("error reading file") {
        let line_str = line.expect("error reading line");
        test_data.push(get_training_tuple(&line_str));
    }

    println!("Loading training data...");
    let mut training_data: Vec<(Matrix, Matrix)> = Vec::new();
    for line in read_lines("./data/mnist_train.csv").expect("error reading file") {
        let line_str = line.expect("error reading line");
        training_data.push(get_training_tuple(&line_str));
    }

    let mut network = Network::new(vec![784, 30, 10]);
    let _ = &network.train(training_data, 60, 10, 0.1, test_data);
}

fn read_lines<P>(file_name: P) -> io::Result<io::Lines<io::BufReader<File>>>
    where P: AsRef<Path> {
    let file = File::open(file_name)?;
    Ok(io::BufReader::new(file).lines())
}

fn get_training_tuple(line: &str) -> (Matrix, Matrix) {
    let expected_num: f64 = line.split(",").into_iter().next().unwrap().parse().unwrap();
    let expected_vec: Vec<Vec<f64>> = (0..10).into_iter()
        .map(|idx| idx as f64)
        .map(|idx| if idx == expected_num { vec![1.] } else { vec![0.] })
        .collect();
    let expected = Matrix::from_vec(expected_vec);
    let input_vec: Vec<f64> = line.split(",")
        .skip(1)
        .map(|n| n.parse::<f64>().unwrap())
        .map(|n| map(n, 0., 255., 0., 1.))
        .collect();

    let input = Matrix {
        num_rows: 784,
        num_cols: 1,
        storage: input_vec.into_boxed_slice(),
    };

    return (input, expected);
}

fn map(value: f64, low1: f64, high1: f64, low2: f64, high2: f64) -> f64 {
    low2 + (value - low1) * ((high2 - low2) / (high1 - low1))
}

#[cfg(test)]
mod tests {
    use neural_network::Matrix;
    use crate::{Network};

    #[test]
    fn network_eval() {
        let b = Network::new(vec![2, 4, 3]);
        let input = Matrix::from_vec(vec![
            vec![0.5],
            vec![0.1],
        ]);
        dbg!(&b);
        dbg!(&input);
        let result2 = b.feed_forward(&input);
        println!("{}", &result2);
    }

    #[test]
    fn mini_batch() {
        let mut network = Network::new(vec![2, 4, 3]);
        let training_data = vec![
            (
                Matrix::from_vec(vec![
                    vec![0.5],
                    vec![0.1],
                ]),
                Matrix::from_vec(vec![
                    vec![0.5],
                    vec![0.1],
                    vec![0.1],
                ]),
            ),
            (
                Matrix::from_vec(vec![
                    vec![0.3],
                    vec![0.4],
                ]),
                Matrix::from_vec(vec![
                    vec![0.7],
                    vec![0.7],
                    vec![0.7],
                ]),
            ),
        ];
        dbg!(&network);
        network.update_mini_batch(&training_data, 3.);
        dbg!(&network);
    }
}