use std::iter;
use std::iter::Sum;
use std::ops::{Add, Mul};
use std::path::Iter;
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

fn sigmoid_vec_mut(x: &mut Vec<f64>) {
    for i in 0..x.len() {
        x[i] = sigmoid(x[i]);
    }
}

fn sigmoid_vec(x: &Vec<f64>) -> Vec<f64> {
    x.iter().map(|y| sigmoid(*y)).collect()
}

fn add(a: &Vec<f64>, b: &Vec<f64>) -> Vec<f64> {
    let mut result: Vec<f64> = Vec::with_capacity(a.len());
    for (i, j) in a.iter().zip(b.iter()) {
        result.push(i + j);
    }
    result
}

fn mult(a: &Vec<f64>, b: &Vec<f64>) -> Vec<f64> {
    let mut result: Vec<f64> = Vec::with_capacity(a.len());
    for (i, j) in a.iter().zip(b.iter()) {
        result.push(i * j);
    }
    result
}

fn sub_vec(a: &Vec<f64>, b: &Vec<f64>) -> Vec<f64> {
    let mut result: Vec<f64> = Vec::with_capacity(a.len());
    for (i, j) in a.iter().zip(b.iter()) {
        result.push(i - j);
    }
    result
}

fn sub_vec_f64(a: f64, b: &Vec<f64>) -> Vec<f64> {
    b.iter().map(|c| c - a).collect()
}

fn dot<T>(a: &Vec<Vec<T>>, b: &Vec<T>) -> Vec<T>
    where T: Mul<Output=T> + Add<Output=T> + Sum + Copy {
    let mut result: Vec<T> = Vec::new();

    for (w, v) in a.iter().zip(iter::repeat(b).take(a.len())) {
        let partial: T = w.iter().zip(v.iter()).map(|(ww, vv)| ww.to_owned() * vv.to_owned()).sum();
        result.push(partial);
    }

    result
}

fn random_vec(capacity: usize, rand: &mut ThreadRng) -> Vec<f64> {
    let normal = Normal::new(0., 1.).unwrap();
    let mut v = Vec::with_capacity(capacity);
    for _ in 0..capacity {
        v.push(normal.sample(rand));
    }
    v
}

fn sigmoid_prime_vec(z: &Vec<f64>) -> Vec<f64> {
    let sv = sigmoid_vec(z);
    mult(&sv, &sub_vec_f64(1., &sv))
}

#[derive(Debug)]
struct Network {
    num_layers: usize,
    sizes: Vec<usize>,
    biases: Vec<Vec<f64>>,
    weights: Vec<Vec<Vec<f64>>>,
}

#[derive(Debug)]
struct NetworkNew {
    num_layers: usize,
    sizes: Vec<usize>,
    biases: Vec<Matrix>,
    weights: Vec<Matrix>,
}

impl NetworkNew {
    fn new(sizes: Vec<usize>) -> NetworkNew {
        let biases: Vec<Matrix> = sizes.iter()
            .skip(1)
            .map(|x| Matrix::with_random_values(*x, 1))
            .collect();

        let weights: Vec<Matrix> = sizes.split_first().unwrap().1.iter().zip(sizes.split_last().unwrap().1.iter())
            .map(|(x, y)| Matrix::with_random_values(*x, *y)).collect();

        NetworkNew {
            num_layers: sizes.len(),
            sizes,
            biases,
            weights,
        }
    }



    fn feed_forward(&self, mut a: Matrix) -> Matrix {
        for (b, w) in self.biases.iter().zip(self.weights.iter()) {
            println!("----------------");
            println!("{}", &w);
            println!("{}", &a);
            println!("{}", &b);
            println!("----------------");
            a = (b + &w.dot(&a)).apply(sigmoid);
        }
        a
    }
}

impl Network {
    fn new(sizes: Vec<usize>) -> Network {
        let mut rand = thread_rng();

        let biases: Vec<Vec<f64>> = sizes.iter()
            .skip(1)
            .map(|x| random_vec(*x, &mut rand))
            .collect();

        let weights: Vec<Vec<Vec<f64>>> = sizes.split_first().unwrap().1.iter().zip(sizes.split_last().unwrap().1.iter())
            .map(|(x, y)| {
                let mut v = Vec::with_capacity(*x);
                for _ in 0..*x {
                    v.push(random_vec(*y, &mut rand));
                }
                v
            }).collect();

        assert_eq!(biases.len(), weights.len());

        Network {
            num_layers: sizes.len(),
            sizes,
            biases,
            weights,
        }
    }

    fn evaluate(self, mut a: Vec<f64>) -> Vec<f64> {
        for (b, w) in self.biases.iter().zip(self.weights.iter()) {
            let mut intermediate = add(b, &dot(w, &a));
            sigmoid_vec_mut(&mut intermediate);
            a = intermediate;
        }
        a
    }

    fn update_mini_batch(self, mini_batch: (Vec<Vec<f64>>, Vec<f64>), eta: f64) {
        let nabla_b: Vec<Vec<f64>> = vec![vec![0.; self.biases.get(0).unwrap().len()]; self.biases.len()];
        let nabla_w: Vec<Vec<f64>> = vec![vec![0.; self.weights.get(0).unwrap().len()]; self.weights.len()];
    }

    fn backprop(self, x: Vec<f64>, y: Vec<f64>) {
        let mut nabla_b: Vec<Vec<f64>> = vec![vec![0.; self.biases.get(0).unwrap().len()]; self.biases.len()];
        let mut nabla_w: Vec<Vec<f64>> = vec![vec![0.; self.weights.get(0).unwrap().len()]; self.weights.len()];

        let activation = x;
        let mut activations = vec![activation];

        let mut zs: Vec<Vec<f64>> = vec![];

        for (b, w) in self.biases.iter().zip(self.weights.iter()) {
            let z = add(b, &dot(w, b));
            let mut activation = z.clone();

            zs.push(z);
            sigmoid_vec_mut(&mut activation);
            activations.push(activation);
        }

        let delta = mult(
            &self.cost_derivative(activations.last().unwrap(), &y),
            &sigmoid_vec(zs.iter().last().unwrap()),
        );

        let idx = nabla_b.len() - 1;
        nabla_b[idx] = delta;
    }

    fn cost_derivative(self, output_activation: &Vec<f64>, y: &Vec<f64>) -> Vec<f64> {
        sub_vec(output_activation, y)
    }
}

fn main() {
    println!("Hello, world!");
}

#[cfg(test)]
mod tests {
    use neural_network::Matrix;
    use crate::{dot, Network, NetworkNew};

    #[test]
    fn dot_works() {
        let a = vec![vec![1, 2, 3], vec![4, 5, 6]];
        let b = vec![7, 8, 9];
        assert_eq!(dot(&a, &b), vec![50, 122]);
    }

    #[test]
    fn network_eval() {
        let b = NetworkNew::new(vec![2, 4, 3]);
        dbg!(&b);
        let result2 = b.feed_forward(Matrix::from_vec(vec![
            vec![0.5],
            vec![0.1],
        ]));
        println!("{}", &result2);
    }
}