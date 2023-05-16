use std::fmt;
use std::ops::{Add, Mul, Sub};
use rand::distributions::Distribution;
use rand::thread_rng;
use rand_distr::Normal;

#[derive(Debug, PartialEq)]
pub struct Matrix {
    pub num_rows: usize,
    pub num_cols: usize,
    pub storage: Box<[f64]>,
}

impl Matrix {
    pub fn with_shape(other: &Self) -> Self {
        Matrix::with_size(other.num_rows, other.num_cols)
    }

    pub fn with_size(num_rows: usize, num_cols: usize) -> Self {
        Matrix {
            num_rows,
            num_cols,
            storage: vec![0.; num_cols * num_rows].into_boxed_slice(),
        }
    }

    pub fn from_vec(vec: Vec<Vec<f64>>) -> Self {
        Matrix {
            num_rows: vec.len(),
            num_cols: vec[0].len(),
            storage: vec.into_iter().flatten().collect::<Vec<f64>>().into_boxed_slice(),
        }
    }

    pub fn with_random_values(num_rows: usize, num_cols: usize) -> Self {
        let storage_size = num_rows * num_cols;
        let mut storage = Vec::with_capacity(storage_size);

        let mut rand = thread_rng();
        let normal = Normal::new(0., 1.).unwrap();

        for _ in 0..storage_size {
            storage.push(normal.sample(&mut rand));
        }

        Matrix {
            num_rows,
            num_cols,
            storage: storage.into_boxed_slice(),
        }
    }

    pub fn apply(&self, fun: fn(f64) -> f64) -> Self {
        let mut new_storage = Vec::with_capacity(self.num_rows * self.num_cols);
        for a in self.storage.iter() {
            new_storage.push(fun(*a));
        }

        Matrix {
            num_rows: self.num_rows,
            num_cols: self.num_cols,
            storage: new_storage.into_boxed_slice(),
        }
    }

    pub fn get(&self, row: usize, col: usize) -> f64 {
        self.storage[row * self.num_cols + col]
    }

    pub fn dot(&self, other: &Self) -> Self {
        assert_eq!(self.num_cols, other.num_rows);

        let mut new_storage = Vec::with_capacity(other.num_rows * other.num_cols);
        for i in 0..self.num_rows {
            for j in 0..other.num_cols {
                let mut acc = 0.;
                for k in 0..self.num_cols {
                    acc += self.get(i, k) * other.get(k, j);
                }
                new_storage.push(acc);
            }
        }
        Matrix {
            num_rows: self.num_rows,
            num_cols: other.num_cols,
            storage: new_storage.into_boxed_slice(),
        }
    }

    fn do_op(&self, other: &Self, op: fn(f64, f64) -> f64) -> Self {
        assert_eq!(self.num_rows, other.num_rows, "Matrices need to have same amount of rows.");
        assert_eq!(self.num_cols, other.num_cols, "Matrices need to have same amount of columns.");

        let mut new_storage = Vec::with_capacity(self.num_rows * self.num_cols);
        for (a, b) in self.storage.iter().zip(other.storage.iter()) {
            new_storage.push(op(*a, *b));
        }
        Matrix {
            num_rows: self.num_rows,
            num_cols: self.num_cols,
            storage: new_storage.into_boxed_slice(),
        }
    }

    fn do_op_const(&self, val: f64, op: fn(f64, f64) -> f64) -> Self {
        let mut new_storage = Vec::with_capacity(self.num_rows * self.num_cols);
        for a in self.storage.iter() {
            new_storage.push(op(*a, val));
        }
        Matrix {
            num_rows: self.num_rows,
            num_cols: self.num_cols,
            storage: new_storage.into_boxed_slice(),
        }
    }
}

impl Add for Matrix {
    type Output = Self;

    fn add(self, other: Self) -> Self {
        self.do_op(&other, |a, b| a + b)
    }
}

impl Add<f64> for Matrix {
    type Output = Self;

    fn add(self, other: f64) -> Self {
        self.do_op_const(other, |a, b| a + b)
    }
}

impl Add<&Matrix> for &Matrix {
    type Output = Matrix;

    fn add(self, rhs: &Matrix) -> Self::Output {
        self.do_op(rhs, |a, b| a + b)
    }
}

impl Sub for Matrix {
    type Output = Self;

    fn sub(self, other: Self) -> Self {
        self.do_op(&other, |a, b| a - b)
    }
}

impl Sub<f64> for Matrix {
    type Output = Self;

    fn sub(self, other: f64) -> Self {
        self.do_op_const(other, |a, b| a - b)
    }
}

impl Mul for Matrix {
    type Output = Self;

    fn mul(self, other: Self) -> Self {
        self.do_op(&other, |a, b| a * b)
    }
}

impl Mul<f64> for Matrix {
    type Output = Self;

    fn mul(self, other: f64) -> Self {
        self.do_op_const(other, |a, b| a * b)
    }
}

impl fmt::Display for Matrix {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let mut lines: Vec<String> = Vec::new();
        for row in 0..self.num_rows {
            let mut values: Vec<String> = Vec::new();
            for col in 0..self.num_cols {
                values.push(self.get(row, col).to_string());
            }
            lines.push(values.join(", "))
        }
        write!(f, "{}", lines.join("\n"))
    }
}

#[cfg(test)]
mod tests {
    use crate::Matrix;

    #[test]
    fn adds() {
        let m1 = Matrix::from_vec(
            vec![
                vec![1., 2., 3.],
                vec![4., 5., 6.],
            ]
        );
        let m2 = Matrix::from_vec(
            vec![
                vec![3., 6., 8.],
                vec![1., 1., 9.],
            ]
        );

        let expected = Matrix::from_vec(
            vec![
                vec![4., 8., 11.],
                vec![5., 6., 15.],
            ]
        );

        let result = m1 + m2;
        assert_eq!(result, expected);
    }

    #[test]
    fn gets() {
        let m = Matrix::from_vec(
            vec![
                vec![1., 2., 3.],
                vec![4., 5., 6.],
                vec![4., 8., 11.],
            ]
        );

        assert_eq!(m.get(0, 0), 1.);
        assert_eq!(m.get(0, 1), 2.);
        assert_eq!(m.get(0, 2), 3.);
        assert_eq!(m.get(1, 0), 4.);
        assert_eq!(m.get(1, 1), 5.);
        assert_eq!(m.get(1, 2), 6.);
        assert_eq!(m.get(2, 0), 4.);
        assert_eq!(m.get(2, 1), 8.);
        assert_eq!(m.get(2, 2), 11.);
    }

    #[test]
    fn prints() {
        let m = Matrix::from_vec(
            vec![
                vec![1., 2., 3.],
                vec![4., 5., 6.],
                vec![4., 8., 11.],
            ]
        );
        print!("{}", m)
    }

    #[test]
    fn dot_1() {
        let a = Matrix::from_vec(vec![
            vec![1., 2.],
            vec![4., 5.],
        ]);
        let b = Matrix::from_vec(vec![
            vec![7., 3.],
            vec![8., 4.],
        ]);
        let expected = Matrix::from_vec(vec![
            vec![23., 11.],
            vec![68., 32.],
        ]);

        let result = a.dot(&b);
        assert_eq!(expected, result);
    }

    #[test]
    fn dot_2() {
        let a = Matrix::from_vec(vec![
            vec![1., 2.],
            vec![3., 4.],
            vec![5., 6.],
        ]);
        let b = Matrix::from_vec(vec![
            vec![7., 8., 9.],
            vec![10., 11., 12.],
        ]);
        let expected = Matrix::from_vec(vec![
            vec![27., 30., 33.],
            vec![61., 68., 75.],
            vec![95., 106., 117.],
        ]);

        let result = a.dot(&b);
        assert_eq!(expected, result);
    }
}
