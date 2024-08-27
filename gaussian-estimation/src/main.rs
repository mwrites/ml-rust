mod plot;

use anyhow::Result;
use ndarray::{Array1, Array2, Axis};
use ndarray_linalg::{Inverse, Determinant};
use rand::Rng;
use rand_distr::{Distribution, Normal};
use std::f64::consts::PI;

fn estimate_gaussian(x: &Array2<f64>) -> (Array1<f64>, Array2<f64>) {
    let mu = x.mean_axis(Axis(0)).unwrap();
    let centered = x - &mu;
    let cov = centered.t().dot(&centered) / (x.nrows() as f64 - 1.0);
    (mu, cov)
}

fn multivariate_gaussian(x: &Array2<f64>, mu: &Array1<f64>, cov: &Array2<f64>) -> Result<Array1<f64>> {
    let k = mu.len() as f64;
    let x_minus_mu = x - mu;
    let cov_inv = cov.inv()?;
    let term = x_minus_mu.dot(&cov_inv).dot(&x_minus_mu.t());
    let det = cov.det()?;
    let coeff = 1.0 / ((2.0 * PI).powf(k / 2.0) * det.sqrt());
    Ok(term.diag().mapv(|x| (-0.5 * x).exp() * coeff))
}

fn main() -> Result<()> {
    let mut rng = rand::thread_rng();
    let normal = Normal::new(14.0, 1.0)?;
    
    let mut data_points: Vec<[f64; 2]> = (0..1000)
        .map(|_| [normal.sample(&mut rng), normal.sample(&mut rng)])
        .collect();

    // Add outliers
    data_points.extend((0..10).map(|_| [rng.gen_range(15.5..16.5), rng.gen_range(15.5..16.5)]));

    let x = Array2::from_shape_vec((data_points.len(), 2), data_points.into_iter().flatten().collect())?;
    let (mu, cov) = estimate_gaussian(&x);

    let p = multivariate_gaussian(&x, &mu, &cov)?;
    let threshold = p.iter().fold(f64::INFINITY, |a, &b| a.min(b)) + 0.01;

    plot::visualize_fit(&x, &mu, &cov, threshold)?;

    println!("Mean: {:?}", mu);
    println!("Covariance: {:?}", cov);
    println!("Threshold: {}", threshold);

    Ok(())
}