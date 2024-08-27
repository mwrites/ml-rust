use std::rc::Rc;
use anyhow::Result;
use candle_core::{Device, Tensor};
use env_logger::Env;
use log::{debug, info};
use rand::prelude::*;
use clap::Parser;
use plotters::prelude::*;

#[derive(Parser, Debug)]

#[command(author, version, about, long_about = None)]
struct Args {
    #[arg(short = 's', long, default_value_t = 300)]
    n_samples: usize,

    #[arg(short = 'c', long, default_value_t = 3)]
    n_clusters: usize,

    #[arg(short = 'f', long, default_value_t = 2)]
    n_features: usize,

    #[arg(short = 'i', long, default_value_t = 15)]
    n_iterations: usize,
}


fn main() -> Result<()> {
    env_logger::Builder::from_env(Env::default().default_filter_or("info")).init();
    
    let args = Args::parse();
    info!("args: {:?}", args);

    let device = Rc::new(Device::cuda_if_available(0)?);
    info!("Using device: {:?}", device);

    let x = Tensor::randn(0f32, 1f32, (args.n_samples, args.n_features), &device)?;
    // let scale = Tensor::full(3f32, (args.n_samples, args.n_features), &device)?;
    // let x = x.mul(&scale)?;
    info!("x: {:?}", x);

    let rand_indices: Vec<usize> = (0..args.n_samples).choose_multiple(&mut thread_rng(), args.n_clusters);
    let mut centroids = Tensor::stack(
        &rand_indices
            .iter()
            .map(|&idx| x.get(idx).unwrap())
            .collect::<Vec<_>>(),
        0,
    )?;
    debug!("centroids: {:?}", centroids);

    let x_vec: Vec<f32> = x.to_vec2()?.into_iter().flatten().collect();
    let x_min = x_vec.iter().step_by(2).min_by(|a, b| a.partial_cmp(b).unwrap()).unwrap();
    let x_max = x_vec.iter().step_by(2).max_by(|a, b| a.partial_cmp(b).unwrap()).unwrap();
    let y_min = x_vec.iter().skip(1).step_by(2).min_by(|a, b| a.partial_cmp(b).unwrap()).unwrap();
    let y_max = x_vec.iter().skip(1).step_by(2).max_by(|a, b| a.partial_cmp(b).unwrap()).unwrap();

    let root = BitMapBackend::new("kmeans_evolution.png", (800, 600)).into_drawing_area();
    root.fill(&WHITE)?;

    let mut chart = ChartBuilder::on(&root)
        .caption("K-means Evolution", ("sans-serif", 50).into_font())
        .margin(5)
        .x_label_area_size(30)
        .y_label_area_size(30)
        .build_cartesian_2d(*x_min..(*x_max + 0.1), *y_min..(*y_max + 0.1))?;

    // Log initial centroids
    let initial_centroids = centroids.to_vec2()?.into_iter().flatten().collect::<Vec<f32>>();
    println!("Initial centroids: {:?}", initial_centroids);

    for iteration in 0..=args.n_iterations {
        let distances = x.unsqueeze(1)?
            .broadcast_sub(&centroids.unsqueeze(0)?)?
            .powf(2.0)?
            .sum_keepdim(2)?
            .sqrt()?;

        let labels = distances.argmin(1)?;
        debug!("labels: {:?}", labels);

        let mut mut_centroids = Vec::<Tensor>::with_capacity(args.n_clusters);
        for k in 0..args.n_clusters {
            // Create a boolean mask for the current cluster
            let mask = labels.eq(k as u32)?.flatten(0, 1)?;

            // Manually convert the boolean mask to indices
            let mut indices = Vec::new();
            for i in 0..mask.dim(0)? {
                if mask.get(i)?.to_scalar::<u8>()? == 1 {
                    indices.push(i as u32);
                }
            }
            let indices_tensor = Tensor::from_vec(indices.clone(), (indices.len(),), &device)?;
            let cluster_points = x.index_select(&indices_tensor, 0)?;

            // calculate the mean of the cluster points
            let feature_means = cluster_points.mean(0)?;
            debug!("feature_means: {:?}", feature_means);
            assert!(feature_means.dims() == [2], "Mean should be a point");

            // update the k-th centroid means
            mut_centroids.push(feature_means);
        }
        centroids = Tensor::stack(&mut_centroids, 0)?;
        info!("centroids: {:?}", centroids.to_vec2::<f32>().unwrap());
        assert!(centroids.dims() == [args.n_clusters, args.n_features], "Centroids shape mismatch");

        // Visualization for each iteration
        chart.draw_series(x_vec.chunks(2).map(|point| Circle::new((point[0], point[1]), 2, BLUE)))?;
        
        // Draw initial centroids in black (larger size)
        chart.draw_series(initial_centroids.chunks(2).map(|point| Circle::new((point[0], point[1]), 3, BLACK.filled())))?;

        // Draw current centroids in red (growing size)
        let current_centroids = centroids.to_vec2()?.into_iter().flatten().collect::<Vec<f32>>();
        chart.draw_series(current_centroids.chunks(2).map(|point| Circle::new((point[0], point[1]), 8, RED.filled())))?;

        root.present()?;

        // Log current centroids on the last iteration
        if iteration == args.n_iterations {
            println!("Final centroids: {:?}", current_centroids);
        }

        if iteration < args.n_iterations {
            chart.plotting_area().fill(&WHITE)?;
        }
    }

    Ok(())
}