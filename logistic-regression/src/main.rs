mod prediction;
use prediction::predict_and_save;

extern crate csv;
use anyhow::Result;
use candle_core::{DType, Device, Tensor};
use candle_nn::{loss, ops, Module, Optimizer, VarBuilder, VarMap};
use clap::Parser;
use log::{debug, info};
use rand::prelude::*;
use std::rc::Rc;

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    #[arg(long, default_value_t = false)]
    progress: bool,
    #[arg(long, default_value = "0.003")]
    learning_rate: f32,
    #[arg(long, default_value = "0.1")]
    regularization: f32,
    #[arg(long, default_value = "100")]
    epochs: i32,
    #[arg(long, default_value = "128")]
    batch_size: usize,
}

struct LogisticRegression {
    linear: candle_nn::Linear,
}

impl LogisticRegression {
    fn new(vs: VarBuilder, feature_cnt: usize) -> Result<Self> {
        let linear = candle_nn::linear(feature_cnt, 10, vs.pp("linear"))?;
        Ok(Self { linear })
    }

    fn loss_and_accuracy(&self, logits: &Tensor, targets: &Tensor) -> Result<(Tensor, f32)> {
        let log_softmax = ops::log_softmax(logits, 1)?;
        let loss = loss::nll(&log_softmax, targets)?;
        let predicted_classes = logits.argmax(1)?;
        let targets_u32 = targets.to_dtype(DType::U32)?;
        let correct_predictions = predicted_classes.eq(&targets_u32)?;
        let accuracy = correct_predictions
            .to_dtype(DType::F32)?
            .mean_all()?
            .to_scalar::<f32>()?;
        Ok((loss, accuracy))
    }
}

impl Module for LogisticRegression {
    fn forward(&self, x: &Tensor) -> Result<candle_core::Tensor, candle_core::Error> {
        Ok(self.linear.forward(x)?)
    }
}

struct MnistNormalizer {
    image_mean: Vec<f32>,
    image_std: Vec<f32>,
}

impl MnistNormalizer {
    fn new() -> Self {
        MnistNormalizer {
            image_mean: vec![0.1307],
            image_std: vec![0.3081],
        }
    }

    pub fn normalize(&self, tensor: &Tensor) -> Result<Tensor> {
        let device = tensor.device();
        let mean = Tensor::from_vec(self.image_mean.clone(), (1,), &device)?;
        let std = Tensor::from_vec(self.image_std.clone(), (1,), &device)?;
        Ok(tensor
            .broadcast_sub(&mean)?
            .broadcast_div(&std)?
            .to_device(device)?)
    }
}

fn main() -> Result<()> {
    let args = Args::parse();
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();
    info!("Arguments parsed: {:?}", args);

    let device = Rc::new(Device::cuda_if_available(0)?);
    info!("Using device: {:?}", device);

    let dataset = candle_datasets::vision::mnist::load()?;
    info!("Dataset loaded");

    let (_, n) = dataset.train_images.shape().dims2()?;
    info!("Number of features per image: {}", n);

    let normalizer = MnistNormalizer::new();
    let training_images = normalizer.normalize(&dataset.train_images.to_device(&device)?)?;
    let training_labels = dataset.train_labels.to_device(&device)?;
    let val_images = normalizer.normalize(&dataset.test_images.to_device(&device)?)?;
    let val_labels = dataset.test_labels.to_device(&device)?;
    info!("Data moved to device and normalized");
    debug!(
        "val_images: {:?}, val_labels: {:.5}", 
        val_images, val_labels
    );

    let varmap = VarMap::new();
    let vs = VarBuilder::from_varmap(&varmap, DType::F32, &device);
    let model = LogisticRegression::new(vs, n)?;
    let mut opt = candle_nn::AdamW::new(
        varmap.all_vars(),
        candle_nn::ParamsAdamW {
            lr: args.learning_rate as f64,
            weight_decay: args.regularization as f64,
            ..Default::default()
        },
    )?;
    info!("Model initialized");

    let (training_size, _) = training_images.shape().dims2()?;
    let n_batches = training_size / args.batch_size;
    info!(
        "Training size: {}, Number of batches: {}",
        training_size, n_batches
    );

    let mut batch_idxs: Vec<usize> = (0..n_batches).collect();

    for epoch in 0..args.epochs {
        let mut sum_train_loss = 0.0;
        let mut sum_train_accuracy = 0.0;
        batch_idxs.shuffle(&mut rand::thread_rng());

        for (i, batch_idx) in batch_idxs.iter().enumerate() {
            let train_data =
                training_images.narrow(0, batch_idx * args.batch_size, args.batch_size)?;
            let train_labels =
                training_labels.narrow(0, batch_idx * args.batch_size, args.batch_size)?;

            let logits = model.forward(&train_data)?;
            let (loss, accuracy) = model.loss_and_accuracy(&logits, &train_labels)?;

            opt.backward_step(&loss)?;

            sum_train_loss += loss.to_scalar::<f32>()?;
            sum_train_accuracy += accuracy;

            if i % 300 == 0 {
                info!("Epoch {}, Batch {}/{}", epoch, i, n_batches);
            }
        }

        let avg_train_loss = sum_train_loss / n_batches as f32;
        let avg_train_accuracy = sum_train_accuracy / n_batches as f32;

        let val_logits = model.forward(&val_images)?;
        let (val_loss, val_accuracy) = model.loss_and_accuracy(&val_logits, &val_labels)?;

        info!(
            "Epoch: {}, Train Loss: {:.5}, Train Accuracy: {:.5}, Val Loss: {:.5}, Val Accuracy: {:.5}",
            epoch, avg_train_loss, avg_train_accuracy, val_loss, val_accuracy
        );
    }

    info!("Training completed");

    predict_and_save(&model, &device, "test.csv", "submission", "subm_logreg")?;

    Ok(())
}
