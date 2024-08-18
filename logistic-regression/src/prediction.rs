use anyhow::{Context, Result};
use arrow::array::UInt32Array;
use arrow::datatypes::{DataType, Field, Schema};
use arrow::record_batch::RecordBatch;
use candle_core::{DType, Device, Tensor};
use candle_nn::{ops, Module};
use csv::WriterBuilder;
use log::info;
use parquet::arrow::arrow_writer::ArrowWriter;
use parquet::file::properties::WriterProperties;
use std::fs::File;
use std::sync::Arc;
use std::path::PathBuf;


pub fn load_test_data(csv_path: &str, device: &Device) -> Result<Tensor> {
    let mut data = Vec::new();
    let mut rdr = csv::Reader::from_path(csv_path)
        .with_context(|| format!("Failed to open CSV file: {}", csv_path))?;

    for (i, result) in rdr.records().enumerate() {
        let record = result.with_context(|| format!("Failed to read record at line {}", i + 1))?;
        let row: Vec<f32> = record
            .iter()
            .enumerate()
            .map(|(j, field)| {
                field
                    .parse::<f32>()
                    .with_context(|| {
                        format!("Failed to parse value at line {}, column {}", i + 1, j + 1)
                    })
                    .map(|v| v / 255.0)
            })
            .collect::<Result<_, _>>()?;
        data.push(row);
    }

    let num_rows = data.len();
    let num_cols = data
        .get(0)
        .map(|row| row.len())
        .context("CSV file is empty")?;
    let flat_data: Vec<f32> = data.into_iter().flatten().collect();

    Tensor::from_slice(&flat_data, (num_rows, num_cols), device)
        .context("Failed to create tensor from data")
}

pub fn save_predictions(
    predictions: &[u32],
    sub_filename: &str,
    sub_filename2: &str,
) -> Result<()> {
    let schema = Arc::new(Schema::new(vec![
        Field::new("ImageId", DataType::UInt32, false),
        Field::new("Label", DataType::UInt32, false),
    ]));

    let batch = RecordBatch::try_new(
        schema.clone(),
        vec![
            // Submission file starts from id 1 not 0
            Arc::new(UInt32Array::from((1..=predictions.len() as u32).collect::<Vec<u32>>())),
            Arc::new(UInt32Array::from(predictions.to_vec())),
        ],
    )?;

    let mut files = Vec::new();

    for name in [sub_filename, sub_filename2] {
        // Parquet file
        let parquet_path = PathBuf::from(format!("{}.parquet", name));
        let file = File::create(&parquet_path)?;
        let mut writer = ArrowWriter::try_new(file, schema.clone(), Some(WriterProperties::builder().build()))?;
        writer.write(&batch)?;
        writer.close()?;
        files.push(parquet_path);

        // CSV file
        let csv_path = PathBuf::from(format!("{}.csv", name));
        let mut csv_writer = WriterBuilder::new().from_path(&csv_path)?;
        csv_writer.write_record(&["ImageId", "Label"])?;
        for (id, &pred) in predictions.iter().enumerate() {
            csv_writer.write_record(&[(id + 1).to_string(), pred.to_string()])?;
        }
        csv_writer.flush()?;
        files.push(csv_path);
    }

    info!("Created submission files:");
    for file in &files {
        info!("  - {:?}", file);
    }

    Ok(())
}


pub fn predict_and_save<M: Module>(
    model: &M,
    device: &Device,
    test_file: &str,
    sub_filename: &str,
    sub_filename2: &str,
) -> Result<()> {
    let test_data = load_test_data(test_file, device)?;
    let test_predictions = model.forward(&test_data)?;

    let probabilities = ops::softmax(&test_predictions, 1)?;
    let predicted_classes = probabilities.argmax(1)?;
    let predicted_classes_vec: Vec<u32> = predicted_classes.to_dtype(DType::U32)?.to_vec1()?;

    save_predictions(&predicted_classes_vec, sub_filename, sub_filename2)
        .context("Failed to save predictions")?;

    Ok(())
}
