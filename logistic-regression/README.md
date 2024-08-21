# Logistic Regression AdamW

Multi Class Logistic Regression

- Dataset: https://huggingface.co/datasets/ylecun/mnist
- Competition: https://www.kaggle.com/competitions/digit-recognizer/


Metal
```bash
❯ cargo run
```

Cuda
```bash
❯ cargo run --features cuda
```


```
     Running `target/debug/logistic-regression`
[2024-08-21T20:44:06Z INFO  logistic_regression] Arguments parsed: Args { progress: false, learning_rate: 0.003, regularization: 0.1, epochs: 100, batch_size: 128 }
[2024-08-21T20:44:06Z INFO  logistic_regression] Using device: Cpu
[2024-08-21T20:44:15Z INFO  logistic_regression] Dataset loaded
[2024-08-21T20:44:15Z INFO  logistic_regression] Number of features per image: 784
[2024-08-21T20:44:17Z INFO  logistic_regression] Data moved to device and normalized
[2024-08-21T20:44:17Z INFO  logistic_regression] Model initialized
[2024-08-21T20:44:17Z INFO  logistic_regression] Training size: 60000, Number of batches: 468
[2024-08-21T20:44:21Z INFO  logistic_regression] Epoch: 0, Train Loss: 0.39955, Train Accuracy: 0.88061, Val Loss: 0.32536, Val Accuracy: 0.90580
```

Help
```bash
❯ cargo run -- -h
```
