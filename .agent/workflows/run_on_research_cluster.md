---
description: Workflow for setting up and running the training on the research cluster
---

# Run on Research Cluster

This workflow guides you through setting up the data, preparing it, and running the training on the research computer.

## 1. Transfer Data
Place your full 15GB dataset into `data/mathwriting-2024-full`.
The structure should look like:
```
data/mathwriting-2024-full/
  train/
    ...inkml files...
  valid/
    ...inkml files...
  ...
```

## 2. Prepare Data
Run the data preparation script. This will scan the `train` and `valid` directories, parse the labels directly from the InkML headers, and create the necessary NumPy arrays for training. It utilizes all available CPU cores to speed up processing.

```bash
python3 prepare_latex_data.py --data_dir data/mathwriting-2024-full --splits train,valid
```

## 3. Train Model
Run the training script. This script is already configured to use **GPU ID 3** only.
It will load the processed data from `data/processed_latex` (generated in the previous step).

```bash
python3 train_latex.py
```

*Note: You can adjust hyperparameters like `--batch_size` or `--epochs` if needed, e.g., `python3 train_latex.py --batch_size 64`.*

## 4. Monitor Training
The script prints progress to the console. Checkpoints are saved to `tf_ckpts/` and the final model weights to `handwriting_model.weights.h5`.

## 5. Generate Samples
Once training is complete (or if you want to test a checkpoint), use the generation script.

```bash
python3 generate.py "\sum_{i=1}^{n} i^2" --output_path sample.png
```

This will create `sample.png` with the generated handwriting.
