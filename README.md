
# Environment Setup

## Create and activate the environment
conda create -n aul python=3.10 -y
conda activate aul

## Install PyTorch with CUDA 12.4 support
conda install pytorch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 pytorch-cuda=12.4 -c pytorch -c nvidia

## Install other dependencies
pip install -r requirements.txt

## How to run

python run.py --config configs/config.json

## Configuration File (`.json`)

All training options are specified in a JSON config. Here's a breakdown of key fields:

### General Settings

| Key                  | Description                                                                 |
|----------------------|-----------------------------------------------------------------------------|
| `mode`               | Mode of operation                                |
| `method_name`        | Unlearning method to use   |
| `loss_type`          | Loss function identifier                                                    |
| `do_init_eval`       | Whether to evaluate before training (used in unlearning)                    |

### Datasets

| Key                 | Description                                             |
|----------------------|---------------------------------------------------------|
| `train_set`        | Path to training data                                     |
| `valid_sets`       | List of validation set paths                              |
| `valid_subset_path`| List of subset info for validation                        |
| `valid_type_path`  | List of type info for validation                          |

### Training Hyperparameters

| Key                           | Description                       |
|-------------------------------|-----------------------------------|
| `num_train_epochs`            | Number of training epochs         |
| `train_batch_size`            | Batch size for training           |
| `eval_batch_size`             | Batch size for evaluation         |
| `gradient_accumulation_steps` | Number of accumulation steps      |
| `learning_rate`               | Learning rate                     |
| `ngpu`                        | Number of GPUs to use             |
| `num_workers`                 | Dataloader workers                |
| `check_val_every_n_epoch`     | Validate every N epochs           |

### Logging & Evaluation

| Key             | Description                                  |
|-----------------|----------------------------------------------|
| `wandb_log`     | Enable Weights & Biases logging              |
| `wandb_project` | Name of the wandb project                    |
| `wandb_run_name`| Specific run name                            |
| `fp16`          | Use 16-bit (mixed) precision training        |
| `strategy`      | PyTorch Lightning strategy (e.g., deepspeed) |

### Model & Input

| Key                   | Description                                  |
|------------------------|----------------------------------------------|
| `model_name_or_path` | Pretrained model path or model name          |
| `tokenizer_name_or_path` | Tokenizer path or name                  |
| `cache_dir`          | Huggingface cache directory                  |
| `input_length`       | Input sequence length                        |
| `output_length`      | Output sequence length                       |
| `target_length`      | Target label length (usually same as output) |

### Unlearning-specific Parameters

| Key              | Description                                      |
|------------------|--------------------------------------------------|
| `suppress_factor`| Factor to suppress forgotten knowledge           |
| `boost_factor`   | Factor to boost retained knowledge               |

