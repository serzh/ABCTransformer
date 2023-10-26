# ABC Transformer

Welcome to the ABC Transformer repository! This repository houses a character-level model for ABC music notation, implemented using the Transformer architecture. I drew inspiration from Andrej Karpathy's NanoGPT project, which you can find [here](https://github.com/karpathy/nanoGPT).

## Getting Started

To run the ABC Transformer, follow these simple steps:

### 1. Installing dependencies

```bash
poetry install
poetry shell
```

### 2. Prepare the Dataset

First, you need to acquire the dataset. You can obtain the same dataset I used from the following website: [ABC Music Notation Dataset](http://www.atrilcoral.com/Partituras_ABC/index_abc.htm). Once you've downloaded the dataset, unzip all the archives into the `data/` directory.

Afterward, run the following command to prepare the data:

```bash
python prepare.py
```

This command will create two essential files: `itos.pkl`, which contains a dictionary for decoding ABC notation, and `corpus.npz`, an npz file containing training and validation datasets.

### 3. Train from Scratch

Now, you're ready to train your ABC Transformer model from scratch. Execute the following command:

```bash
python train.py new
```

This will initiate the training process, and a checkpoint file will be generated upon completion.

### 4. Sample from a Checkpoint

You can generate music samples from a trained checkpoint using the following command:

```bash
python train.py sample checkpoint-v0.pt $'S:4\n' 12
```

This command will produce a sample, and you can provide your own input and seed for customization. Input and seed are both optional.

### 5. Train from a Checkpoint

If you want to continue training your model for additional steps, open the 'train.py' file and increase the value of `max_steps`. Afterward, run the following command to continue training from a checkpoint:

```bash
python train.py cont <checkpoint-file>
```

Feel free to modify the model parameters by editing the 'train.py' file. You can find all the parameter declarations at the top of the file. Here's a configuration that worked well for me:

```python
# Add your recommended parameters here
```

Enjoy exploring the world of ABC music notation with the ABC Transformer! If you have any questions or need assistance, don't hesitate to reach out.

Happy music generation! 🎶