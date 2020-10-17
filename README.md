# Neural Reverse Engineering of Stripped Binaries using Augmented Control Flow Graphs 

This is the official implementation of `Nero-GNN`, the prototype described in: [Yaniv David](http://www.cs.technion.ac.il/~yanivd/), [Uri Alon](http://urialon.cswp.cs.technion.ac.il), and [Eran Yahav](http://www.cs.technion.ac.il/~yahave/), "Neural Reverse Engineering of Stripped Binaries using Augmented Control Flow Graphs״, will appear in OOPSLA '2020, [PDF](https://arxiv.org/abs/1902.09122). 

Our evaluation dataset and other resources are available [here](https://doi.org/10.5281/zenodo.4081641) (Zenodo). These will be used and further explained next.

<center style="padding: 40px"><img width="90%" src="https://github.com/urialon/rPigeoNN/raw/labels2_uri_gnn_lstm_refactor/images/ACSG.png" /></center>

## Requirements
  * [python3.6](https://www.linuxbabe.com/ubuntu/install-python-3-6-ubuntu-16-04-16-10-17-04) 
  * TensorFlow 1.13.1 ([install](https://www.tensorflow.org/install/install_linux)) or using:

```bash
pip install tensorflow-gpu==1.13.1 # for the GPU version
```

Note that CUDA >= 10.1 is required for tensorflow-gpu version 1.13 and above. ([See this link for more information](https://www.tensorflow.org/install/gpu#install_cuda_with_apt))

or:

```bash
pip install tensorflow==1.13.1 # for the CPU version
```
    
To check existing TensorFlow version, run:
```bash
python3 -c 'import tensorflow as tf; print(tf.__version__)'
```
 
## Generating Representations for Binary Procedures

[Our dataset](https://zenodo.org/record/4099685/files/nero_dataset_binaries.tar.gz) was created by compiling several GNU source-code packages into binary executables. 
The packages are split into three sets: training, validation and test (each in its own directory in the extracted archive).

| :construction:        | We are working on sharing a stable and easy to use version of our binary representations generation system. <BR> Stay tuned for updates on this repository.        | :construction:        |
|---------------|:------------------------:|---------------|


Performing a thorough cleanup and deduplication process (detailed in
[our paper](https://arxiv.org/abs/1902.09122)) resulted in a dataset containing
67,246 samples. The procedure representations for these samples can be found
[in this archive](https://zenodo.org/record/4095276/files/procedure_representations.tar.gz).

Extracting the procedure representations archive will create the folder `procedure_representations` and inside it two more folders:
1. `raw`: The raw representations for all the binary procedures in the above dataset. Each procedure is represented by one line in the relevant file for each set (training.json, validation.json and test.json) 
1. `preprocessed`: The raw procedure representations preprocessed for training.

The `preprocessed` directory contains:
1. Files for training the model: `data.dict` and `data.train` (the dictionary and preprocessed training set samples accordingly)
1. `data.val` - The (preprocessed) validation set samples.
1. `data.test` - The (preprocessed) test set samples.

## Training New Models

As we show in [our paper](https://arxiv.org/abs/1902.09122), `Nero-GNN` is the best variation of our approach, and so we focus on and showcase it here.

### Training from scratch

Training a `Nero-GNN` model is performed by running the following command line:
```bash
python3 -u nero.py --data procedure_representations/processed/data \
--test procedure_representations/processed/data.val --save new_model/model \
--gnn_layers NUM_GNN_LAYERS
```

Where `NUM_GNN_LAYERS` is the number of GNN layers. In the paper, we found `NUM_GNN_LAYERS=4` to perform best.
The paths to the (training) `--data` and (validation) `--test` arguments can be changed to point to a new dataset.
Here, we provide the dataset that [we used in the paper](#generating-representations-for-binary-procedures). 

### Trained models

Trained models are available [in this archive](https://zenodo.org/record/4095276/files/nero_gnn_model.tar.gz).
Extracting it will create the `gnn` directory composed of:
1. Trained model (the `dictionaries.bin` & `model_iter495.*` files, storing the 495th training iteration)
1. Training log.
1. Prediction results log.

## Evaluation

Evaluation of a trained model is performed using the following command line: 
```bash
python3 -u nero.py --test procedure_representations/data.test \
--load gnn/model_iter495 \
--gnn_layers NUM_GNN_LAYERS
```

if `model_iter495` is the checkpoint that performed best on the validation set during training (this is the case in the provided [trained model](#trained-models)).
The value of `NUM_GNN_LAYERS` should be the same as in training.

### Additional Flags
* Use the `--no_arg` flag during training **and** testing, to train a "no-values" model (as in Table 4 in [our paper](https://arxiv.org/abs/1902.09122))
* Use the `--no_api` flag during training **and** testing, to train an "obfuscated" model (as in Table 2 in [our paper](https://arxiv.org/abs/1902.09122)) - a model that does not use the API names (assuming they are obfuscated).


### Understanding the prediction process and its results

This section provides a name prediction walkthrough for an example from our test set ([further explained here](#generating-representations-for-binary-procedures).
For readability, we start straight from the graph representation (similar to the one depicted in Fig.2(c) in [our paper](https://arxiv.org/abs/1902.09122)) and skip the rest of the steps.

The `get_tz` procedure from the `find` executable is part of `findutils` package.
This procedure is represented as a json found at line 1715 in `procedure_representations/raw/test.json`.

This json can be pretty-printed by running:
```bash
awk 'NR==1715' procedure_representations/raw/test.json | python -m json.tool
```

This json represents the procedure's graph:
* The graph nodes are basic blocks named `ob<x>` (where x is a number with an optional postfix, e.g., `initialize`).
* The json contains data regarding edges between the nodes, and the abstracted call sites in each node.
* In this json we see that the first node, `ob-1.initialize`, contains a call to the External api call `getenv` marked by `Egetenv`. This api call is made with the argument which was resolved to the concrete string `TZ`.
* Other calls to `memcpy` and `strlen` are made with the abstract value `CONST` as their argument. This `CONST` abstract value is called `STK` in the paper (see page 14).
* Note that `Nxmemdup` is a Normal (internal) call. This name was taken from the debug information and kept here for our debugging purposes and is stripped again before being used in the training/prediction steps.

The name prediction for this procedure by our Nero-GNN can be found in line 876 of the models prediction log file:

```bash
head -n 1 gnn_model/predictions_iter495_F1_45.5.txt  && awk 'NR==876' gnn_model/predictions_iter495_F1_45.5.txt
```

Which results in:
```csv
PredCode,package,Original,Predicted,Thrown
[+],get*tz@find@findutils,get*tz,get*tz,['BLANK'; 'BLANK'; 'BLANK'; 'BLANK']
```

This line starts with the prediction code: `+`,`±` or `-` for full, partial, or no match accordingly. Then, the procedure information, ground truth, prediction and truncated sub-tokens follow:

* **Procedure name:** `get_tz`
* **Executable:** `find`
* **Package name:** `findutils`
* **Ground truth procedure name:** `['get', 'tz']`
* **Model's prediction:** `['get', 'tz']`
* **Truncated suffix subtokens:** `['BLANK', 'BLANK', 'BLANK', 'BLANK']`

Note that we truncate the prediction after the first `BLANK` or `UNKNOWN` sub-token prediction. 
