# Neural Reverse Engineering of Stripped Binaries using Augmented Control Flow Graphs 

This is the official implementation of `Nero-GNN`, the prototype described in: [Yaniv David](http://www.cs.technion.ac.il/~yanivd/), [Uri Alon](http://urialon.cswp.cs.technion.ac.il), and [Eran Yahav](http://www.cs.technion.ac.il/~yahave/), "Neural Reverse Engineering of Stripped Binaries using Augmented Control Flow Graphs״, will appear in OOPSLA '2020, [PDF](https://arxiv.org/abs/1902.09122). 

Our evaluation dataset and other resources are available [here](https://doi.org/10.5281/zenodo.4081641) (Zenodo). These will be used and further explained next.

![An overview of the data-gen process](https://github.com/tech-srl/Nero/blob/main/images/ACSG.png?raw=true "Data-generation Process")

Table of Contents
=================
  * [Requirements](#requirements)
  * [Data generation](#generating-representations-for-binary-procedures)
  * [GNN neural model](#predicting-procedure-names-using-neural-models)
  * [Evaluation](#evaluation)
  * [Citation](#citation)

## Requirements 

### Data Generation Specific Requirements

* [python3.8](https://www.python.org/downloads/)
* [LLVM version 10](https://llvm.org/docs/GettingStarted.html) and the llvmlite & llvmcpy python packages (other versions might work. 3.x will not).
* [IDA-PRO](https://www.hex-rays.com/products/ida/) (tested with version 6.95).
* [angr](http://angr.io), and the simuvex package. 
* A few more python packages: scandir, tqdm, jsonpickle, parmap, python-magic, pyelftools, setproctitle.

Using a licensed IDA-PRO installation for Linux, all of these requirements were verified as compatible for running on an Ubuntu 20 machine (and with some more effort even on Ubuntu 16).

For Ubuntu 20, you can use the `requirements.txt` file in this repository to install all python packages against the native python3.8 version:

```bash
pip3 install -r requirements.txt
```

LLVM version 10 can be installed with:
```bash
sudo apt get install llvm-10
```

The IDA-python scripts (in `datagen/ida/py2`) were tested against the python 2.7 version bundled with IDA-PRO 6.95, and should work with newer versions at least up-to 7.4 (more info [here](https://www.hex-rays.com/products/ida/support/ida74_idapython_python3.shtml)). Please [file a bug](https://github.com/tech-srl/Nero/issues) if it doesn't.

The jsonpickle python package also needs to be installed for use by this bundled python version:

1. Download the package: 
```bash 
wget https://files.pythonhosted.org/packages/32/d5/2f47f03d3f64c31b0d7070b488274631d7567c36e81a9f744e6638bb0f0d/jsonpickle-0.9.6.tar.gz
```
2. Extract only the package sources: 
```bash 
tar -xvf jsonpickle-0.9.6.tar.gz jsonpickle-0.9.6/jsonpickle/
```
3. Move it to the IDA-PRO python directory: 
```bash 
mv jsonpickle-0.9.6/jsonpickle /opt/ida-6.95/idal64/python/
```

Note that, when installed as root, IDA-PRO defaults to installing in `/opt/ida-6.95/idal64`. Other paths will require adjusting here and in other scripts.

### Neural Model Specific Requirements

  * [python3.6](https://www.python.org/downloads/). (For using the same Ubuntu 20 machine for training and data generation we recommend using [virtualenv](http://thomas-cokelaer.info/blog/2014/08/installing-another-python-version-into-virtualenv/))
  * These two python packages: jsonpickle, scipy
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

[Our binaries dataset](https://zenodo.org/record/4099685/files/nero_dataset_binaries.tar.gz) was created by compiling several GNU source-code packages into binary executables and performing a thorough cleanup and deduplication process (detailed in [our paper](https://arxiv.org/abs/1902.09122)).
 
The packages are split into three sets: training, validation and test (each in its own directory in the extracted archive: `TRAIN`, `VALIDATE` & `TEST` resp.).

To obtain preprocessed representations for these binaries you can either download our preprocessed dataset, or create a new dataset from our or any other binaries dataset.

### Creating Representations

#### Indexing

Indexing, i.e., analyzing the binaries and creating augmented control flow graphs based representations for them is performed using:

```bash
python3 -u index_binaries.py --input-dir TRAIN --output-dir TRAIN_INDEXED
```

where `TRAIN` is the directory holding the binaries to index, and results are placed in `TRAIN_INDEXED`. 

To index successfully, binaries must contain debug information and adhere to this file name structure: 
```
<compiler>-<compiler version>__O<Optimization level(u for default)>__<Package name>[-<optional package version>]__<Executable name> 
```
For example: "gcc-5__Ou__cssc__sccs".

Some notes on the indexing process and its results:
1. The indexing process might take several hours. We recommend running it on a machine with multiple CPU-cores and adequate RAM.
1. The number of procedures created might depend on the timeout value selected for procedure indexing (controlled by `--index-timeout` with the default of 30 minutes).
1. Procedures containing features not supported by the indexing engine (e.g., vector operations) or CFGs with more than 1000 unique CFG paths will not be indexed.
1. The created representations might have some minor discrepancies when compared with those published in zenodo. These include JSON field ordering and formating. These discrepancies are the result of porting this prototype to Python3 towards its publication.
1. To change the path to the IDA-PRO installation use `--idal64-path`.

#### Filter and collect

Next, to filter and collect all the indexed procedures into one JSON file:
```bash
python3 -u collect_and_filter.py --input-dir TRAIN_INDEXED --output-file=train.json
```

This will filter and collect indexed procedures from `TRAIN_INDEXED` (which should hold the indexed binaries for training from the last step) and store them in `train.json`.

#### Preprocess for use by the model

Finally, to preprocess raw representations, preparing them for use by the neural model, use:

```bash
python3 preprocess.py -trd train.json -ted test.json -vd validation.json -o data
```

This will preprocess the training(`train.json`), validation(`validation.json`) and test(`test.json`) files. Note that this step require TensorFlow and other components mentioned [here](#neural-model-specific-requirements).

### Using Prepared Representations

The procedure representations for the binaries in our dataset can be found
[in this archive](https://zenodo.org/record/4095276/files/procedure_representations.tar.gz).

Extracting the procedure representations archive will create the folder `procedure_representations` and inside it two more folders:
1. `raw`: The raw representations for all the binary procedures in the above dataset. Each procedure is represented by one line in the relevant file for each set (training.json, validation.json and test.json) 
1. `preprocessed`: The raw procedure representations preprocessed for training.

The `preprocessed` directory contains:
1. Files for training the model: `data.dict` and `data.train` (the dictionary and preprocessed training set samples accordingly)
1. `data.val` - The (preprocessed) validation set samples.
1. `data.test` - The (preprocessed) test set samples.

## Predicting Procedure Names Using Neural Models

As we show in [our paper](https://arxiv.org/abs/1902.09122), `Nero-GNN` is the best variation of our approach, and so we focus on and showcase it here.

### Training From Scratch

Training a `Nero-GNN` model is performed by running the following command line:
```bash
python3 -u gnn.py --data procedure_representations/processed/data \
--test procedure_representations/processed/data.val --save new_model/model \
--gnn_layers NUM_GNN_LAYERS
```

Where `NUM_GNN_LAYERS` is the number of GNN layers. In the paper, we found `NUM_GNN_LAYERS=4` to perform best.
The paths to the (training) `--data` and (validation) `--test` arguments can be changed to point to a new dataset.
Here, we provide the dataset that [we used in the paper](#generating-representations-for-binary-procedures). 

We trained our models using a `Tesla V100` GPU. Other GPUs might require changing the number of GNN layers or other dims to fit into the available RAM.

### Using Pre-Trained Models

Trained models are available [in this archive](https://zenodo.org/record/4095276/files/nero_gnn_model.tar.gz).
Extracting it will create the `gnn` directory composed of:
1. Trained model (the `dictionaries.bin` & `model_iter495.*` files, storing the 495th training iteration)
1. Training log.
1. Prediction results log.

## Evaluation

Evaluation of a trained model is performed using the following command line: 
```bash
python3 -u gnn.py --test procedure_representations/data.test \
--load gnn/model_iter495 \
--gnn_layers NUM_GNN_LAYERS
```

if `model_iter495` is the checkpoint that performed best on the validation set during training (this is the case in the provided [trained model](#trained-models)).
The value of `NUM_GNN_LAYERS` should be the same as in training.

### Additional Flags
* Use the `--no_arg` flag during training **and** testing, to train a "no-values" model (as in Table 4 in [our paper](https://arxiv.org/abs/1902.09122))
* Use the `--no_api` flag during training **and** testing, to train an "obfuscated" model (as in Table 2 in [our paper](https://arxiv.org/abs/1902.09122)) - a model that does not use the API names (assuming they are obfuscated).


### Understanding the Prediction Process and Its Results

This section provides a name prediction walk-through for an example from our test set ([further explained here](#generating-representations-for-binary-procedures).
For readability, we start straight from the graph representation (similar to the one depicted in Fig.2(c) in [our paper](https://arxiv.org/abs/1902.09122)) and skip the rest of the steps.

The `get_tz` procedure from the `find` executable is part of `findutils` package.
This procedure is represented as a json found at line 1715 in `procedure_representations/raw/test.json`.

This json can be pretty-printed by running:
```bash
awk 'NR==1715' procedure_representations/raw/test.json | python3 -m json.tool
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


## Citation 

[Neural Reverse Engineering of Stripped Binaries using Augmented Control Flow Graphs](https://arxiv.org/pdf/1902.09122)

```
@article{
    David2020,
    title = {Neural Reverse Engineering of Stripped Binaries Using Augmented Control Flow Graphs},
    author = {David, Yaniv and Alon, Uri and Yahav, Eran},
    doi = {10.1145/3428293},
    journal = {Proceedings of the ACM on Programming Languages},
    number = {OOPSLA},
    title = {{Neural reverse engineering of stripped binaries using augmented control flow graphs}},
    volume = {4},
    year = {2020}
}
```