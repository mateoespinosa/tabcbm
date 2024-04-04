# TabCBM: Concept-based Interpretable Neural Networks for Tabular Data
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://github.com/mateoespinosa/tabcbm/blob/main/LICENSE) [![Python 3.7+](https://img.shields.io/badge/python-3.7+-green.svg)](https://www.python.org/downloads/release/python-370/) [![Paper](https://img.shields.io/badge/-Paper-red)](https://openreview.net/pdf?id=TIsrnWpjQ0) [![Poster](https://img.shields.io/badge/-Poster-yellow)](https://github.com/mateoespinosa/tabcbm/blob/main/media/poster.pdf)


![TabCBM Architecture](figures/tabcbm_architecture.png)


This repository contains the official implementation of our **TMLR** paper
[*"TabCBM: Concept-based Interpretable Neural Networks for Tabular Data"*](https://openreview.net/pdf?id=TIsrnWpjQ0) and its corresponding version in [**ICML's Workshop on Interpretable Machine Learning for Healthcare (IMLH 2023)**](https://openreview.net/forum?id=2YG1aTEaj4).

This work was done by [Mateo Espinosa Zarlenga](https://mateoespinosa.github.io/),
[Zohreh Shams](https://zohrehshams.com/),
[Michael Edward Nelson](https://uk.linkedin.com/in/michael-nelson-443029137),
[Been Kim](https://beenkim.github.io/),
and [Mateja Jamnik](https://www.cl.cam.ac.uk/~mj201/)


#### TL;DR

There has been significant efforts in recent years on designing neural
architectures that can explain their predictions using high-level units
of information referred to as "concepts". Nevertheless, these methods have thus
far never been deployed or designed to be applicable for tabular tasks, leaving
crucial domains such as those in healthcare and genomics out of the scope of
concept-based interpretable models. In this work, we first provide the a novel
definition of what a concept entails in a general tabular domain and then propose
**Tabular Concept Bottleneck Models (TabCBMs)**, a family of interpretable
self-explaining neural architectures capable of discovering high-level concept
explanations for tabular tasks **without sacrificing state-of-the-art performance**.

#### Abstract

Concept-based interpretability addresses the opacity of deep neural networks by
constructing an explanation for a model's prediction using high-level units of
information referred to as concepts. Research in this area, however, has been mainly
focused on image and graph-structured data, leaving high-stakes tasks whose data is
tabular out of reach of existing methods. In this paper, we address this gap by
introducing the first definition of what a high-level concept may entail in tabular
data. We use this definition to propose Tabular Concept Bottleneck Models (TabCBMs),
a family of interpretable self-explaining neural architectures capable of learning
high-level concept explanations for tabular tasks. As our method produces
concept-based explanations both when partial concept supervision or no concept
supervision is available at training time, it is adaptable to settings where concept
annotations are missing. We evaluate our method in both synthetic and real-world tabular
tasks and show that TabCBM outperforms or performs competitively compared to
state-of-the-art methods, while providing a high level of interpretability as measured by
its ability to discover known high-level concepts. Finally, we show that TabCBM can
discover important high-level concepts in synthetic datasets inspired by critical tabular
tasks (e.g., single-cell RNAseq) and allows for human-in-the-loop concept interventions
in which an expert can identify and correct mispredicted concepts to boost the model's
performance.


# Installation

You can locally install this package by first cloning this repository:
```bash
$ git clone https://github.com/mateoespinosa/tabcbm
```
We provide an automatic mechanism for this installation using
Python's setup process with our standalone `setup.py`. To install our package,
therefore, you only need to move into the cloned directory (`cd tabcbm`) and run:
```bash
$ python setup.py install
```
After running this, you should by able to import our package locally
using
```python
import tabcbm
```

# Usage

## High-level Usage
In this repository, we include a standalone TensorFlow implementation of Tabular
Concept Bottleneck Models (TabCBMs) which can be easily trained from scratch
given a set of samples that may or may not have binary concept annotations.

In order to use our model's implementation, you first need to install all our
code's requirements (listed in `requirements.txt`) or by following the
installation instructions above.

After you have installed all dependencies, you should be able to import
`TabCBM` as a standalone keras Model as follows:

```python
from tabcbm.models.tabcbm import TabCBM

#####
# Define your pytorch dataset objects
#####

x_train = ...  # Numpy np.ndarray with shape (batch, features) containing samples
y_train = ...   # Numpy np.ndarray with shape (batch) containing integer labels

#####
# Construct the model's hyperparameters
#####
n_concepts = ...  # Number of concepts we wish to discover in the task

tab_cbm_params = dict(
    features_to_concepts_model=..., # Provide Keras model to be used for the feature to latent code model (e.g., $\phi$)
    concepts_to_labels_model=..., # Provide Keras model to be used for the concept-scores-to-label model (e.g., f)
    loss_fn=..., # An appropiate loss function following tensorflow's loss function APIs
    latent_dims=32, # Size of latent space for concept embeddings
    n_concepts=n_concepts,  # Number of concepts we wish to discover in the task
    n_supervised_concepts=0, # Change to another number if concept supervision is expected (i.e., we have a c_train matrix)

    # Loss hypers
    coherence_reg_weight=..., # Scalar loss weight for the coherence regularizer
    diversity_reg_weight=..., # Scalar loss weight for the diversity regularizer
    feature_selection_reg_weight=..., # Scalar loss weight for the specificity regularizer
    concept_prediction_weight=..., # Scalar loss weight hyper for the concept predictive loss (only relevant if n_supervised_concepts != 0)
)

#####
# Perform its self-supervised pre-training first
#####

ss_tabcbm = TabCBM(
    self_supervised_mode=True,
    **tab_cbm_params,
)
ss_tabcbm.compile(optimizer=tf.keras.optimizers.Adam(1e-3))
ss_tabcbm._compute_self_supervised_loss(x_test[:2, :])
ss_tabcbm.set_weights(ss_tabcbm.get_weights())

ss_tabcbm.fit(
    x=x_train,
    y=y_train,
    validation_split=0.2,
    epochs=100,
    batch_size=256
)

#####
# And now do its supervised training stage
#####

tabcbm = TabCBM(
    self_supervised_mode=False,
    concept_generators=ss_tabcbm.concept_generators,
    prior_masks=ss_tabcbm.feature_probabilities,
    **tab_cbm_params,
)
tabcbm.compile(optimizer=optimizer_gen())
tabcbm._compute_supervised_loss(
    x_test[:2, :],
    y_test[:2],
    c_true=(
        c_train_real[:2, :]
        if c_train_real is not None else None
    ),
)
tabcbm.fit(
    x=x_train,
    y=y_train,
    validation_split=0.2,
    epochs=100,
    batch_size=256
)
```


For a **step-by-step example** showing how to generate a dataset and configure a TabCBM
**for training on your own custom dataset**, please see our [Synth-Nonlin example notebook](https://github.com/mateoespinosa/tabcbm/blob/main/examples/synth_nonlin_train_walkthrough.ipynb).


Further documentation on this model's parameters, as well as test
scripts showing examples of usage, will be incoorporated with the aforementioned
refactor that is currently going on. Until then, feel free to raise an issue
or reach out if you want specific details on a parameter of TabCBM.

## Class Arguments

Our **TabCBM module** takes the following initialization key arguments:
- `features_to_concepts_model` (tf.keras.Model): A tensorflow model mapping
input features with shape $(B, ...)$ to a set of latent codes with shape
$(B, m)$. This is the latent code encoder $\phi$ used in Figure 1 of our paper.
- `concepts_to_labels_model` (tf.keras.Model): A tensorflow model mapping a set
of concept scores in $[0, 1]$ with shape $(B, k')$ to a set of
output probabilities for each of the task classes (i.e., a tensor with shape
$(B, L)$). This is the label predictor $f$ used in Figure 1 of our paper.
- `latent_dims` (int): The dimensionality $m$ of the latent code and the concept
embeddings.
- `n_concepts` (int): Number of total concepts to use for this model. This number
must include both supervised concepts (if any) and unsupervised/discovered
concepts.
- `masking_values` (np.ndarray or None): The values to use when masking
each of the input features. This should be an array with $n$ elments in it, one
for each input feauture. If None, as it is defaulted, we use an array with all
zeros (i.e., we will mask all samples using zero masks).
- `features_to_embeddings_model` (tf.keras.Model or None): An optional
tensorflow model used to preprocess the input features before passing them to
the feature_to_concepts_model. This argument can be used to incoorporate
learnable embedding-based models when working with categorical features where we
would like to learn embeddings for each of the categories in each discrete
feature. If not provided (i.e., set to None) then we assume no input
preprocessing is needed. If provided, then it is expected that the effective
input shape of `features_to_concepts_model` is the effective output shape of
`features_to_embeddings_model`.
- `cov_mat` (np.ndarray or None): Empirical $(n \times n)$ covariance matrix for
the input training features. This covariance is used for learning correlated gate
maskings that take into account cross-feature correlations to avoid leakage when
a feature is masked (as in SEFS). If not provided (i.e., set to None) then we
will assume that all features are independent of each other (i.e., the
covariance matrix is the identity matrix).

The different components of TabCBM's loss can be configured through the following
arguments:
- `loss_fn` (Callable[tf.Tensor, tf.Tensor]): A loss function to use for the
downstream task. This is a differientable TF function that takes the true labels
`y_true` and the predicted labels `y_pred` for a batch of `B` inputs, and
returns a vector of size `(B)` describing the loss for each sample. Defaults to
TF's unreduced categorical cross entropy.
- `coherence_reg_weight` (float): Weight for the coherence regulariser (called
$\lambda_\text{co}$ in the paper). Defaults to 0.1.
- `diversity_reg_weight` (float): Weight for the diversity regulariser (called
$\lambda_\text{div}$ in the paper). Defaults to 5.
- `feature_selection_reg_weight` (float): Weight for the specificity regulariser
(called $\lambda_\text{spect}$ in the paper). Defaults to 5.
- `top_k` (int): Number of k-nearest neighbors to use when computing the
coherency loss. This argument is important to fine tune and must be less than
the batch size. Defaults to 32.

If some ground-truth concepts labels are provided during training, then this
can be indicated through the following arguments:
- `n_supervised_concepts` (int): Number of concepts that will be provided
supervision for. If non-zero, then we expect, for each sample, to be provided
with a vector of `n_supervised_concepts` with binary concept annotations. The
value of `n_supervised_concepts` should be less than `n_concepts`. Defaults to
0 (i.e., no ground-truth concepts provided).
- `concept_prediction_weight` (int): When provided with ground-truth concepts
during training, this value specifies the weight of the concept prediction loss
used during training for the supervised concepts. Defaults to 0 (i.e., no
ground-truth concepts provided).

A quick note that if ground truth concepts are provided during training, then
the first `n_supervised_concepts` concept scores will correspond to the provided
concepts in the same order they are given in the training concept label vector.
Moreover, our TabCBM implementation supports partial concept annotations (i.e.,
some samples may have some concepts annotatated and some may not). This can be
done by setting unknown concept labels as `NaN`s in the corresponding training
samples.

Our TabCBM's class also provides arguments to aid with the end-to-end
incorporation of the Self-supervised pipeline as part of its training (as shown
in the example above). These arguments are:
- `self_supervised_mode` (bool): Whether or not this model's mask generator
modules have been pretrained. If True, then it will use the SEFS pre-text
self-supervised task to pretrain these modules when one calls the `.fit(...)`
function. Otherwise, if `False`, it assumes mask generators have already been
pre-trained and `.fit(...)` will proceed directly to end-to-end training of the
entire TabCBM model. See TabCBM notebook example to see how to use this
parameter for pretraining. Defaults to False.
- `g_model` (tf.keras.Model or None): Model to be used for reconstructing the
input features from the learnt latent code during self-supervised pretraining.
If not provided (i.e., set to None) then it defaults to a simple 3-layer ReLU
MLP model with a hidden layer with 500 activation in it. Notice that this model
is only relevant during self-supervised pretraining but it is irrelevant during
end-to-end TabCBM training and any subsequent inferences.
- `gate_estimator_weight`: (float): Weight to be used self-supervised mask
generator pretraining for the regularizer penalizing the model for not correctly
predicting the mask applied to the sample. Defaults to 1.
- `include_bn` (bool): Whether or not we include a learnable batch normalization
layer that preprocesses the input features before any concept embeddings/scores.
Defaults to False.
- `rec_model_units` (List[int]): The size of the layers for the MLP used
for the the reconstruction model during the self-supervised pre-training stagte.
Defaults to [64].

Finally, we provide some control over the architecture used for the concept
generators via the following arguments:
- `concept_generator_units` (List[int]): The size of the layers for the MLP used
for the concept generator models (i.e., $\rho^{(i)}$ models.). Defaults to [64].
- `concept_generators` (list[tf.keras.Model]): A list of `n_concepts` TF model
to be used as concept generators $\rho$. If not provided (i.e., set to None),
then will instantiate each concept generator using a ReLU MLP with layer sizes
`concept_generator_units`.
- `prior_masks` (np.ndarray or None): Initial values for TabCBM's masks logit
probabilities. If not provided (i.e., set to None), then we will randomly
initialize every mask's logit probability to a value uniformly at random from
$[-1, 1]$.


# Experiment Reproducibility

## Running Experiments

To reproduce the experiments discussed in our paper, please use our
`run_experiments.py` script in `experiments/` after installing the package as
indicated above. You should then be able to run all experiments by running this
script with the appropiate config from the `experiments/configs/` directory.
For example, to run our experiments on the `Synth-Linear` dataset
(see our paper), you can execute the following command:

```bash
$ python experiments/run_experiments.py dot -c experiments/configs/linear_tab_synth_config.yaml -o results/synth_linear
```
This should generate a summary of all the results after execution has
terminated and dump all results/trained models/logs into the given
output directory (`results/synth_linear/` in this case).
