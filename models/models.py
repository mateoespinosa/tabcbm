import sklearn
import scipy
import tensorflow as tf
import numpy as np
import concepts_xai.methods.SENN.base_senn as SENN
import concepts_xai.methods.SENN.aggregators as aggregators
import concepts_xai.methods.VAE.betaVAE as beta_vae
import concepts_xai.methods.VAE.losses as vae_losses

def replace_with_embedding(
    input_tensor,
    input_shape,
    emb_dims,
    emb_in_size,
    emb_out_size=1,
):
    # Then time to add an embedding here!
    inputs = []
    special_dims = sorted(
        list(zip(emb_dims, emb_in_size)),
        key=lambda x: x[0],
    )
    current = 0
    special_idx = 0
    next_post = special_dims[0][0]
    while current < input_shape[-1]:
        if current == next_post:
            emb_input = tf.cast(
                input_tensor[:, current],
                tf.int64,
            )
            inputs.append(
                tf.keras.layers.Embedding(
                    input_dim=special_dims[special_idx][1],
                    output_dim=emb_out_size,
                )(emb_input)
            )

            # And update all trackers
            special_idx += 1
            if special_idx < len(special_dims):
                next_post = special_dims[special_idx][0]
            else:
                next_post = input_shape[-1]
            current += 1
        else:
            inputs.append(
                input_tensor[:, current:next_post],
            )
            current = next_post
    return tf.concat(
        inputs,
        axis=-1,
    )

def construct_encoder(
    input_shape,
    units,
    latent_dims,
    include_bn=False,
    latent_act=None,  # Original paper used "sigmoid" but this is troublesome in deep architectures
    emb_dims=None,
    emb_in_size=None,
    emb_out_size=1,
    return_embedding_extractor=False,
):
    encoder_inputs = tf.keras.Input(shape=input_shape)
    encoder_compute_graph = encoder_inputs
    if (emb_dims is not None) and (emb_in_size is not None):
        encoder_compute_graph = replace_with_embedding(
            input_tensor=encoder_compute_graph,
            input_shape=input_shape,
            emb_dims=emb_dims,
            emb_in_size=emb_in_size,
            emb_out_size=emb_out_size,
        )
        if return_embedding_extractor:
            emb_extractor = tf.keras.Model(
                encoder_inputs,
                encoder_compute_graph,
                name="embedding_extractor",
            )
            after_emb_inputs = encoder_compute_graph
    if include_bn:
        encoder_compute_graph = tf.keras.layers.BatchNormalization(
            axis=-1,
            momentum=0.99,
            epsilon=0.001,
            center=False,
            scale=False,
        )(encoder_compute_graph)
    
    # Include the fully connected bottleneck here
    for i, units in enumerate(units):
        encoder_compute_graph = tf.keras.layers.Dense(
            units,
            activation='relu',
            name=f"encoder_dense_{i}",
        )(encoder_compute_graph)
        if include_bn:
            encoder_compute_graph = tf.keras.layers.BatchNormalization(
                axis=-1,
                momentum=0.99,
                epsilon=0.001,
                center=True,
                scale=False,  # False as this will be merged into the upcoming fully FC layer
            )(encoder_compute_graph)
    
    # TIme to generate the latent code here
    encoder_compute_graph = tf.keras.layers.Dense(
        latent_dims,
        activation=latent_act,
        name="encoder_bypass_channel",
    )(encoder_compute_graph)
    
    encoder = tf.keras.Model(
        encoder_inputs,
        encoder_compute_graph,
        name="encoder",
    )
    if return_embedding_extractor:
        emb_to_code = tf.keras.Model(
            after_emb_inputs,
            encoder_compute_graph,
            name="emb_to_code",
        )
        return encoder, emb_extractor, emb_to_code
    return encoder

############################################################################
## Build concepts-to-labels model
############################################################################

def construct_decoder(units, num_outputs):
    decoder_layers = [
        tf.keras.layers.Dense(
            units,
            activation=tf.nn.relu,
            name=f"decoder_dense_{i+1}",
        ) for i, units in enumerate(units)
    ]
    return tf.keras.Sequential(decoder_layers + [
        tf.keras.layers.Dense(
            num_outputs if num_outputs > 2 else 1,
            activation=None,
            name="decoder_model_output",
        )
    ])


# Construct the complete model
def construct_end_to_end_model(
    input_shape,
    encoder,
    decoder,
    num_outputs,
    learning_rate=1e-3,
):
    model_inputs = tf.keras.Input(shape=input_shape)
    latent = encoder(model_inputs)
    if isinstance(latent, list):
        if len(latent) > 1:
            compacted_vector = tf.keras.layers.Concatenate(axis=-1)(
                latent
            )
        else:
            compacted_vector = latent[0]
    else:
        compacted_vector = latent
    model_compute_graph = decoder(compacted_vector)
    # Now time to collapse all the concepts again back into a single vector
    model = tf.keras.Model(
        model_inputs,
        model_compute_graph,
        name="complete_model",
    )
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate),
        loss=(
            tf.keras.losses.BinaryCrossentropy(from_logits=True) if (num_outputs <= 2)
            else tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        ),
        metrics=[
            "accuracy" if (num_outputs <= 2)
            else "sparse_categorical_accuracy"
        ],
    )
    return model, encoder, decoder


############################################################################
## SENN-specific constructors
############################################################################


def construct_senn_coefficient_model(units, num_concepts, num_outputs):
    decoder_layers = [tf.keras.layers.Flatten()] + [
        tf.keras.layers.Dense(
            units,
            activation=tf.nn.relu,
            name=f"coefficient_model_dense_{i+1}",
        ) for i, units in enumerate(units)
    ]
    return tf.keras.Sequential(decoder_layers + [
        tf.keras.layers.Dense(
            num_concepts * num_outputs,
            activation=None,
            name="coefficient_model_output",
        ),
        tf.keras.layers.Reshape([num_outputs, num_concepts])
    ])

def construct_senn_encoder(
    input_shape,
    units,
    end_activation="sigmoid",
    latent_dims=0,
    latent_act=None,  # Original paper used "sigmoid" but this is troublesome in deep architectures
    emb_dims=None,
    emb_in_size=None,
    emb_out_size=1,
):
    encoder_inputs = tf.keras.Input(shape=input_shape)
    encoder_compute_graph = encoder_inputs
    if (emb_dims is not None) and (emb_in_size is not None):
        encoder_compute_graph = replace_with_embedding(
            input_tensor=encoder_compute_graph,
            input_shape=input_shape,
            emb_dims=emb_dims,
            emb_in_size=emb_in_size,
            emb_out_size=emb_out_size,
        )
    for i, units in enumerate(units):
        encoder_compute_graph = tf.keras.layers.Dense(
            units,
            activation='relu',
            name=f"encoder_dense_{i}",
        )(encoder_compute_graph)

    mean = tf.keras.layers.Dense(latent_dims, activation=None, name="means")(encoder_compute_graph)
    log_var = tf.keras.layers.Dense(latent_dims, activation=None, name="log_var")(encoder_compute_graph)
    senn_encoder = tf.keras.Model(
        encoder_inputs,
        mean,
        name="senn_encoder",
    )
    vae_encoder = tf.keras.Model(
        encoder_inputs,
        [mean, log_var],
        name="vae_encoder",
    )
    return senn_encoder, vae_encoder


def construct_vae_decoder(
    units,
    output_shape,
    latent_dims,
):
    """CNN decoder architecture used in the 'Challenging Common Assumptions in the Unsupervised Learning
       of Disentangled Representations' paper (https://arxiv.org/abs/1811.12359)

       Note: model is uncompiled
    """

    latent_inputs = tf.keras.Input(shape=(latent_dims,))
    model_out = latent_inputs
    for unit in units:
        model_out = tf.keras.layers.Dense(
            unit,
            activation='relu',
        )(model_out)
    model_out = tf.keras.layers.Dense(
        output_shape,
        activation=None,
    )(model_out)

    return tf.keras.Model(
        inputs=latent_inputs,
        outputs=[model_out],
    )

def get_reconstruction_fn(concept_decoder):
    def reconstruction_loss_fn(y_true, y_pred):
        #return vae_losses.bernoulli_fn_wrapper()(y_true, concept_decoder(y_pred))
        return tf.reduce_sum(
            tf.square(y_true - concept_decoder(y_pred)),
            [-1]
        )
    return reconstruction_loss_fn


def construct_senn_model(
    concept_encoder,
    concept_decoder,
    coefficient_model,
    num_outputs,
    regularization_strength=0.1,
    learning_rate=1e-3,
    sparsity_strength=2e-5,
):
    senn_model = SENN.SelfExplainingNN(
        encoder_model=concept_encoder,
        coefficient_model=coefficient_model,
        aggregator_fn=(
            aggregators.multiclass_additive_aggregator if (num_outputs >= 2)
            else aggregators.scalar_additive_aggregator
        ),
        task_loss_fn=(
            tf.keras.losses.BinaryCrossentropy(from_logits=True) if (num_outputs < 2)
            else tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        ),
        reconstruction_loss_fn=get_reconstruction_fn(concept_decoder),
        regularization_strength=regularization_strength,
        sparsity_strength=sparsity_strength,
        name="SENN",
        metrics=[
            tf.keras.metrics.BinaryAccuracy() if (num_outputs < 2)
            else tf.keras.metrics.SparseTopKCategoricalAccuracy(k=1)
        ],
    )
    senn_model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate),
    )
    return senn_model