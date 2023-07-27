"""
Description of the model architectures we use for our TabCBM experiments.
"""

import tensorflow as tf


def construct_encoder(
    input_shape,
    units,
    latent_dims,
    latent_act=None,
):
    encoder_inputs = tf.keras.Input(shape=input_shape)
    encoder_compute_graph = encoder_inputs

    # Include the fully connected bottleneck here
    for i, units in enumerate(units):
        encoder_compute_graph = tf.keras.layers.Dense(
            units,
            activation='relu',
            name=f"encoder_dense_{i}",
        )(encoder_compute_graph)

    # TIme to generate the latent code here
    encoder_compute_graph = tf.keras.layers.Dense(
        latent_dims,
        activation=latent_act,
        name="encoder_bypass_channel",
    )(encoder_compute_graph)

    return tf.keras.Model(
        encoder_inputs,
        encoder_compute_graph,
        name="encoder",
    )

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
            "binary_accuracy" if (num_outputs <= 2)
            else "sparse_categorical_accuracy"
        ],
    )
    return model, encoder, decoder
