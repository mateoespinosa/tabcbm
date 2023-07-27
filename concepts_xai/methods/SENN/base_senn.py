"""
Taken from https://github.com/dmitrykazhdan/concept-based-xai.

All credit due to Kazhdan et al. (arXiv:2104.06917).
"""

"""
Tensorflow implementation of Self-Explaining Neural Networks (SENN) by
Alvarez-Melis and Jaakkola (NeurIPS 2018) [1].

[1] https://papers.nips.cc/paper/2018/file/3e9f0fc9b2f89e043bc6233994dfcf76-Paper.pdf

"""

import tensorflow as tf
from tensorflow.python.keras.engine import data_adapter


class SelfExplainingNN(tf.keras.Model):
    """
    Main class implementation for SENN.
    """

    def __init__(
        self,
        encoder_model,
        coefficient_model,
        aggregator_fn,
        task_loss_fn,
        regularization_strength=1e-1, # "lambda" regulatizer parameter
        sparsity_strength=2e-5,  # "zeta" autoencoder parameter
        reconstruction_loss_fn=None,
        task_loss_weight=1,
        # Not using tf.norm(x, ord='fro', axis=[-2, -1]) as it is prone to instabilities during
        # automatic differientation!
        robustness_norm_fn=lambda x: tf.sqrt(tf.reduce_sum(tf.square(x),  axis=[-2, -1]) + 1e-15),
        metrics=None,
        **kwargs,
    ):
        super(SelfExplainingNN, self).__init__(**kwargs)
        self.task_loss_weight = task_loss_weight
        self.encoder_model = encoder_model
        self.coefficient_model = coefficient_model
        self.aggregator_fn = aggregator_fn
        self.task_loss_fn = task_loss_fn
        self.regularization_strength = regularization_strength
        self.sparsity_strength = sparsity_strength
        self.reconstruction_loss_fn = reconstruction_loss_fn
        self.robustness_norm_fn = robustness_norm_fn
        if self.reconstruction_loss_fn is not None:
            self.metrics_dict = {
                "loss": tf.keras.metrics.Mean(
                    name="loss"
                ),
                "task_loss": tf.keras.metrics.Mean(
                    name="task_loss"
                ),
                "robustness_loss": tf.keras.metrics.Mean(
                    name="robustness_loss"
                ),
                "reconstruction_loss": tf.keras.metrics.Mean(
                    name="reconstruction_loss"
                ),
            }
        else:
            self.metrics_dict = {
                "loss": tf.keras.metrics.Mean(
                    name="loss"
                ),
                "task_loss": tf.keras.metrics.Mean(
                    name="task_loss"
                ),
                "robustness_loss": tf.keras.metrics.Mean(
                    name="robustness_loss"
                ),
            }
        self._extra_metrics = []
        for metric in (metrics or []):
            if isinstance(metric, (tuple, list)):
                if len(metric) != 2:
                    raise ValueError(
                        f"Expected metrics to be a list of  tuples "
                        f"(name, metric) or TF metric objects. Instead we "
                        f"received {metric}."
                    )
                name, metric = metric
            else:
                name = metric.name
            self.metrics_dict[name] = metric
            self._extra_metrics.append((name, metric))

    @property
    def metrics(self):
        return [
            self.metrics_dict[name] for name in self.metrics_dict.keys()
        ]

    def update_metrics(self, losses):
        for (name, loss) in losses:
            self.metrics_dict[name].update_state(loss)

    def call(self, inputs):
        # First compute our concepts
        concepts = self.encoder_model(inputs)  # (batch, n_concepts)
        concepts = tf.debugging.check_numerics(concepts, "concepts has NaN!")
        # Then all of the theta weights which we will use for our concepts
        thetas = self.coefficient_model(inputs) # (batch, n_outputs, n_concepts)
        thetas = tf.debugging.check_numerics(thetas, "thetas has NaN!")
        if len(thetas.shape) < 3:
            # Then the number of classes/outputs is 1 so let's make it explicit
            thetas = tf.expand_dims(thetas, axis=1)
        # Make sure the dimensions match
        tf.debugging.assert_equal(
            tf.shape(concepts)[-1],
            tf.shape(thetas)[-1],
            message=(
                "The last dimension of the returned concept tensor and the "
                "theta tensor must be the same (they both should correspond to "
                "the number of concepts used in the SENN). We found "
                f"{tf.shape(concepts)[-1]} entries in concepts.shape[-1] while "
                f"{tf.shape(thetas)[-1]} entries in thetas.shape[-1]."
            )
        )
        predictions = self.aggregator_fn(
            thetas=thetas,
            concepts=concepts,
        )
        return predictions, (concepts, thetas)

    def train_step(self, inputs):
        # This will allow us to compute the task specific loss
        inputs = data_adapter.expand_1d(inputs)
        x, y, sample_weight = data_adapter.unpack_x_y_sample_weight(inputs)

        with tf.GradientTape() as outter_tape:
            total_loss = 0
            # Do a nesting of tapes as we will need to compute gradients and
            # jacobian in order for one
            with tf.GradientTape(
                persistent=True,
                watch_accessed_variables=False,
            ) as inner_tape:
                # First compute predictions and their corresponding explanation
                inner_tape.watch(x)
                preds, (concepts, thetas) = self(x)
            # This gives us the task specific loss
            task_loss = self.task_loss_fn(y, preds)
            total_loss += task_loss * self.task_loss_weight

            if self.reconstruction_loss_fn is not None:
                # Now compute the encoder reconstruction loss similarly
                reconstruction_loss = self.reconstruction_loss_fn(
                    x,
                    concepts,
                )
                total_loss += self.sparsity_strength * reconstruction_loss

            # Finally, time to add the robustness loss. This is the trickiest
            # and most expensive one as it requires the computation of a
            # jacobian
            # Gradient of f(x) with respect to x should have shape
            # (batch, n_outputs, <shape_of_input_x>).
            # Note that we use a jacobian computation rather than a gradient
            # computation (as done in the paper) as we support general
            # multi-dimensional outputs
            if len(preds.shape) < 3:
                # Jacobian requires a 2D input
                preds = tf.expand_dims(preds, axis=1)
            preds = tf.debugging.check_numerics(preds, "preds has NaN!")
            x = tf.debugging.check_numerics(x, "x has NaN!")
            f_grad_x = inner_tape.batch_jacobian(preds, x)
            f_grad_x = tf.debugging.check_numerics(f_grad_x, "f_grad_x has NaN!")

            # Reshape to (batch, n_outputs, flatten_x_shape)
            f_grad_x = tf.reshape(
                f_grad_x,
                [tf.shape(x)[0], tf.shape(preds)[-1], -1]
            )
            f_grad_x = tf.debugging.check_numerics(f_grad_x, "f_grad_x has NaN!")

            # Jacobian of h(x) with respect to x should have shape
            # (batch, n_concepts, <shape_of_input_x>)
            h_jacobian_x = inner_tape.batch_jacobian(
                concepts,
                x,
            )
            h_jacobian_x = tf.debugging.check_numerics(h_jacobian_x, "h_jacobian_x has NaN!")
            # Reshape to (batch, n_concepts, flatten_x_shape)
            h_jacobian_x = tf.reshape(
                h_jacobian_x,
                [tf.shape(x)[0], tf.shape(concepts)[-1], -1]
            )
            robustness_loss = self.robustness_norm_fn(
                f_grad_x - tf.matmul(
                    # No need to transpose thetas as they already have the
                    # number of outputs as its first non-batch dimension (i.e.,
                    # its shape is (batch, n_outputs, n_concepts))
                    thetas,
                    # h_jacobian_x shape: (batch, n_concepts, flatten_x_shape)
                    h_jacobian_x,
                )  # matmul shape: (batch, n_outputs, flatten_x_shape)
            )
            robustness_loss = tf.debugging.check_numerics(robustness_loss, "robustness_loss has NaN!")
            robustness_loss = tf.math.reduce_mean(robustness_loss)
            robustness_loss = tf.debugging.check_numerics(robustness_loss, "robustness_loss mean has NaN!")
            total_loss += self.regularization_strength * robustness_loss
            total_loss = tf.debugging.check_numerics(total_loss, "total_loss acc has NaN!")

        # Compute gradients and proceed with SGD
        gradients = outter_tape.gradient(total_loss, self.trainable_variables)
        self.optimizer.apply_gradients(
            zip(gradients, self.trainable_variables)
        )

        # And update all of our metrics
        self.update_metrics([
            ("loss", total_loss),
            ("task_loss", task_loss),
            ("robustness_loss", robustness_loss),
        ])
        if self.reconstruction_loss_fn is not None:
            self.update_metrics([
                ("reconstruction_loss", reconstruction_loss),
            ])
        for (name, metric) in self._extra_metrics:
            self.metrics_dict[name].update_state(y, preds, sample_weight)
        return {
            name: val.result()
            for name, val in self.metrics_dict.items()
        }

    def test_step(self, inputs):
        # This will allow us to compute the task specific loss
        inputs = data_adapter.expand_1d(inputs)
        x, y, sample_weight = data_adapter.unpack_x_y_sample_weight(inputs)
        with tf.GradientTape() as outter_tape:
            total_loss = 0
            # Do a nesting of tapes as we will need to compute gradients and
            # jacobian
            with tf.GradientTape(
                persistent=True,
                watch_accessed_variables=False,
            ) as inner_tape:
                # First compute predictions and their corresponding explanation
                inner_tape.watch(x)
                preds, (concepts, thetas) = self(x)
            # This gives us the task specific loss
            task_loss = self.task_loss_fn(y, preds)
            total_loss += task_loss * self.task_loss_weight

            if self.reconstruction_loss_fn is not None:
                # Now compute the encoder reconstruction loss similarly
                reconstruction_loss = self.reconstruction_loss_fn(
                    x,
                    concepts,
                )
                total_loss += self.sparsity_strength * reconstruction_loss

            # Finally, time to add the robustness loss. This is the trickiest
            # and most expensive one as it requires the computation of a
            # jacobian
            # Gradient of f(x) with respect to x should have shape
            # (batch, n_outputs, <shape_of_input_x>).
            # Note that we use a jacobian computation rather than a gradient
            # computation (as done in the paper) as we support general
            # multi-dimensional outputs
            if len(preds.shape) < 3:
                # Jacobian requires a 2D input
                preds = tf.expand_dims(preds, axis=1)
            f_grad_x = inner_tape.batch_jacobian(preds, x)
            # Reshape to (batch, n_outputs, flatten_x_shape)
            f_grad_x = tf.reshape(
                f_grad_x,
                [tf.shape(x)[0], tf.shape(preds)[-1], -1]
            )

            # Jacobian of h(x) with respect to x should have shape
            # (batch, n_concepts, <shape_of_input_x>)
            h_jacobian_x = inner_tape.batch_jacobian(
                concepts,
                x,
            )
            # Reshape to (batch, n_concepts, flatten_x_shape)
            h_jacobian_x = tf.reshape(
                h_jacobian_x,
                [tf.shape(x)[0], tf.shape(concepts)[-1], -1]
            )
            robustness_loss = self.robustness_norm_fn(
                f_grad_x - tf.matmul(
                    # No need to transpose thetas as they already have the
                    # number of outputs as its first non-batch dimension (i.e.,
                    # its shape is (batch, n_outputs, n_concepts))
                    thetas,
                    # h_jacobian_x shape: (batch, n_concepts, flatten_x_shape)
                    h_jacobian_x,
                )  # matmul shape: (batch, n_outputs, flatten_x_shape)
            )
            robustness_loss = tf.math.reduce_mean(robustness_loss)
            total_loss += self.regularization_strength * robustness_loss

        # And update all of our metrics
        self.update_metrics([
            ("loss", total_loss),
            ("task_loss", task_loss),
            ("robustness_loss", robustness_loss),
        ])
        if self.reconstruction_loss_fn is not None:
            self.update_metrics([
                ("reconstruction_loss", reconstruction_loss),
            ])
        for (name, metric) in self._extra_metrics:
            self.metrics_dict[name].update_state(y, preds, sample_weight)
        return {
            name: val.result()
            for name, val in self.metrics_dict.items()
        }
