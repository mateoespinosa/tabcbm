"""
The code in this file related to the Gaussian copula sampling is taken
from https://github.com/chl8856/SEFS. All credit given to Lee et al. for their
SEFS work.

Everything else is writen by the TabCBM's authors.
"""

import concepts_xai.evaluation.metrics.completeness as completeness
import logging
import numpy as np
import scipy
import tensorflow as tf

def log(x):
    return tf.math.log(x + 1e-6)

def Gaussian_CDF(x): #change from erf function
    return 0.5 * (1. + tf.math.erf(x / tf.math.sqrt(2.)))

def copula_generation(X, batch_size):
    cov = np.corrcoef(X.T)
    L = scipy.linalg.cholesky(cov, lower=True)
    epsilon = np.random.normal(
        loc=0,
        scale=1,
        size=[np.shape(L)[0], batch_size],
    )
    g = np.matmul(L, epsilon)
    return g.T

class TabCBM(tf.keras.Model):

    def __init__(
        self,
        # Important architectural and task-specific arguments
        features_to_concepts_model,  # What we call \phi in the paper
        concepts_to_labels_model,  # The "label predictor" which we call f
        latent_dims,
        n_concepts,
        mean_inputs,
        features_to_embeddings_model=None,
        cov_mat=None,

        # Self-supervised (SS) related arguments
        # Only set this to true if the SS stage has already been completed
        # for the mask generators
        self_supervised_mode=False,
        g_model=None,
        gate_estimator_weight=1,
        include_bn=False,

        # Loss configuration
        loss_fn=tf.keras.losses.sparse_categorical_crossentropy,
        coherence_reg_weight=0.1,
        diversity_reg_weight=5,
        feature_selection_reg_weight=5,
        top_k=32,  # Used for coherence loss

        # Concept supervision arguments
        # Only relevant if concept supervision is provided
        n_supervised_concepts=0,
        concept_prediction_weight=0,

        # Minor architectural arguments
        concept_generator_units=[64],
        rec_model_units=[64],
        concept_generators=None,
        prior_masks=None, # If provided, it must have as many elements as concepts

        # Evaluation-related arguments
        acc_metric=None,

        # Misc flags/parameters for debugging/deep dive modifications of the
        # method
        eps=1e-5,
        threshold=0.5,
        temperature=1,
        end_to_end_training=True,
        normalized_scores=True,
        forward_deterministic=True,
        **kwargs,
    ):
        super().__init__(**kwargs)

        # Set initial state from parameters
        self.normalized_scores = normalized_scores
        self.self_supervise_mode = self_supervised_mode
        self.concepts_to_labels_model = concepts_to_labels_model
        self.features_to_concepts_model = features_to_concepts_model
        self.features_to_embeddings_model = features_to_embeddings_model
        self.eps = eps
        self.threshold = threshold
        self.n_concepts = n_concepts
        self.n_supervised_concepts = n_supervised_concepts
        self.forward_deterministic = forward_deterministic
        assert self.n_supervised_concepts <= self.n_concepts, (
            f"We were indicated that {self.n_supervised_concepts} concepts "
            f"will be given for supervision yet we only are learning a total "
            f"of {self.n_concepts} concepts. We cannot supervise more concepts "
            f"than the number of concepts we learn/extract."
        )
        self.loss_fn = loss_fn
        self.top_k = top_k
        self.temperature = temperature
        self.gate_estimator_weight = gate_estimator_weight
        self.coherence_reg_weight = coherence_reg_weight
        self.diversity_reg_weight = diversity_reg_weight
        self.feature_selection_reg_weight = feature_selection_reg_weight
        self.concept_prediction_weight = concept_prediction_weight

        self.latent_dims = latent_dims
        self.end_to_end_training = end_to_end_training
        input_shape = self.features_to_concepts_model.inputs[0].shape[1:]
        self.self_supervised_selection_prob = self.add_weight(
            name=f"ss_probability_vector",
            shape=(self.n_concepts, input_shape[-1],),
            dtype=tf.float32,
            initializer=tf.keras.initializers.RandomUniform(
                minval=0.4,
                maxval=0.6,
            ),
            trainable=False,
        )

        # Initialize the g model which will be in charge of reconstructing
        # the model latent activations from the concept scores alone
        self.g_model = g_model
        if self.g_model is None:
            self.g_model = completeness._get_default_model(
                num_concepts=n_concepts,
                num_hidden_acts=self.latent_dims,
            )

        assert len(input_shape) == 1, \
            f'Expected a 1D input yet we got shape {input_shape}'

        # Setup our metrics
        self.metric_names = [
            "loss",
            "accuracy",
            "task_loss",
            "reg_loss_closest",
            "reg_loss_similarity",
            "prob_sparsity_loss",
            "avg_mask_rec_loss",
            "avg_features_rec_loss",
            "avg_concept_size",
            "max_probability",
            "min_probability",
            "mean_probability",
        ]
        if self.n_supervised_concepts != 0:
            self.metric_names.append("avg_concept_prediction")
            self.metric_names.append("concept_pred_loss")
            self.metric_names.append("avg_concept_accuracy")
            self.metric_names.append("mean_concept_task_acc")
        self.metrics_dict = {
            name: tf.keras.metrics.Mean(name=name)
            for name in self.metric_names
        }
        self._acc_metric = (
            acc_metric or (
                lambda y_true, y_pred: \
                    tf.keras.metrics.sparse_top_k_categorical_accuracy(
                        y_true,
                        y_pred,
                        k=1,
                    )
            )
        )

        # And we will initialize some models to generate concept vectors
        # from the masked input features
        if cov_mat is not None:
            self.cov_mat = cov_mat
        else:
            # else we assume all concepts are independent of each other
            logging.warning(
                "Assuming independence between features in "
                "TabCBM training."
            )
            self.cov_mat = np.eye(input_shape[-1], dtype=np.float32)
        try:
            self.L = scipy.linalg.cholesky(self.cov_mat, lower=True).astype(
                np.float32
            )
        except:
            # Else, if it is not decomposable, assume full independence
            print(
                "[WARNING] Assuming independence between features in TabCBM "
                "training."
            )
            self.cov_mat = np.eye(input_shape[-1], dtype=np.float32)
            self.L = scipy.linalg.cholesky(self.cov_mat, lower=True).astype(
                np.float32
            )
        self.mean_inputs = mean_inputs
        given = concept_generators is not None
        self.concept_generators = concept_generators or []
        self.rec_values_models = []
        self.rec_mask_models = []

        if prior_masks is None:
            # Then we randomly initialize the logits
            initial_mask = tf.keras.initializers.RandomUniform(
                minval=-1,
                maxval=1,
            )
        else:
            # otherwise a prior has been imposed so let's take advantage
            # of it!
            initial_mask = lambda *args, **kwargs: prior_masks
        self.feature_probabilities = self.add_weight(
            name=f"probability_vector_logits",
            shape=(self.n_concepts, input_shape[-1],),
            dtype=tf.float32,
            initializer=initial_mask,
            trainable=True,
        )

        rec_model_mask = None
        rec_model_vars = None
        for i in range(self.n_concepts):
            # For each concept we will have a simple MLP that maps
            # the masked input to the concept's input latent space. This will
            # then used to produce a score for the input of interest for
            # each concept
            if not given:
                layers = [
                    tf.keras.layers.Dense(
                        acts,
                        activation='relu',
                    )
                    for acts in concept_generator_units
                ]
                layers += [
                    tf.keras.layers.Dense(
                        self.latent_dims,
                        activation=None,
                    ),
                ]

                if include_bn:
                    # Then include a batch normalization layer at the begining
                    layers = [tf.keras.layers.BatchNormalization(
                        axis=-1,
                        momentum=0.99,
                        epsilon=0.001,
                        center=False,
                        scale=False,
                    )] + layers
                self.concept_generators.append(
                    tf.keras.models.Sequential(
                        layers,
                        name=f"concept_generators_{i}",
                    )
                )
                self.concept_generators[-1].compile()

            if self_supervised_mode:
                # Initialize the mask and value reconstruction models which will
                # be in charge of reconstructing the input from its masked
                # version
                if (
                    (rec_model_vars is None) or
                    (rec_model_mask is None)
                ):
                    layers = (
                        [
                            tf.keras.layers.Dense(
                                acts,
                                activation='relu',
                            )
                            for acts in rec_model_units
                        ] +
                        [
                            tf.keras.layers.Dense(
                                input_shape[-1],
                                activation=None,  # The sigmoid will be implicit
                                                  # in the loss function
                            ),
                        ]
                    )
                    rec_model_mask = tf.keras.models.Sequential(
                        layers,
                        name="rec_mask_model",
                    )
                    rec_model_mask.compile()

                    rec_model_vars = tf.keras.models.Sequential(
                        (
                            [
                                tf.keras.layers.Dense(
                                    acts,
                                    activation='relu',
                                )
                                for acts in rec_model_units
                            ] +
                            [
                                tf.keras.layers.Dense(
                                    input_shape[-1],
                                    activation=None,
                                ),
                            ]
                        ),
                        name=f"rec_values_model_{i}",
                    )
                    rec_model_vars.compile()


                self.rec_values_models.append(rec_model_vars)
                self.rec_mask_models.append(rec_model_mask)

    @property
    def metrics(self):
        return [self.metrics_dict[name] for name in self.metric_names]

    def update_metrics(self, losses):
        for (loss_name, loss) in losses:
            self.metrics_dict[loss_name].update_state(loss)

    def _multi_bernoulli_sample(self, pi, shape):
        # Sample from a standard Gaussian first to perform the
        # reparameterization trick
        epsilon = tf.random.normal(shape, 0, 1)
        v = tf.transpose(tf.linalg.matmul(self.L, epsilon))
        u = Gaussian_CDF(v)
        return tf.cast(u <= pi, tf.float32)

    def _relaxed_multi_bernoulli_sample(self, pi, shape):
        # Sample from a standard Gaussian first to perform the
        # reparameterization trick
        epsilon = tf.random.normal(shape, 0, 1)
        v = tf.transpose(tf.linalg.matmul(self.L, epsilon))
        u = Gaussian_CDF(v)
        return tf.nn.sigmoid(
            1.0/self.temperature * (
                log(pi) - log(1. - pi) + log(u) - log(1. - u)
            )
        )

    def mask_features(self, x, training=False):
        masked_xs = []
        if self.features_to_embeddings_model is not None:
            # Then let's apply our embedding generator as we may have
            # some variables which are categorical in nature
            x = self.features_to_embeddings_model(x)
        for i in range(self.n_concepts):
            if training or (not self.forward_deterministic):
                gate_vector = self._relaxed_multi_bernoulli_sample(
                    tf.nn.sigmoid(self.feature_probabilities[i, :]),
                    shape=tf.stack([tf.shape(self.L)[0], tf.shape(x)[0]], axis=0),
                )
            else:
                # Else we do a deterministic non-differientable unit mask
                gate_vector = tf.nn.sigmoid(self.feature_probabilities[i, :])

            # Extend gate vector so that it can be broadcasted across all
            # samples in the batch of x
            masked_xs.append(
                gate_vector * x + (1 - gate_vector) * tf.expand_dims(
                    self.mean_inputs,
                    axis=0,
                )
            )
        return masked_xs

    def compute_concept_matrix(self, x, training=False):
        masked_xs = self.mask_features(x, training=training)
        concept_vectors = []
        for concept_generator, masked_x in zip(
            self.concept_generators,
            masked_xs,
        ):
            # This will be a sample of size [B, latent_dims]
            concept_vectors.append(concept_generator(masked_x))

        # We need to stack them all in the first dimension so that
        # we obtain a variable with size [B, n_concepts, latent_dims]
        return tf.stack(concept_vectors, axis=1), masked_xs

    def _emb_concept_scores(self, x, compute_reg_terms=False, training=False):
        # First we compute the concept matrix
        # Shape: [B, n_concepts, latent_dims]
        concept_matrix, masked_xs = self.compute_concept_matrix(
            x,
            training=training,
        )
        # Shape: [B, n_concepts, latent_dims]
        concept_matrix_norm = tf.math.l2_normalize(concept_matrix, axis=1)

        # Then for each concept, we have a corresponding masked inputs which we
        # will pass through the feature to concepts model (which cannot be
        # trained)
        concept_probs = []
        bottleneck_acts = []
        concept_prob_norms = []

        for i, masked_x in enumerate(masked_xs):
            # Shape: [B, latent_dims]
            latent = self.features_to_concepts_model(masked_x)
            # Shape: [B, latent_dims]
            latent_norm = tf.math.l2_normalize(latent, axis=-1)

            # Compute the concept probability scores
            # Shape: [B, 1]
            concept_prob = tf.sigmoid(tf.squeeze(
                tf.matmul(
                    # [B, 1, latent_dim]
                    concept_matrix_norm[:, i:(i+1), :],
                    # [B, latent_dim, 1]
                    tf.expand_dims(latent, axis=-1),
                ),
                axis=-1,
            ))
            concept_probs.append(concept_prob)
            # Shape: [B, 1, 1]
            broad_prob = tf.expand_dims(concept_prob, axis=-1)
            # Add in the bottleneck the linear combination of the two semantic
            # embeddings
            # Shape: [B, 1, latent_dims//2]
            bottleneck_acts.append(
                broad_prob * concept_matrix_norm[:, i:(i+1), concept_matrix_norm.shape[-1]//2:] +
                (1 - broad_prob) * concept_matrix_norm[:, i:(i+1), :concept_matrix_norm.shape[-1]//2]
            )
            # Shape: [B, 1]
            concept_prob_norm = tf.sigmoid(tf.squeeze(
                tf.matmul(
                    # [B, 1, latent_dim]
                    concept_matrix_norm[:, i:(i+1), :],
                    # [B, latent_dim, 1]
                    tf.expand_dims(latent_norm, axis=-1),
                ),
                axis=-1,
            ))
            concept_prob_norms.append(concept_prob_norm)
        # Shape: [B, n_concepts]
        concept_prob = tf.concat(concept_probs, axis=1)
        # Shape: [B, n_concepts, latent_dim//2]
        bottleneck = tf.concat(bottleneck_acts, axis=1)
        # Shape: [B, n_concepts * (latent_dim//2)]
        bottleneck = tf.reshape(bottleneck, [tf.shape(bottleneck)[0], -1])
        # Shape: [B, n_concepts]
        concept_prob_norm = tf.concat(concept_prob_norms, axis=1)

        if not compute_reg_terms:
            return concept_prob, bottleneck

        # Compute the regularization loss terms
        # Shape: [n_concepts, B]
        reshaped_concept_probs = tf.transpose(concept_prob_norm)
        completement_shape = tf.math.maximum(
            tf.shape(reshaped_concept_probs)[-1] - self.top_k,
            0,
        )

        reg_loss_closest = tf.reduce_mean(
            tf.nn.top_k(
                reshaped_concept_probs,
                k=tf.math.minimum(
                    self.top_k,
                    tf.shape(reshaped_concept_probs)[-1]
                ),
                sorted=True,
            ).values
        )
        reg_loss_similarity = tf.reduce_mean(
            tf.math.abs(
                tf.linalg.matmul(
                    concept_matrix_norm,
                    tf.transpose(concept_matrix_norm, perm=[0, 2, 1]),
                ) - tf.expand_dims(tf.eye(self.n_concepts), axis=0)
            )
        )

        return (
            concept_prob,
            bottleneck,
            reg_loss_closest/self.n_concepts,
            reg_loss_similarity/self.n_concepts,
        )

    def _concept_scores(self, x, compute_reg_terms=False, training=False):
        # First we compute the concept matrix
        # Shape: [B, n_concepts, latent_dims]
        concept_matrix, masked_xs = self.compute_concept_matrix(
            x,
            training=training,
        )
        # Shape: [B, n_concepts, latent_dims]
        concept_matrix_norm = tf.math.l2_normalize(concept_matrix, axis=1)

        # Then for each concept, we have a corresponding masked inputs which we
        # will pass through the feature to concepts model (which cannot be
        # trained)
        concept_probs = []
        concept_prob_norms = []

        for i, masked_x in enumerate(masked_xs):
            # Shape: [B, latent_dims]
            latent = self.features_to_concepts_model(masked_x)
            # Shape: [B, latent_dims]
            latent_norm = tf.math.l2_normalize(latent, axis=-1)

            # Compute the concept probability scores
            # Shape: [B, 1]
            concept_prob = tf.squeeze(
                tf.matmul(
                    # [B, 1, latent_dim]
                    concept_matrix_norm[:, i:(i+1), :],
                    # [B, latent_dim, 1
                    tf.expand_dims(latent, axis=-1),
                ),
                axis=-1,
            )
            concept_probs.append(concept_prob)
            # Shape: [B, 1]
            concept_prob_norm = tf.squeeze(
                tf.matmul(
                    # [B, 1, latent_dim]
                    concept_matrix_norm[:, i:(i+1), :],
                    # [B, latent_dim, 1
                    tf.expand_dims(latent_norm, axis=-1),
                ),
                axis=-1,
            )
            concept_prob_norms.append(concept_prob_norm)
        # Shape: [B, n_concepts]
        concept_prob = tf.concat(concept_probs, axis=1)
        # Shape: [B, n_concepts]
        concept_prob_norm = tf.concat(concept_prob_norms, axis=1)
        # Threshold them if they are below the given threshold value
        if self.normalized_scores:
            concept_prob = tf.sigmoid(concept_prob)
        elif self.threshold is not None:
            concept_prob = concept_prob * tf.cast(
                (concept_prob_norm > self.threshold),
                tf.float32,
            )

        if not compute_reg_terms:
            return concept_prob, concept_prob

        # Compute the regularization loss terms
        # Shape: [n_concepts, B]
        reshaped_concept_probs = tf.transpose(concept_prob_norm)
        completement_shape = tf.math.maximum(
            tf.shape(reshaped_concept_probs)[-1] - self.top_k,
            0,
        )

        reg_loss_closest = tf.reduce_mean(
            tf.nn.top_k(
                reshaped_concept_probs,
                k=tf.math.minimum(
                    self.top_k,
                    tf.shape(reshaped_concept_probs)[-1]
                ),
                sorted=True,
            ).values
        )
        reg_loss_similarity = tf.reduce_mean(
            tf.math.abs(
                tf.linalg.matmul(
                    concept_matrix_norm,
                    tf.transpose(concept_matrix_norm, perm=[0, 2, 1]),
                ) - tf.expand_dims(tf.eye(self.n_concepts), axis=0)
            )
        )

        return (
            concept_prob,
            concept_prob, # Bottleneck
            reg_loss_closest/self.n_concepts,
            reg_loss_similarity/self.n_concepts,
        )

    def concept_scores(self, x, compute_reg_terms=False, training=False):
        return self._concept_scores(
            x=x,
            compute_reg_terms=compute_reg_terms,
            training=training,
        )


    def _compute_self_supervised_loss(self, x):
        total_loss = 0.0
        avg_mask_rec_loss = 0.0
        avg_features_rec_loss = 0.0
        if self.features_to_embeddings_model is not None:
            # Then let's apply our embedding generator as we may have
            # some variables which are categorical in nature
            x = self.features_to_embeddings_model(x)
        for i in range(self.n_concepts):
            mask = self._multi_bernoulli_sample(
                self.self_supervised_selection_prob[i, :],
                shape=tf.stack([tf.shape(self.L)[0], tf.shape(x)[0]], axis=0),
            )
            # Extend gate vector so that it can be broadcasted across all
            # samples in the batch of x
            masked_x = (
                mask * x + (1 - mask) * tf.expand_dims(
                    self.mean_inputs,
                    axis=0,
                )
            )

            # Now let's generate the concept vector for this concept
            concept_vector = self.concept_generators[i](masked_x)

            # Try and predict the original mask from this concept vector
            mask_pred = self.rec_mask_models[i](concept_vector)
            mask_rec_loss = tf.keras.metrics.binary_crossentropy(
                mask,
                mask_pred,
                from_logits=True,
            )
            total_loss += self.gate_estimator_weight * tf.math.reduce_mean(
                mask_rec_loss
            )
            avg_mask_rec_loss += (
                self.gate_estimator_weight * tf.math.reduce_mean(mask_rec_loss)
            )

            # Try and predict the original values of the sample as well
            features_pred = self.rec_values_models[i](concept_vector)
            features_rec_loss = tf.keras.metrics.mean_squared_error(
                x,
                features_pred,
            )
            total_loss += tf.math.reduce_mean(features_rec_loss)
            avg_features_rec_loss += tf.math.reduce_mean(features_rec_loss)

        return total_loss, [
            ("loss", total_loss),
            ("avg_mask_rec_loss", avg_mask_rec_loss/self.n_concepts),
            ("avg_features_rec_loss", avg_features_rec_loss/self.n_concepts),
        ]

    def _compute_supervised_loss(
        self,
        x,
        y_true,
        c_true=None,
        training_decoder=False,
    ):
        # First, compute the concept scores for the given samples
        (
            scores,
            bottleneck,
            reg_loss_closest,
            reg_loss_similarity,
        ) = self.concept_scores(
            x,
            compute_reg_terms=True,
            training=True,
        )

        # Then predict the labels after reconstructing the activations
        # from the scores via the g model
        y_pred = self.concepts_to_labels_model(
            self.g_model(bottleneck),
            training=training_decoder,
        )
        # Compute the task loss
        if (len(y_true.shape) == 1) and (y_pred.shape[-1] == 1):
            y_pred = tf.squeeze(y_pred, axis=-1)
        log_prob_loss = tf.reduce_mean(self.loss_fn(y_true, y_pred))
        # Compute the probability sparsity losses for all concepts
        prob_sparsity_loss = tf.math.reduce_mean(
            # Using mean instead of sum to avoid explosure in case of large
            # inputs and avoid unstable training
            tf.math.reduce_mean(
                tf.math.sigmoid(self.feature_probabilities),
                axis=-1,
            )
        )

        # And the probability diversity
        normalized_probs = tf.math.l2_normalize(
            tf.math.sigmoid(self.feature_probabilities),
            axis=-1,
        )

        # It may be the case that we actually get supervision for some of the
        # concepts!
        if self.n_supervised_concepts != 0:
            assert c_true is not None, (
                "Expected concepts to be provided during training if the "
                "concept prediction weight is non-zero!"
            )
            concept_pred_loss = 0
            seen_concepts = 0
            for aligned_idx, concept_idx in enumerate(range(c_true.shape[-1])):
                selected_samples = tf.math.logical_not(
                    tf.math.is_nan(c_true[:, concept_idx])
                )
                seen_concepts += 1
                concept_pred_loss += tf.cond(
                    tf.math.reduce_any(selected_samples),
                    lambda: tf.keras.metrics.binary_crossentropy(
                        tf.cast(
                            tf.boolean_mask(
                                c_true[:, concept_idx],
                                selected_samples,
                                axis=0,
                            ),
                            tf.float32,
                        ),
                        tf.cast(
                            tf.boolean_mask(
                                scores[:, aligned_idx],
                                selected_samples,
                                axis=0,
                            ),
                            tf.float32,
                        ),
                    ),
                    lambda: 0.0,
                )
            concept_pred_loss = concept_pred_loss / (seen_concepts + 1e-15)
        else:
            concept_pred_loss = 0

        # And include them into the total loss
        total_loss = (
            log_prob_loss -
            self.coherence_reg_weight * reg_loss_closest +
            self.diversity_reg_weight * reg_loss_similarity +
            self.feature_selection_reg_weight * prob_sparsity_loss +
            self.concept_prediction_weight * concept_pred_loss
        )

        # And report all metrics together with the main loss
        task_acc = self._acc_metric(
            tf.cast(y_true, tf.float32),
            tf.cast(y_pred, tf.float32),
        )
        total_metrics = [
            ("loss", total_loss),
            ("task_loss", log_prob_loss),
            ("accuracy", task_acc),
        ]
        if self.n_supervised_concepts != 0:
            avg_concept_acc = 0
            seen_concepts = 0
            for aligned_idx, concept_idx in enumerate(range(c_true.shape[-1])):
                selected_samples = tf.math.logical_not(
                    tf.math.is_nan(c_true[:, concept_idx])
                )
                seen_concepts += 1
                avg_concept_acc += tf.cond(
                    tf.math.reduce_any(selected_samples),
                    lambda: tf.keras.metrics.binary_accuracy(
                        tf.cast(
                            tf.boolean_mask(
                                c_true[:, concept_idx],
                                selected_samples,
                                axis=0,
                            ),
                            tf.float32
                        ),
                        tf.cast(
                            tf.boolean_mask(
                                scores[:, aligned_idx],
                                selected_samples,
                                axis=0,
                            ),
                            tf.float32
                        ),
                    ),
                    lambda: 0.0,
                )

            concept_acc = avg_concept_acc/(seen_concepts + 1e-15)
            total_metrics += [
                ("concept_pred_loss", concept_pred_loss),
                ("avg_concept_accuracy", concept_acc),
                ("mean_concept_task_acc", (task_acc + concept_acc)/2),
            ]
        total_metrics += [
            ("reg_loss_closest", self.coherence_reg_weight * reg_loss_closest),
            ("reg_loss_similarity", self.diversity_reg_weight * reg_loss_similarity),
            ("prob_sparsity_loss", self.feature_selection_reg_weight * prob_sparsity_loss),
            ("max_probability", tf.math.reduce_max(tf.nn.sigmoid(self.feature_probabilities))),
            ("min_probability", tf.math.reduce_min(tf.nn.sigmoid(self.feature_probabilities))),
            ("mean_probability", tf.math.reduce_mean(tf.nn.sigmoid(self.feature_probabilities))),
            (
                "avg_concept_size",
                tf.math.reduce_mean(
                    tf.math.reduce_sum(
                        tf.cast(
                            tf.nn.sigmoid(self.feature_probabilities) > 0.5,
                            tf.float32,
                        ),
                        axis=-1,
                    )
                )
            ),
        ]

        return total_loss, total_metrics

    def train_step(self, inputs):
        if self.n_supervised_concepts != 0:
            # Then we expect some sort of concept supervision!
            x, (y, c) = inputs
        else:
            x, y = inputs
            c = None
        # We first need to make sure that we set our concepts to labels model
        # so that we do not optimize over its parameters
        prev_trainable = self.concepts_to_labels_model.trainable
        if not self.end_to_end_training:
            self.concepts_to_labels_model.trainable = False
        with tf.GradientTape() as tape:
            if self.self_supervise_mode:
                loss, metrics = self._compute_self_supervised_loss(x)
                trainable_vars = []
                for i, generator in enumerate(self.concept_generators):
                    trainable_vars += generator.trainable_variables
                    trainable_vars += \
                        self.rec_mask_models[i].trainable_variables
                    trainable_vars += \
                        self.rec_values_models[i].trainable_variables
            else:
                loss, metrics = self._compute_supervised_loss(
                    x,
                    y,
                    c_true=c,
                    # Only train the decoder if requested by the user
                    training_decoder=self.end_to_end_training,
                )
                embedding_vars = []
                if self.features_to_embeddings_model is not None:
                    embedding_vars = \
                        self.features_to_embeddings_model.trainable_variables
                trainable_vars = (
                    self.g_model.trainable_variables +
                    [self.feature_probabilities] + (
                        self.concepts_to_labels_model.trainable_variables
                        if self.end_to_end_training else []
                    ) + (
                        self.features_to_concepts_model.trainable_variables
                        if self.end_to_end_training else []
                    ) + embedding_vars
                )
                for generator in self.concept_generators:
                    trainable_vars += generator.trainable_variables

        gradients = tape.gradient(loss, trainable_vars)
        self.optimizer.apply_gradients(
            zip(gradients, trainable_vars)
        )
        self.update_metrics(metrics)

        # And recover the previous step of the concept to labels model
        self.concepts_to_labels_model.trainable = prev_trainable

        return {
            name: self.metrics_dict[name].result()
            for name in self.metric_names
        }

    def test_step(self, inputs):
        if self.n_supervised_concepts != 0:
            x, (y, c) = inputs
        else:
            x, y = inputs
            c = None
        if self.self_supervise_mode:
            loss, metrics = self._compute_self_supervised_loss(x)
        else:
            loss, metrics = self._compute_supervised_loss(
                x,
                y,
                c_true=c,
                training_decoder=False,
            )
        return {
            name: val
            for name, val in metrics
            if ("probability" not in name) and ("avg_concept_size" not in name)
        }

    def call(self, x, **kwargs):
        concept_scores, bottleneck = self.concept_scores(
            x,
            training=kwargs.get('training', False),
        )
        predicted_labels = self.concepts_to_labels_model(
            self.g_model(bottleneck),
            training=False,
        )
        return predicted_labels, concept_scores

    def predict_bottleneck(self, x, **kwargs):
        concept_scores, bottleneck = self.concept_scores(
            x,
            training=kwargs.get('training', False),
        )
        return concept_scores, bottleneck

    def from_bottleneck(self, bottleneck):
        return self.concepts_to_labels_model(
            self.g_model(bottleneck),
            training=False,
        )

    def intervene(self, x, concepts_intervened, concept_values, **kwargs):
        concept_scores, bottleneck = self.concept_scores(
            x,
            kwargs.get('training', False),
        )
        for i, concept_idx in enumerate(concepts_intervened):
            # Update the specific concept appropriately
            bottleneck[:, concept_idx] = concept_values[i]
        predicted_labels = self.concepts_to_labels_model(
            self.g_model(bottleneck),
            training=False,
        )
        return predicted_labels, concept_scores