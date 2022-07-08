"""

Module containing our Tabular Concept Bottleneck
implementation.

"""

import concepts_xai.evaluation.metrics.completeness as completeness
import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K
import scipy

################################################################################
## Helper Functions
################################################################################

def log(x): 
    return tf.math.log(x + 1e-6)

def div(x, y):
    return tf.div(x, (y + 1e-6))
    
def Gaussian_CDF(x): #change from erf function
    return 0.5 * (1. + tf.math.erf(x / tf.math.sqrt(2.))) 

def copula_generation(X, batch_size):
    cov = np.corrcoef(X.T)
    L = scipy.linalg.cholesky(cov, lower=True)
    epsilon = np.random.normal(loc=0., scale=1., size=[np.shape(L)[0], batch_size])
    g = np.matmul(L, epsilon)
    return g.T

################################################################################
## Main Model
################################################################################

class TabCBM(tf.keras.Model):
    """
    TODO
    """

    def __init__(
        self,
        self_supervised_mode,
        features_to_concepts_model,
        concepts_to_labels_model,
        latent_dims,
        n_concepts,
        cov_mat,
        mean_inputs,
        n_exclusive_concepts=0,
        g_model=None,
        self_supervised_selection_prob=None,
        gate_estimator_weight=1,
        threshold=0.5,
        loss_fn=tf.keras.losses.sparse_categorical_crossentropy,
        top_k=32,
        temperature=1,
        coherence_reg_weight=0.1,
        diversity_reg_weight=0.1,
        prob_diversity_reg_weight=0.1,
        contrastive_reg_weight=0.1,
        feature_selection_reg_weight=1,
        seed=None,
        eps=1e-5,
        end_to_end_training=False,
        acc_metric=None,
        initial_mask_probability=0.2,
        normalized_scores=True,
        **kwargs,
    ):
        super(TabCBM, self).__init__(**kwargs)
        
        # Set initial state from parameters
        self.normalized_scores = normalized_scores
        self.self_supervise_mode = self_supervised_mode
        self.concepts_to_labels_model = concepts_to_labels_model
        self.features_to_concepts_model = features_to_concepts_model
        self.eps = eps
        self.threshold = threshold
        self.n_concepts = n_concepts
        self.n_exclusive_concepts = n_exclusive_concepts
        self.loss_fn = loss_fn
        self.top_k = top_k
        self.temperature = temperature
        self.gate_estimator_weight = gate_estimator_weight
        self.prob_diversity_reg_weight = prob_diversity_reg_weight
        self.coherence_reg_weight = coherence_reg_weight
        self.diversity_reg_weight = diversity_reg_weight
        self.contrastive_reg_weight = contrastive_reg_weight
        self.feature_selection_reg_weight = feature_selection_reg_weight
        self.latent_dims = latent_dims
        self.end_to_end_training = end_to_end_training
        input_shape = self.features_to_concepts_model.inputs[0].shape[1:]
        self.self_supervised_selection_prob = self.add_weight(
            name=f"ss_probability_vector",
            shape=(self.n_concepts, input_shape[-1],),
            dtype=tf.float32,
            initializer=tf.keras.initializers.RandomUniform(
                minval=0.4,
                maxval=0.6, #0.8,
            ), #tf.constant_initializer(initial_mask_probability),
            trainable=False,
        )
        
        # Initialize the g model which will be in charge of reconstructing
        # the model latent activations from the concept scores alone
        self.g_model = g_model
        if self.g_model is None:
            self.g_model = completeness._get_default_model(
                num_concepts=n_concepts,
                num_hidden_acts=latent_dims,
            )

        assert len(input_shape) == 1, \
            f'Expected a 1D input yet we got shape {input_shape}'

        # Setup our metrics
        self.metric_names = [
            "loss",
            "accuracy",
            "task_loss",
            "reg_loss_closest",
            "reg_loss_complement",
            "reg_loss_similarity",
            "prob_sparsity_loss",
            "prob_diversity_loss",
            "avg_mask_rec_loss",
            "avg_features_rec_loss",
            "max_probability",
            "min_probability",
            "mean_probability",
        ]
        self.metrics_dict = {
            name: tf.keras.metrics.Mean(name=name)
            for name in self.metric_names
        }
        self._acc_metric = (
            acc_metric or (
                lambda y_true, y_pred: tf.keras.metrics.sparse_top_k_categorical_accuracy(
                    y_true,
                    y_pred,
                    k=1,
                )
            )
        )
        
        # And we will initialize some models to generate concept vectors
        # from the masked input features
        self.cov_mat = cov_mat
        self.L = scipy.linalg.cholesky(self.cov_mat, lower=True).astype(np.float32)
        self.mean_inputs = mean_inputs
        self.concept_generators = []
        self.rec_values_models = []
        self.rec_mask_models = []
        for i in range(self.n_concepts):
            # For each concept we will have a simple MLP that maps
            # the masked input to the concept's input latent space. This will
            # then used to produce a score for the input of interest for
            # each concept
            self.concept_generators.append(
                tf.keras.models.Sequential(
                    [
                        tf.keras.layers.Dense(
                            100,
                            activation='relu'
                        ),
                        tf.keras.layers.Dense(
                            100,
                            activation='relu'
                        ),
                        tf.keras.layers.Dense(
                            self.latent_dims,
                            activation=None,
                        ),
                    ],
                    name=f"concept_generators_{i}",
                )
            )
            self.concept_generators[-1].compile()
            
            self.feature_probabilities = self.add_weight(
                name=f"probability_vector_logits",
                shape=(self.n_concepts, input_shape[-1],),
                dtype=tf.float32,
                initializer=tf.keras.initializers.RandomUniform(
                    minval=-1,
                    maxval=1,
                ),
                trainable=True,
            )
            
            # Initialize the mask and value reconstruction models which
            # will be in charge of reconstructing the input from its
            # masked version
            self.rec_values_models.append(tf.keras.models.Sequential(
                [
                    tf.keras.layers.Dense(
                        100,
                        activation='relu'
                    ),
                    tf.keras.layers.Dense(
                        100,
                        activation='relu'
                    ),
                    tf.keras.layers.Dense(
                        100,
                        activation='relu'
                    ),
                    tf.keras.layers.Dense(
                        input_shape[-1],
                        activation=None,
                    ),
                ],
                name=f"rec_values_model_{i}",
            ))
            self.rec_values_models[-1].compile()
            
            self.rec_mask_models.append(tf.keras.models.Sequential(
                [
                    tf.keras.layers.Dense(
                        100,
                        activation='relu'
                    ),
                    tf.keras.layers.Dense(
                        100,
                        activation='relu'
                    ),
                    tf.keras.layers.Dense(
                        100,
                        activation='relu'
                    ),
                    tf.keras.layers.Dense(
                        input_shape[-1],
                        # The sigmoid will be implicit in the loss function
                        activation=None,
                    ),
                ],
                name="rec_mask_model",
            ))
            self.rec_mask_models[-1].compile()
                
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
        epsilon = tf.debugging.check_numerics(epsilon, "y_pred has nan!")
        v = tf.transpose(tf.linalg.matmul(self.L, epsilon))
        v = tf.debugging.check_numerics(v, "v has nan!")
        u = Gaussian_CDF(v)    
        u = tf.debugging.check_numerics(u, "u has nan!")
        return tf.nn.sigmoid(
            1.0 / self.temperature * (
                log(pi) - log(1. - pi) + log(u) - log(1. - u)
            )
        )
        # TESTING TO SEE WHAT HAPPENS IF WE DO HARD THRESHOLDING
#         return tf.cast(u <= pi, tf.float32)
#         tf.cast(
#             tf.nn.sigmoid(
#                 1.0/self.temperature * (log(pi) - log(1. - pi) + log(u) - log(1. - u))
#             ) >= 0.5,
#             tf.float32,
#         )

    def mask_features(self, x):
        masked_xs = []
        for i in range(self.n_concepts):
            gate_vector = self._relaxed_multi_bernoulli_sample(
                tf.nn.sigmoid(self.feature_probabilities[i, :]),
                shape=tf.stack([tf.shape(self.L)[0], tf.shape(x)[0]], axis=0),
            )
            
            # Extend gate vector so that it can be broadcasted across all
            # samples in the batch of x
            masked_xs.append(
                gate_vector * x + (1 - gate_vector) * tf.expand_dims(
                    self.mean_inputs,
                    axis=0,
                )
            )
        return masked_xs
    
    def compute_concept_matrix(self, x):
        masked_xs = self.mask_features(x)
        concept_vectors = []
        for concept_generator, masked_x in zip(
            self.concept_generators,
            masked_xs,
        ):
            # This will be a sample of size [B, latent_dims]
            concept_vectors.append(
                tf.debugging.check_numerics(
                    concept_generator(masked_x),
                    f"concept_generator_{len(concept_vectors)} has nan"
                )
            )
            
        # We need to stack them all in the first dimension so that
        # we obtain a variable with size [B, n_concepts, latent_dims]
        return tf.stack(concept_vectors, axis=1), masked_xs

    def concept_scores(self, x, compute_reg_terms=False):
        # First we compute the concept matrix
        concept_matrix, masked_xs = self.compute_concept_matrix(x) # [B, n_concepts, latent_dims]
        concept_matrix = tf.debugging.check_numerics(
            concept_matrix,
            "concept_matrix has nan!",
        )
        concept_matrix_norm = tf.math.l2_normalize(concept_matrix, axis=1) # [B, n_concepts, latent_dims]
        concept_matrix_norm = tf.debugging.check_numerics(
            concept_matrix_norm,
            "concept_matrix_norm has nan!",
        )
        
        # Then for each concept, we have a corresponding masked inputs which we will pass through
        # the feature to concepts model (which cannot be trained)
        concept_probs = []
        concept_prob_norms = []
        
        for i, masked_x in enumerate(masked_xs):
            latent = self.features_to_concepts_model(masked_x)  # [B, latent_dims]
            latent = tf.debugging.check_numerics(
                latent,
                "latent has nan!",
            )
            latent_norm = tf.math.l2_normalize(latent, axis=-1) # [B, latent_dims]
            latent_norm = tf.debugging.check_numerics(
                latent_norm,
                "latent_norm has nan!",
            )
            
            # Compute the concept probability scores
            concept_prob = tf.squeeze(
                tf.matmul(
                    concept_matrix_norm[:, i:(i+1), :],  # [B, 1, latent_dim]
                    tf.expand_dims(latent, axis=-1),  # [B, latent_dim, 1
                ),
                axis=-1,
            ) # Shape: [B, 1]
            concept_prob = tf.debugging.check_numerics(
                concept_prob,
                "concept_prob has nan!",
            )
            concept_probs.append(concept_prob)
            concept_prob_norm = tf.squeeze(
                tf.matmul(
                    concept_matrix_norm[:, i:(i+1), :],  # [B, 1, latent_dim]
                    tf.expand_dims(latent_norm, axis=-1),  # [B, latent_dim, 1
                ),
                axis=-1,
            ) # Shape: [B, 1]
            concept_prob_norm = tf.debugging.check_numerics(
                concept_prob_norm,
                "concept_prob has nan!",
            )
            concept_prob_norms.append(concept_prob_norm)
        concept_prob = tf.concat(concept_probs, axis=1) # [B, n_concepts]
        concept_prob_norm = tf.concat(concept_prob_norms, axis=1) # [B, n_concepts]
        if self.n_exclusive_concepts:
            # Then we will make sure the n_exclusive_concepts first
            # concepts are mutually exclusive via a softmax layer
            exclusive_concepts = concept_prob[:, :self.n_exclusive_concepts]
            exclusive_concepts = tf.nn.softmax(exclusive_concepts, axis=-1)
            # The rest of concepts are considered to be binary concepts so
            # let's set them to be sigmoided
            other_concepts = tf.math.sigmoid(
                concept_prob[:, self.n_exclusive_concepts:]
            )
            concept_prob = tf.concat(
                [exclusive_concepts, other_concepts],
                axis=-1,
            )

            exclusive_concepts = concept_prob_norm[:, :self.n_exclusive_concepts]
            exclusive_concepts = tf.nn.softmax(exclusive_concepts, axis=-1)
            # The rest of concepts are considered to be binary concepts so
            # let's set them to be sigmoided
            other_concepts = tf.math.sigmoid(
                concept_prob_norm[:, self.n_exclusive_concepts:]
            )
            concept_prob_norm = tf.concat(
                [exclusive_concepts, other_concepts],
                axis=-1,
            )

        # Threshold them if they are below the given threshold value
        elif self.normalized_scores:
            concept_prob = tf.sigmoid(concept_prob)
        elif self.threshold is not None:
            concept_prob = concept_prob * tf.cast(
                (concept_prob_norm > self.threshold),
                tf.float32,
            )
        concept_prob = tf.debugging.check_numerics(concept_prob, "concept_prob in redone has nan!")

        if not compute_reg_terms:
            return concept_prob

        # Compute the regularization loss terms
        reshaped_concept_probs = tf.transpose(concept_prob_norm) # Shape: [n_concepts, B]
        reshaped_concept_probs = tf.debugging.check_numerics(
            reshaped_concept_probs,
            "reshaped_concept_probs has nan!",
        )
        completement_shape = tf.math.maximum(
            tf.shape(reshaped_concept_probs)[-1] - self.top_k,
            0,
        )

        # Idea to try: minimize frobenius norm of the difference between
        #              the concept matrices of the closest k inputs
        #              and a representative sample of them (i.e., make sure they are all
        #              as close as possible if they are close samples)
#         # Compute batch pairwise distances
#         r = tf.reduce_sum(x*x, 1)
#         # turn r into column vector
#         r = tf.reshape(r, [-1, 1])
#         D = r - 2*tf.matmul(x, tf.transpose(x)) + tf.transpose(r)
#         closest_inds = tf.nn.top_k(
#             D,
#             k=tf.math.minimum(
#                 self.top_k,
#                 tf.shape(D)[-1], #tf.shape(reshaped_concept_probs)[-1]
#             ),
#             sorted=True,
#         ).indices
#         pairwise_concept_sims = tf.linalg.matmul(
#             concept_matrix_norm,
#             tf.transpose(concept_matrix_norm, perm=[0, 2, 1]),
#         )
#         reg_loss_closest = tf.reduce_mean(
#             concept_prob_norm[closest_inds, :]
#         )
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
        reg_loss_complement = tf.cond(
            tf.equal(completement_shape, 0),
            lambda: 0.0,
            lambda: tf.reduce_mean(
                -tf.nn.top_k(
                    -reshaped_concept_probs,
                    k=completement_shape,
                    sorted=True,
                ).values
            )
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
            reg_loss_closest/self.n_concepts,
            reg_loss_complement/self.n_concepts,
            reg_loss_similarity/self.n_concepts,
        )
    
    def _compute_self_supervised_loss(self, x):
        total_loss = 0.0
        avg_mask_rec_loss = 0.0
        avg_features_rec_loss = 0.0
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
            avg_mask_rec_loss += self.gate_estimator_weight * tf.math.reduce_mean(
                mask_rec_loss
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
    
    def _compute_supervised_loss(self, x, y_true, training_decoder=False):
        # First, compute the concept scores for the given samples
        scores, reg_loss_closest, reg_loss_complement, reg_loss_similarity = self.concept_scores(
            x,
            compute_reg_terms=True
        )

        # Then predict the labels after reconstructing the activations
        # from the scores via the g model
        y_pred = self.concepts_to_labels_model(
            self.g_model(scores),
            training=training_decoder,
        )
        y_pred = tf.debugging.check_numerics(y_pred, "y_pred has nan!")
        # Compute the task loss
        if (len(y_true.shape) == 1) and (y_pred.shape[-1] == 1):
            y_pred = tf.squeeze(y_pred, axis=-1)
        log_prob_loss = tf.reduce_mean(self.loss_fn(y_true, y_pred))
        # Compute the probability sparsity losses for all concepts
        prob_sparsity_loss = tf.math.reduce_mean(
            # Using mean instead of sum to avoid explosure in case of large inputs and
            # avoid unstable training
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
        prob_diversity_loss = tf.math.reduce_mean(
            tf.matmul(
                normalized_probs,
                tf.transpose(normalized_probs)
            ) - tf.eye(self.n_concepts)
        )
        
        # And include them into the total loss
        total_loss = (
            log_prob_loss -
            self.coherence_reg_weight * reg_loss_closest +
            self.contrastive_reg_weight * reg_loss_complement +
            self.diversity_reg_weight * reg_loss_similarity + 
            self.feature_selection_reg_weight * prob_sparsity_loss +
            self.prob_diversity_reg_weight * prob_diversity_loss
        )
        
        # And report all metrics together with the main loss
        return total_loss, [
            ("loss", total_loss),
            ("task_loss", log_prob_loss),
            ("accuracy", self._acc_metric(tf.cast(y_true, tf.float32), tf.cast(y_pred, tf.float32))),
            ("reg_loss_closest", self.coherence_reg_weight * reg_loss_closest),
            ("reg_loss_complement", self.contrastive_reg_weight * reg_loss_complement),
            ("reg_loss_similarity", self.diversity_reg_weight * reg_loss_similarity),
            ("prob_sparsity_loss", self.feature_selection_reg_weight * prob_sparsity_loss),
            ("prob_diversity_loss", self.prob_diversity_reg_weight * prob_diversity_loss),
            ("max_probability", tf.math.reduce_max(tf.nn.sigmoid(self.feature_probabilities))),
            ("min_probability", tf.math.reduce_min(tf.nn.sigmoid(self.feature_probabilities))),
            ("mean_probability", tf.math.reduce_mean(tf.nn.sigmoid(self.feature_probabilities))),
        ]

    def train_step(self, inputs):
        x, y = inputs
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
                    trainable_vars += self.rec_mask_models[i].trainable_variables
                    trainable_vars += self.rec_values_models[i].trainable_variables
            else:
                loss, metrics = self._compute_supervised_loss(
                    x,
                    y,
                    # Only train the decoder if requested by the user
                    training_decoder=self.end_to_end_training,
                )
                trainable_vars = (
                    self.g_model.trainable_variables + 
                    [self.feature_probabilities] + (
                        self.concepts_to_labels_model.trainable_variables
                        if self.end_to_end_training else []
                    ) + (
                        self.features_to_concepts_model.trainable_variables
                        if self.end_to_end_training else []
                    )
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
        x, y = inputs
        if self.self_supervise_mode:
            loss, metrics = self._compute_self_supervised_loss(x)
        else:
            loss, metrics = self._compute_supervised_loss(
                x,
                y,
                training_decoder=False,
            )
        return {
            name: val
            for name, val in metrics
        }

    def call(self, x, **kwargs):
        concept_scores = self.concept_scores(x)
        predicted_labels = self.concepts_to_labels_model(
            self.g_model(concept_scores),
            training=False,
        )
        return predicted_labels, concept_scores
