import concepts_xai.evaluation.metrics.completeness as completeness
import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K
import scipy

'''
Re-implementation of the "On Completeness-aware Concept-Based Explanations in
Deep Neural Networks":

1) See https://arxiv.org/abs/1910.07969 for the original paper

2)  See https://github.com/chihkuanyeh/concept_exp for the original paper
    implementation

'''


class TopicModel(tf.keras.Model):
    """Base class of a topic model."""

    def __init__(
        self,
        concepts_to_labels_model,
        n_channels,
        n_concepts,
        g_model=None,
        threshold=0.5,
        loss_fn=tf.keras.losses.sparse_categorical_crossentropy,
        top_k=32,
        lambda1=0.1,
        lambda2=0.1,
        seed=None,
        eps=1e-5,
        data_format="channels_last",
        allow_gradient_flow_to_c2l=False,
        acc_metric=None,
        initial_topic_vector=None,
        **kwargs,
    ):
        super(TopicModel, self).__init__(**kwargs)

        initializer = tf.keras.initializers.RandomUniform(
            minval=-0.5,
            maxval=0.5,
            seed=seed,
        )

        # Initialize our topic vector tensor which we will learn
        # as part of our training
        if initial_topic_vector is not None:
            self.topic_vector = self.add_weight(
                name="topic_vector",
                shape=(n_channels, n_concepts),
                dtype=tf.float32,
                initializer=lambda *args, **kwargs: initial_topic_vector,
                trainable=True,
            )
        else:
            self.topic_vector = self.add_weight(
                name="topic_vector",
                shape=(n_channels, n_concepts),
                dtype=tf.float32,
                initializer=tf.keras.initializers.RandomUniform(
                    minval=-0.5,
                    maxval=0.5,
                    seed=seed,
                ),
                trainable=True,
            )

        # Initialize the g model which will be in charge of reconstructing
        # the model latent activations from the concept scores alone
        self.g_model = g_model
        if self.g_model is None:
            self.g_model = completeness._get_default_model(
                num_concepts=n_concepts,
                num_hidden_acts=n_channels,
            )

        # Set the concept-to-label predictor model
        self.concepts_to_labels_model = concepts_to_labels_model

        # Set remaining model hyperparams
        self.eps = eps
        self.threshold = threshold
        self.n_concepts = n_concepts
        self.loss_fn = loss_fn
        self.top_k = top_k
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.n_channels = n_channels
        self.allow_gradient_flow_to_c2l = allow_gradient_flow_to_c2l
        assert data_format in ["channels_last", "channels_first"], (
            f'Expected data format to be either "channels_last" or '
            f'"channels_first" however we obtained "{data_format}".'
        )
        if data_format == "channels_last":
            self._channel_axis = -1
        else:
            raise ValueError(
                'Currently we only support "channels_last" data_format'
            )

        self.metric_names = ["loss", "mean_sim", "accuracy"]
        self.metrics_dict = {
            name: tf.keras.metrics.Mean(name=name)
            for name in self.metric_names
        }
        self._acc_metric = (
            acc_metric or (lambda y_true, y_pred: tf.keras.metrics.sparse_top_k_categorical_accuracy(
                y_true=y_true,
                y_pred=y_pred,
                k=1,
            ))
        )

    @property
    def metrics(self):
        return [self.metrics_dict[name] for name in self.metric_names]

    def update_metrics(self, losses):
        for (loss_name, loss) in losses:
            self.metrics_dict[loss_name].update_state(loss)

    def concept_scores(self, x, compute_reg_terms=False):
        # Compute the concept representation by first normalizing across the
        # channel dimension both the concept vectors and the input
        assert x.shape[self._channel_axis] == self.n_channels, (
            f'Expected input to have {self.n_channels} elements in its '
            f'channels axis (defined as axis {self._channel_axis}). '
            f'Instead, we found the input to have shape {x.shape}.'
        )

        x_norm = tf.math.l2_normalize(x, axis=self._channel_axis)
        topic_vector_norm = tf.math.l2_normalize(self.topic_vector, axis=0)
        # Compute the concept probability scores
        topic_prob = K.dot(x, topic_vector_norm)
        topic_prob_norm = K.dot(x_norm, topic_vector_norm)

        # Threshold them if they are below the given threshold value
        if self.threshold is not None:
            topic_prob = topic_prob * tf.cast(
                (topic_prob_norm > self.threshold),
                tf.float32,
            )
        topic_prob_sum = tf.reduce_sum(
            topic_prob,
            axis=self._channel_axis,
            keepdims=True,
        )
        # And normalize the actual scores
        topic_prob = topic_prob / (topic_prob_sum + self.eps)
        if not compute_reg_terms:
            return topic_prob

        # Compute the regularization loss terms
        reshaped_topic_probs = tf.transpose(
            tf.reshape(topic_prob_norm, (-1, self.n_concepts))
        )
        reg_loss_1 = tf.reduce_mean(
            tf.nn.top_k(
                reshaped_topic_probs,
                k=tf.math.minimum(
                    self.top_k,
                    tf.shape(reshaped_topic_probs)[-1]
                ),
                sorted=True,
            ).values
        )

        reg_loss_2 = tf.reduce_mean(
            K.dot(tf.transpose(topic_vector_norm), topic_vector_norm) -
            tf.eye(self.n_concepts)
        )
        return topic_prob, reg_loss_1, reg_loss_2

    def _compute_loss(self, x, y_true, training):
        # First, compute the concept scores for the given samples
        scores, reg_loss_1, reg_loss_2 = self.concept_scores(
            x,
            compute_reg_terms=True
        )

        # Then predict the labels after reconstructing the activations
        # from the scores via the g model
        y_pred = self.concepts_to_labels_model(
            self.g_model(scores),
            training=training,
        )

        # Compute the task loss
        log_prob_loss = tf.reduce_mean(self.loss_fn(y_true, y_pred))

        # And include them into the total loss
        total_loss = (
            log_prob_loss -
            self.lambda1 * reg_loss_1 +
            self.lambda2 * reg_loss_2
        )

        # Compute the accuracy metric to track
        return total_loss, reg_loss_1, self._acc_metric(y_true, y_pred)

    def train_step(self, inputs):
        x, y = inputs

        # We first need to make sure that we set our concepts to labels model
        # so that we do not optimize over its parameters
        prev_trainable = self.concepts_to_labels_model.trainable
        if not self.allow_gradient_flow_to_c2l:
            self.concepts_to_labels_model.trainable = False
        with tf.GradientTape() as tape:
            loss, mean_sim, acc = self._compute_loss(
                x,
                y,
                # Only train the decoder if requested by the user
                training=self.allow_gradient_flow_to_c2l,
            )

        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(
            zip(gradients, self.trainable_variables)
        )
        self.update_metrics([
            ("loss", loss),
            ("mean_sim", mean_sim),
            ("accuracy", acc),
        ])

        # And recover the previous step of the concept to labels model
        self.concepts_to_labels_model.trainable = prev_trainable

        return {
            name: self.metrics_dict[name].result()
            for name in self.metric_names
        }

    def test_step(self, inputs):
        x, y = inputs
        loss, mean_sim, acc = self._compute_loss(x, y, training=False)
        self.update_metrics([
            ("loss", loss),
            ("mean_sim", mean_sim),
            ("accuracy", acc)
        ])

        return {
            name: self.metrics_dict[name].result()
            for name in self.metric_names
        }

    def call(self, x, **kwargs):
        concept_scores = self.concept_scores(x)
        predicted_labels = self.concepts_to_labels_model(
            self.g_model(concept_scores),
            training=False,
        )
        return predicted_labels, concept_scores


class AutoencodingTopicModel(TopicModel):
    """
    Base class of a topic model with regularizer which incentivizes learnt
    topics to be able to predict the input features.
    """

    def __init__(
        self,
        features_to_concepts_model,
        concepts_to_labels_model,
        n_channels,
        n_concepts,
        rec_model=None,
        g_model=None,
        threshold=0.5,
        loss_fn=tf.keras.losses.sparse_categorical_crossentropy,
        top_k=32,
        lambda1=0.1,
        lambda2=0.1,
        seed=None,
        eps=1e-5,
        data_format="channels_last",
        allow_gradient_flow_to_c2l=False,
        acc_metric=None,
        initial_topic_vector=None,
        rec_strength=0.5,
        rec_loss_fn=tf.keras.losses.mean_squared_error,
        rec_input_transform=lambda x: x,
        **kwargs,
    ):
        super(AutoencodingTopicModel, self).__init__(
            concepts_to_labels_model=concepts_to_labels_model,
            n_channels=n_channels,
            n_concepts=n_concepts,
            g_model=g_model,
            threshold=threshold,
            loss_fn=loss_fn,
            top_k=top_k,
            lambda1=lambda1,
            lambda2=lambda2,
            seed=seed,
            eps=eps,
            data_format=data_format,
            allow_gradient_flow_to_c2l=allow_gradient_flow_to_c2l,
            acc_metric=acc_metric,
            initial_topic_vector=initial_topic_vector,
            **kwargs,
        )
        self.rec_strength = rec_strength
        self.rec_loss_fn = rec_loss_fn
        self.features_to_concepts_model = features_to_concepts_model
        input_shape = self.features_to_concepts_model.inputs[0].shape[1:]
        self.rec_model = rec_model or tf.keras.models.Sequential([
            tf.keras.layers.Dense(
                64,
                activation='relu'
            ),
            tf.keras.layers.Dense(
                128,
                activation='relu'
            ),
            tf.keras.layers.Dense(
                256,
                activation='relu'
            ),
            tf.keras.layers.Dense(
                np.prod(input_shape),
                activation=None,
            ),
            tf.keras.layers.Reshape(input_shape)
        ])
        self.rec_input_transform = rec_input_transform
        self.metric_names.append("rec_loss")
        self.metrics_dict["rec_loss"] = tf.keras.metrics.Mean(
            name="rec_loss",
        )

    def _compute_loss(self, x, y_true, training):
        # First, compute the concept scores for the given samples
        prev_trainable = self.features_to_concepts_model.trainable
        self.features_to_concepts_model.trainable = False
        latent = self.features_to_concepts_model(x)
        scores, reg_loss_1, reg_loss_2 = self.concept_scores(
            latent,
            compute_reg_terms=True
        )

        # Then predict the labels after reconstructing the activations
        # from the scores via the g model
        y_pred = self.concepts_to_labels_model(
            self.g_model(scores),
            training=training,
        )

        # Compute the task loss
        log_prob_loss = tf.reduce_mean(self.loss_fn(y_true, y_pred))

        # And the reconstruction loss
        rec_logits = self.rec_model(scores)
        x_trans = self.rec_input_transform(x)
        rec_loss = tf.reduce_mean(
            self.rec_loss_fn(
                x_trans,
                rec_logits,
            )
        )

        # And include them into the total loss
        total_loss = (
            log_prob_loss -
            self.lambda1 * reg_loss_1 +
            self.lambda2 * reg_loss_2 +
            self.rec_strength * rec_loss
        )

        # Compute the accuracy metric to track
        self.update_metrics([("rec_loss", rec_loss)])
        self.features_to_concepts_model.trainable = prev_trainable
        return total_loss, reg_loss_1, self._acc_metric(y_true, y_pred)

    def call(self, x, **kwargs):
        concept_scores = self.concept_scores(
            self.features_to_concepts_model(x)
        )
        predicted_labels = self.concepts_to_labels_model(
            self.g_model(concept_scores),
            training=False,
        )
        return predicted_labels, concept_scores


class SplitAutoencodingTopicModel(AutoencodingTopicModel):
    """
    """

    def __init__(
        self,
        features_to_concepts_model,
        concepts_to_labels_model,
        n_channels,
        n_concepts,
        n_exclusive_concepts=0,
        rec_model=None,
        g_model=None,
        threshold=0.5,
        loss_fn=tf.keras.losses.sparse_categorical_crossentropy,
        top_k=32,
        lambda1=0.1,
        lambda2=0.1,
        seed=None,
        eps=1e-5,
        data_format="channels_last",
        allow_gradient_flow_to_c2l=False,
        acc_metric=None,
        initial_topic_vector=None,
        rec_strength=0.5,
        rec_loss_fn=tf.keras.losses.mean_squared_error,
        rec_input_transform=lambda x: x,
        **kwargs,
    ):
        super(SplitAutoencodingTopicModel, self).__init__(
            features_to_concepts_model=features_to_concepts_model,
            concepts_to_labels_model=concepts_to_labels_model,
            n_channels=n_channels,
            n_concepts=n_concepts,
            rec_model=rec_model,
            g_model=g_model,
            threshold=threshold,
            loss_fn=loss_fn,
            top_k=top_k,
            lambda1=lambda1,
            lambda2=lambda2,
            seed=seed,
            eps=eps,
            data_format=data_format,
            allow_gradient_flow_to_c2l=allow_gradient_flow_to_c2l,
            acc_metric=acc_metric,
            initial_topic_vector=initial_topic_vector,
            rec_strength=rec_strength,
            rec_loss_fn=rec_loss_fn,
            rec_input_transform=rec_input_transform,
        )
        self.n_exclusive_concepts = n_exclusive_concepts


    def concept_scores(self, x, compute_reg_terms=False):
        # Compute the concept representation by first normalizing across the
        # channel dimension both the concept vectors and the input
        assert x.shape[self._channel_axis] == self.n_channels, (
            f'Expected input to have {self.n_channels} elements in its '
            f'channels axis (defined as axis {self._channel_axis}). '
            f'Instead, we found the input to have shape {x.shape}.'
        )

        x_norm = tf.math.l2_normalize(x, axis=self._channel_axis)
        topic_vector_norm = tf.math.l2_normalize(self.topic_vector, axis=0)
        # Compute the concept probability scores
        topic_prob = K.dot(x, topic_vector_norm)
        topic_prob_norm = K.dot(x_norm, topic_vector_norm)

        if self.n_exclusive_concepts:
            # Then we will make sure the n_exclusive_concepts first
            # concepts are mutually exclusive via a softmax layer
            exclusive_concepts = topic_prob[:, :self.n_exclusive_concepts]
            exclusive_concepts = tf.nn.softmax(exclusive_concepts, axis=-1)
            # The rest of concepts are considered to be binary concepts so
            # let's set them to be sigmoided
            other_concepts = tf.math.sigmoid(
                topic_prob[:, self.n_exclusive_concepts:]
            )
            topic_prob = tf.concat(
                [exclusive_concepts, other_concepts],
                axis=-1,
            )

            exclusive_concepts = topic_prob_norm[:, :self.n_exclusive_concepts]
            exclusive_concepts = tf.nn.softmax(exclusive_concepts, axis=-1)
            # The rest of concepts are considered to be binary concepts so
            # let's set them to be sigmoided
            other_concepts = tf.math.sigmoid(
                topic_prob_norm[:, self.n_exclusive_concepts:]
            )
            topic_prob_norm = tf.concat(
                [exclusive_concepts, other_concepts],
                axis=-1,
            )

        # Threshold them if they are below the given threshold value
        elif self.threshold is not None:
            topic_prob = topic_prob * tf.cast(
                (topic_prob_norm > self.threshold),
                tf.float32,
            )

        if not compute_reg_terms:
            return topic_prob

        # Compute the regularization loss terms
        reshaped_topic_probs = tf.transpose(
            tf.reshape(topic_prob_norm, (-1, self.n_concepts))
        )
        reg_loss_1 = tf.reduce_mean(
            tf.nn.top_k(
                reshaped_topic_probs,
                k=tf.math.minimum(
                    self.top_k,
                    tf.shape(reshaped_topic_probs)[-1]
                ),
                sorted=True,
            ).values
        )

        reg_loss_2 = tf.reduce_mean(
            K.dot(tf.transpose(topic_vector_norm), topic_vector_norm) -
            tf.eye(self.n_concepts)
        )
        return topic_prob, reg_loss_1, reg_loss_2

    def _compute_loss(self, x, y_true, training):
        # First, compute the concept scores for the given samples
        prev_trainable = self.features_to_concepts_model.trainable
        self.features_to_concepts_model.trainable = False
        latent = self.features_to_concepts_model(x)
        scores, reg_loss_1, reg_loss_2 = self.concept_scores(
            latent,
            compute_reg_terms=True
        )

        # Then predict the labels after reconstructing the activations
        # from the scores via the g model
        y_pred = self.concepts_to_labels_model(
            self.g_model(scores),
            training=training,
        )

        # Compute the task loss
        log_prob_loss = tf.reduce_mean(self.loss_fn(y_true, y_pred))

        # And the reconstruction loss
        rec_logits = self.rec_model(scores)
        x_trans = self.rec_input_transform(x)
        rec_loss = tf.reduce_mean(
            self.rec_loss_fn(
                x_trans,
                rec_logits,
            )
        )

        # And include them into the total loss
        total_loss = (
            log_prob_loss -
            self.lambda1 * reg_loss_1 +
            self.lambda2 * reg_loss_2 +
            self.rec_strength * rec_loss
        )

        # Compute the accuracy metric to track
        self.update_metrics([("rec_loss", rec_loss)])
        self.features_to_concepts_model.trainable = prev_trainable
        return total_loss, reg_loss_1, self._acc_metric(y_true, y_pred)

    def call(self, x, **kwargs):
        concept_scores = self.concept_scores(
            self.features_to_concepts_model(x)
        )
        predicted_labels = self.concepts_to_labels_model(
            self.g_model(concept_scores),
            training=False,
        )
        return predicted_labels, concept_scores


class InterpretableNonlinearModule(tf.keras.layers.Layer):
    """Base class of a topic model."""

    def __init__(
        self,
        n_concepts,
        g_model=None,
        
        rec_model=None,
        rec_strength=0.5,
        rec_loss_fn=tf.keras.losses.mean_squared_error,
        rec_input_transform=lambda x: x,
        rec_shape=None,

        threshold=0.5,
        top_k=32,
        lambda1=0.1,
        lambda2=0.1,
        lambda3=0.1,
        seed=None,
        eps=1e-5,
        data_format="channels_last",
        initial_topic_vector=None,
        n_exclusive_concepts=0,
        **kwargs,
    ):
        super(InterpretableNonlinearModule, self).__init__(**kwargs)
        self.initial_topic_vector = initial_topic_vector

        # Set remaining model hyperparams
        self.g_model = g_model
        self.eps = eps
        self.threshold = threshold
        self.n_concepts = n_concepts
        self.top_k = top_k
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.lambda3 = lambda3
        self.rec_strength = rec_strength
        self.rec_model = rec_model
        self.rec_loss_fn = rec_loss_fn
        self.rec_input_transform = rec_input_transform
        self.rec_shape = rec_shape
        self.n_exclusive_concepts = n_exclusive_concepts
        self.seed = seed

        assert data_format in ["channels_last", "channels_first"], (
            f'Expected data format to be either "channels_last" or '
            f'"channels_first" however we obtained "{data_format}".'
        )
        if data_format == "channels_last":
            self._channel_axis = -1
        else:
            raise ValueError(
                'Currently we only support "channels_last" data_format'
            )

    def build(self, input_shape):
        if isinstance(input_shape, (list, tuple)):
            # Then we were provided the sample corresponding to the latent code
            # for regularization purposes
            input_shape, sample_input_sample = input_shape
        else:
            input_shape, sample_input_sample = input_shape, None
        n_channels = input_shape[self._channel_axis]
        initializer = tf.keras.initializers.RandomUniform(
            minval=-0.5,
            maxval=0.5,
            seed=self.seed,
        )
        # Initialize our topic vector tensor which we will learn
        # as part of our training
        if self.initial_topic_vector is not None:
            self.concept_matrix = self.add_weight(
                name="concept_matrix",
                shape=(n_channels, self.n_concepts),
                dtype=tf.float32,
                initializer=lambda *args, **kwargs: self.initial_topic_vector,
                trainable=True,
            )
        else:
            self.concept_matrix = self.add_weight(
                name="concept_matrix",
                shape=(n_channels, self.n_concepts),
                dtype=tf.float32,
                initializer=tf.keras.initializers.RandomUniform(
                    minval=-0.5,
                    maxval=0.5,
                    seed=self.seed,
                ),
                trainable=True,
            )

        # Initialize the g model which will be in charge of reconstructing
        # the model latent activations from the concept scores alone
        if self.g_model is None:
            self.g_model = completeness._get_default_model(
                num_concepts=self.n_concepts,
                num_hidden_acts=n_channels,
            )

        # Initialize reconstruction loss model
        if self.rec_strength:
            if self.rec_shape is not None:
                rec_out_shape = self.rec_shape
            else:
                rec_out_shape = input_shape
            self.rec_model = self.rec_model or tf.keras.models.Sequential([
                tf.keras.layers.Dense(
                    64,
                    activation='relu'
                ),
                tf.keras.layers.Dense(
                    128,
                    activation='relu'
                ),
                tf.keras.layers.Dense(
                    256,
                    activation='relu'
                ),
                tf.keras.layers.Dense(
                    np.prod(rec_out_shape),
                    activation=None,
                ),
                tf.keras.layers.Reshape(rec_out_shape)
            ])

    def concept_scores(self, x, compute_reg_terms=False):
        # Compute the concept representation by first normalizing across the
        # channel dimension both the concept vectors and the input

        x_norm = tf.math.l2_normalize(x, axis=self._channel_axis)
        topic_vector_norm = tf.math.l2_normalize(self.concept_matrix, axis=0)
        # Compute the concept probability scores
        topic_prob = K.dot(x, topic_vector_norm)
        topic_prob_norm = K.dot(x_norm, topic_vector_norm)
        if self.n_exclusive_concepts:
            # Then we will make sure the n_exclusive_concepts first
            # concepts are mutually exclusive via a softmax layer
            exclusive_concepts = topic_prob[:, :self.n_exclusive_concepts]
            exclusive_concepts = tf.nn.softmax(exclusive_concepts, axis=-1)
            # The rest of concepts are considered to be binary concepts so
            # let's set them to be sigmoided
            other_concepts = tf.math.sigmoid(
                topic_prob[:, self.n_exclusive_concepts:]
            )
            topic_prob = tf.concat(
                [exclusive_concepts, other_concepts],
                axis=-1,
            )

            exclusive_concepts = topic_prob_norm[:, :self.n_exclusive_concepts]
            exclusive_concepts = tf.nn.softmax(exclusive_concepts, axis=-1)
            # The rest of concepts are considered to be binary concepts so
            # let's set them to be sigmoided
            other_concepts = tf.math.sigmoid(
                topic_prob_norm[:, self.n_exclusive_concepts:]
            )
            topic_prob_norm = tf.concat(
                [exclusive_concepts, other_concepts],
                axis=-1,
            )

        # Threshold them if they are below the given threshold value
        elif self.threshold is not None:
            topic_prob = topic_prob * tf.cast(
                (topic_prob_norm > self.threshold),
                tf.float32,
            )

        if not compute_reg_terms:
            return topic_prob

        # Compute the regularization loss terms
        reshaped_topic_probs = tf.transpose(
            tf.reshape(topic_prob_norm, (-1, self.n_concepts))
        )
#         reshaped_topic_probs = tf.transpose(
#             tf.reshape(topic_prob, (-1, self.n_concepts))
#         )
        completement_shape = tf.math.maximum(tf.shape(reshaped_topic_probs)[-1] - self.top_k, 0)
        reg_loss_closest = tf.reduce_mean(
            tf.nn.top_k(
                reshaped_topic_probs,
                k=tf.math.minimum(
                    self.top_k,
                    tf.shape(reshaped_topic_probs)[-1]
                ),
                sorted=True,
            ).values
        )
        reg_loss_complement = tf.cond(
            tf.equal(completement_shape, 0),
            lambda: 0.0,
            lambda: tf.reduce_mean(
                -tf.nn.top_k(
                    -reshaped_topic_probs,
                    k=completement_shape,
                    sorted=True,
                ).values
            )
        )
        reg_loss_similarity = tf.reduce_mean(
            tf.math.abs(
                K.dot(tf.transpose(topic_vector_norm), topic_vector_norm) -
                tf.eye(self.n_concepts)
            )
        )
#         reg_loss_similarity = tf.reduce_mean(
#             K.dot(tf.transpose(topic_vector_norm), topic_vector_norm) -
#             tf.eye(self.n_concepts)
#         )
        return topic_prob, reg_loss_closest, reg_loss_complement, reg_loss_similarity

    def call(self, inputs, **kwargs):
        if isinstance(inputs, (list, tuple)):
            # Then we were provided the sample corresponding to the latent code
            # for regularization purposes
            latent, input_sample = inputs
        else:
            latent, input_sample = inputs, None

        # First and foremost: compute concept scores
        concept_scores, reg_loss_closest, reg_loss_complement, reg_loss_similarity = self.concept_scores(
            latent,
            compute_reg_terms=True
        )

        # Construct the reconstruction loss
        if self.rec_strength:
            rec_logits = self.rec_model(concept_scores)
            if (input_sample is not None) and (self.rec_shape is not None):
                assert input_sample.shape[1:] == self.rec_shape, (
                    f'Expected second input to InterpretableNonlinearModule to '
                    f'have the same shape as its rec_shape argument '
                    f'({self.rec_shape}). Instead we got an input with shape '
                    f'{input_sample.shape}.'
                )
                x = input_sample
            else:
                x = latent
            x_trans = self.rec_input_transform(x)
            rec_loss = tf.reduce_mean(
                self.rec_loss_fn(
                    x_trans,
                    rec_logits,
                )
            )
        else:
            rec_loss = 0

        # Add all the losses this layer will contribute during training
        self.add_loss(
            -self.lambda1 * reg_loss_closest +
            self.lambda2 * reg_loss_similarity +
            self.lambda3 * reg_loss_complement +
            self.rec_strength * rec_loss
        )
        self.add_metric(reg_loss_closest, name='reg_loss_closest')
        self.add_metric(reg_loss_complement, name='reg_loss_complement')
        self.add_metric(reg_loss_similarity, name='reg_loss_similarity')
        if self.rec_strength:
            self.add_metric(rec_loss, name='rec_loss')

        # And return both the output hidden code of this module as well as
        # its predicted concept scores
        return self.g_model(concept_scores), concept_scores
