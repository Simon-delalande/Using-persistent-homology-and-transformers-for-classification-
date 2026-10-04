"""Masked, permutation-invariant attention classifier for persistence diagrams."""

import tensorflow as tf
from tensorflow import keras


@keras.utils.register_keras_serializable(package="TDA")
class DiagramEmbedding(keras.layers.Layer):
    """Project birth/death coordinates and replace the final slot with learned CLS."""

    def __init__(self, embedding_dim=32, **kwargs):
        super().__init__(**kwargs)
        self.embedding_dim = embedding_dim
        self.projection = keras.layers.Dense(embedding_dim)

    def build(self, input_shape):
        self.cls = self.add_weight(
            name="cls", shape=(1, 1, self.embedding_dim),
            initializer=keras.initializers.RandomNormal(stddev=0.02),
        )
        self.projection.build(input_shape[0])
        super().build(input_shape)

    def call(self, inputs):
        values, mask = inputs
        # The last slot is reserved for CLS by DiagramPreprocessor.
        points = tf.where(mask[:, :-1, None], values[:, :-1], 0.0)
        points = self.projection(points)
        points = tf.where(mask[:, :-1, None], points, 0.0)
        cls = tf.broadcast_to(self.cls, [tf.shape(values)[0], 1, self.embedding_dim])
        return tf.concat([points, cls], axis=1)

    def get_config(self):
        return {**super().get_config(), "embedding_dim": self.embedding_dim}


@keras.utils.register_keras_serializable(package="TDA")
class EncoderLayer(keras.layers.Layer):
    """Pre-normalized attention and feed-forward residual blocks with key masking."""

    def __init__(self, embedding_dim=32, num_heads=2, feedforward_dim=64,
                 dropout=0.1, **kwargs):
        super().__init__(**kwargs)
        if embedding_dim % num_heads:
            raise ValueError("embedding_dim must be divisible by num_heads.")
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.feedforward_dim = feedforward_dim
        self.dropout_rate = dropout
        self.norm1 = keras.layers.LayerNormalization(epsilon=1e-6)
        self.norm2 = keras.layers.LayerNormalization(epsilon=1e-6)
        self.attention = keras.layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=embedding_dim // num_heads, dropout=dropout,
        )
        self.ffn = keras.Sequential([
            keras.layers.Dense(feedforward_dim, activation="relu"),
            keras.layers.Dense(embedding_dim),
        ])
        self.dropout1 = keras.layers.Dropout(dropout)
        self.dropout2 = keras.layers.Dropout(dropout)

    def build(self, input_shape):
        values_shape, _ = input_shape
        self.norm1.build(values_shape)
        self.norm2.build(values_shape)
        self.attention.build(values_shape, values_shape)
        self.ffn.build(values_shape)
        self.dropout1.build(values_shape)
        self.dropout2.build(values_shape)
        super().build(input_shape)

    def call(self, inputs, training=False):
        x, mask = inputs
        normalized = self.norm1(x)
        # All queries see only valid keys, including CLS. Broadcast over queries.
        attended = self.attention(
            normalized, normalized, attention_mask=mask[:, None, :], training=training,
        )
        x = x + self.dropout1(attended, training=training)
        x = x + self.dropout2(self.ffn(self.norm2(x)), training=training)
        # Reset padded query states after each block; never infer masks from zeros.
        return tf.where(mask[:, :, None], x, 0.0)

    def get_config(self):
        return {
            **super().get_config(), "embedding_dim": self.embedding_dim,
            "num_heads": self.num_heads, "feedforward_dim": self.feedforward_dim,
            "dropout": self.dropout_rate,
        }


@keras.utils.register_keras_serializable(package="TDA")
class CLSReadout(keras.layers.Layer):
    """Read the final CLS position for any input sequence length."""

    def call(self, x):
        return x[:, -1, :]


def build_attention_model(*, embedding_dim=32, num_heads=2, num_layers=2,
                          feedforward_dim=64, dropout=0.1):
    """Build a set classifier with dynamic length and no positional encoding."""
    if num_layers < 1:
        raise ValueError("num_layers must be positive.")
    values = keras.Input(shape=(None, 2), name="values")
    mask = keras.Input(shape=(None,), dtype="bool", name="valid_mask")
    x = DiagramEmbedding(embedding_dim, name="diagram_embedding")([values, mask])
    for index in range(num_layers):
        x = EncoderLayer(
            embedding_dim, num_heads, feedforward_dim, dropout, name=f"encoder_{index}",
        )([x, mask])
    x = CLSReadout(name="cls_readout")(x)
    x = keras.layers.LayerNormalization(epsilon=1e-6, name="readout_norm")(x)
    x = keras.layers.Dense(32, activation="relu", name="classifier_hidden")(x)
    x = keras.layers.Dropout(dropout)(x)
    predictions = keras.layers.Dense(3, activation="softmax", name="classes")(x)
    return keras.Model(
        inputs={"values": values, "valid_mask": mask}, outputs=predictions,
        name="diagram_attention",
    )
