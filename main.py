import tensorflow as tf
from tensorflow.keras import layers, Model
import numpy as np


def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

def lens_to_mask(lens, max_length):
    seq = tf.range(max_length)
    return tf.expand_dims(lens, axis=-1) > seq

def repeat(tensor, repeats):
    return tf.repeat(tf.expand_dims(tensor, axis=0), repeats=repeats, axis=0)

def pack_with_inverse(tensor, pattern):
    shape = tf.shape(tensor)

    def inverse(tensor):
        return tf.reshape(tensor, shape)

    return tf.reshape(tensor, [-1, shape[-1]]), inverse


def MLP(dim, dim_out=None, expand_factor=2):
    dim_out = default(dim_out, dim)
    dim_inner = dim_out * expand_factor
    return tf.keras.Sequential([
        layers.Dense(dim_inner, activation='gelu'),
        layers.Dense(dim_out)
    ])

# B-spline
class BSpline(tf.keras.layers.Layer):
    def __init__(self, learned=False):
        super(BSpline, self).__init__()
        matrix = tf.constant([
            [-1,  3, -3,  1],
            [ 3, -6,  3,  0],
            [-3,  0,  3,  0],
            [ 1,  4,  1,  0]
        ], dtype=tf.float32) / 6.0

        self.coeff = tf.Variable(matrix, trainable=learned)

    def call(self, control_points, num_times, lens=None):
        batch = tf.shape(control_points)[0]

        if exists(lens):
            times = tf.range(num_times, dtype=tf.float32) / (lens - 1)[:, None]
            times = tf.clip_by_value(times, 0., 1.)
        else:
            times = tf.linspace(0., 1., num_times)
            times = tf.tile(times[None, :], [batch, 1])

        powers = tf.range(4, dtype=tf.float32)[::-1]
        times = tf.pow(tf.expand_dims(times, axis=-1), powers)

        return tf.matmul(tf.matmul(times, self.coeff), control_points)


class SplineBasedTransformer(tf.keras.layers.Layer):
    def __init__(
        self, dim, enc_depth, model_dim=None, dec_depth=None, dim_head=64, heads=8, dropout=0.,
        num_control_points=4, always_mlp_project=False, **kwargs
    ):
        super(SplineBasedTransformer, self).__init__()
        self.dim = dim
        model_dim = default(model_dim, dim)
        dec_depth = default(dec_depth, enc_depth)

        self.num_control_points = num_control_points
        self.control_point_latents = tf.Variable(tf.zeros((num_control_points, model_dim)), trainable=True)
        self.bspliner = BSpline()

        self.mlp_in = MLP(dim, model_dim) if always_mlp_project or dim != model_dim else tf.keras.layers.Layer()

        # Transformer Encoder and Decoder
        self.encoder = [layers.MultiHeadAttention(num_heads=heads, key_dim=dim_head) for _ in range(enc_depth)]
        self.decoder = [layers.MultiHeadAttention(num_heads=heads, key_dim=dim_head) for _ in range(dec_depth)]

        self.to_control_points = layers.Dense(model_dim)
        self.mlp_out = MLP(model_dim, dim) if always_mlp_project or dim != model_dim else tf.keras.layers.Layer()

    def decode_from_latents(self, control_points, num_times, mask=None, lens=None):
        assert num_times >= 2

        splined_from_latent_controls = self.bspliner(control_points, num_times, lens=lens)

        if exists(lens) and not exists(mask):
            mask = lens_to_mask(lens, num_times)

        decoded = splined_from_latent_controls
        for layer in self.decoder:
            decoded = layer(decoded, decoded, attention_mask=mask)

        return self.mlp_out(decoded)

    def call(self, data, lens=None, return_loss=False, return_latents=False):
        batch, num_points = tf.shape(data)[0], tf.shape(data)[1]

        x = self.mlp_in(data)
        mask = lens_to_mask(lens, num_points) if exists(lens) else None

        latents = tf.tile(self.control_point_latents[None, :, :], [batch, 1, 1])
        encoder_input = tf.concat([latents, x], axis=1)

        encoded = encoder_input
        for layer in self.encoder:
            encoded = layer(encoded, encoded, attention_mask=mask)

        control_points = self.to_control_points(latents)
        recon = self.decode_from_latents(control_points, num_points, mask, lens)

        if return_loss:
            return tf.reduce_mean(tf.keras.losses.mse(recon, data))

        if return_latents:
            return recon, control_points

        return recon



class ImageAutoencoderWrapper(tf.keras.layers.Layer):
    def __init__(self, image_size, patch_size, spline_transformer, channels=3):
        super(ImageAutoencoderWrapper, self).__init__()
        self.num_times = (image_size // patch_size) ** 2

        image_patch_dim = channels * patch_size ** 2
        dim = spline_transformer.dim

        self.patch_to_tokens = tf.keras.Sequential([
            layers.Reshape((image_size // patch_size, image_size // patch_size, image_patch_dim)),
            layers.Dense(dim)
        ])

        self.spline_transformer = spline_transformer

        self.tokens_to_patch = tf.keras.Sequential([
            layers.Dense(image_patch_dim),
            layers.Reshape((image_size, image_size, channels))
        ])

    def call(self, images, return_loss=False, return_latents=False):
        tokens = self.patch_to_tokens(images)
        recon = self.spline_transformer(tokens, return_loss=return_loss, return_latents=return_latents)

        if return_loss:
            return recon

        return self.tokens_to_patch(recon)
