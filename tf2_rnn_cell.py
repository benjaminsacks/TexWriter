"""
This module provides a Keras-compatible RNN cell for handwriting synthesis,
based on the architecture described in Alex Graves' paper, "Generating
Sequences with Recurrent Neural Networks."

The `KerasLSTMAttentionCell` is a custom `tf.keras.layers.Layer` that
functions as an RNN cell. It is designed to be unrolled manually within a
TensorFlow `while_loop` due to its complex state and the dynamic nature of
the attention mechanism.
"""
import tensorflow as tf
import numpy as np


class KerasLSTMAttentionCell(tf.keras.layers.Layer):
    """
    A Keras-compatible implementation of an LSTM cell with a content-based
    attention mechanism, designed for sequence generation tasks like handwriting
    synthesis.

    This cell encapsulates a three-layer LSTM stack and a Gaussian Mixture
    Model (GMM) based attention mechanism. At each timestep, it processes
    an input stroke and the previous state to produce an output and a new state.

    The state is a complex tuple containing:
    - Hidden (h) and cell (c) states for each of the three LSTMs.
    - `kappa`: The center points for the Gaussian attention windows.
    - `w`: The context vector, which is a weighted sum of the input characters.
    - `phi`: The attention weights applied to the input characters.
    """

    def __init__(self, lstm_size, num_attn_mixture_components, vocab_size, dropout_rate=0.0, **kwargs):
        """
        Initializes the cell and its internal layers.

        Args:
            lstm_size (int): The number of units in each LSTM layer.
            num_attn_mixture_components (int): The number of Gaussian mixtures
                to use for the attention mechanism.
            vocab_size (int): The number of unique characters in the vocabulary.
            dropout_rate (float): Dropout rate for outputs.
        """
        super().__init__(**kwargs)
        self.lstm_size = lstm_size
        self.num_attn_mixture_components = num_attn_mixture_components
        self.vocab_size = vocab_size
        self.dropout_rate = dropout_rate

        # Internal layers of the cell (no internal recurrent dropout to avoid while_loop issues)
        self.lstm1 = tf.keras.layers.LSTMCell(self.lstm_size, name='lstm_cell_1')
        self.lstm2 = tf.keras.layers.LSTMCell(self.lstm_size, name='lstm_cell_2')
        self.lstm3 = tf.keras.layers.LSTMCell(self.lstm_size, name='lstm_cell_3')
        # Dense layer to predict attention parameters (alpha, beta, kappa)
        self.attention_layer = tf.keras.layers.Dense(
            3 * self.num_attn_mixture_components, name='attention_params'
        )

    @property
    def state_size(self):
        """
        Returns a list of integer sizes for each component of the cell's state.
        A `None` entry indicates a dynamic size, which is handled in the `call` method.
        """
        return [
            self.lstm_size,                    # h1: Hidden state of LSTM 1
            self.lstm_size,                    # c1: Cell state of LSTM 1
            self.lstm_size,                    # h2: Hidden state of LSTM 2
            self.lstm_size,                    # c2: Cell state of LSTM 2
            self.lstm_size,                    # h3: Hidden state of LSTM 3
            self.lstm_size,                    # c3: Cell state of LSTM 3
            self.num_attn_mixture_components,  # kappa: Attention window centers
            self.vocab_size,                   # w: Context vector (window)
            None,                              # phi: Attention weights (dynamic)
        ]

    @property
    def output_size(self):
        """Returns the size of the cell's output."""
        return self.lstm_size

    def get_initial_state(self, batch_size, dtype):
        """
        Creates the initial zero-filled state for the cell.
        Placeholders are created for dynamically sized state components.
        """
        def get_zero_state(size, name):
            # For dynamically sized states, create a minimal placeholder that
            # will be properly shaped in the first iteration of the `call` method.
            if size is None:
                return tf.zeros([batch_size, 1], dtype=dtype, name=name)
            return tf.zeros([batch_size, size], dtype=dtype, name=name)

        return [get_zero_state(s, f'initial_state_{i}') for i, s in enumerate(self.state_size)]

    def build(self, input_shape):
        """
        Builds the internal layers of the cell with the correct input shapes.
        This method is called by Keras automatically the first time the layer is used.
        """
        # Shape for LSTM1 input: concat([prev_w, timestep_input])
        s1_in_shape = tf.TensorShape((None, self.vocab_size + 3))
        self.lstm1.build(s1_in_shape)

        # Shape for Attention layer input: concat([prev_w, timestep_input, s1_out])
        attention_input_shape = tf.TensorShape((None, self.vocab_size + 3 + self.lstm_size))
        self.attention_layer.build(attention_input_shape)

        # Shape for LSTM2 input: concat([timestep_input, s1_out, w])
        s2_in_shape = tf.TensorShape((None, 3 + self.lstm_size + self.vocab_size))
        self.lstm2.build(s2_in_shape)

        # Shape for LSTM3 input: concat([timestep_input, s2_out, w])
        s3_in_shape = tf.TensorShape((None, 3 + self.lstm_size + self.vocab_size))
        self.lstm3.build(s3_in_shape)

        super().build(input_shape)  # Mark this layer as built

    def call(self, inputs, states, training=False):
        """
        Performs one timestep calculation for the RNN cell.

        Args:
            inputs (dict): A dictionary containing the inputs for the current timestep.
                'strokes': The stroke data for the current timestep. (batch, 3)
                'attention_values': The one-hot encoded character sequence. (batch, char_len, vocab_size)
                'c_len': The length of each character sequence in the batch. (batch,)
            states (list): The list of state tensors from the previous timestep.
            training (bool): Whether the model is in training mode.

        Returns:
            A tuple (output, new_states):
                output (Tensor): The output of the cell for the current timestep.
                new_states (list): The list of state tensors for the next timestep.
        """
        # --- Unpack inputs and states ---
        timestep_input = inputs['strokes']
        attention_values = inputs['attention_values']
        attention_values_lengths = inputs['c_len']
        h1, c1, h2, c2, h3, c3, prev_kappa, prev_w, prev_phi = states

        batch_size = tf.shape(attention_values)[0]
        char_len = tf.shape(attention_values)[1]
        vocab_size = tf.shape(attention_values)[2]

        # --- Initialize dynamic states on the first timestep ---
        # If prev_w has a dummy shape from get_initial_state, resize it.
        prev_w = tf.cond(
            tf.shape(prev_w)[-1] != vocab_size,
            lambda: tf.zeros([batch_size, vocab_size], dtype=tf.float32),
            lambda: prev_w
        )
        # If prev_phi has a dummy shape, resize it based on character length.
        prev_phi = tf.cond(
            tf.shape(prev_phi)[-1] != char_len,
            lambda: tf.zeros([batch_size, char_len], dtype=tf.float32),
            lambda: prev_phi
        )

        # --- LSTM 1 ---
        # Input to the first LSTM is the concatenation of the previous context vector `w`
        # and the current stroke data `x(t)`.
        s1_in = tf.concat([prev_w, timestep_input], axis=1)
        s1_out, [s1_h, s1_c] = self.lstm1(s1_in, states=[h1, c1])
        
        # Apply dropout to LSTM output
        if training and self.dropout_rate > 0:
            s1_out = tf.nn.dropout(s1_out, rate=self.dropout_rate)

        # --- Attention Mechanism ---
        # The attention parameters are predicted by a dense layer based on the
        # previous context vector `w`, current stroke `x(t)`, and LSTM 1 output `h1(t)`.
        attention_inputs = tf.concat([prev_w, timestep_input, s1_out], axis=1)
        attention_params = self.attention_layer(attention_inputs)
        alpha, beta, kappa = tf.split(tf.nn.softplus(attention_params), 3, axis=1)

        # Update kappa: The positions of the attention windows move across the text.
        kappa = prev_kappa + kappa / 25.0
        
        # --- CLAMPING FIX for "Runaway Attention" ---
        # Clamp kappa to be at most char_len (plus small margin) to prevent looking at padding.
        # attention_values_lengths is shape (batch,)
        # kappa is shape (batch, k)
        max_k = tf.cast(attention_values_lengths, dtype=tf.float32)
        max_k = tf.expand_dims(max_k, 1) # (batch, 1)
        # Allow looking slightly past the end (e.g. +1 char) to see "stop"
        kappa = tf.minimum(kappa, max_k + 1.0)
        # ---------------------------------------------

        beta = tf.clip_by_value(beta, 1e-2, np.inf) # Prevent beta from becoming too small

        # --- Gaussian Mixture Attention Calculation ---
        # `u` represents the indices of the character sequence.
        u = tf.cast(tf.range(char_len), dtype=tf.float32)
        u = tf.reshape(u, [1, 1, -1])  # Shape: (1, 1, char_len)

        # Expand dims for broadcasting
        kappa_expanded = tf.expand_dims(kappa, 2)  # (batch, k, 1)
        alpha_expanded = tf.expand_dims(alpha, 2)  # (batch, k, 1)
        beta_expanded = tf.expand_dims(beta, 2)    # (batch, k, 1)

        # `phi` represents the attention weights over the characters for this timestep.
        # It's a sum of `k` Gaussian distributions.
        phi = tf.reduce_sum(
            alpha_expanded * tf.exp(-tf.square(kappa_expanded - u) / beta_expanded),
            axis=1
        )  # Shape: (batch, char_len)

        # Mask phi for padded characters in the sequence.
        sequence_mask = tf.sequence_mask(attention_values_lengths, maxlen=char_len, dtype=tf.float32)
        phi = phi * sequence_mask

        # --- Context Vector (Window) ---
        # `w` is the context vector, calculated as the weighted sum of the one-hot
        # encoded characters, using `phi` as the weights.
        phi_expanded = tf.expand_dims(phi, 2)  # (batch, char_len, 1)
        w = tf.reduce_sum(phi_expanded * attention_values, axis=1)  # (batch, vocab_size)

        # --- LSTM 2 ---
        # Input is the current stroke, output of LSTM 1, and the new context vector.
        s2_in = tf.concat([timestep_input, s1_out, w], axis=1)
        s2_out, [s2_h, s2_c] = self.lstm2(s2_in, states=[h2, c2])
        
        # Apply dropout to LSTM output
        if training and self.dropout_rate > 0:
            s2_out = tf.nn.dropout(s2_out, rate=self.dropout_rate)

        # --- LSTM 3 ---
        # Input is the current stroke, output of LSTM 2, and the new context vector.
        s3_in = tf.concat([timestep_input, s2_out, w], axis=1)
        s3_out, [s3_h, s3_c] = self.lstm3(s3_in, states=[h3, c3])
        
        # Apply dropout to LSTM output
        if training and self.dropout_rate > 0:
            s3_out = tf.nn.dropout(s3_out, rate=self.dropout_rate)

        # --- Assemble New State and Return ---
        new_states = [s1_h, s1_c, s2_h, s2_c, s3_h, s3_c, kappa, w, phi]
        output = s3_out

        return output, new_states