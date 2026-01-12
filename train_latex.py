"""
Trains a handwriting synthesis model for LaTeX generation.

This script uses a Recurrent Neural Network (RNN) with a Mixture Density Network (MDN)
output to model the distribution of pen strokes, conditioned on a LaTeX character
sequence. It is based on the paper "Generating Sequences with Recurrent Neural
Networks" by Alex Graves.

The script handles:
- Loading processed training data (strokes and character sequences).
- Defining the `HandwritingRNN` model, which includes a custom RNN cell with
  an attention mechanism.
- A training loop with mini-batching, gradient clipping, and checkpointing.
- Calculating a custom loss function for the MDN output.
"""
import os
import sys
import json
import argparse
import numpy as np
import tensorflow as tf
from tqdm import tqdm

# Add the handwriting-synthesis-master directory to the Python path
# sys.path.append(os.path.join(os.getcwd(), 'handwriting-synthesis-master'))

# Set the GPU to use
os.environ["CUDA_VISIBLE_DEVICES"] = "3"

from tf2_rnn_cell import KerasLSTMAttentionCell
import drawing


# --- Model Definition ---

class HandwritingRNN(tf.keras.Model):
    """
    A sequence-to-sequence model for handwriting generation, adapted for TF2.
    """
    def __init__(self, lstm_size=400, output_mixture_components=20,
                 attention_mixture_components=10, vocab_size=80, **kwargs):
        super().__init__(**kwargs)
        self.lstm_size = lstm_size
        self.output_mixture_components = output_mixture_components
        self.attention_mixture_components = attention_mixture_components
        self.vocab_size = vocab_size

        # Custom RNN cell with attention
        self.cell = KerasLSTMAttentionCell(
            lstm_size=self.lstm_size,
            num_attn_mixture_components=self.attention_mixture_components,
            vocab_size=self.vocab_size
        )

        # Dense layer to predict the parameters of the MDN output
        self.output_layer = tf.keras.layers.Dense(
            1 + self.output_mixture_components * 6, name='mdn_output'
        )

    def build(self, input_shape):
        """Builds the model's layers."""
        stroke_shape, char_shape, _, _ = input_shape
        # The input to the RNN cell is a dictionary
        cell_input_shape = {
            'strokes': (stroke_shape[0], stroke_shape[2]),
            'attention_values': char_shape,
            'c_len': ()
        }
        self.cell.build(cell_input_shape)
        # The input to the output layer is the output of the RNN cell
        self.output_layer.build((None, self.lstm_size))
        self.built = True


    @tf.function(reduce_retracing=True)
    def call(self, inputs):
        """
        Forward pass of the model.
        """
        x, c, x_len, c_len = inputs
        batch_size = tf.shape(x)[0]
        max_stroke_len = tf.shape(x)[1]

        # One-hot encode the character sequence for the attention mechanism
        c_one_hot = tf.one_hot(c, depth=self.vocab_size)

        # Get initial state for the RNN cell
        initial_state = self.cell.get_initial_state(batch_size, dtype=tf.float32)

        # Manual unrolling of the RNN using tf.while_loop
        outputs = tf.TensorArray(tf.float32, size=max_stroke_len)
        t = tf.constant(0)

        # Define shape invariants for the loop variables
        state_shape_invariants = []
        for s in self.cell.state_size:
            if s is None:
                state_shape_invariants.append(tf.TensorShape([None, None]))
            else:
                state_shape_invariants.append(tf.TensorShape([None, s]))
        
        shape_invariants = [
            tf.TensorShape([]),  # t
            tf.TensorShape(None),  # outputs
        ] + state_shape_invariants

        def loop_cond(t, outputs, *state):
            return tf.less(t, max_stroke_len)

        def loop_body(t, outputs, *state):
            cell_inputs = {
                'strokes': x[:, t, :],
                'attention_values': c_one_hot,
                'c_len': c_len
            }
            output, new_state_list = self.cell(cell_inputs, list(state))
            outputs = outputs.write(t, output)
            return [t + 1, outputs] + new_state_list

        final_loop_vars = tf.while_loop(
            loop_cond,
            loop_body,
            loop_vars=[t, outputs] + initial_state,
            shape_invariants=shape_invariants
        )
        
        outputs = final_loop_vars[1]

        # Stack the outputs and apply the final dense layer
        outputs = tf.transpose(outputs.stack(), [1, 0, 2])
        mdn_params = self.output_layer(outputs)
        return mdn_params

    def loss_function(self, y_true, y_pred):
        """
        Loss function for the Mixture Density Network.
        """
        mdn_params = y_pred
        # Split the MDN parameters
        pi, mu1, mu2, sigma1, sigma2, rho, eos_prob = tf.split(
            mdn_params,
            [
                self.output_mixture_components,
                self.output_mixture_components,
                self.output_mixture_components,
                self.output_mixture_components,
                self.output_mixture_components,
                self.output_mixture_components,
                1
            ],
            axis=-1
        )

        # Normalize mixture weights
        pi = tf.nn.softmax(pi, axis=-1)

        # Apply activations to sigmas and rho
        sigma1 = tf.exp(sigma1)
        sigma2 = tf.exp(sigma2)
        rho = tf.tanh(rho)

        # Calculate the bivariate Gaussian distribution
        x_data, y_data, eos_data = tf.split(y_true, 3, axis=-1)

        z_x = (x_data - mu1) / sigma1
        z_y = (y_data - mu2) / sigma2
        z = tf.square(z_x) + tf.square(z_y) - 2 * rho * z_x * z_y
        
        # Denominator of the Gaussian
        norm = 2 * np.pi * sigma1 * sigma2 * tf.sqrt(1 - tf.square(rho))
        
        # PDF of the Gaussian
        pdf = tf.exp(-z / (2 * (1 - tf.square(rho)))) / norm

        # Weighted sum of PDFs
        loss_stroke = tf.reduce_sum(pi * pdf, axis=-1)
        loss_stroke = -tf.math.log(tf.maximum(loss_stroke, 1e-10))

        # Loss for end-of-stroke prediction
        loss_eos = tf.keras.losses.binary_crossentropy(
            eos_data, tf.sigmoid(eos_prob), from_logits=False
        )
        
        return tf.reduce_mean(loss_stroke + loss_eos)

    @tf.function(reduce_retracing=True)
    def sample(self, text, char_map, initial_bias=1.0):
        """Generate a handwriting sample from a text string."""
        # Convert text to character indices
        c = np.array([char_map.get(char, 0) for char in text], dtype=np.int32)
        c = tf.constant(c, dtype=tf.int32)[tf.newaxis, :]
        c_len = tf.constant([len(text)], dtype=tf.int32)

        batch_size = 1
        max_output_len = 1000

        c_one_hot = tf.one_hot(c, self.vocab_size)
        initial_state = self.cell.get_initial_state(batch_size, dtype=tf.float32)

        current_stroke = tf.zeros((batch_size, 3), dtype=tf.float32)
        strokes = tf.TensorArray(tf.float32, size=0, dynamic_size=True)
        state = initial_state

        t = tf.constant(0)
        end_of_sequence = tf.constant(False, dtype=tf.bool)

        state_shape_invariants = [tf.TensorShape([None, s]) if s is not None else tf.TensorShape([None, None]) for s in self.cell.state_size]
        
        loop_vars = [t, strokes, current_stroke, end_of_sequence] + initial_state

        def loop_cond(t, strokes, current_stroke, end_of_sequence, *state):
            return tf.logical_and(tf.less(t, max_output_len), tf.logical_not(end_of_sequence))

        def loop_body(t, strokes, current_stroke, end_of_sequence, *state):
            cell_inputs = {
                'strokes': current_stroke[:, :3],
                'attention_values': c_one_hot,
                'c_len': c_len
            }
            output, new_state_list = self.cell(cell_inputs, list(state))
            mdn_params = self.output_layer(output)

            pi, mu1, mu2, sigma1, sigma2, rho, eos_prob = tf.split(
                mdn_params,
                [
                    self.output_mixture_components, self.output_mixture_components, self.output_mixture_components,
                    self.output_mixture_components, self.output_mixture_components, self.output_mixture_components, 1
                ],
                axis=-1
            )
            
            pi = tf.nn.softmax(pi * (1 + initial_bias), axis=-1)
            sigma1, sigma2 = tf.exp(sigma1), tf.exp(sigma2)
            rho = tf.tanh(rho)

            mixture_idx = tf.random.categorical(tf.math.log(pi), 1, dtype=tf.int32)
            mixture_idx = tf.squeeze(mixture_idx, axis=-1)

            mu1_s = tf.gather(mu1, mixture_idx, batch_dims=1, axis=1)
            mu2_s = tf.gather(mu2, mixture_idx, batch_dims=1, axis=1)
            sigma1_s = tf.gather(sigma1, mixture_idx, batch_dims=1, axis=1)
            sigma2_s = tf.gather(sigma2, mixture_idx, batch_dims=1, axis=1)
            rho_s = tf.gather(rho, mixture_idx, batch_dims=1, axis=1)

            z = tf.random.normal((batch_size, 2))
            x = mu1_s + sigma1_s * z[:, 0:1]
            y = mu2_s + sigma2_s * (rho_s * z[:, 0:1] + tf.sqrt(1 - tf.square(rho_s)) * z[:, 1:2])

            eos_val = 1.0 / (1.0 + tf.exp(-eos_prob))
            eos = tf.cast(tf.random.uniform((batch_size, 1)) < eos_val, tf.float32)
            
            end_of_sequence = tf.logical_and(tf.greater(t, 0), tf.reduce_all(eos > 0))

            current_stroke = tf.concat([x, y, eos], axis=1)
            strokes = strokes.write(t, tf.squeeze(current_stroke))
            
            return [t + 1, strokes, current_stroke, end_of_sequence] + new_state_list

        final_loop_vars = tf.while_loop(
            loop_cond,
            loop_body,
            loop_vars=loop_vars,
            shape_invariants=[
                tf.TensorShape([]),
                tf.TensorShape(None),
                tf.TensorShape([None, None]),
                tf.TensorShape([])] + state_shape_invariants
        )
        
        strokes = final_loop_vars[1]


        return strokes.stack()


# --- Data Loading ---

def load_data(data_dir):
    """Loads the processed handwriting data from .npy files."""
    x = np.load(os.path.join(data_dir, 'x.npy'))
    x_len = np.load(os.path.join(data_dir, 'x_len.npy'))
    c = np.load(os.path.join(data_dir, 'c.npy'))
    c_len = np.load(os.path.join(data_dir, 'c_len.npy'))
    with open(os.path.join(data_dir, 'char_map.json'), 'r') as f:
        char_map = json.load(f)
    return x, x_len, c, c_len, char_map

# --- Training ---

def train(args):
    """Main training routine."""
    # Load data
    x, x_len, c, c_len, char_map = load_data(args.data_dir)
    vocab_size = len(char_map)

    # Create tf.data.Dataset
    dataset = tf.data.Dataset.from_tensor_slices((x, c, x_len, c_len))
    dataset = dataset.shuffle(buffer_size=len(x)).batch(args.batch_size)

    # Initialize model, optimizer, and checkpoint manager
    model = HandwritingRNN(
        lstm_size=args.lstm_size,
        output_mixture_components=args.output_mixture_components,
        attention_mixture_components=args.attention_mixture_components,
        vocab_size=vocab_size
    )
    optimizer = tf.keras.optimizers.Adam(learning_rate=args.learning_rate)

    # Build the model by passing a sample batch
    sample_x, sample_c, sample_x_len, sample_c_len = next(iter(dataset))
    model((sample_x, sample_c, sample_x_len, sample_c_len))

    # Checkpoint setup
    ckpt = tf.train.Checkpoint(optimizer=optimizer, model=model)
    ckpt_manager = tf.train.CheckpointManager(
        ckpt, directory=args.checkpoint_dir, max_to_keep=5
    )

    # Restore from the latest checkpoint if it exists
    if ckpt_manager.latest_checkpoint:
        tqdm.write(f"Restoring from {ckpt_manager.latest_checkpoint}...")
        # Use expect_partial to avoid errors if the model architecture has changed
        # slightly (e.g., adding a new layer).
        ckpt.restore(ckpt_manager.latest_checkpoint).expect_partial()
        tqdm.write("Model restored.")
    else:
        tqdm.write("Initializing from scratch.")

    @tf.function(reduce_retracing=True)
    def train_step(batch):
        x_batch, c_batch, x_len_batch, c_len_batch = batch
        with tf.GradientTape() as tape:
            mdn_params = model((x_batch, c_batch, x_len_batch, c_len_batch), training=True)
            # The target for the loss is the input stroke data, shifted by one step
            loss = model.loss_function(x_batch[:, 1:, :], mdn_params[:, :-1, :])

        grads = tape.gradient(loss, model.trainable_variables)
        # Apply gradient clipping to prevent exploding gradients
        grads, _ = tf.clip_by_global_norm(grads, 10.0)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))
        return loss

    # Training loop
    tqdm.write("Starting training...")
    for epoch in range(args.epochs):
        with tqdm(total=len(x) // args.batch_size, desc=f"Epoch {epoch+1}/{args.epochs}") as pbar:
            for batch in dataset:
                loss = train_step(batch)
                pbar.update(1)
                pbar.set_postfix({'loss': f'{loss.numpy():.4f}'})

        # Save checkpoint after each epoch
        if (epoch + 1) % args.save_every == 0:
            save_path = ckpt_manager.save()
            tqdm.write(f"Saved checkpoint for epoch {epoch+1}: {save_path}")
            model.save_weights(args.model_weights_path)
            tqdm.write(f"Saved model weights to {args.model_weights_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train a handwriting synthesis model for LaTeX.")
    parser.add_argument('--data_dir', type=str, default='data/processed_latex', help='Directory with processed data.')
    parser.add_argument('--checkpoint_dir', type=str, default='tf_ckpts', help='Directory to save checkpoints.')
    parser.add_argument('--model_weights_path', type=str, default='handwriting_model.weights.h5', help='Path to save final model weights.')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training.')
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs.')
    parser.add_argument('--learning_rate', type=float, default=1e-3, help='Optimizer learning rate.')
    parser.add_argument('--save_every', type=int, default=1, help='Save checkpoint every N epochs.')
    # Model Hyperparameters
    parser.add_argument('--lstm_size', type=int, default=400, help='Size of LSTM layers.')
    parser.add_argument('--output_mixture_components', type=int, default=20, help='Number of GMM output components.')
    parser.add_argument('--attention_mixture_components', type=int, default=10, help='Number of GMM attention components.')

    args = parser.parse_args()
    train(args)
