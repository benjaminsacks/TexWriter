"""
Generates a handwriting sample from a trained HandwritingRNN model.

This script loads a pre-trained model and a character map, then uses the model's
`sample` method to generate stroke data for a given input text. The resulting
strokes are then rendered into a PNG image.

Usage:
    python generate.py "Your text here" --output_path "sample.png"
"""
import os
import sys
import json
import argparse
import tensorflow as tf

# Add the subdirectory to the Python path to allow direct import
# of modules from the original handwriting-synthesis project.
sys.path.append(os.path.join(os.getcwd(), 'handwriting-synthesis-master'))
import drawing

# Import the model definition.
# The KerasLSTMAttentionCell is also needed for the model to be loaded correctly.
from train_latex import HandwritingRNN
from tf2_rnn_cell import KerasLSTMAttentionCell

def generate_handwriting(
    text,
    model_weights_path,
    char_map_path,
    output_filename,
    lstm_size,
    output_mixture_components,
    attention_mixture_components,
    bias
):
    """
    Loads the trained model, generates handwriting for the given text,
    and saves it as an image.

    Args:
        text (str): The text to convert into handwriting.
        model_weights_path (str): Path to the saved model weights (.h5 file).
        char_map_path (str): Path to the character map JSON file.
        output_filename (str): Path to save the output image.
        lstm_size (int): The size of the LSTM layers in the model.
        output_mixture_components (int): The number of GMM components for the output.
        attention_mixture_components (int): The number of GMM components for attention.
        bias (float): The sampling bias.
    """
    if not os.path.exists(model_weights_path):
        print(f"ERROR: Model weights file not found at '{model_weights_path}'.")
        print("Please run 'python train_latex.py' to train and save the model first.")
        return

    print("Loading character map...")
    with open(char_map_path, 'r') as f:
        char_map = json.load(f)
    vocab_size = len(char_map)

    print("Initializing model...")
    model = HandwritingRNN(
        lstm_size=lstm_size,
        output_mixture_components=output_mixture_components,
        attention_mixture_components=attention_mixture_components,
        vocab_size=vocab_size
    )

    print("Building model and loading weights...")
    # To load weights, the model must be "built" first. We can do this by
    # calling it once with dummy data that has the correct structure and types.
    dummy_x = tf.zeros([1, 1, 3], dtype=tf.float32)
    dummy_c = tf.zeros([1, len(text)], dtype=tf.int32)
    dummy_x_len = tf.constant([1], dtype=tf.int32)
    dummy_c_len = tf.constant([len(text)], dtype=tf.int32)

    try:
        # The model call needs to be wrapped in a tf.function to be compatible
        # with the way the training script builds the model.
        @tf.function
        def build_model(inputs):
            return model(inputs)
        
        build_model((dummy_x, dummy_c, dummy_x_len, dummy_c_len))
        model.load_weights(model_weights_path)
        print(f"Successfully loaded weights from {model_weights_path}")
    except Exception as e:
        print(f"Error loading weights: {e}")
        print("Ensure the model architecture in this script matches the saved weights.")
        return

    # --- Inference/Sampling ---
    print(f"\nGenerating handwriting for: '{text}'")
    # The `sample` method is decorated with @tf.function in the model class
    sampled_offsets = model.sample(text, char_map, initial_bias=bias)

    print(f"Drawing and saving sample to {output_filename}")
    drawing.draw(
        sampled_offsets.numpy(),
        save_file=output_filename
    )
    print("Done.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Generate handwriting from a trained model.")
    parser.add_argument(
        'text',
        type=str,
        help='The text to generate handwriting for.'
    )
    parser.add_argument(
        '--output_path',
        type=str,
        default='generated_sample.png',
        help='File path to save the generated image.'
    )
    parser.add_argument(
        '--model_weights',
        type=str,
        default='handwriting_model.weights.h5',
        help='Path to the trained model weights file.'
    )
    parser.add_argument(
        '--char_map',
        type=str,
        default='data/processed_latex/char_map.json',
        help='Path to the character map JSON file.'
    )
    # The following hyperparameters must match the trained model
    parser.add_argument('--lstm_size', type=int, default=400, help='Size of LSTM layers.')
    parser.add_argument('--output_mixture_components', type=int, default=20, help='Number of GMM output components.')
    parser.add_argument('--attention_mixture_components', type=int, default=10, help='Number of GMM attention components.')
    parser.add_argument('--bias', type=float, default=0.5, help='Sampling bias.')

    args = parser.parse_args()

    generate_handwriting(
        text=args.text,
        model_weights_path=args.model_weights,
        char_map_path=args.char_map,
        output_filename=args.output_path,
        lstm_size=args.lstm_size,
        output_mixture_components=args.output_mixture_components,
        attention_mixture_components=args.attention_mixture_components,
        bias=args.bias
    )