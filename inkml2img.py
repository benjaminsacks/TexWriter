"""
A utility script to convert InkML files to PNG images and compare with model generation.

Capabilities:
1. Convert a batch of .inkml files to images.
2. Convert a batch of .inkml files to images and stitch them together vertically
   with generated images using the model (compare mode).

Usage:
    # Convert only
    python inkml2img.py --input_dir data/samples --output_dir output_images

    # Compare (Generate & Stitch)
    python inkml2img.py --mode compare \
                        --input_dir data/samples \
                        --output_dir output_comparison \
                        --model_weights handwriting_model.weights.h5 \
                        --char_map data/processed_latex/char_map.json \
                        --bias 0.5
"""
import os
import sys
import glob
import argparse
import json
import xml.etree.ElementTree as ET
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

# Project imports
import drawing
from train_latex import HandwritingRNN
from tf2_rnn_cell import KerasLSTMAttentionCell


def get_traces_data(inkml_file_path, xmlns='{http://www.w3.org/2003/InkML}'):
    """
    Parses an InkML file to extract stroke data and associated labels.
    """
    traces_data = []
    try:
        tree = ET.parse(inkml_file_path)
        root = tree.getroot()
    except ET.ParseError as e:
        print(f"Error parsing {inkml_file_path}: {e}")
        return []

    # Find all traces with their IDs
    traces_all = [{
        'id': trace_tag.get('id'),
        'coords': [
            [float(axis_coord) for axis_coord in coord.split(' ')]
            for coord in (trace_tag.text or '').strip().replace('\n', '').split(',')
            if coord
        ]
    } for trace_tag in root.findall(xmlns + 'trace')]

    # Sort traces by ID for easy lookup
    traces_all.sort(key=lambda trace_dict: int(trace_dict['id']))

    # Find the main trace group wrapper
    trace_group_wrapper = root.find(xmlns + 'traceGroup')

    if trace_group_wrapper is not None:
        # Process trace groups with labels
        for traceGroup in trace_group_wrapper.findall(xmlns + 'traceGroup'):
            label_element = traceGroup.find(xmlns + 'annotation')
            if label_element is None or label_element.text is None:
                continue
            
            label = label_element.text
            traces_curr = []
            for traceView in traceGroup.findall(xmlns + 'traceView'):
                trace_data_ref = int(traceView.get('traceDataRef'))
                if 0 <= trace_data_ref < len(traces_all):
                    traces_curr.append(traces_all[trace_data_ref]['coords'])
            
            if traces_curr:
                traces_data.append({'label': label, 'trace_group': traces_curr})
    else:
        # Handle files with no explicit trace groups
        label_element = root.find(xmlns + 'annotation')
        global_label = label_element.text if label_element is not None else 'unlabeled'
        for trace in traces_all:
            traces_data.append({'label': global_label, 'trace_group': [trace['coords']]})

    return traces_data


def get_latex_label(inkml_file_path, xmlns='{http://www.w3.org/2003/InkML}'):
    """
    Robustly extracts the LaTeX label from an InkML file, preferring 'normalizedLabel'.
    """
    try:
        tree = ET.parse(inkml_file_path)
        root = tree.getroot()
    except Exception:
        return None

    ns = {'inkml': xmlns.strip('{}')}
    
    # Look for normalizedLabel first, then label
    label = None
    for annotation in root.findall('inkml:annotation', ns):
        atype = annotation.get('type')
        if atype == 'normalizedLabel':
            label = annotation.text
            break
        elif atype == 'label' and label is None:
            label = annotation.text
            
    return label


def plot_strokes(ax, traces, color='#284054', linewidth=2):
    """
    Helper to plot traces on a given matplotlib axis.
    traces: list of list of [x, y] coordinates.
    """
    for stroke in traces:
        if not isinstance(stroke, list) and not isinstance(stroke, np.ndarray):
            continue
        
        data = np.array(stroke)[:, :2]
        if data.shape[0] < 2:
            ax.plot(data[0, 0], data[0, 1], 'o', c=color)
        else:
            x, y = zip(*data)
            ax.plot(x, y, linewidth=linewidth, c=color)

    ax.invert_yaxis()
    ax.set_aspect('equal', adjustable='box')
    ax.axis('off')


def inkml_to_png(inkml_path, output_path, color='#284054'):
    """Converts a single InkML file to a PNG image."""
    traces = get_traces_data(inkml_path)
    if not traces:
        print(f"No traces found in {inkml_path}, skipping.")
        return

    plt.figure()
    
    # Flatten traces for simple plotting (ignoring groups)
    all_strokes = []
    for elem in traces:
        all_strokes.extend(elem['trace_group'])
        
    plot_strokes(plt.gca(), all_strokes, color=color)

    try:
        plt.savefig(output_path, bbox_inches='tight', dpi=150)
    except Exception as e:
        print(f"Could not save image {output_path}: {e}")
    finally:
        plt.close()


def load_model(weights_path, char_map_path, lstm_size=800, output_mixture_components=20, attention_mixture_components=10):
    """Loads the trained handwriting model."""
    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"Model weights not found at {weights_path}")
    
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

    # Build model with dummy data
    dummy_x = tf.zeros([1, 1, 3], dtype=tf.float32)
    dummy_c = tf.zeros([1, 1], dtype=tf.int32)
    dummy_x_len = tf.constant([1], dtype=tf.int32)
    dummy_c_len = tf.constant([1], dtype=tf.int32)

    @tf.function
    def build_model(inputs):
        return model(inputs)
    
    build_model((dummy_x, dummy_c, dummy_x_len, dummy_c_len))
    model.load_weights(weights_path)
    print(f"Loaded weights from {weights_path}")
    
    return model, char_map


def generate_and_stitch(inkml_path, output_path, model, char_map, bias=0.5):
    """
    Generates a sample for the InkML's label and stitches it with the original.
    """
    # 1. Get Actual Data
    label = get_latex_label(inkml_path)
    if not label:
        print(f"Skipping {inkml_path}: No label found")
        return

    traces_data = get_traces_data(inkml_path)
    real_strokes = []
    for elem in traces_data:
        real_strokes.extend(elem['trace_group'])

    if not real_strokes:
        print(f"Skipping {inkml_path}: No stroke data found")
        return

    # 2. Generate Data
    # print(f"Generating for: '{label}'")
    try:
        sampled_offsets = model.sample(label, char_map, initial_bias=bias)
        generated_strokes = drawing.offsets_to_coords(sampled_offsets.numpy())
        # Convert to list of checks for plot_strokes compat if needed, 
        # or just format it as a single stroke list (drawing.offsets_to_coords returns one big array with eos column)
        
        # Split generated strokes based on EOS
        gen_stroke_list = []
        # drawing.offsets_to_coords returns shape (N, 3), 3rd col is EOS
        # process to split
        splits = np.where(generated_strokes[:, 2] == 1)[0] + 1
        gen_stroke_list = np.split(generated_strokes[:, :2], splits)
        # remove empty arrays
        gen_stroke_list = [s for s in gen_stroke_list if s.shape[0] > 0]
        
    except Exception as e:
        print(f"Generation failed for '{label}': {e}")
        return

    # 3. Plot Side-by-Side (Vertical)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6))
    
    # Plot Real
    plot_strokes(ax1, real_strokes, color='#284054')
    ax1.set_title(f"Actual: {label[:50]}...", fontsize=10)
    
    # Plot Generated
    # We can use plot_strokes since we converted it to list of arrays
    # But generated strokes might need denormalization or special handling?
    # drawing.draw does: denoise -> interpolate -> align.
    # We should probably at least denoise/align to make it look 'standard'?
    # Actually, model output is offsets. offsets_to_coords reconstructs it.
    # usually raw model output is already "normalized" in scale.
    # Let's just plot it raw first.
    
    plot_strokes(ax2, gen_stroke_list, color='#D14040') # Red for generated
    ax2.set_title("Generated", fontsize=10)

    try:
        plt.tight_layout()
        plt.savefig(output_path, dpi=150)
    except Exception as e:
        print(f"Could not save comparison {output_path}: {e}")
    finally:
        plt.close()


def main():
    parser = argparse.ArgumentParser(description="Convert/Compare InkML files.")
    parser.add_argument('--input_dir', type=str, default='data/samples', help='Directory containing .inkml files.')
    parser.add_argument('--output_dir', type=str, help='Output directory. Defaults to input_dir + "_out".')
    parser.add_argument('--mode', type=str, choices=['convert', 'compare'], default='convert', help='Operation mode.')
    
    # Generation args
    parser.add_argument('--model_weights', type=str, default='handwriting_model.weights.h5')
    parser.add_argument('--char_map', type=str, default='data/processed_latex/char_map.json')
    parser.add_argument('--bias', type=float, default=0.5)
    
    args = parser.parse_args()

    if not args.output_dir:
        args.output_dir = args.input_dir.rstrip('/') + '_out'

    if not os.path.isdir(args.input_dir):
        print(f"Error: Input directory '{args.input_dir}' not found.")
        sys.exit(1)

    os.makedirs(args.output_dir, exist_ok=True)
    
    files = sorted(glob.glob(os.path.join(args.input_dir, '**', '*.inkml'), recursive=True))
    print(f"Found {len(files)} InkML files in {args.input_dir}")

    # Initialize model if in compare mode
    model = None
    char_map = None
    if args.mode == 'compare':
        try:
            model, char_map = load_model(args.model_weights, args.char_map)
        except Exception as e:
            print(f"Failed to load model: {e}")
            sys.exit(1)

    for i, file_path in enumerate(files):
        # Create output filename
        rel_path = os.path.relpath(file_path, args.input_dir)
        out_name = os.path.splitext(rel_path)[0] + '.png'
        out_path = os.path.join(args.output_dir, out_name)
        
        os.makedirs(os.path.dirname(out_path), exist_ok=True)

        print(f"[{i+1}/{len(files)}] Processing {os.path.basename(file_path)}...")
        
        if args.mode == 'compare':
            generate_and_stitch(file_path, out_path, model, char_map, bias=args.bias)
        else:
            inkml_to_png(file_path, out_path)

    print("Done.")

if __name__ == "__main__":
    main()