"""
Prepares handwriting data for LaTeX synthesis training.

This script processes a dataset of InkML files and their corresponding LaTeX labels
to create NumPy arrays suitable for training a sequence-to-sequence model.

The pipeline is as follows:
1.  Load labels from a .jsonl file that maps InkML filenames to LaTeX strings.
2.  Create and save a character map (a dictionary mapping each character to an integer).
3.  For each InkML file:
    a. Parse the XML to extract the raw stroke coordinates.
    b. Process the strokes: align, denoise, convert to relative offsets, and normalize.
    c. Encode the LaTeX label using the character map.
4.  Pad all sequences to a fixed maximum length.
5.  Save the processed strokes (x), stroke lengths (x_len), encoded characters (c),
    and character lengths (c_len) as .npy files.
"""
import os
import sys
import json
import argparse
from xml.etree import ElementTree
import numpy as np
from tqdm import tqdm

# Add the subdirectory to the Python path to allow direct import
# of modules from the original handwriting-synthesis project.
sys.path.append(os.path.join(os.getcwd(), 'handwriting-synthesis-master'))
import drawing

# --- Helper Functions ---

def load_labels(labels_file, data_dir):
    """
    Loads labels from a .jsonl file.

    Args:
        labels_file (str): Path to the .jsonl file.
        data_dir (str): The root directory of the dataset.

    Returns:
        list: A list of tuples, where each tuple contains an absolute file path
              to an .inkml file and its corresponding LaTeX label.
    """
    labels = []
    print(f"Loading labels from {labels_file}...")
    with open(labels_file, 'r') as f:
        for line in f:
            entry = json.loads(line)
            # Make the file path absolute
            full_path = os.path.join(data_dir, entry['filename'])
            if os.path.exists(full_path):
                labels.append((full_path, entry['label']))
    if not labels:
        raise ValueError(f"No valid file paths found from {labels_file}. Check paths and `data_dir`.")
    return labels

def create_char_map(transcriptions):
    """Creates a character-to-index mapping from a list of transcriptions."""
    all_chars = set(''.join(transcriptions))
    char_map = {char: i for i, char in enumerate(sorted(list(all_chars)))}
    # Ensure a padding character exists for bucketing.
    if ' ' not in char_map:
        char_map[' '] = len(char_map)
    return char_map

def get_stroke_sequence(filename):
    """
    Parses an InkML file to extract, process, and normalize a stroke sequence.

    The processing steps, adapted from the original `handwriting-synthesis`
    project, are:
    1.  Parse XML and extract (x, y) coordinates for each stroke.
    2.  Combine all strokes into a single sequence, adding an end-of-stroke flag.
    3.  Align the drawing to the center.
    4.  Denoise the strokes by removing small jitter.
    5.  Convert absolute coordinates to relative offsets (delta x, delta y).
    6.  Truncate to a maximum length.
    7.  Normalize the offsets to have zero mean and unit variance.

    Args:
        filename (str): Path to the .inkml file.

    Returns:
        np.ndarray: A processed stroke sequence of shape (N, 3), where each
                    row is [dx, dy, end_of_stroke_flag].
    """
    try:
        tree = ElementTree.parse(filename).getroot()
        namespace = {'inkml': 'http://www.w3.org/2003/InkML'}
        traces = tree.findall('inkml:trace', namespace)
    except ElementTree.ParseError as e:
        print(f"Warning: Could not parse {filename}: {e}")
        return np.array([])

    coords = []
    for trace in traces:
        points = (trace.text or '').strip().split(',')
        stroke = []
        for point_str in points:
            parts = point_str.strip().split()
            if len(parts) >= 2:
                # We use x and -y because y-coordinates are typically inverted in image space.
                stroke.append([float(parts[0]), -1 * float(parts[1])])
        
        if not stroke:
            continue

        # Add the end-of-stroke flag (0 for intermediate points, 1 for the last point).
        for i, point in enumerate(stroke):
            coords.append([point[0], point[1], int(i == len(stroke) - 1)])

    if not coords:
        return np.array([])

    # Convert to numpy and apply processing functions from drawing.py
    coords = np.array(coords)
    coords = drawing.align(coords)
    coords = drawing.denoise(coords)
    offsets = drawing.coords_to_offsets(coords)
    offsets = offsets[:drawing.MAX_STROKE_LEN]
    offsets = drawing.normalize(offsets)
    return offsets

def main(args):
    """Main function to run the data preparation pipeline."""
    # 1. Load labels
    try:
        labels = load_labels(args.labels_file, args.data_dir)
        print(f"Found {len(labels)} samples.")
    except (FileNotFoundError, ValueError) as e:
        print(f"Error: {e}")
        sys.exit(1)

    # 2. Create character map
    transcriptions = [label for _, label in labels]
    char_map = create_char_map(transcriptions)
    
    os.makedirs(args.output_dir, exist_ok=True)
    char_map_path = os.path.join(args.output_dir, 'char_map.json')
    print(f"Dumping character map to {char_map_path}...")
    with open(char_map_path, 'w') as f:
        json.dump(char_map, f, indent=2)

    # 3. Initialize numpy arrays for storing processed data
    num_samples = len(labels)
    x = np.zeros([num_samples, drawing.MAX_STROKE_LEN, 3], dtype=np.float32)
    x_len = np.zeros([num_samples], dtype=np.int16)
    c = np.full([num_samples, drawing.MAX_CHAR_LEN], fill_value=char_map[' '], dtype=np.int8)
    c_len = np.zeros([num_samples], dtype=np.int8)
    
    valid_indices = []

    print("Processing strokes and transcriptions...")
    for i, (stroke_fname, transcription) in enumerate(tqdm(labels, total=num_samples)):
        # Process strokes
        x_i = get_stroke_sequence(stroke_fname)
        if x_i.shape[0] == 0:
            continue

        # Basic validity check from the original project (removes outlier strokes)
        if np.any(np.linalg.norm(x_i[:, :2], axis=1) > 60):
            continue

        # Process and encode transcription
        encoded_c = [char_map[char] for char in transcription if char in char_map]
        encoded_c = encoded_c[:drawing.MAX_CHAR_LEN]

        # Store in numpy arrays
        x[i, :len(x_i), :] = x_i
        x_len[i] = len(x_i)
        c[i, :len(encoded_c)] = encoded_c
        c_len[i] = len(encoded_c)
        
        valid_indices.append(i)
            
    print(f"Successfully processed {len(valid_indices)} valid samples.")

    # Filter out any samples that were skipped or failed processing
    x = x[valid_indices]
    x_len = x_len[valid_indices]
    c = c[valid_indices]
    c_len = c_len[valid_indices]

    # 5. Save the final arrays
    print(f"Saving processed data to {args.output_dir}...")
    np.save(os.path.join(args.output_dir, 'x.npy'), x)
    np.save(os.path.join(args.output_dir, 'x_len.npy'), x_len)
    np.save(os.path.join(args.output_dir, 'c.npy'), c)
    np.save(os.path.join(args.output_dir, 'c_len.npy'), c_len)

    print("Data preparation complete.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Prepare LaTeX handwriting data for training.")
    parser.add_argument(
        '--data_dir',
        type=str,
        default='mathwriting-2024-excerpt',
        help='Root directory of the dataset.'
    )
    parser.add_argument(
        '--labels_file',
        type=str,
        default='mathwriting-2024-excerpt/symbols.jsonl',
        help='Path to the .jsonl file containing file paths and labels.'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='data/processed_latex',
        help='Directory to save the processed .npy files and char_map.json.'
    )
    
    args = parser.parse_args()
    main(args)