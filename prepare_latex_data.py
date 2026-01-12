"""
Prepares handwriting data for LaTeX synthesis training.

This script processes the MathWriting dataset (InkML files) to create NumPy arrays 
suitable for training a sequence-to-sequence model.

It directly reads labels from the InkML files in the specified splits (e.g., train, valid).

The pipeline is as follows:
1.  Iterate through .inkml files in the specified splits.
2.  Parse each file to extract:
    - The LaTeX label (preferring 'normalizedLabel', then 'label').
    - The raw stroke coordinates.
3.  Process the strokes: align, denoise, convert to relative offsets, and normalize.
4.  Build a character map from all found labels.
5.  Encode labels and pad sequences.
6.  Save the processed data (x, x_len, c, c_len) and char_map.
"""
import os
import sys
import json
import argparse
import glob
import multiprocessing
from xml.etree import ElementTree
import numpy as np
from tqdm import tqdm

import drawing

# --- Helper Functions ---

def get_drawing_and_label(filename):
    """
    Parses an InkML file to extract the label and processed stroke sequence.
    """
    try:
        tree = ElementTree.parse(filename).getroot()
        namespace = {'inkml': 'http://www.w3.org/2003/InkML'}
    except ElementTree.ParseError as e:
        print(f"Warning: Could not parse {filename}: {e}")
        return None, None

    # --- Extract Label ---
    # Look for normalizedLabel first, then label
    label = None
    for annotation in tree.findall('inkml:annotation', namespace):
        atype = annotation.get('type')
        if atype == 'normalizedLabel':
            label = annotation.text
            break
        elif atype == 'label' and label is None:
            label = annotation.text
    
    if label is None:
        return None, None

    # --- Extract Strokes ---
    traces = tree.findall('inkml:trace', namespace)
    coords = []
    for trace in traces:
        points = (trace.text or '').strip().split(',')
        stroke = []
        for point_str in points:
            parts = point_str.strip().split()
            if len(parts) >= 2:
                # We use x and -y because y-coordinates are typically inverted in image space
                stroke.append([float(parts[0]), -1 * float(parts[1])])
        
        if not stroke:
            continue

        # Add the end-of-stroke flag (0 for intermediate points, 1 for the last point)
        for i, point in enumerate(stroke):
            coords.append([point[0], point[1], int(i == len(stroke) - 1)])

    if not coords:
        return None, None

    # --- Process Strokes (using drawing.py) ---
    coords = np.array(coords)
    coords = drawing.align(coords)
    coords = drawing.denoise(coords)
    offsets = drawing.coords_to_offsets(coords)
    offsets = offsets[:drawing.MAX_STROKE_LEN]
    offsets = drawing.normalize(offsets)

    return offsets, label

def create_char_map(transcriptions):
    """Creates a character-to-index mapping from a list of transcriptions."""
    all_chars = set(''.join(transcriptions))
    # Sort for determinism
    char_map = {char: i for i, char in enumerate(sorted(list(all_chars)))}
    # Ensure a padding character exists (usually space or 0)
    # We use the existing logic where ' ' is mapped if present, or added.
    # However, for padding, we often want index 0 or a specific index.
    # The training script uses padding values from this map.
    if ' ' not in char_map:
        char_map[' '] = len(char_map)
    return char_map

def main(args):
    """Main function to run the data preparation pipeline."""
    
    # 1. Collect Data
    splits = args.splits.split(',')
    all_data = [] # List of tuples: (offsets, label, filename)

    print(f"Scanning splits: {splits} in {args.data_dir}...")
    
    valid_files = []
    for split in splits:
        split_dir = os.path.join(args.data_dir, split)
        if not os.path.isdir(split_dir):
            print(f"Warning: Split directory not found: {split_dir}")
            continue
        
        # Find all .inkml files
        files = glob.glob(os.path.join(split_dir, '*.inkml'))
        valid_files.extend(files)

    print(f"Found {len(valid_files)} InkML files. Processing...")
    
    # Process files (Parsing XML and calculating strokes)
    # Use multiprocessing to speed up processing
    num_processes = multiprocessing.cpu_count()
    print(f"Processing with {num_processes} processes...")
    
    with multiprocessing.Pool(processes=num_processes) as pool:
        # returns list of (offsets, label) or (None, None)
        results = list(tqdm(pool.imap(get_drawing_and_label, valid_files), total=len(valid_files)))
    
    # Filter out None results and invalid strokes
    for offsets, label in results:
        if offsets is None or label is None:
            continue
        
        # Basic validity check (removes outlier strokes)
        if offsets.shape[0] == 0 or np.any(np.linalg.norm(offsets[:, :2], axis=1) > 60):
            continue
            
        all_data.append((offsets, label))

    print(f"Successfully processed {len(all_data)} valid samples.")

    if not all_data:
        print("No valid data found. Exiting.")
        sys.exit(1)

    # 2. Create character map
    transcriptions = [item[1] for item in all_data]
    char_map = create_char_map(transcriptions)
    
    os.makedirs(args.output_dir, exist_ok=True)
    char_map_path = os.path.join(args.output_dir, 'char_map.json')
    print(f"Dumping character map to {char_map_path}...")
    with open(char_map_path, 'w') as f:
        json.dump(char_map, f, indent=2)

    # 3. Create NumPy arrays
    num_samples = len(all_data)
    
    # Initialize arrays
    x = np.zeros([num_samples, drawing.MAX_STROKE_LEN, 3], dtype=np.float32)
    x_len = np.zeros([num_samples], dtype=np.int16)
    c = np.full([num_samples, drawing.MAX_CHAR_LEN], fill_value=char_map.get(' ', 0), dtype=np.int8)
    c_len = np.zeros([num_samples], dtype=np.int8)
    
    print("Encoding character sequences and building arrays...")
    for i, (offsets, label) in enumerate(tqdm(all_data)):
        # Fill x (strokes)
        x[i, :len(offsets), :] = offsets
        x_len[i] = len(offsets)
        
        # Fill c (chars)
        encoded_c = [char_map[char] for char in label if char in char_map]
        encoded_c = encoded_c[:drawing.MAX_CHAR_LEN]
        
        c[i, :len(encoded_c)] = encoded_c
        c_len[i] = len(encoded_c)

    # 4. Save arrays
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
        help='Root directory of the dataset (containing train/valid subfolders).'
    )
    parser.add_argument(
        '--splits',
        type=str,
        default='train,valid',
        help='Comma-separated list of subdirectories to include (e.g., "train,valid").'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='data/processed_latex',
        help='Directory to save the processed .npy files and char_map.json.'
    )
    
    args = parser.parse_args()
    main(args)