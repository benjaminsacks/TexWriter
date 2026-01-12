"""
A utility script to convert InkML files to PNG images.

This script processes a directory of InkML files, which contain digital ink data,
and renders each file as a PNG image, preserving the original strokes. It is
useful for visualizing the raw handwriting data.

Usage:
    python inkml2img.py /path/to/inkml_folder /path/to/output_png_folder
"""
import os
import glob
import argparse
import xml.etree.ElementTree as ET
import numpy as np
import matplotlib.pyplot as plt


def get_traces_data(inkml_file_path, xmlns='{http://www.w3.org/2003/InkML}'):
    """
    Parses an InkML file to extract stroke data and associated labels.

    Args:
        inkml_file_path (str): The absolute path to the .inkml file.
        xmlns (str): The XML namespace for the InkML format.

    Returns:
        list: A list of dictionaries, where each dictionary contains the
              'label' and the 'trace_group' (a list of strokes, where each
              stroke is a list of [x, y] coordinates).
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
        # Handle files with no explicit trace groups (e.g., validation data)
        # by treating each trace as its own group.
        label_element = root.find(xmlns + 'annotation')
        global_label = label_element.text if label_element is not None else 'unlabeled'
        for trace in traces_all:
            traces_data.append({'label': global_label, 'trace_group': [trace['coords']]})

    return traces_data


def inkml_to_png(inkml_path, output_path, color='#284054'):
    """
    Converts a single InkML file to a PNG image.

    Args:
        inkml_path (str): Path to the input InkML file.
        output_path (str): Path to save the output PNG file.
        color (str): The color of the strokes.
    """
    # Each InkML can contain multiple symbols; we render them all on one image.
    traces = get_traces_data(inkml_path)
    if not traces:
        print(f"No traces found in {inkml_path}, skipping.")
        return

    plt.figure()
    for elem in traces:
        for stroke in elem['trace_group']:
            # Ensure stroke is a list of points
            if not isinstance(stroke, list) or not all(isinstance(p, list) and len(p) >= 2 for p in stroke):
                continue
            
            data = np.array(stroke)[:, :2]
            if data.shape[0] < 2: # Need at least two points to draw a line
                plt.plot(data[0, 0], data[0, 1], 'o', c=color)
            else:
                x, y = zip(*data)
                plt.plot(x, y, linewidth=2, c=color)

    # Configure plot aesthetics
    plt.gca().invert_yaxis()
    plt.gca().set_aspect('equal', adjustable='box')
    plt.axis('off')

    # Save the figure
    try:
        plt.savefig(output_path, bbox_inches='tight', dpi=150)
    except Exception as e:
        print(f"Could not save image {output_path}: {e}")
    finally:
        plt.close() # Close the figure to free memory


def main():
    """Main function to run the conversion process."""
    parser = argparse.ArgumentParser(description="Convert a directory of InkML files to PNG images.")
    parser.add_argument('input_dir', type=str, help='The directory containing .inkml files.')
    parser.add_argument('output_dir', type=str, help='The directory to save the output .png files.')
    args = parser.parse_args()

    if not os.path.isdir(args.input_dir):
        print(f"Error: Input directory not found at '{args.input_dir}'")
        sys.exit(1)

    os.makedirs(args.output_dir, exist_ok=True)
    print(f"Output will be saved in '{args.output_dir}'")

    # Find all .inkml files recursively
    files = sorted(glob.glob(os.path.join(args.input_dir, '**', '*.inkml'), recursive=True))
    if not files:
        print(f"No .inkml files found in '{args.input_dir}'.")
        return

    print(f"Found {len(files)} InkML files to convert.")

    for file_path in files:
        # Create a corresponding output path
        relative_path = os.path.relpath(file_path, args.input_dir)
        output_filename = os.path.splitext(relative_path)[0] + '.png'
        output_path = os.path.join(args.output_dir, output_filename)
        
        # Ensure the output subdirectory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        print(f"Converting {file_path} -> {output_path}")
        inkml_to_png(file_path, output_path)

    print("Conversion complete.")


if __name__ == "__main__":
    main()