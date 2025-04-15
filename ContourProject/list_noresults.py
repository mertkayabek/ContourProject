import os
import argparse

def check_directory(base_dir):
    """
    Check each subdirectory in base_dir for a 'visualizations' folder.
    Return names of directories where the 'visualizations' folder contains fewer than 7 files.
    """
    folders_with_few_files = []
    
    try:
        # Check each item in the base directory
        for item in os.listdir(base_dir):
            item_path = os.path.join(base_dir, item)
            
            # Check if it's a directory
            if os.path.isdir(item_path):
                # Check if it contains a 'visualizations' folder
                vis_path = os.path.join(item_path, 'visualizations')
                
                if os.path.isdir(vis_path):
                    # Count number of files in the visualizations folder
                    file_count = len([f for f in os.listdir(vis_path) if os.path.isfile(os.path.join(vis_path, f))])
                    
                    # If fewer than 7 files, add parent folder name to our list
                    if file_count < 7:
                        folders_with_few_files.append(item)
    except Exception as e:
        print(f"Error checking directory {base_dir}: {e}")
    
    return folders_with_few_files

def save_to_file(folder_list, output_file):
    """Save the list of folder names to a text file, one folder per line."""
    try:
        with open(output_file, 'w') as f:
            for folder in folder_list:
                f.write(folder + '\n')
        return True
    except Exception as e:
        print(f"Error saving to file {output_file}: {e}")
        return False

def main():
    # Set up command line argument parsing
    parser = argparse.ArgumentParser(description='Check for folders with few visualization files.')
    parser.add_argument('--dir', '-d', default=os.getcwd(), 
                        help='Base directory to check (default: current directory)')
    parser.add_argument('--output', '-o', default='folders_with_few_visualizations2.txt',
                        help='Output file name (default: folders_with_few_visualizations.txt)')
    args = parser.parse_args()
    
    # Override the directory with a hardcoded path
    args.dir = "/home_student/kayabek/sw/Results/parameter_sweep_20250411_193145"  # Replace with your actual directory path
    # Check if the base directory exists
    if not os.path.isdir(args.dir):
        print(f"Error: Directory '{args.dir}' does not exist or is not accessible.")
        return
    
    # Check directories and get list of folders with few visualizations
    print(f"Checking directories in {args.dir}...")
    folder_list = check_directory(args.dir)
    # Set the output file path by combining a specific directory with the filename
    output_dir = "/home_student/kayabek/sw/Results"  # Set your desired output directory
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)
    # Combine directory and filename for full output path
    args.output = os.path.join(output_dir, os.path.basename(args.output))
    # Save results to file
    if save_to_file(folder_list, args.output):
        print(f"Found {len(folder_list)} folders with fewer than 7 visualization files.")
        print(f"Results saved to {args.output}")

if __name__ == '__main__':
    main()