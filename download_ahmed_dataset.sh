#!/bin/bash
# This Bash script downloads the Ahmed files from HuggingFace repository to a local directory.
# Only the STL files (.stl) and VTP files (.vtp) are downloaded (no volume files for surface-only training).
# It uses a function, download_run_files, to check for the existence of two specific files (".stl", ".vtp") in a run directory.
# If a file doesn't exist, it's downloaded from the repository. If it does exist, the download is skipped.
# The script runs multiple downloads in parallel, both within a single run and across multiple runs.
# It also includes checks to prevent overloading the system by limiting the number of parallel downloads.

# Set the local directory to download the files
LOCAL_DIR="/data/ahmed_data/raw"  # <--- This is the directory where the files will be downloaded.

# Set the repository URL and temporary directory
REPO_URL="https://huggingface.co/datasets/neashton/ahmedml"
TEMP_DIR="/data/ahmed_temp"

# Create the local directory if it doesn't exist
mkdir -p "$LOCAL_DIR"

# Clone repository once if not already done
if [ ! -d "$TEMP_DIR" ]; then
    echo "Cloning Ahmed repository..."
    git clone "$REPO_URL" "$TEMP_DIR"
    cd "$TEMP_DIR"
    git lfs pull
fi

# Function to download files for a specific run
download_run_files() {
    local i=$1
    RUN_DIR="run_$i"
    RUN_LOCAL_DIR="$LOCAL_DIR/$RUN_DIR"
    SOURCE_DIR="$TEMP_DIR/$RUN_DIR"
    
    # Skip if source directory doesn't exist
    if [ ! -d "$SOURCE_DIR" ]; then
        return
    fi
    
    # Create the run directory if it doesn't exist
    mkdir -p "$RUN_LOCAL_DIR"
    
    # Check if the .stl file exists before downloading
    if [ ! -f "$RUN_LOCAL_DIR/ahmed_$i.stl" ]; then
        if [ -f "$SOURCE_DIR"/*.stl ]; then
            cp "$SOURCE_DIR"/*.stl "$RUN_LOCAL_DIR/ahmed_$i.stl" &
        fi
    else
        echo "File ahmed_$i.stl already exists, skipping download."
    fi
    
    # Check if the .vtp file exists before downloading
    if [ ! -f "$RUN_LOCAL_DIR/boundary_$i.vtp" ]; then
        if [ -f "$SOURCE_DIR"/*.vtp ]; then
            cp "$SOURCE_DIR"/*.vtp "$RUN_LOCAL_DIR/boundary_$i.vtp" &
        fi
    else
        echo "File boundary_$i.vtp already exists, skipping download."
    fi
    
    wait # Ensure that both files for this run are downloaded before moving to the next run
}

# Loop through the run folders and download the files
for i in $(seq 1 500); do
    download_run_files "$i" &
    
    # Limit the number of parallel jobs to avoid overloading the system
    if (( $(jobs -r | wc -l) >= 8 )); then
        wait -n # Wait for the next background job to finish before starting a new one
    fi
done

# Wait for all remaining background jobs to finish
wait

# Cleanup temporary directory
rm -rf "$TEMP_DIR"

echo "Ahmed dataset download completed!"
