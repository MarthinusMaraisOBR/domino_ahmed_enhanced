#!/bin/bash
# Download remaining Ahmed files (runs 101-500)

LOCAL_DIR="/data/ahmed_data/raw"
TEMP_DIR="/data/ahmed_temp_remaining"

echo "🚀 Downloading remaining Ahmed files (runs 101-500)..."
echo "📊 This will download ~33GB more data"

# Clean up temp directory
rm -rf "$TEMP_DIR"

# Clone with sparse checkout
git clone --filter=blob:none --no-checkout https://huggingface.co/datasets/neashton/ahmedml "$TEMP_DIR"
cd "$TEMP_DIR"

# Configure sparse checkout for runs 101-500
git sparse-checkout init --cone

echo "🎯 Configuring sparse checkout for runs 101-500..."
for i in $(seq 101 500); do
    echo "run_$i/ahmed_$i.stl" >> .git/info/sparse-checkout
    echo "run_$i/boundary_$i.vtp" >> .git/info/sparse-checkout
done

# Checkout the remaining files
echo "📥 Downloading runs 101-500 (this will take time)..."
git checkout HEAD

# Verify what we downloaded
echo "✅ Download completed! Verifying files..."
stl_count=$(find . -name "ahmed_*.stl" | wc -l)
vtp_count=$(find . -name "boundary_*.vtp" | wc -l)
echo "📊 Downloaded: $stl_count STL files, $vtp_count boundary VTP files"

# Copy files to final location
echo "📂 Organizing files for DoMINO..."
copied_count=0
for i in $(seq 101 500); do
    if [ -f "run_$i/ahmed_$i.stl" ] && [ -f "run_$i/boundary_$i.vtp" ]; then
        mkdir -p "$LOCAL_DIR/run_$i"
        cp "run_$i/ahmed_$i.stl" "$LOCAL_DIR/run_$i/"
        cp "run_$i/boundary_$i.vtp" "$LOCAL_DIR/run_$i/"
        ((copied_count++))
        
        # Progress update every 50 runs
        if [ $((copied_count % 50)) -eq 0 ]; then
            echo "  ✅ Copied $copied_count additional runs so far..."
        fi
    fi
done

# Cleanup temp directory
cd /
rm -rf "$TEMP_DIR"

echo ""
echo "🎉 Remaining Ahmed download completed!"
echo "📊 Total dataset now contains:"
echo "  STL files: $(find "$LOCAL_DIR" -name "*.stl" | wc -l)"
echo "  VTP files: $(find "$LOCAL_DIR" -name "*.vtp" | wc -l)"
echo "  Total size: $(du -sh "$LOCAL_DIR" | cut -f1)"
echo "  Run directories: $(find "$LOCAL_DIR" -type d -name "run_*" | wc -l)"

# Show range of runs
echo ""
echo "📋 Dataset range:"
echo "  First run: $(find "$LOCAL_DIR" -type d -name "run_*" | sort -V | head -1 | xargs basename)"
echo "  Last run: $(find "$LOCAL_DIR" -type d -name "run_*" | sort -V | tail -1 | xargs basename)"

# Verify a few sample files from the new range
echo ""
echo "📋 Sample verification from new downloads:"
for run in run_101 run_200 run_300 run_400 run_500; do
    if [ -d "$LOCAL_DIR/$run" ]; then
        echo "📁 $run: $(ls "$LOCAL_DIR/$run/" | wc -l) files ($(du -sh "$LOCAL_DIR/$run" | cut -f1))"
    fi
done

echo ""
echo "🎯 Complete Ahmed dataset ready!"
echo "📝 Next steps:"
echo "  1. cd src"
echo "  2. python process_data.py"
echo "  3. python train.py"
echo ""
echo "💡 With 500 cases, you now have the FULL Ahmed dataset for maximum training diversity!"
