#!/bin/bash
# Download ONLY the essential Ahmed files (STL + boundary VTP)

LOCAL_DIR="/data/ahmed_data/raw"
TEMP_DIR="/data/ahmed_temp"

echo "🚀 Downloading essential Ahmed files only (STL + boundary VTP)..."
echo "📊 This will download ~5-10GB instead of 50GB"

# Clean up
rm -rf "$LOCAL_DIR" "$TEMP_DIR"
mkdir -p "$LOCAL_DIR"

# Clone with sparse checkout
git clone --filter=blob:none --no-checkout https://huggingface.co/datasets/neashton/ahmedml "$TEMP_DIR"
cd "$TEMP_DIR"

# Configure sparse checkout for ONLY the files we need
git sparse-checkout init --cone

# Download only STL and boundary VTP files (exclude images, CSVs, slices)
echo "🎯 Configuring sparse checkout for essential files only..."
for i in $(seq 1 100); do  # Start with first 100 runs
    echo "run_$i/ahmed_$i.stl" >> .git/info/sparse-checkout
    echo "run_$i/boundary_$i.vtp" >> .git/info/sparse-checkout
done

# Checkout only the essential files
echo "📥 Downloading essential files (this may take time but uses much less space)..."
git checkout HEAD

# Verify what we downloaded
echo "✅ Download completed! Verifying files..."
stl_count=$(find . -name "ahmed_*.stl" | wc -l)
vtp_count=$(find . -name "boundary_*.vtp" | wc -l)
echo "📊 Downloaded: $stl_count STL files, $vtp_count boundary VTP files"

# Copy files to final location with proper structure
echo "📂 Organizing files for DoMINO..."
for i in $(seq 1 100); do
    if [ -f "run_$i/ahmed_$i.stl" ] && [ -f "run_$i/boundary_$i.vtp" ]; then
        mkdir -p "$LOCAL_DIR/run_$i"
        cp "run_$i/ahmed_$i.stl" "$LOCAL_DIR/run_$i/"
        cp "run_$i/boundary_$i.vtp" "$LOCAL_DIR/run_$i/"
        echo "  ✅ Copied run_$i files"
    fi
done

# Cleanup temp directory
cd /
rm -rf "$TEMP_DIR"

echo ""
echo "🎉 Essential Ahmed download completed!"
echo "📊 Final results:"
echo "  STL files: $(find "$LOCAL_DIR" -name "*.stl" | wc -l)"
echo "  VTP files: $(find "$LOCAL_DIR" -name "*.vtp" | wc -l)"
echo "  Total size: $(du -sh "$LOCAL_DIR" | cut -f1)"

# Verify sample files
echo ""
echo "📋 Sample verification:"
ls -lh "$LOCAL_DIR/run_1/" 2>/dev/null || echo "No files in run_1"
ls -lh "$LOCAL_DIR/run_2/" 2>/dev/null || echo "No files in run_2"

echo ""
echo "🚀 Ready for DoMINO training!"
echo "📝 Next steps:"
echo "  1. cd src"
echo "  2. python process_data.py"
echo "  3. python train.py"
