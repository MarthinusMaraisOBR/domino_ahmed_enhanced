#!/bin/bash
# Selective Ahmed download - STL + boundary VTP files ONLY (no cutting planes)

LOCAL_DIR="/data/ahmed_data/raw"
echo "🚀 Starting selective Ahmed download (STL + boundary VTP only)..."

# Clear any previous downloads
rm -rf "$LOCAL_DIR"
mkdir -p "$LOCAL_DIR"

# Download files directly to final location
for i in $(seq 1 50); do  # Start with first 50 runs for testing
    echo "📦 Downloading run_$i..."
    
    RUN_DIR="$LOCAL_DIR/run_$i"
    mkdir -p "$RUN_DIR"
    
    # Download STL file (geometry)
    echo "  📐 Downloading STL geometry..."
    huggingface-cli download neashton/ahmedml "run_$i" \
        --include="*.stl" \
        --local-dir-use-symlinks False \
        --local-dir "$LOCAL_DIR" 2>/dev/null
    
    # Download ALL VTP files first, then filter
    echo "  📊 Downloading VTP files..."
    huggingface-cli download neashton/ahmedml "run_$i" \
        --include="*.vtp" \
        --local-dir-use-symlinks False \
        --local-dir "$LOCAL_DIR" 2>/dev/null
    
    # Now filter - keep only boundary VTP, remove cutting plane VTPs
    if [ -d "$RUN_DIR" ]; then
        echo "  🔍 Filtering VTP files (keeping only boundary data)..."
        
        # List all VTP files downloaded
        vtp_files=$(find "$RUN_DIR" -name "*.vtp" 2>/dev/null)
        
        if [ -n "$vtp_files" ]; then
            echo "    Found VTP files:"
            for vtp in $vtp_files; do
                filename=$(basename "$vtp")
                echo "      - $filename"
                
                # Keep files that look like boundary data, remove cutting planes
                if [[ "$filename" == *"boundary"* ]] || [[ "$filename" == *"surface"* ]] || [[ "$filename" == *"wall"* ]]; then
                    echo "        ✅ Keeping boundary file: $filename"
                    # Rename to standard format if needed
                    if [ "$filename" != "boundary_$i.vtp" ]; then
                        mv "$vtp" "$RUN_DIR/boundary_$i.vtp"
                        echo "        📝 Renamed to: boundary_$i.vtp"
                    fi
                elif [[ "$filename" == *"slice"* ]] || [[ "$filename" == *"plane"* ]] || [[ "$filename" == *"cut"* ]]; then
                    echo "        🗑️  Removing cutting plane: $filename"
                    rm "$vtp"
                else
                    # If unsure, check file size - boundary files are usually larger
                    file_size=$(stat -f%z "$vtp" 2>/dev/null || stat -c%s "$vtp" 2>/dev/null)
                    if [ "$file_size" -gt 10000 ]; then  # > 10KB, likely boundary data
                        echo "        ✅ Keeping (large file, likely boundary): $filename"
                        mv "$vtp" "$RUN_DIR/boundary_$i.vtp"
                    else
                        echo "        🗑️  Removing (small file, likely metadata): $filename"
                        rm "$vtp"
                    fi
                fi
            done
        else
            echo "    ⚠️  No VTP files found for run_$i"
        fi
        
        # Rename STL file to standard format
        stl_files=$(find "$RUN_DIR" -name "*.stl" 2>/dev/null)
        if [ -n "$stl_files" ]; then
            for stl in $stl_files; do
                if [ "$(basename "$stl")" != "ahmed_$i.stl" ]; then
                    mv "$stl" "$RUN_DIR/ahmed_$i.stl"
                    echo "    📝 Renamed STL to: ahmed_$i.stl"
                fi
            done
        fi
    fi
        
    # Check progress every 10 runs
    if [ $((i % 10)) -eq 0 ]; then
        echo ""
        echo "✅ Completed $i runs"
        echo "📊 Size so far: $(du -sh "$LOCAL_DIR" | cut -f1)"
        echo "📊 STL files: $(find "$LOCAL_DIR" -name "*.stl" | wc -l)"
        echo "📊 Boundary VTP files: $(find "$LOCAL_DIR" -name "boundary_*.vtp" | wc -l)"
        echo "📊 Total VTP files: $(find "$LOCAL_DIR" -name "*.vtp" | wc -l)"
        echo ""
    fi
done

echo ""
echo "✅ Selective download completed!"
echo "📊 Final statistics:"
echo "📄 STL files: $(find "$LOCAL_DIR" -name "*.stl" | wc -l)"
echo "📄 Boundary VTP files: $(find "$LOCAL_DIR" -name "boundary_*.vtp" | wc -l)"
echo "📄 Other VTP files: $(find "$LOCAL_DIR" -name "*.vtp" ! -name "boundary_*" | wc -l)"
echo "💾 Total size: $(du -sh "$LOCAL_DIR" | cut -f1)"

# Verify files have actual content
echo ""
echo "📋 Sample file verification:"
for run in run_1 run_2 run_3; do
    if [ -d "$LOCAL_DIR/$run" ]; then
        echo "📁 $run:"
        ls -lh "$LOCAL_DIR/$run/" | grep -E '\.(stl|vtp)$' || echo "  No files found"
    fi
done

echo ""
echo "🎯 Ready for surface-only DoMINO training!"
echo "📝 Next steps:"
echo "  1. cd src"
echo "  2. python process_data.py"
echo "  3. python train.py"
