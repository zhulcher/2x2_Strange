#!/bin/bash

# Set the target directory (change as needed)
TARGET_DIR="/sdf/data/neutrino/zhulcher/BNBNUMI/dan_carber3_files/"

# Loop over all .root files recursively
find "$TARGET_DIR" -type f -name "*.root" | while read -r file; do
    echo "Processing: $file"
    python print_particle_record.py "$file"
done