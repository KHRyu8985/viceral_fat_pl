#!/bin/bash

# Directory containing the data
DATA_DIR="/home/kanghyun/viceral_fat_pl/data/KU-PET-CT"
YAML_FILE="/home/kanghyun/viceral_fat_pl/data/KU-PET-CT/data_splits.yaml"

# Create a temporary file for the new YAML content
TEMP_YAML=$(mktemp)

# Process each subject directory
for subject_dir in "${DATA_DIR}"/*/; do
    if [ -d "$subject_dir" ]; then
        subject_name=$(basename "$subject_dir")
        
        # Check if combined_seg.nii.gz exists
        if [ ! -f "${subject_dir}/combined_seg.nii.gz" ]; then
            echo "Removing directory: $subject_dir"
            rm -rf "$subject_dir"
            
            # Remove the subject from the YAML file
            grep -v "^[[:space:]]*- ${subject_name}$" "$YAML_FILE" > "$TEMP_YAML"
            mv "$TEMP_YAML" "$YAML_FILE"
        fi
    fi
done

echo "Processing complete"
