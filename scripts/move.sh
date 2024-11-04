#! /bin/bash

SOURCE_DIR="/home/kanghyun/viceral_fat_pl/data/KU-PET-CT2"
TARGET_DIR="/home/kanghyun/viceral_fat_pl/data/KU-PET-CT"

# Loop through all directories in source
for subject_dir in "$SOURCE_DIR"/*/ ; do
    echo "$subject_dir"
    if [ -d "$subject_dir" ]; then
        # Extract subject number from directory name
        subject_num=$(basename "$subject_dir")
        echo "$subject_num"
        # Only proceed if target directory already exists
        if [ -d "$TARGET_DIR/$subject_num" ]; then
            echo "Target directory exists"
            # Check if source file exists and copy only if it exists
            if [ -f "$subject_dir/combined_image.nii.gz" ]; then
                cp "$subject_dir/combined_image.nii.gz" "$TARGET_DIR/$subject_num/combined_seg.nii.gz"
                echo "Successfully copied and renamed file for subject: $subject_num"
            fi
        fi
    fi
done

echo "Copy operation completed."

