import nibabel as nib
import numpy as np
from pathlib import Path
import yaml

def check_seg_values(yaml_path):
    # Load the YAML file
    with open(yaml_path, "r") as file:
        data = yaml.safe_load(file)
    
    # Get the first fold's data and just the first case
    fold = data["cross_validation_splits"][0]["fold_1"]
    first_case = fold["train"][0]  # Just get the first training case
    
    # Check the segmentation file
    base_dir = Path(yaml_path).parent
    seg_path = base_dir / first_case / "combined_seg.nii.gz"
    seg_data = nib.load(seg_path).get_fdata()
    
    print(f"Maximum value found: {np.max(seg_data)}")
    print(f"All unique values found: {sorted(np.unique(seg_data))}")

# Run the check
yaml_path = "data/KU-PET-CT/data_splits.yaml"
check_seg_values(yaml_path)