import autorootcwd
import os
import random
import yaml
from sklearn.model_selection import KFold
import click
from collections import OrderedDict

def get_subject_list(data_path):
    return [d for d in os.listdir(data_path) if os.path.isdir(os.path.join(data_path, d)) and d.isdigit()]

def create_cross_validation_splits(subjects, n_splits=5, val_size=10):
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    cv_splits = []
    
    for fold, (train_idx, test_idx) in enumerate(kf.split(subjects), 1):
        train = [subjects[i] for i in train_idx]
        test = [subjects[i] for i in test_idx]
        
        # Randomly select validation subjects from train
        val = random.sample(train, val_size)
        train = [s for s in train if s not in val]
        
        cv_splits.append({
            f'fold_{fold}': {
                'train': train,
                'val': val,
                'test': test
            }
        })
    
    return cv_splits

def format_subjects_as_strings(subjects):
    return [f"'{subject}'" for subject in subjects]

@click.command()
@click.option('--data_path', default= "data/KU-PET-CT", type=click.Path(exists=True), required=True, help='Path to the data folder')
@click.option('--output_file', type=click.Path(), default='data_splits.yaml', help='Output YAML file name')
def main(data_path, output_file):
    subjects = get_subject_list(data_path)
    
    # Create 5-fold cross-validation splits for all subjects
    cv_splits = create_cross_validation_splits(subjects)
    
    # Format all subject IDs as strings
    for fold in cv_splits:
        for split_name in fold[list(fold.keys())[0]]:
            fold[list(fold.keys())[0]][split_name] = format_subjects_as_strings(fold[list(fold.keys())[0]][split_name])
    
    # Prepare data for YAML
    data = {
        'cross_validation_splits': cv_splits
    }
    
    # Construct the full output path
    output_path = os.path.join(data_path, output_file)
    
    # Write to YAML file
    with open(output_path, 'w') as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=False)
    
    # Remove extra quotes from the YAML file
    with open(output_path, 'r') as f:
        content = f.read()
    content = content.replace("'''", "'")
    with open(output_path, 'w') as f:
        f.write(content)
    
    click.echo(f"Data splits have been written to {output_path}")
    click.echo(f"Total subjects: {len(subjects)}")
    click.echo(f"Cross-validation: {len(cv_splits)} folds")
    click.echo(f"Each fold - Train: {len(cv_splits[0][f'fold_1']['train'])}, "
               f"Validation: {len(cv_splits[0][f'fold_1']['val'])}, "
               f"Test: {len(cv_splits[0][f'fold_1']['test'])}")

if __name__ == '__main__':
    main()
