import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight
import numpy as np
import tqdm
import joblib
from collections import OrderedDict


def preprocess_data(data_file, output_dir):
    """
    Exercice : Fonction pour prétraiter les données brutes et les préparer pour l'entraînement de modèles.

    Objectifs :
    1. Charger les données brutes à partir d’un fichier CSV.
    2. Nettoyer les données (par ex. : supprimer les valeurs manquantes).
    3. Encoder les labels catégoriels (colonne `family_accession`) en entiers.
    4. Diviser les données en ensembles d’entraînement, de validation et de test selon une logique définie.
    5. Sauvegarder les ensembles prétraités et des métadonnées utiles.

    Indices :
    - Utilisez `LabelEncoder` pour encoder les catégories.
    - Utilisez `train_test_split` pour diviser les indices des données.
    - Utilisez `to_csv` pour sauvegarder les fichiers prétraités.
    - Calculez les poids de classes en utilisant les comptes des classes.
    """

    # Step 1: Load the data
    print('Loading Data')
    data=pd.read_csv(data_file)

    # Step 2: Handle missing values
    data = data.dropna()

    # Step 3: Encode the 'family_accession' to numeric labels
    label_encoder = LabelEncoder()
    data['class_encoded'] = label_encoder.fit_transform(data['family_accession'])

    # Save the label encoder
    joblib.dump(label_encoder, f"{output_dir}/label_encoder.pkl")

    # Save the label mapping to a text file
    with open(f"{output_dir}/label_mapping.txt", 'w') as f:
        for label, index in zip(label_encoder.classes_, range(len(label_encoder.classes_))):
            f.write(f"{label}: {index}\n")

    # Step 4: Distribute data
    # For each unique class:
    # - If count == 1: go to test set
    # - If count == 2: 1 to dev, 1 to test
    # - If count == 3: 1 to train, 1 to dev, 1 to test
    # - Else: stratified split (train/dev/test)

    train_indices = []
    dev_indices = []
    test_indices = []

    print("Distributing data")
    for cls in tqdm.tqdm(label_encoder.classes_):
        indices = data[data['family_accession'] == cls].index.tolist()
        num_samples = len(indices)

    # Step 5: Convert index lists to numpy arrays
    if num_samples == 1:
        test_indices.append(indices[0])
    elif num_samples == 2:
        dev_indices.append(indices[0])
        test_indices.append(indices[1])
    elif num_samples == 3:
        train_indices.append(indices[0])
        dev_indices.append(indices[1])
        test_indices.append(indices[2])
    else:
        train, temp = train_test_split(indices, test_size=0.4, stratify=data.loc[indices, 'class_encoded'])
        dev, test = train_test_split(temp, test_size=0.5, stratify=data.loc[temp, 'class_encoded'])
        train_indices.extend(train)
        dev_indices.extend(dev)
        test_indices.extend(test)


    # Step 6: Create DataFrames from the selected indices
    # Step 7: Drop unused columns: family_id, sequence_name, etc.

    train_data = data.loc[train_indices].drop(columns=['family_id', 'sequence_name'])
    dev_data = data.loc[dev_indices].drop(columns=['family_id', 'sequence_name'])
    test_data = data.loc[test_indices].drop(columns=['family_id', 'sequence_name'])

    # Step 8: Save train/dev/test datasets as CSV
    train_data.to_csv(f"{output_dir}/train_data.csv", index=False)
    dev_data.to_csv(f"{output_dir}/dev_data.csv", index=False)
    test_data.to_csv(f"{output_dir}/test_data.csv", index=False)

    # Step 9: Calculate class weights from the training set
    # Step 10: Normalize weights and scale
    # class_counts = ...
    # class_weights = ...
    class_counts = data['class_encoded'].value_counts()
    class_weights = class_weight.compute_class_weight(
    'balanced', classes=np.unique(data['class_encoded']), y=data['class_encoded']
    )
    class_weights_dict = dict(zip(np.unique(data['class_encoded']), class_weights))

    # Step 11: Save the class weights
    with open(f"{output_dir}/class_weights.txt", 'w') as f:
        for class_id, weight in class_weights_dict.items():
            f.write(f"Class {class_id}: {weight}\n")

    pass


if __name__ == "__main__":
    import argparse

    print("Script démarré")

    import sys
    sys.argv = [
        "preprocess.py", 
        "--data_file", "./Data-Lakes-tp1-student/data/bronze/random_split/combined_df.csv",
        "--output_dir", "./Data-Lakes-tp1-student/data/bronze/random_split/"
    ]

    parser = argparse.ArgumentParser(description="Preprocess protein data")
    parser.add_argument("--data_file", type=str, required=True, help="Path to train CSV file")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save the preprocessed files")
    args = parser.parse_args()

    preprocess_data(args.data_file, args.output_dir)
