import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
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
    # data = pd.read_csv(...)

    # Step 2: Handle missing values
    # data = data.dropna()

    # Step 3: Encode the 'family_accession' to numeric labels
    # label_encoder = LabelEncoder()
    # data['class_encoded'] = label_encoder.fit_transform(...)

    # Save the label encoder
    # joblib.dump(...)

    # Save the label mapping to a text file
    # with open(...)

    # Step 4: Distribute data
    # For each unique class:
    # - If count == 1: go to test set
    # - If count == 2: 1 to dev, 1 to test
    # - If count == 3: 1 to train, 1 to dev, 1 to test
    # - Else: stratified split (train/dev/test)

    print("Distributing data")
    # for cls in tqdm.tqdm(...):

        # Logic for assigning indices to train/dev/test

    # Step 5: Convert index lists to numpy arrays

    # Step 6: Create DataFrames from the selected indices

    # Step 7: Drop unused columns: family_id, sequence_name, etc.

    # Step 8: Save train/dev/test datasets as CSV
    # df.to_csv(...)

    # Step 9: Calculate class weights from the training set
    # class_counts = ...
    # class_weights = ...

    # Step 10: Normalize weights and scale

    # Step 11: Save the class weights
    # with open(...)

    pass


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Preprocess protein data")
    parser.add_argument("--data_file", type=str, required=True, help="Path to train CSV file")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save the preprocessed files")
    args = parser.parse_args()

    preprocess_data(args.data_file, args.output_dir)
