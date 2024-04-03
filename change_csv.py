import pandas as pd

# Charger le fichier CSV
df = pd.read_csv('predicted_test_categories_200.csv')

# Ajouter une colonne ID avec des identifiants uniques
df['ID'] = range(1, len(df)+1)

# Renommer la colonne 'Category' en 'Label'
df.rename(columns={'Category': 'Label'}, inplace=True)

# Sélectionner les colonnes dans l'ordre désiré
df = df[['ID', 'Label']]

# Sauvegarder le nouveau CSV
df.to_csv('predicted_test_categories_200_with_header.csv', index=False)
