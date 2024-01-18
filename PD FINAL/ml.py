# ml.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.impute import SimpleImputer
import pickle

# Load your dataset
df = pd.read_csv('COMP.csv', encoding='ISO-8859-1')
df = df.dropna()

# Feature selection (excluding 'plugin', 'codec', 'level')
features = df.columns.difference(['plugin', 'codec', 'level'])

# Split the data into training and testing sets
X_train, X_test, y_plugin_train, y_plugin_test, y_codec_train, y_codec_test, y_level_train, y_level_test = train_test_split(
    df[features], df['plugin'], df['codec'], df['level'], test_size=0.2, random_state=42)

# Define preprocessing steps
numeric_features = X_train.select_dtypes(include=['int64', 'float64']).columns
categorical_features = X_train.select_dtypes(include=['object']).columns

numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean'))
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# Choose models (Decision Tree Classifiers)
model_plugin = Pipeline(steps=[('preprocessor', preprocessor),
                                ('classifier', DecisionTreeClassifier())])
model_codec = Pipeline(steps=[('preprocessor', preprocessor),
                               ('classifier', DecisionTreeClassifier())])
model_level = Pipeline(steps=[('preprocessor', preprocessor),
                               ('classifier', DecisionTreeClassifier())])

# Train the models
model_plugin.fit(X_train, y_plugin_train)
model_codec.fit(X_train, y_codec_train)
model_level.fit(X_train, y_level_train)

# Save the models using pickle
with open('model_plugin.pkl', 'wb') as model_file:
    pickle.dump(model_plugin, model_file)

with open('model_codec.pkl', 'wb') as model_file:
    pickle.dump(model_codec, model_file)

with open('model_level.pkl', 'wb') as model_file:
    pickle.dump(model_level, model_file)
