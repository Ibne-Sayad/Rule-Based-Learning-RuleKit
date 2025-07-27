import pandas as pd
import os

# Rulefetcher Data Handler Module
# This module provides functionality for loading, cleaning, and preprocessing data
class DataHandler:
    def __init__(self, input_path: str):
        self.input_path = input_path
        self.df = None

    def load_data(self):
        # Loads data from the specified CSV file.
        if not os.path.exists(self.input_path):
            raise FileNotFoundError(f"File '{self.input_path}' not found.")

        try:
            self.df = pd.read_csv(self.input_path)
        except pd.errors.ParserError as e:
            raise ValueError(f"Error parsing file: {e}")

        return self.df

    def clean_columns(self):
        #Cleans column names by stripping whitespace and converting to lowercase
        if self.df is None:
            raise ValueError("Data not loaded. Call load_data() first.")

        # Strip whitespace and lowercase column names
        self.df.columns = [col.strip().lower() for col in self.df.columns]
        return self.df

    def rename_columns(self, rename_dict: dict):
        # Renames columns based on a provided dictionary.
        if self.df is None:
            raise ValueError("Data not loaded. Call load_data() first.")
        
        # Normalize current column names before renaming
        current_cols = {col: col.strip().lower() for col in self.df.columns}
        inverse_mapping = {v: k for k, v in current_cols.items()}
        
        # Normalize keys of rename_dict for matching
        normalized_rename = {
            k.strip().lower(): v for k, v in rename_dict.items()
        }

        # Apply renaming only if the column exists
        valid_renames = {
            col: new_name
            for col, new_name in normalized_rename.items()
            if col in self.df.columns
        }

        self.df.rename(columns=valid_renames, inplace=True)
        return self.df
    
    def drop_columns(self, columns_to_drop: list):
        # Drops specified columns from the DataFrame.
        if self.df is None:
            raise ValueError("Data not loaded. Call load_data() first.")

        # Convert all columns to lowercase for matching
        cols_in_df = set(self.df.columns)
        to_drop = [col for col in columns_to_drop if col in cols_in_df]

        if not to_drop:
            print("No matching columns found to drop.")
        else:
            self.df.drop(columns=to_drop, inplace=True)
            print(f"Dropped columns: {to_drop}")

        return self.df


    def handle_missing_values(self, strategy="keep"):
        # Handles missing values based on the specified strategy.
        if self.df is None:
            raise ValueError("Data not loaded. Call load_data() first.")

        if strategy == "drop":
            self.df.dropna(inplace=True)
        elif strategy == "keep":
            self.df = self.df.astype(str).fillna("__MISSING__")
        else:
            raise ValueError("Invalid missing value strategy. Choose 'drop' or 'keep'.")

        return self.df

    def encode_categoricals(self):
        # Encodes categorical columns as strings.
        if self.df is None:
            raise ValueError("Data not loaded. Call load_data() first.")

        for col in self.df.select_dtypes(include=['object']).columns:
            self.df[col] = self.df[col].astype(str)

        return self.df
