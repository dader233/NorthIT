import pandas as pd
import numpy as np


class DataPreprocessor:
    def __init__(self, df):
        if not isinstance(df, pd.DataFrame):
            raise TypeError("Wrong input data")
        
        self.df = df.copy()
        self.transform_info = {
            'removed_cols': [],
            'imputation': {},
            'og_cols' : [],
            'one_hot_cols' : [],
            'normalization' : {}
            
        }
        
    def remove_missing(self, threshold=0.5):
        if not 0 <= threshold <= 1:
            raise ValueError("Wrong threshold, must be between 0 and 1")
        
        missing_ratio = self.df.isnull().sum() / len(self.df)
        cols_remove = missing_ratio[missing_ratio > threshold].index.tolist()
        
        if cols_remove:
            self.df =  self.df.drop(columns=cols_remove)
            self.transform_info['removed_cols'] = cols_remove
            
        for col in self.df.columns:
            if self.df[col].isnull().any():
                if self.df[col].dtype in ['int64', 'float64']:
                    fill_val = self.df[col].median()
                    fill_type = "median"
                else:
                    if not self.df[col].mode().empty:
                        fill_val = self.df[col].mode()[0]
                    else:
                        fill_val = "Unknown"
                    fill_type = "mode"
                
                self.transform_info["imputation"][col] ={
                    "fill_type": fill_type,
                    "fill_val": fill_val,
                }
                self.df[col] = self.df[col].fillna(fill_val)
        return self
    
    def encode_categorical(self):
        cols = self.df.select_dtypes(include=['object', 'category', 'string']).columns.tolist()
        self.transform_info['og_cats'] = cols
        
        if cols:
            dummies = pd.get_dummies(data=self.df[cols])
            self.transform_info['one_hot_cols'] = dummies.columns.tolist()
            self.df = self.df.drop(columns=cols)
            self.df = pd.concat([self.df, dummies],axis=1) 
        
        return self
    def normalize_numeric(self):
        pass


