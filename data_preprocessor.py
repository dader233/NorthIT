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
        self.transform_info['og_cols'] = cols
        
        if cols:
            dummies = pd.get_dummies(data=self.df[cols])
            self.transform_info['one_hot_cols'] = dummies.columns.tolist()
            self.df = self.df.drop(columns=cols)
            self.df = pd.concat([self.df, dummies],axis=1) 
        
        return self
    def normalize_numeric(self, method='minmax'):
        if method not in ['minmax', 'std']:
            raise ValueError("Wrong method, must be minmax or std")
        
        cols = self.df.select_dtypes(include=['number']).columns.tolist()
        for col in cols:
            if method == 'minmax':
                min_val = self.df[col].min()
                max_val = self.df[col].max()
                if max_val > min_val:
                    self.df[col] = (self.df[col] - min_val) / (max_val - min_val)
                    self.transform_info['normalization'][col]= {
                        'method' : 'minmax',
                        'min' : min_val,
                        'max' : max_val
                    }
            else:
                mean_val = self.df[col].mean()
                std_val = self.df[col].std()
                if std_val > 0:
                    self.df[col] = (self.df[col] - mean_val) / std_val
                    self.transform_info['normalization'][col] = {
                        'method' : 'std',
                        'mean' : mean_val,
                        'std' : std_val
                    }
        return self

    def fit_transform(self, threshold=0.5, method='minmax'):
        self.remove_missing(threshold)
        self.encode_categorical()
        self.normalize_numeric(method)
        return self.df
    
    def get_transform_info(self):
        return self.transform_info
    
    def apply_new_data(self, new_df):
        res_def = new_df.copy()
        remove_cols = [col for col in self.transform_info['removed_cols'] if col in res_def.columns]
        
        if remove_cols:
            res_def = res_def.drop(columns=remove_cols)
            
        for col, info in self.transform_info['imputation'].items():
            if col in res_def.columns:
                res_def[col] = res_def[col].fillna(info['fill_val'])

        for col, params in self.transform_info['normalization'].items():
            if col in res_def.columns:
                if params['method'] == 'minmax':
                    res_def[col] = (res_def[col] - params['min']) / (params['max'] - params['min'])
                else:
                    res_def[col] = (res_def[col] - params['mean']) / params['std']
        return res_def


                


