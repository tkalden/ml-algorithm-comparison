
import pandas as pd
from sklearn.preprocessing import StandardScaler


class clean_data:

    def __init__(self,df):
        self.df = df

    def encode_data(self,features):
        for f in features:
            # convert to cateogry dtype
            self.df[f] = self.df[f].astype('category')
            # convert to category codes
            self.df[f] = self.df[f].cat.codes

    def hot_endcode_data(self,categorical):
        for var in categorical:
            self.df = pd.concat([self.df, pd.get_dummies(self.df[var], prefix=var)], axis=1)
            del self.df[var]
    
    def return_processed_data(self):
        return self.df

    def drop_colums(self, columns):
        return self.df.drop(columns, axis=1, inplace=True)
    
    def scale_continous_variable(self,continuous):
        scaler = StandardScaler()
        for var in continuous:
            self.df[var] = self.df[var].astype('float64')
            self.df[var] = scaler.fit_transform(self.df[var].values.reshape(-1, 1))

