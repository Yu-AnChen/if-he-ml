import numpy as np
import pandas as pd
from joblib import Parallel, delayed
import pickle


class GridDf:
    def __init__(
        self,
        df,
        grid_size,
        y_column_name='Yt',
        x_column_name='Xt',
    ) -> None:
        self.df = df
        self.grid_size = grid_size
        self.y_column_name = y_column_name
        self.x_column_name = x_column_name
        self.category_counts = {}
    
    @property
    def grid_shape(self):
        return (len(self.y_bins)-1, len(self.x_bins)-1)

    @property
    def y_bins(self):
        return np.arange(0, self.y_column.max()+self.grid_size, self.grid_size)

    @property
    def x_bins(self):
        return np.arange(0, self.x_column.max()+self.grid_size, self.grid_size)

    @property
    def y_column(self):
        return self.df[self.y_column_name]
    
    @property
    def x_column(self):
        return self.df[self.x_column_name]
    
    @property
    def cut_y(self):
        return pd.cut(self.y_column, self.y_bins)
    
    @property
    def cut_x(self):
        return pd.cut(self.x_column, self.x_bins)

    def grid_category_count(self, column_name):
        cat = self.df[column_name].astype('category')
        categories = cat.cat.categories
        grouped = cat.groupby([self.cut_y, self.cut_x])
        counts = grouped.size().rename(f"Counts")

        count_series = []
        for c in categories:
            sub_df = self.df[cat == c]
            cut_y = pd.cut(sub_df[self.y_column_name], self.y_bins)
            cut_x = pd.cut(sub_df[self.x_column_name], self.x_bins)
            count_series.append(
                sub_df
                    .groupby([cut_y, cut_x])
                    .size()
                    .rename(f"{column_name}_{c}")
            )
        cat_count = pd.DataFrame(counts).join(count_series)
        rs, cs = self.multiindex_to_coor(cat_count)
        cat_count.loc[:, 'row_s'] = rs
        cat_count.loc[:, 'col_s'] = cs
        self.category_counts.update({column_name: cat_count})
        return cat_count
    
    # seems to be slow when number of categories is low
    # https://github.com/pandas-dev/pandas/issues/46202
    def grid_category_count_slow(self, column_name):
        cat = self.df[column_name].astype('category')
        grouped = cat.groupby([self.cut_y, self.cut_x])
        counts = grouped.size().rename(f"Counts")
        cat_count = pd.DataFrame(counts).join(
            cat
                .groupby([self.cut_y, self.cut_x])
                .value_counts(sort=False)
                .unstack(-1)
                .rename(columns=lambda x: f"{column_name}_{x}")
        ).fillna(0).astype(int)
        rs, cs = self.multiindex_to_coor(cat_count)
        cat_count.loc[:, 'row_s'] = rs
        cat_count.loc[:, 'col_s'] = cs
        self.category_counts.update({column_name: cat_count})
        return cat_count

    def calculate_similarity(self, column_name, similarity_target):
        try:
            cat_count = self.category_counts[column_name]
        except KeyError:
            cat_count = self.grid_category_count(column_name)
        column_mask = cat_count.columns.str.contains(f"{column_name}_")
        assert len(similarity_target) == sum(column_mask)
        compositions = cat_count.loc[:, column_mask].values
        cat_count.loc[:, f"Similarity"] = (
            np.dot(compositions, similarity_target) /
            (np.linalg.norm(compositions, axis=1) * np.linalg.norm(similarity_target))
        )
        return cat_count
    
    @staticmethod
    def multiindex_to_coor(df):
        return [
            df.index.map(lambda x: x[i].left).values.astype(int)
            for i in range(2)
        ]
    
    def crop(self, df, array, n_jobs=-1):
        row_s, col_s = self.multiindex_to_coor(df)
        try:
            return Parallel(n_jobs=n_jobs, require=None, verbose=1)(
                delayed(lambda rs, cs: array[rs:rs+self.grid_size, cs:cs+self.grid_size])(rs, cs) 
                for rs, cs in zip(row_s, col_s)
            )
        except pickle.PicklingError:
            print('cannot be pickled')
            return Parallel(n_jobs=n_jobs, require='sharedmem', verbose=1)(
                delayed(lambda rs, cs: array[rs:rs+self.grid_size, cs:cs+self.grid_size])(rs, cs) 
                for rs, cs in zip(row_s, col_s)
            )