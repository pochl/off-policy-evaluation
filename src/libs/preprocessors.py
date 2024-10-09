from sklearn.preprocessing import OrdinalEncoder
from src.libs.dataclass import DataClass
from copy import deepcopy
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


def random_split(data: DataClass, test_size=0.3):

    interaction_train, interaction_test = train_test_split(data.interactions, test_size=test_size, stratify=data.interactions['feedback'], random_state=42)

    return DataClass(interaction_train, data.users, data.items), DataClass(interaction_test, data.users, data.items)


class IDEncoder:

    def __init__(self) -> None:
        
        self.user_encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value= -1, dtype=int)
        self.item_encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value= -1, dtype=int)

    def fit(self, data: DataClass):

        self.all_users_fitted, self.all_items_fitted = self._get_all_users_items(data)

        self.user_encoder.fit(self.all_users_fitted[:,None])
        self.item_encoder.fit(self.all_items_fitted[:,None])
        
    def transform(self, user_ids: list = None, item_ids: list = None, data: DataClass = None):

        if user_ids is not None:
            return self.user_encoder.transform(np.array(user_ids)[:,None]).flatten()
        
        if item_ids is not None:
            return self.item_encoder.transform(np.array(item_ids)[:,None]).flatten()

        if data is not None:

            data_ = deepcopy(data)

            if data_.interactions is not None:
                data_.interactions['user_id'] = self.user_encoder.transform(data_.interactions['user_id'].to_numpy()[:,None]).flatten()
                data_.interactions['item_id'] = self.item_encoder.transform(data_.interactions['item_id'].to_numpy()[:,None]).flatten()

            if data_.users is not None:
                data_.users['user_id'] = self.user_encoder.transform(data_.users['user_id'].to_numpy()[:,None]).flatten()

            if data_.items is not None:
                data_.items['item_id'] = self.item_encoder.transform(data_.items['item_id'].to_numpy()[:,None]).flatten()

            return data_
    
    def fit_transform(self, data: DataClass):

        self.fit(data)

        return self.transform(data=data)
    
    def inverse_transform(self, rank_df: pd.DataFrame):
        pass
        
    def _get_all_users_items(self, data: DataClass):

        if data.users is not None:
            all_users = data.users['user_id'].to_numpy()
        else:
            all_users = data.interactions['user_id'].unique()

        if data.items is not None:
            all_items = data.items['item_id'].to_numpy()
        else:
            all_items = data.interactions['item_id'].unique()

        return all_users, all_items