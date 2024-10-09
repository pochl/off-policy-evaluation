from src.libs.preprocessors import IDEncoder
from src.libs.dataclass import DataClass
from sklearn.metrics.pairwise import cosine_similarity
from typing import Iterable
import numpy as np
import pandas as pd


class CBF:

    def __init__(self) -> None:

        self.id_encoder = IDEncoder()

    def fit(self, data: DataClass):
        """Computes users' embeddings by averaging embedding of items that the user liked.

        Args:
            feedback_matrix (_type_): _description_
        """

        data_enc = self.id_encoder.fit_transform(data)
        self.seen_items = data_enc.interactions.groupby('user_id')['item_id'].agg(list).sort_index().values.tolist()
        
        emb_cols = [col for col in data_enc.items.columns if ('emd_' in col)]

        self.items_emd = data_enc.items[emb_cols].to_numpy()

        merged_df = pd.merge(
            data_enc.interactions[data_enc.interactions['feedback'] == 1],
            data_enc.items,
            on='item_id'
        )

        merged_df = merged_df[['user_id'] + emb_cols]
        
        self.users_emd = merged_df.groupby('user_id').mean().sort_index().to_numpy()
        

    def score(self, user_id=None, data: DataClass = None):

        user_idx = self._process_user_id(user_id)

        return cosine_similarity(self.users_emd[user_idx], self.items_emd)
    
    def rank(self, user_id=None, data: DataClass = None, exclude_seen=True):

        user_idx = self._process_user_id(user_id)

        scores = self.score(user_id)

        if exclude_seen:
            for i, u in enumerate(user_idx):
                scores[i, self.seen_items[u]] = -np.inf

        return np.argsort(-scores, axis=1)
    
    def recommend(self, user_id=None, top_n=None):
        
        ranked_items = self.rank(user_id)

        if top_n is not None:
            return ranked_items[:, :top_n]
        
        return ranked_items
    
    def _process_user_id(self, user_id):

        if user_id is not None:
            if isinstance(user_id, Iterable):
                user_idx = self.id_encoder.user_encoder.transform(user_id)
            else:
                user_idx = np.array([user_idx])
        else:
            user_idx = np.array(range(len(self.id_encoder.user_encoder.categories_[0])))

        return user_idx