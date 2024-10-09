import numpy as np

from obp.utils import sample_action_fast


def prepare_cls_data(context, action_context, action_log, action_tar):

    action_emb_log = action_context[action_log]

    action_emb_tar = action_context[action_tar]

    context_combined = np.vstack((context, context))
    action_emb = np.vstack((action_emb_log, action_emb_tar))
    target = np.concatenate((np.zeros(len(context)), np.ones(len(context))))

    return context_combined, action_emb, target