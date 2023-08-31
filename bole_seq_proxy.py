import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data._utils import collate
from recbole.data import Interaction
from recbole.trainer import Trainer
from recbole.utils import EvaluatorType,  InputType
from recbole.model.sequential_recommender import *
import scipy.sparse as sp
from torch import nn
import copy
import io
import types
import yaml,json

from main import RSSequentialDataSet

from main import ModelAbstract

default_cfg = dict(MAX_ITEM_LIST_LENGTH= 50 ,
USER_ID_FIELD= "user_id" ,
ITEM_ID_FIELD= "item_id" ,
LIST_SUFFIX='_list',
ITEM_LIST_FIELD= 'item_list',
ITEM_LIST_LENGTH_FIELD='item_length',
TIME_FIELD= "timestamp",
LABEL_FIELD='rating',
NEG_PREFIX='neg_',
loss_decimal_place=4,
device='cpu',
saved=True,
checkpoint_dir='checkpoint',
eval_batch_size=50,
show_progress=True,
seq_len=None,
log_wandb=None,
metric_decimal_place=4,
eval_type=EvaluatorType.RANKING,
MODEL_INPUT_TYPE=InputType.PAIRWISE,
eval_args={"mode": "uni10","order":"TO","group_by_user": True},
# topk=[],
parameters={},
train_neg_sample_args={'dynamic':'none'}
)


class BoleInteraction(dict):
    def __init__(self, *args, **kwargs):
        dict.__init__(self, *args, **kwargs)
        self.__dict__ = self
    def allowDotting(self, state=True):
        if state:
            self.__dict__ = self
        else:
            self.__dict__ = dict()
    def to(self,device):
        for key in self.keys():
            if  isinstance(self[key],(torch.Tensor,nn.Module)):
                self[key] = self[key].to(device)
        return self

    @property
    def length(self,):
        return len(self['user_id'])

    @property
    def interaction(self):
        return self

    @property
    def final_config_dict(self):
        return self


def BoleSASRec():
    return MakeClass('BoleSASRec', (ModelAbstract, SASRec))


def BoleLightSANs():
    return MakeClass('BoleLightSANs', (ModelAbstract, LightSANs))

def BoleFID():
    return MakeClass('BoleFID', (ModelAbstract, FID))

def BoleTiSASRec():
    return MakeClass('BoleTiSASRec', (ModelAbstract, TiSASRec))

class BoleSequentialDataset:
    def __init__(self,config,ds):
        self.uid_field = config['USER_ID_FIELD']
        self.iid_field = config['ITEM_ID_FIELD']
        self.seq_len = config['max_seq_length']
        self.ds = ds

    @property
    def user_num(self):
        return  self.num(self.uid_field)


    def num(self,field):
        if field ==self.uid_field:
            return max(self.ds['users']) + 1
        if field ==self.iid_field:
            return max(self.ds['items']) + 1
        return len(self.ds[field])




def train_collate_fn(batch):
    """
    batch {'user_id':xx, 'item_id':xx,'seq_len':xx,'item_seq':[xx,xx],'time_seq':[xx],'neg_items':[xx,xx]}

    Merge all dictionaries in one batch into one：
    {'user_id':np.array([xx,xx]),
    'item_id':np.array([xx,xx]),
    'seq_len':np.array([xx,xx]),
    'item_seq':np.array([[xx,xx],[xx,xx]]),
    'time_seq':np.array([xx,xx]),
    'neg_items':np.array([xx,xx])}

     Change the keyword to a version that RecBole recognizes

    """
    d = collate.default_collate(batch)
    d['item_id_list'] = d.pop('item_seq')
    d['item_length'] = d.pop('seq_len')
    d['neg_item_id']=d.pop('neg_items')
    d['timestamp_list'] = d.pop('time_seq')
    return BoleInteraction(d)

def valid_collate_fn(batch):
    """
    batch :{'user_id':xx, 'item_id':xx,'seq_len':xx,'item_seq':[xx,xx],'time_seq':[xx],'neg_items':[xx,xx]}

    return  (interaction,row_idx, positive_u, positive_i)
    """
    batch_positive = []
    batch_negative =[]
    for b in batch:
        # {'user_id':xx, 'item_id':xx,'seq_len':xx,'item_seq':[xx,xx],'time_seq':[xx],'neg_items':[xx,xx]}
        neg_items = b['neg_items'] # A list of products with a negative sample.
        batch_positive.append(dict(b))
        for neg_item_id in neg_items: # Based on the negative sampling results neg_items negative samples are generated
            b.update({'item_id':neg_item_id})
            batch_negative.append(dict(b))

    interaction = train_collate_fn(batch_positive+batch_negative)

    n_positive = len(batch) # Positive sample size
    user_id = interaction['user_id'] # Positive sample user ID
    uniq_user_id = user_id[:n_positive].tolist()
    row_idx = np.array([uniq_user_id.index(uid) for uid in user_id])

    positive_u = np.arange(n_positive)  #The position of the rows of positive samples in the batch matrix，
    positive_i = np.array(interaction['item_id'][:n_positive])

    return (interaction,row_idx, positive_u, positive_i)



def MakeClass(className: str, super_classes):

    ModelAbstract, BaseClassifier = super_classes

    assert issubclass(ModelAbstract, ModelAbstract), '''
       
    assert issubclass(BaseClassifier, SequentialRecommender), '''

    def __init__(self,):
        ModelAbstract.__init__(self)
        if torch.cuda.is_available():
            self.device = 'cuda'


    def train(self, ds=None, valid_ds=None, valid_funcs=None, cb_progress=lambda x: None):
        if ds is None :
            return BaseClassifier.train(self)
        if isinstance(ds, bool):
            return BaseClassifier.train(self,ds)

        params = vars(self)
        cfg = dict(default_cfg)
        cfg.update(params)
        cfg = BoleInteraction(cfg)
        cfg['model'] = type(self).__name__  #Bole needs to explicitly specify the model name in the cfg
        BaseClassifier.__init__(self,cfg,BoleSequentialDataset(cfg,ds))

        self.to(self.device)

        train_data = RSSequentialDataSet(ds, self.max_seq_length, 1)
        valid_data = RSSequentialDataSet(valid_ds, self.max_seq_length)
        valid_data.item_num = max(ds['items'])+1
        train_loader = DataLoader(train_data, batch_size=cfg['train_batch_size'], collate_fn=train_collate_fn,
                                  shuffle=True)
        valid_loader = DataLoader(valid_data, batch_size=cfg['train_batch_size'], collate_fn=valid_collate_fn,
                                  shuffle=False)
        cfg['checkpoint_dir'] = self.checkpoint_path
        trainer = Trainer(cfg, self)
        trainer.tensorboard = SummaryWriter(self.tensorboard_path)
        best_valid_score, best_valid_result = trainer.fit(
            train_loader, valid_loader, saved=cfg['saved'], show_progress=cfg['show_progress']
        )
        return None

    def predict(self, ds, cb_progress=lambda x: None):

        if  isinstance(ds, (BoleInteraction, Interaction)):
            return BaseClassifier.predict(self,ds)

        ratings = ds['target']
        data = RSSequentialDataSet(ds, self.max_seq_length)

        d = DataLoader(data, batch_size=self.train_batch_size, collate_fn=valid_collate_fn,
                   shuffle=False)
        scores = []
        user_ids = []
        item_ids = []
        with torch.no_grad():
            for interaction, row_idx, positive_u, positive_i  in d:
                user_ids.extend(interaction['user_id'].numpy())
                item_ids.extend(interaction['item_id'].numpy())
                s = self.predict(interaction.to(self.device))
                s = s.cpu().numpy()
                scores.extend(s)

        return sp.csr_matrix((scores,(user_ids,item_ids)),shape=ratings.shape)

    def __str__(self):
        return ModelAbstract.__str__(self)

    funcs= dict( __str__=__str__,
            __init__=__init__,
            predict=predict,
            train=train
    )
    return type(className, super_classes, funcs)()
