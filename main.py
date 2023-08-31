import heapq
import importlib
import io
import math
import os
import zipfile
import copy
import types
import yaml,json
import pandas as pd
import numpy as np
import scipy.sparse as sp


random_seed = 1
num_neg_samples = 50
data_dir='data/small_data'
model_config='fid.yaml'
checkpoint_dir = 'check_point_dir'

def parse_data( data_dir):
    """
    :param data_dir: The directory of files where the dataset is located (one directory is a dataset)
    Each directory contains a rating .csv file with 4 columns, one row representing an interactive sample
    user  item  rating  timestamp
    ---------------------------------The following shows the first five lines of the sample rating .csv file
    196	242	3	881250949
    186	302	3	891717742
    22	377	1	878887116
    244	51	2	880606923
    166	346	1	886397596

    :return: Dataframe，

    Contains four columns of 'user', 'item', 'rating', 'timestamps', and one row represents an interaction

    """
    data = None
    if os.path.isfile(data_dir):
        z = zipfile.ZipFile(data_dir, "r")
        for filename in z.namelist():
            if filename == 'data.csv':
                names = ['user', 'item', 'rating', 'timestamp']
                data = pd.read_csv(z.open(filename), header=None, names=names, sep='\t', dtype=np.int)
        assert data is not None, "The data .csv file could not be found"

    else:
        names = ['user', 'item', 'rating', 'timestamp']
        data = pd.read_csv(data_dir + "/data.csv", header=None, names=names, sep='\t', dtype=np.int)

    return data


def split( data: pd.DataFrame):
    """
    Divide the dataframes read by the ScenarioSeqRec.parse_data into training sets, validation sets, and test machines.
    The data division process is as follows:

    user1     ---------------------------------------------------------------------> time
    The user browses the product ID：  1     4      5      1      7      6            18          13
    Historical browsing time：       123   124    235    343   465     543          766          872
    Training set:                  |<--           train_seq             --> |
    Validation set:               |<--         input X                 --> |<--  target -->|
    Test set:                    |<--                input X                          --> |<--target-->|

    :param data:pd.DataFrame
     A row represents an interaction between a user and an item。
     Contains ['user', 'item', 'rating', 'timestamp'] four columns, one row for one sample.
    :return: (d_train, d_valid, d_test)
    d_train=dict{users:[x,x,x], #A list of users' IDs
                seq_items:[[x,x,x],[x,x,],[x,x,x]] # The list of products browsed by the user
                seq_timestamps:[[x,x,x],[x,x,x],[x,x,x]],# The timestamp list of the goods viewed by the user
                items:[x,x,x,x,]}# A list of all product IDs present in the system
    d_valid=dict{users:[x,x,x],
                seq_items:[[x,x,x],[x,x,],[x,x,x]]
                seq_timestamps:[[x,x,x],[x,x,x],[x,x,x]]，
                items:[x,x,x,x,],
                target：sp.csr_matrix(), #The rows and columns of the sparse matrix represent users and products
                                        # The value of the element is 1, which indicates that users of the row will interact with the products in the column in the future.
                                        # The value of the element is -1, which indicates that the user of the row will not interact with the products in the column in the future. (Obtained by negative sampling)
                                        # This variable is also entered as the groundth parameter of metric.
                 }

    d_test： The structure is consistent with the d_valid
    """

    item_unique = data['item'].unique().tolist()

    user_unique = data['user'].unique().tolist()
    data = data.sort_values(by=['user', 'timestamp']).reset_index(drop=True)

    n_users = max(user_unique)+1
    n_items = max(item_unique)+1

    # Generate a user sequence
    usergroup = data.groupby('user')
    item_list = usergroup['item'].apply(list)
    timestamp_list = usergroup['timestamp'].apply(list)

    # Divide the training set, test set, and validation set
    train_users = []
    train_seq_items = []
    train_seq_timestamps = []

    # Validation set
    valid_users = []
    valid_seq_items = []
    valid_seq_timestamps = []
    valid_target_item = []
    valid_neg_items = []

    # Test set
    test_users = []
    test_seq_items = []
    test_seq_timestamps = []
    test_target_item = []
    test_neg_items = []

    np.random.seed(random_seed)
    all_items = set(item_unique)
    for user_id, visited_items in item_list.iteritems():
        visited_time = timestamp_list[user_id]
        if len(visited_items) < 5:
            continue

        neg_items = list(all_items - set(visited_items))
        # The neg_items is too large and needs to be sampled, reducing the number。
        if len(neg_items) > num_neg_samples:
            neg_items = np.random.choice(neg_items,(num_neg_samples,),replace=False)
        else:
            neg_items = np.random.choice(neg_items, (num_neg_samples,), replace=True)

        # Test data, the last data browsed by the user is used as a positive sample of the test data, and the previous data is used as input
        test_users.append(user_id)
        test_seq_items.append(visited_items[:-1])
        test_seq_timestamps.append(visited_time[:-1])
        test_target_item.append(visited_items[-1])
        test_neg_items.append(neg_items)

        valid_users.append(user_id)
        valid_seq_items.append(visited_items[:-2])
        valid_seq_timestamps.append(visited_time[:-2])
        valid_target_item.append(visited_items[-2])
        valid_neg_items.append(neg_items)

        train_users.append(user_id)
        train_seq_items.append(visited_items[:-2])
        train_seq_timestamps.append(visited_time[:-2])

    d_train = {'users': train_users, 'seq_items': train_seq_items, 'seq_timestamps': train_seq_timestamps,
                  'items': item_unique}

    def convert_to_csr(users,target_item,neg_items,n_users,n_items):
        """
        According to the user(users), the target-item of the user's evaluation and the negative sample set
        that the user has not accessed, a coefficient matrix is constructed to use as a label.

        users: The user of the test
        target_item： The products (positive sample) visited by each user correspond one-to-one with users
        neg_items: A list of products that each user does not visit (negative sample list), and one list corresponds to a negative sample set of one user
        """

        # p_iid positve item id , n_iids a list of negative item ids
        uids,iids,ratings = [],[],[]
        for uid, p_iid, n_iids in zip(users, target_item, neg_items):
            uids.extend([uid]+[uid]*len(n_iids))
            iids.extend([p_iid] + list(n_iids))
            ratings.extend([1]+[-1]*len(n_iids))
        return sp.csr_matrix((ratings,(uids,iids)),shape=(n_users,n_items))

    valid_ratings = convert_to_csr(valid_users,valid_target_item,valid_neg_items,n_users,n_items)
    d_valid =  {'users': valid_users, 'seq_items': valid_seq_items, 'seq_timestamps': valid_seq_timestamps,
                'items': item_unique,'target':valid_ratings}

    test_ratings = convert_to_csr(test_users, test_target_item, test_neg_items, n_users, n_items)
    d_test = {'users': test_users, 'seq_items': test_seq_items, 'seq_timestamps': test_seq_timestamps,
                           'items': item_unique,'target':test_ratings}

    return d_train, d_valid, d_test


class RSSequentialDataSet:
    """
    Convert the d_train or d_valid into a list of samples needed for timing recommendations,
    d_train=dict{users:[x,x,x],
                    seq_items:[[x,x,x],[x,x,],[x,x,x]]
                    seq_timestamps:[[x,x,x],[x,x,],[x,x,x]],
                    items:[x,x,x,x,]}
     d_valid=dict{users:[x,x,x],
                    seq_items:[[x,x,x],[x,x,],[x,x,x]]
                    seq_timestamps:[[x,x,x],[x,x,],[x,x,x]]，
                    items:[x,x,x,x,],
                    target：sp.csr_matrix(),
                     }

    A sample contains inputs:
    user_id  :  User ID
    seq_items： A list of the user's historical interactive product IDs
    seq_times： The time corresponding to the list of user historical interactions
    item_id  :  Product ID of the user's next interaction (positive sample)
    neg_items:  Product ID (negative sample set) the next time the user does not interact
    seq_len:    The length of the user's historical interaction sequence


    3 parameters need to be set：
        （1） ScenarioSeqRec.parsedata : The data returned
        （2） The maximum length of the seq_items
        （3） neg_items : Negative sample size

    eg：
    data = RSSequentialDataSet(ds,10,1)
    print(len(data)) Returns the total sample size
    print(data[0]) Returns the 0th sample。
    """

    def __init__(self, ds,max_item_len, neg_samples = 10):
        """
        ds： d_train or d_train type of dictionary.
        {users: [x,x,x]
        seq_items:[[x,x,x],[x,x,x],[x,x,x]]
        seq_timestamps:[[x,x,x],[x,x,x],[x,x,x]]
        items: [x,x,x]
        target: sp.csr_matrix
        }
        max_item_len: The maximum number of historical user interactions to retain
        neg_samples: Negative sample size
        """
        self.ds = ds
        self.max_item_len = max_item_len
        self.neg_samples = neg_samples
        self.random_seed = 1
        np.random.seed(self.random_seed)

        self.users = ds['users']
        self.items = ds['items']
        self.all_items = set(self.items)

        n = max(self.users)+1

        self.seq_items = [[]]*n
        self.seq_timestamps = [[]] * n
        for i,uid in enumerate(self.users):
            self.seq_items[uid] = ds['seq_items'][i]
            self.seq_timestamps[uid] = ds['seq_timestamps'][i]



        self.index = [] # list of (user_idx, item_idx, seq_slice ),
        # user_id : User ID used to record test data,
        # item_id : The product ID used to record the test data,
        # seq_slice : Used to record the interception range of historical data in the sequence seq_items and seq_timestamps corresponding to the user ID.
        #         Note that not all data in SEQ is taken as input.

        if 'target' in ds: #If there are target items, it is in the testing phase, otherwise it is in the training phase
            self._build_test_index()
        else:
            self._build_train_index()

    def _build_test_index(self):
        """
        Build an index on the test set


        #              - - - - - - - - - - - - - - - - - - - - -> time
        # seq:       [   a   b   c   d   e   f   g  h  i  j  k ]     length  l
        # sample(l-1)：        |<--   x(length：max_len)    -->|  y（target_item）  |
        """

        # Find all positive sample IDs, and item IDs
        ratings = self.ds['target'].tolil()
        self.neg_items = [[]]*ratings.shape[0]

        user_ids = self.users
        item_ids = []
        seq_slice = []
        for u in self.users:
            #Process lists of positive samples, negative samples
            udata = ratings.data[u]
            uitems =ratings.rows[u]
            p_idx =  udata.index(1)
            item_ids.append( uitems.pop(p_idx))
            self.neg_items[u] = uitems
            # Index the user's history
            seq = self.seq_items[u]
            # ----------------------------->  seq
            #       |i|<--  max_item_len -->|
            l = len(seq)
            i = l - self.max_item_len
            i = max(i,0)
            seq_slice.append(slice(i,l))

        self.index = list(zip(user_ids,item_ids,seq_slice))




    def _build_train_index(self):
        """
                The needle training set builds the index

        #              - - - - - - - - - - - - - - - - - - - - -> time
        # seq:      [   a   b   c   d   e   f   g  h  i  j  k ]
        # Sample 1:   | x | y|
        # Sample2:   |<-- x-->| y |
        # Sample3:  |<--    X   -->| y |
        # Sample4: |<--      x     -->| y |
        #Sample5  |<--         x      -->| y |
        #
        # Sample(l-1)：  |<--      x(length：max_len)        -->| y |
        """

        for u in self.users:
            user_ids = []
            item_ids = []
            seq_slice = []
            seq = self.seq_items[u]
            l = len(seq)
            user_ids = [u] * (l-1)
            item_ids = seq[1:]
            for i in range(1,l-1):
                if i < self.max_item_len:
                    seq_slice.append(slice(0,i))
                else:
                    seq_slice.append(slice(i-self.max_item_len, i ))

            self.index.extend(list(zip(user_ids,item_ids,seq_slice)))



    def __len__(self):
        return len(self.index)

    def _get_item(self,idx):
        if isinstance(idx, slice):
            data = [self._get_item(i) for i in range(idx.start,idx.stop,idx.step)]
            # (user_id, item_id,seq_len,item_seq,time_seq,neg_items)
            data = list(zip(*data))

            return data
        else:
            user_id,item_id ,slic = self.index[idx]
            item_seq = np.zeros((self.max_item_len,),dtype=np.int)
            time_seq = np.zeros((self.max_item_len,),dtype=np.int)
            seq_len = slic.stop - slic.start
            item_seq[:seq_len] = self.seq_items[user_id][slic]
            time_seq[:seq_len] = self.seq_timestamps[user_id][slic]

            if 'target' in self.ds:
                neg_items = np.array(self.neg_items[user_id])
            else:
                neg_items = list(self.all_items - set(self.seq_items[user_id] + [item_id]))
                if self.neg_samples == 1:
                    neg_items = np.random.choice(neg_items)
                else:
                    neg_items = np.random.choice(neg_items,(self.neg_samples,))
            return (user_id, item_id,seq_len,item_seq,time_seq,neg_items)


    def __getitem__(self, idx):
        headers = ('user_id', 'item_id','seq_len','item_seq','time_seq','neg_items')
        data= self._get_item(idx)
        return dict(list(zip(headers,data)))

class MetricBase:
    def __init__(self):
        self.bigger= True # The bigger the metric, the better, if False, it means that the smaller the better
    def __str__(self):
        return self.__class__.__name__


    def __call__(self,groundtruth,pred ) ->float:
        raise Exception("Not callable for an abstract function")

class NDCG5(MetricBase):

    def __init__(self, topks=None):
        self.topk = 5
        self.topks = topks  # [2,5],
        if topks is not None:
            self.topk = max(topks)

    def cal_ndcg(self, rank, ground_truth):
        """
        @param rank: list/tuple
        @param ground_truth: list/tuple
        @return: np.array (shape: len_rank)
        """
        len_rank = len(rank)
        len_gt = len(ground_truth)
        idcg_len = min(len_gt, len_rank)
        idcg = np.cumsum(1.0 / np.log2(np.arange(2, len_rank + 2)))
        if len_rank > idcg_len: idcg[idcg_len:] = idcg[idcg_len - 1]
        dcg = np.cumsum([1.0 / np.log2(idx2 + 2) if item in ground_truth else 0.0 for idx2, item in enumerate(rank)])
        return dcg / idcg

    def __call__(self, pred, ground_truth) -> float:
        """
        @param pred: csr_matrix
        @param ground_truth: csr_matrix
        @return: float/ np.array (When self.topks are present)
        """
        gt = ground_truth.tolil()
        pred = pred.tolil()
        users = list(set(ground_truth.nonzero()[0]))
        result = []
        for uid in users:
            pos_items = [gt.rows[uid][i] for i, x in enumerate(gt.data[uid]) if x == 1]
            rank_pair = heapq.nlargest(self.topk,
                                       zip(pred.data[uid], pred.rows[uid]))
            _, topk_items = list(zip(*rank_pair))
            result.append(self.cal_ndcg(topk_items, pos_items))
        if self.topks is not None:
            return np.mean(result, axis=0)[[topk-1 for topk in self.topks]]  # (n,topk) => (topk,) => (len(topks),)
        else:
            return np.mean(result, axis=0)[self.topk-1]  # (n,topk) => (topk,) => (1,)


class NDCG10(MetricBase):
    """
    eg:
        pred/ground_truth:
            [1,2,3,4,7] / [2,7,9];  命中item 2,7; idcg_1 = 1/log_2(2+0) + 1/log_2(2+1) + 1/log_2(2+2)=,
                                                dcg_1 = 1/log_2(2+1) + 1/log_2(2+4)=
                                                ndcg_1 = dcg_1/idcg_1 =
            [1,3,4,7,9] / [4,7];    命中item 4,7;  idcg_2 = 1/log_2(2+0) + 1/log_2(2+1) = ,
                                                dcg_2 = 1/log_2(2+2) + 1/log_2(2+3)= ,
                                                ndcg_2 = dcg_2/idcg_2
        NDCG@5 = 1/2 * (ndcg_1 + ndcg_2)
    """
    def __init__(self, topks=None):
        self.topk = 10

    def __call__(self, pred: sp.isspmatrix_csr, ground_truth: sp.isspmatrix_csr) -> float:
        """
        @param pred: csr_matrix
        @param ground_truth: csr_matrix
        @return: float
        """
        gt = ground_truth.tolil()
        pred = pred.tolil()
        users = list(set(ground_truth.nonzero()[0]))
        ndcg = []
        for uid in users:
            pos_items = [gt.rows[uid][i] for i, x in enumerate(gt.data[uid]) if x > 0]
            rank_pair = heapq.nlargest(self.topk,
                                       zip(pred.data[uid], pred.rows[uid]))
            _, topk_items = list(zip(*rank_pair))
            idcg_len = min(len(topk_items), len(pos_items))
            idcg = sum([1.0 / math.log2(idx) for idx in range(2, idcg_len + 2)])
            dcg = sum(
                [1.0 / math.log2(idx2 + 2) if item in pos_items else 0.0 for idx2, item in enumerate(topk_items)])
            ndcg.append(dcg / idcg)
        return sum(ndcg) / len(ndcg)

# Ranking-based
class HR5(MetricBase):

    def __init__(self, topks=None):
        self.topk = 5

    def __call__(self, pred, ground_truth) -> float:
        """
        @param pred: csr_matrix, pred
        @param ground_truth: csr_matrix
        @return: float
        """
        gt = ground_truth.tolil()
        pred = pred.tolil()
        users = list(set(ground_truth.nonzero()[0]))
        hr = []
        for uid in users:
            pos_items = [gt.rows[uid][i] for i, x in enumerate(gt.data[uid]) if x == 1]
            rank_pair = heapq.nlargest(self.topk,
                                       zip(pred.data[uid], pred.rows[uid]))
            _, topk_items = list(zip(*rank_pair))
            hits = [1 if item in pos_items else 0 for item in topk_items]
            hr.append(sum(hits) / len(pos_items))
        return sum(hr) / len(hr)


class HR10(MetricBase):
    """
    eg:
        pred/ground_truth:
            [1,2,3,4,7] / [2,7,9];  hit item 2, 7; hr_1 = 2/3
            [1,3,4,8,9] / [4,7];    hit item 4;   hr_2 = 1/2
        HR@5 = 1/2 * (2/3 + 1/2)
    """
    def __init__(self, topks=None):
        self.topk = 10

    def __call__(self, pred:sp.isspmatrix_csr,ground_truth:sp.isspmatrix_csr) -> float:
        """
        @param pred: csr_matrix, pred
        @param ground_truth: csr_matrix
        @return: float
        """
        gt = ground_truth.tolil()
        pred = pred.tolil()
        users = list(set(ground_truth.nonzero()[0]))
        hr = []
        for uid in users:
            pos_items = [gt.rows[uid][i] for i, x in enumerate(gt.data[uid]) if x > 0]
            rank_pair = heapq.nlargest(self.topk,
                                       zip(pred.data[uid], pred.rows[uid]))
            _, topk_items = list(zip(*rank_pair))
            hits = [1 if item in pos_items else 0 for item in topk_items]
            hr.append(sum(hits))
        return sum(hr) / len(hr)


class ModelAbstract:
    def __init__(self):
        self.checkpoint_path = None
        self.tensorboard_path = None
        self.cache_dir=None
    def train(self, ds,valid_ds = None,test_ds=None,valid_funcs=None,cb_progress=lambda x:None):
        return None
    def predict(self,ds,cb_progress=lambda x:None):
        return None

    def save(self,fio:io.IOBase):
        #  save model to a file (can be memory file io.BytesIO())
        return None

    def load(self,fio:io.IOBase):
        #  load model from a file (can be memory file io.BytesIO())
        return None

    def class_name(self):

        return str(self.__class__)[8:-2].split('.')[-1].lower()

    def __str__(self):
        parameters_dic=copy.deepcopy(self.__dict__)
        parameters=get_parameters_js(parameters_dic)
        return dict_to_yamlstr({self.class_name():parameters})

    def __getitem__(self, key):
        if isinstance(key,str) and hasattr(self,key):
            return getattr(self, key)
        else:
            return None
    def __setitem__(self, key,value):
        if isinstance(key,str):
            return setattr(self, key, value)
        else:
            return None

def dict_to_yamlstr(d:dict)->str:
    with io.StringIO() as mio:
        json.dump(d, mio)
        mio.seek(0)
        if hasattr(yaml, 'full_load'):
            y = yaml.full_load(mio)
        else:
            y = yaml.load(mio)
        return yaml.dump(y)

def get_parameters_js(js) -> dict:
    ans = None
    if isinstance(js, (dict)):
        ans = dict([(k,get_parameters_js(v)) for (k,v) in js.items() if not isinstance(v, types.BuiltinMethodType)])
    elif isinstance(js, (float, int, str)):
        ans = js
    elif isinstance(js, (list, set, tuple)):
        ans = [get_parameters_js(x) for x in js]
    elif js is None:
        ans = None
    else:
        ans = {get_full_class_name(js): get_parameters_js(js.__dict__)}
    return ans

def get_full_class_name(c)->str:
    s = str(type(c))
    return s[8:-2]


def myloads(jstr):

    if hasattr(yaml, 'full_load'):
       js = yaml.full_load(io.StringIO(jstr))
    else:
       js = yaml.load(io.StringIO(jstr))
    if isinstance(js, str):
        return {js: {}}
    else:
        return js



def need_import(value):

    if isinstance(value, str) and len(value) > 3 and value[0] == value[-1] == '_' and not value == "__init__":
        return True
    else:
        return False


def create_obj_from_json(js):

    if isinstance(js, dict):
        rtn_dict = {}
        for key, values in js.items():
            if need_import(key):
                assert values is None or isinstance(values,
                                                    dict), f"The value of the object {key} to be imported must be dict or None"
                assert len(js) == 1, f"{js} contains {key} objects that need to be imported, and cannot contain other key-value pairs"
                key = key[1:-1]
                cls = my_import(key)
                if "__init__" in values:
                    assert isinstance(values, dict), f"__init__ keyword, put into a dictionary object, as an initialization function of the parent class {key}"
                    init_params = create_obj_from_json(values['__init__'])
                    if isinstance(init_params, dict):
                        obj = cls(**init_params)
                    else:
                        obj = cls(init_params)
                    values.pop("__init__")
                else:
                    obj = cls()

                for k, v in values.items():
                    setattr(obj, k, create_obj_from_json(v))
                return obj
            rtn_dict[key] = create_obj_from_json(values)
        return rtn_dict
    elif isinstance(js, (set, list)):
        return [create_obj_from_json(x) for x in js]
    elif isinstance(js,str):
        if need_import(js):
            cls_name = js[1:-1]
            return my_import(cls_name)()
        else:
            return js
    else:
        return js


def my_import(name):
    components = name.split('.')
    model_name = '.'.join(components[:-1])
    class_name = components[-1]
    mod = importlib.import_module(model_name)
    cls = getattr(mod, class_name)
    return cls

def enclose_class_name(value):
    if isinstance(value,dict):
        assert len(value)==1, " Must one class"
        for k,v in value.items():
            if k[0]==k[-1]=="_":
                return {k:v}
            else:
                return {f"_{k}_":v}
    elif isinstance(value,str):
        if value[0]==value[-1]=="_":
            return value
        else:
            return f"_{value}_"
    else:
        return value

def set_seed(seed=2333):

    import random,os, torch, numpy as np
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def main():
    data = parse_data(data_dir)
    d_train, d_valid, d_test = split(data)
    metrics = [HR5(),HR10(),NDCG5(),NDCG10()]
    valid_funs = metrics
    with open(model_config, 'rb') as infile:
        cfg = yaml.safe_load(infile)

    set_seed()
    algorithm = create_obj_from_json(enclose_class_name({cfg['algorithm']:cfg['algorithm_parameters']}))
    algorithm.checkpoint_path=checkpoint_dir
    algorithm.train(d_train, d_valid, valid_funs)
    pred = algorithm.predict(d_test)
    results = [m(pred, d_test['target']) for m in metrics]
    headers = [str(m) for m in metrics]
    print(dict(zip(headers, results)))


if __name__ == '__main__':
    main()