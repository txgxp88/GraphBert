#########--------------------------------------------------

import pandas as pd
import torch
from torch_geometric.data import Data
from sklearn.preprocessing import LabelEncoder
import torch.nn.functional as F

def step_1(dir_path):

    content_path = dir_path.joinpath('Data', 'Cora', 'cora.content')
    cites_path = dir_path.joinpath('Data', 'Cora', 'cora.cites')

    content_df = pd.read_csv(content_path, sep='\t', header=None)

    paper_ids = content_df[0].tolist()  
    features = torch.tensor(content_df.iloc[:, 1:-1].values, dtype=torch.float) 
    labels_raw = content_df.iloc[:, -1].tolist()  

    label_encoder = LabelEncoder()
    labels = torch.tensor(label_encoder.fit_transform(labels_raw), dtype=torch.long)

    id_map = {pid: i for i, pid in enumerate(paper_ids)}

    cites_df = pd.read_csv(cites_path, sep='\t', header=None, names=['source', 'target'])

    cites_df = cites_df[cites_df['source'].isin(id_map) & cites_df['target'].isin(id_map)]

    src = cites_df['source'].map(id_map).tolist()
    dst = cites_df['target'].map(id_map).tolist()

    edge_index = torch.tensor([src, dst], dtype=torch.long)

    data = Data(x=features, edge_index=edge_index, y=labels)


    ########## one-hot encoding
    num_classes = len(data.y.unique())  # Cora has 7 classes

    # F.one_hot for one-hot coding
    y_one_hot = F.one_hot(data.y, num_classes=num_classes).float()  # cora dataset shape: [2708, 7]

    data.num_nodes = data.x.shape[0]
    data.node_list = list(range(data.num_nodes))  # initial nodes
    data.y_one_hot = y_one_hot


    return data

