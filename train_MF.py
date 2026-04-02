
import torch
import pickle
import numpy as np
import pandas as pd
import torch.nn as nn
from models import NCF, MF
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import mean_squared_error
from sklearn.metrics import accuracy_score, f1_score
from utils import recall_at_k, ndcg_k, save_epoch_triples
import random 
import os
import torch
from torch.utils.data import Dataset
import numpy as np



def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if using multi-GPU
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def evaluate_implicit(predictions, test_df, threshold=0.5):
    pred_binary = (predictions >= threshold).astype(int)
    labels = test_df['feedback'].values

    acc = accuracy_score(labels, pred_binary)
    f1 = f1_score(labels, pred_binary)

    print(f"Accuracy: {acc:.4f}, F1: {f1:.4f}")

    return {'Accuracy': acc, 'F1': f1}, predictions, labels


def informed_neg_sampling(df, n_items, neg_per_pos=4, seed=None):
    """Negative samples from label=0 items if available, otherwise random."""
    if seed is not None:
        np.random.seed(seed)
    neg_rows = []
    pos_rows = df[df['feedback'] == 1]
    neg_label_rows = df[df['feedback'] == 0]

    # Map from user -> set of items with label 0
    user_zero_items = neg_label_rows.groupby('user_id')['item_id'].apply(set).to_dict()
    all_item_ids = np.arange(n_items)

    for u, g in pos_rows.groupby('user_id'):
        pos_items = set(g['item_id'])
        n_neg = len(g) * neg_per_pos

        # Candidate negative items: first try user-specific label-0, else random
        cand_neg_items = list(user_zero_items.get(u, set()))
        if len(cand_neg_items) >= n_neg:
            sampled_neg_items = np.random.choice(list(set(cand_neg_items) - pos_items), size=n_neg, replace=False)
        else:
            sampled_neg_items = list(set(cand_neg_items) - pos_items)
            # Fill up remaining with random negatives not seen as positive or label-0
            remaining = n_neg - len(sampled_neg_items)
            exclude = pos_items.union(set(cand_neg_items))
            possible_items = list(set(all_item_ids) - exclude)
            if len(possible_items) < remaining:
                # If too few possible, sample with replacement
                random_neg = np.random.choice(possible_items, size=remaining, replace=True)
            else:
                random_neg = np.random.choice(possible_items, size=remaining, replace=False)
            sampled_neg_items = list(sampled_neg_items) + list(random_neg)

        for j in sampled_neg_items:
            neg_rows.append([u, j, 0.0])

    neg_df = pd.DataFrame(neg_rows, columns=['user_id', 'item_id', 'feedback'])
    return pd.concat([pos_rows, neg_df], ignore_index=True)


def uniform_neg_sampling(df, n_items, neg_per_pos=4):
    """Return a dataframe that has `neg_per_pos` zeros for every positive row."""
    neg_rows = []
    pos_rows = df[df['feedback'] == 1]

    for u, g in pos_rows.groupby('user_id'):
        pos_items = set(g['item_id'])
        for _ in range(len(g) * neg_per_pos):
            j = np.random.randint(n_items)
            while j in pos_items:
                j = np.random.randint(n_items)
            neg_rows.append([u, j, 0.0])

    neg_df = pd.DataFrame(neg_rows, columns=['user_id','item_id','feedback'])
    return pd.concat([pos_rows, neg_df], ignore_index=True)


def evaluate(predictions, test_df):
    
    mse = mean_squared_error(test_df['rating'], predictions)
    rmse = np.sqrt(mse)
    
    print(f"Test RMSE: {rmse:.4f}")
    print(f"Test MSE: {mse:.4f}")

    rate_dict_ori = {}
    for ratings in test_df['rating']:
        if int(ratings) not in rate_dict_ori:
            rate_dict_ori[int(ratings)] = 0
        rate_dict_ori[int(ratings)] +=1

    print('Original Test data distro: ', rate_dict_ori)

    rate_dict_pred = {}
    for ratings in predictions:
        if int(ratings) not in rate_dict_pred:
            rate_dict_pred[int(ratings)] = 0
        rate_dict_pred[int(ratings)] +=1

    print('Pred Test data distro: ', rate_dict_pred)
    
    return {'rmse': rmse, 'mse': mse},  predictions, test_df['rating']

def calculate_rmse(predictions, targets):
    mse = torch.mean((predictions - targets) ** 2)  # Mean Squared Error
    rmse = torch.sqrt(mse)  # Root Mean Squared Error
    return rmse


class BPRTripletDataset(Dataset):
    def __init__(self, df, n_items, seed=42):
        """
        df: DataFrame with columns ['user_id', 'item_id', 'feedback'] where feedback=1 is positive.
        n_items: total number of items
        """
        self.user_pos = {}
        self.n_items = n_items
        self.seed = seed

        for user, group in df[df['feedback'] == 1].groupby('user_id'):
            self.user_pos[user] = set(group['item_id'])

        self.users = []
        self.pos_items = []
        for user, pos_items in self.user_pos.items():
            for pos in pos_items:
                self.users.append(user)
                self.pos_items.append(pos)

    def __len__(self):
        return len(self.users)

    def __getitem__(self, idx):
        user = self.users[idx]
        pos_item = self.pos_items[idx]

        # deterministic negative per (epoch seed, idx)
        rng = np.random.RandomState(self.seed + idx)

        while True:
            neg_item = rng.randint(self.n_items)
            if neg_item not in self.user_pos[user]:
                break

        return (
            torch.tensor(user, dtype=torch.long),
            torch.tensor(pos_item, dtype=torch.long),
            torch.tensor(neg_item, dtype=torch.long),
        )



def get_acc_f1(test_df):
    user_ids = test_df['user_id'].values
    item_ids = test_df['item_id'].values
    pred_rating = model.predict(user_ids, item_ids)
    out = evaluate_implicit(pred_rating, test_df)

def get_ranks(model, test_df, train_user_dict, save_model = False, best_score = None, model_saving_path = None, opt = None, user_batch_size =512):
    model.eval()

    unique_users = test_df['user_id'].unique().tolist()
    answers = [test_df[test_df['user_id'] == user]['item_id'].tolist() for user in unique_users]

    all_ranking_scores = [] 
    with torch.no_grad():
        for start in range(0, len(unique_users), user_batch_size):
            user_in = unique_users[start: start + user_batch_size]
            ranking_score = model.predict_for_rank(user_in)  
            all_ranking_scores.append(ranking_score)         

    ranking_score = np.concatenate(all_ranking_scores, axis=0)

    for index, user in enumerate(unique_users):
        train_items = train_user_dict[int(user)]
        for item in train_items:
            ranking_score[index][item] = -np.inf

    ind = np.argpartition(ranking_score, -50)[:, -50:]
    arr_ind = ranking_score[np.arange(len(ranking_score))[:, None], ind]
    arr_ind_argsort = np.argsort(arr_ind)[np.arange(len(ranking_score)), ::-1]
    pred_list = ind[np.arange(len(ranking_score))[:, None], arr_ind_argsort]


    recall, ndcg = [], []
    for k in [10, 20, 50]:
        recall.append(recall_at_k(answers, pred_list, k))
        ndcg.append(ndcg_k(answers, pred_list, k))

    
    if save_model and recall[1] > best_score:
        best_score = recall[1]
        print('Best model found! Saving...')
        torch.save({
            'model': model.state_dict(),
            'optimizer': opt.state_dict()
        }, model_saving_path + '/best_model_noisy_bpr_02.pth')

    print("Recall@10:" '{:.4f}'.format(recall[0]), "NDCG@10:" '{:.4f}'.format(ndcg[0]))
    print("Recall@20:" '{:.4f}'.format(recall[1]), "NDCG@20:" '{:.4f}'.format(ndcg[1]))
    print("Recall@50:" '{:.4f}'.format(recall[2]), "NDCG@50:" '{:.4f}'.format(ndcg[2]))


if __name__ == "__main__":

    seed = 42
    set_seed(seed)

    dir_path = "./yelp"
    train_df = pickle.load(open(f"{dir_path}/02_modelled_noisy_data_full.p", "rb"))
    test_df = pickle.load(open(f"{dir_path}/test_df.p", "rb"))

    u_map = pickle.load(open(f"{dir_path}/user_id_map.p", "rb"))
    i_map = pickle.load(open(f"{dir_path}/item_id_map.p", "rb"))

    model_saving_path = dir_path + "_models"
    os.makedirs(model_saving_path, exist_ok=True)

    triples_saving_path = os.path.join(model_saving_path, "all_epoch_triples")
    os.makedirs(triples_saving_path, exist_ok=True)

    epochs = 50
    emb_dim = 100
    n_users = 31668
    n_items = 38048
    weight_decay = 0.01
    learning_rate = 0.01
    batch_size = 2048

    model = MF(n_users, n_items, emb_dim).cuda()
    opt = torch.optim.Adam(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay / batch_size
    )

    test_df = test_df[test_df['feedback'] == 1]
    user_item_dict = train_df.groupby('user_id')['item_id'].apply(list).to_dict()

    best_score = 0.0

    for epoch in range(epochs):
        loss_at_epoch = []
        model.train()

        # make epoch-specific deterministic dataset
        # so each epoch can have different but reproducible negatives
        train_dataset = BPRTripletDataset(train_df, n_items, seed=seed + epoch)

        # save all triples for this epoch
        triples_file = os.path.join(triples_saving_path, f"triples_epoch_{epoch:03d}.pkl")
        save_epoch_triples(train_dataset, triples_file)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        for batch in train_loader:
            users = batch[0].cuda()
            pos_items = batch[1].cuda()
            neg_items = batch[2].cuda()

            loss = model.bpr_loss(users, pos_items, neg_items)

            opt.zero_grad()
            loss.backward()
            opt.step()

            loss_at_epoch.append(loss.item())

        print('Epoch: {}, Loss: {} '.format(
            epoch, sum(loss_at_epoch) / (len(loss_at_epoch) * batch_size))
        )

        if epoch % 5 == 0:
            print("[TEST]")
            current_score = get_ranks(
                model, test_df, user_item_dict, True, best_score, model_saving_path, opt
            )

            # if get_ranks returns score, use this
            if current_score is not None and current_score > best_score:
                best_score = current_score
                torch.save(
                    {
                        "model": model.state_dict(),
                        "optimizer": opt.state_dict(),
                        "epoch": epoch,
                        "triples_file": triples_file,
                    },
                    model_saving_path + '/best_model_noisy_bpr_02.pth'
                )

    print('Final Testing!!!!')

    model = MF(n_users, n_items, emb_dim).cuda()

    checkpoint = torch.load(model_saving_path + '/best_model_noisy_bpr_02.pth')
    model.load_state_dict(checkpoint['model'])

    model.eval()
    get_ranks(model, test_df, user_item_dict)

    print("Best checkpoint epoch:", checkpoint["epoch"])
    print("Triples used for that checkpoint:", checkpoint["triples_file"])