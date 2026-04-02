from dataloader_lightGCN import Loader
from models import LightGCN, PureMF
import utils
import torch
from torch import nn, optim
import numpy as np
import argparse
import random 
import os
import pickle


def parse_args():
    parser = argparse.ArgumentParser(description="Go lightGCN")
    parser.add_argument('--bpr_batch', type=int,default=2048,
                        help="the batch size for bpr loss training procedure")
    parser.add_argument('--recdim', type=int,default=64,
                        help="the embedding size of lightGCN")
    parser.add_argument('--layer', type=int,default=3,
                        help="the layer num of lightGCN")
    parser.add_argument('--lr', type=float,default=0.001,
                        help="the learning rate")
    parser.add_argument('--decay', type=float,default=1e-7,
                        help="the weight decay for l2 normalizaton")
    parser.add_argument('--dropout', type=int,default=0,
                        help="using the dropout or not")
    parser.add_argument('--keepprob', type=float,default=0.6,
                        help="the batch size for bpr loss training procedure")
    parser.add_argument('--a_fold', type=int,default=100,
                        help="the fold num used to split large adj matrix, like gowalla")
    parser.add_argument('--testbatch', type=int,default=100,
                        help="the batch size of users for testing")
    parser.add_argument('--n_users', type=int,default=100,
                        help="the batch size of users for testing")
    parser.add_argument('--n_items', type=int,default=100,
                        help="the batch size of users for testing")
    parser.add_argument('--dataset', type=str,default='amazon-book',
                        help="available datasets: [lastfm, gowalla, yelp2018, amazon-book]")
    parser.add_argument('--path', type=str,default="./checkpoints",
                        help="path to save weights")
    parser.add_argument('--gpu', type=str,default="0",
                        help="path to save weights")
    parser.add_argument('--topks', nargs='?',default="[10, 20, 50]",
                        help="@k test list")
    parser.add_argument('--tensorboard', type=int,default=0,
                        help="enable tensorboard")
    parser.add_argument('--comment', type=str,default="lgn")
    parser.add_argument('--load', type=int,default=0)
    parser.add_argument('--epochs', type=int,default=300)
    parser.add_argument('--multicore', type=int, default=0, help='whether we use multiprocessing or not in test')
    parser.add_argument('--pretrain', type=int, default=0, help='whether we use pretrained weight or not')
    parser.add_argument('--seed', type=int, default=2020, help='random seed')
    parser.add_argument('--model', type=str, default='lgn', help='rec-model, support [mf, lgn]')

    return parser.parse_args()


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if using multi-GPU
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def BPR_train_original(dataset, model, opt, epoch, device, config):
    model.train()

    S = utils.UniformSample_original(dataset)
    users = torch.Tensor(S[:, 0]).long()
    Items = torch.Tensor(S[:, 1]).long()
    labels = torch.Tensor(S[:, 2]).long()

    triples_file = os.path.join(config['triples_saving_path'], f"triples_epoch_{epoch:03d}.pkl")
    utils.save_epoch_triples_lightgcn(users, Items, labels, triples_file)


    users = users.to(device)
    Items = Items.to(device)
    labels = labels.to(device)
    users, Items, labels = utils.shuffle(users, Items, labels)
    total_batch = len(users) // config['bpr_batch_size'] + 1
    aver_loss = 0.
    for (batch_i,
         (batch_users,
          batch_items,
          batch_labels)) in enumerate(utils.minibatch(users,
                                                   Items,
                                                   labels,
                                                   batch_size=config['bpr_batch_size'])):
        
        loss, reg_loss = model.bpr_loss(batch_users, batch_items, batch_labels)
        reg_loss = reg_loss*config['decay']
        loss = loss + reg_loss
        # loss, _ = model.compute_bce_loss(batch_users, batch_items, batch_labels)


        opt.zero_grad()
        loss.backward()
        opt.step()

        aver_loss += loss
        # print(f'BPRLoss/BPR', loss, epoch * int(len(users) / config['bpr_batch_size']) + batch_i)
    aver_loss = aver_loss / total_batch
    return f"loss{aver_loss:.3f}", triples_file

def get_ranks(model, test_df, train_user_dict, user_batch_size = 512):

    unique_users = test_df['user_id'].unique().tolist()
    answers = [test_df[test_df['user_id'] == user]['item_id'].tolist() for user in unique_users]

    all_ranking_scores = []
    for start in range(0, len(unique_users), user_batch_size):
        user_in = unique_users[start: start + user_batch_size]
        ranking_score = model.predict_for_rank(user_in)  
        all_ranking_scores.append(ranking_score)         

    ranking_score = np.concatenate(all_ranking_scores, axis=0)

    for index, user in enumerate(unique_users):
        if int(user) in train_user_dict:
            train_items = train_user_dict[int(user)]
            for item in train_items:
                ranking_score[index][item] = -np.inf

    ind = np.argpartition(ranking_score, -50)[:, -50:]
    arr_ind = ranking_score[np.arange(len(ranking_score))[:, None], ind]
    arr_ind_argsort = np.argsort(arr_ind)[np.arange(len(ranking_score)), ::-1]
    pred_list = ind[np.arange(len(ranking_score))[:, None], arr_ind_argsort]


    recall, ndcg = [], []
    for k in [10, 20, 50]:
        recall.append(utils.recall_at_k(answers, pred_list, k))
        ndcg.append(utils.ndcg_k(answers, pred_list, k))

    print("Recall@10:" '{:.4f}'.format(recall[0]), "NDCG@10:" '{:.4f}'.format(ndcg[0]))
    print("Recall@20:" '{:.4f}'.format(recall[1]), "NDCG@20:" '{:.4f}'.format(ndcg[1]))
    print("Recall@50:" '{:.4f}'.format(recall[2]), "NDCG@50:" '{:.4f}'.format(ndcg[2]))


def test_one_batch(X, topks):
    sorted_items = X[0].numpy()
    groundTrue = X[1]
    r = utils.getLabel(groundTrue, sorted_items)
    pre, recall, ndcg = [], [], []
    for k in topks:
        ret = utils.RecallPrecision_ATk(groundTrue, r, k)
        pre.append(ret['precision'])
        recall.append(ret['recall'])
        ndcg.append(utils.NDCGatK_r(groundTrue,r,k))
    return {'recall':np.array(recall), 
            'precision':np.array(pre), 
            'ndcg':np.array(ndcg)}
        
            
def Test(dataset, Recmodel, device, epoch, config, save_model = False, best_score = None, model_saving_path = None, optimizer = None, triplets = None):
    u_batch_size = config['test_u_batch_size']
    testDict = dataset.testDict
    # eval mode with no dropout
    Recmodel = Recmodel.eval()
    max_K = max(config['topks'])
    results = {'precision': np.zeros(len(config['topks'])),
               'recall': np.zeros(len(config['topks'])),
               'ndcg': np.zeros(len(config['topks']))}
    with torch.no_grad():
        users = list(testDict.keys())
        try:
            assert u_batch_size <= len(users) / 10
        except AssertionError:
            print(f"test_u_batch_size is too big for this dataset, try a small one {len(users) // 10}")
        users_list = []
        rating_list = []
        groundTrue_list = []
        # auc_record = []
        # ratings = []
        total_batch = len(users) // u_batch_size + 1
        for batch_users in utils.minibatch(users, batch_size=u_batch_size):
            allPos = dataset.getUserPosItems(batch_users)
            groundTrue = [testDict[u] for u in batch_users]
            batch_users_gpu = torch.Tensor(batch_users).long()
            batch_users_gpu = batch_users_gpu.to(device)

            rating = Recmodel.getUsersRating(batch_users_gpu)
            #rating = rating.cpu()
            exclude_index = []
            exclude_items = []
            for range_i, items in enumerate(allPos):
                exclude_index.extend([range_i] * len(items))
                exclude_items.extend(items)
            rating[exclude_index, exclude_items] = -(1<<10)
            _, rating_K = torch.topk(rating, k=max_K)
            rating = rating.cpu().numpy()
            del rating
            users_list.append(batch_users)
            rating_list.append(rating_K.cpu())
            groundTrue_list.append(groundTrue)
        assert total_batch == len(users_list)
        X = zip(rating_list, groundTrue_list)

        pre_results = []
        for x in X:
            pre_results.append(test_one_batch(x, config['topks']))
        scale = float(u_batch_size/len(users))
        for result in pre_results:
            results['recall'] += result['recall']
            results['precision'] += result['precision']
            results['ndcg'] += result['ndcg']
        results['recall'] /= float(len(users))
        results['precision'] /= float(len(users))
        results['ndcg'] /= float(len(users))


        if save_model and results['recall'][1] > best_score:
            new_best = results['recall'][1]
            print('Best model found! Saving checkpoint...')
            # prepare checkpoint dict
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': Recmodel.state_dict(),
                'best_score': new_best,
                'triples_file': triplets,
                'config': config
            }
            # include optimizer state if provided
            if optimizer is not None:
                checkpoint['optimizer_state_dict'] = optimizer.state_dict()

            torch.save(checkpoint, os.path.join(model_saving_path, 'best_model_checkpoint.pth'))
            best_score = new_best

        print(f'Test/Recall@{config['topks']}',
                        {str(config['topks'][i]): results['recall'][i] for i in range(len(config['topks']))}, epoch)
        print(f'Test/Precision@{config['topks']}',
                        {str(config['topks'][i]): results['precision'][i] for i in range(len(config['topks']))}, epoch)
        print(f'Test/NDCG@{config['topks']}',
                        {str(config['topks'][i]): results['ndcg'][i] for i in range(len(config['topks']))}, epoch)

        return best_score



if __name__ == "__main__":

    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    config = {}
    config['bpr_batch_size'] = args.bpr_batch
    config['latent_dim_rec'] = args.recdim
    config['lightGCN_n_layers']= args.layer
    config['dropout'] = args.dropout
    config['keep_prob']  = args.keepprob
    config['A_n_fold'] = args.a_fold
    config['test_u_batch_size'] = args.testbatch
    config['multicore'] = args.multicore
    config['lr'] = args.lr
    config['decay'] = args.decay
    config['pretrain'] = args.pretrain
    config['A_split'] = False
    config['bigdata'] = False
    config['dataset_name'] = args.dataset
    config['topks'] = eval(args.topks)
    config['n_users'] = args.n_users
    config['n_items'] = args.n_items

    set_seed(args.seed)

    GPU = torch.cuda.is_available()
    device = torch.device('cuda' if GPU else "cpu")

    dataset_name = args.dataset
    model_name = args.model

    model_saving_path = dataset_name +'_models'
    os.makedirs(model_saving_path , exist_ok=True)

    triples_saving_path = os.path.join(dataset_name, "all_epoch_triples")
    os.makedirs(triples_saving_path, exist_ok=True)

    config['triples_saving_path'] = triples_saving_path

    TRAIN_epochs = args.epochs 
    LOAD = args.load
    PATH = args.path
    topks = eval(args.topks)

    dataset = Loader(config, device, path=f"/home/ubuntu/unlearning_rec_exps/recsys_implicit/datasets/amazon_data/")

    Recmodel = LightGCN(config, dataset)
    Recmodel = Recmodel.to(device)

    # opt = optim.Adam(Recmodel.parameters(), lr=config['lr'], weight_decay = config['decay']/config['bpr_batch_size'])
    opt = optim.Adam(Recmodel.parameters(), lr = config['lr'])

    best_score = 0.0
    for epoch in range(TRAIN_epochs):

        output_information, triples_file = BPR_train_original(dataset, Recmodel, opt, epoch, device, config)

        if epoch %5 == 0:
            print("[TEST]")
            best_score = Test(dataset, Recmodel, device, epoch, config, True, best_score, model_saving_path, optimizer=opt, triplets=triples_file)

        print(f'EPOCH[{epoch+1}/{TRAIN_epochs}] {output_information}')
        last_checkpoint = {
            'epoch': epoch,
            'model_state_dict': Recmodel.state_dict(),
            'optimizer_state_dict': opt.state_dict(),
            'triples_file': triples_file,
            'config': config,

            'best_score': best_score
        }
        torch.save(last_checkpoint, os.path.join(model_saving_path, 'last_epoch_checkpoint.pth'))



    print("Ori [TEST]")
    best_ckpt_path = os.path.join(model_saving_path, 'best_model_checkpoint.pth')
    checkpoint = torch.load(best_ckpt_path, map_location=device)
    # create a fresh model instance and load the saved state dict
    dataset = Loader(checkpoint.get('config', config), device, path=f"/home/ubuntu/unlearning_rec_exps/recsys_implicit/datasets/amazon_data/")
    eval_model = LightGCN(checkpoint.get('config', config), dataset)
    eval_model.load_state_dict(checkpoint['model_state_dict'])
    eval_model = eval_model.to(device).eval()
    dir_path = f"/home/ubuntu/sess_rec_exps/unlearning_rec_exps/datasets/amazon_data/"
    train_df=pickle.load(open(f"{dir_path}/02_modelled_noisy_data_full.p", "rb"))
    test_df=pickle.load(open(f"{dir_path}/test_df.p", "rb"))

    train_df['feedback'] = 1
    test_df['feedback'] = 1

    # test_df = test_df[test_df['feedback'] == 1]

    # train_df = train_df[train_df['rating'] >= 4]
    # test_df = test_df[test_df['rating'] >= 4]
    user_item_dict = train_df.groupby('user_id')['item_id'].apply(list).to_dict()

    get_ranks(eval_model, test_df, user_item_dict)









