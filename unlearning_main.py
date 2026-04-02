import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from typing import Dict, Tuple, List, Set, Optional
from collections import defaultdict
import math
import random
import contextlib
import pickle
from models import MF
import random 
import time
from tqdm import tqdm
from train_MF import get_ranks, set_seed


class InteractionDataset(Dataset):
    """Dataset of (user, item, label). label in {0,1}."""
    def __init__(self, triples: List[Tuple[int,int,int]]):
        self.data = triples

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        u, i, y = self.data[idx]
        return torch.tensor(u), torch.tensor(i), torch.tensor(y, dtype=torch.long)


class DeletedTripletDataset(Dataset):
    def __init__(self, pickle_path: str):
        data = pickle.load(open(pickle_path, "rb"))

        self.users = torch.as_tensor(data["users"], dtype=torch.long)
        self.pos_items = torch.as_tensor(data["pos_items"], dtype=torch.long)
        self.neg_items = torch.as_tensor(data["neg_items"], dtype=torch.long)

        assert len(self.users) == len(self.pos_items) == len(self.neg_items)

    def __len__(self):
        return len(self.users)

    def __getitem__(self, idx):
        return self.users[idx], self.pos_items[idx], self.neg_items[idx]


def truncated_svd(mat: torch.Tensor, rank: int) -> Tuple[torch.Tensor, torch.Tensor]:
    # mat: (m x n). Return A (m x r), B (n x r) s.t. A @ B.T ≈ mat
    U, S, Vh = torch.linalg.svd(mat, full_matrices=False)
    r = min(rank, S.numel())
    U_r = U[:, :r]
    S_r = S[:r]
    V_r = Vh[:r, :].T
    A = U_r * S_r.sqrt()
    B = V_r * S_r.sqrt()
    return A, B

class EmbeddingRowAdapter(nn.Module):
    """
    Low-rank adapter attached to selected rows of an embedding table:
      E[idx] <- E[idx] + A_rows @ B^T
    """
    def __init__(self, emb: nn.Embedding, idx: torch.Tensor, rank_emb):
        super().__init__()
        self.emb = emb  # base table; should be frozen externally
        self.register_buffer("idx", idx.long())        # (m,)

        self.A_rows = nn.Parameter(torch.empty(idx.size(0), rank_emb))                 # (m, r)
        self.B      = nn.Parameter(torch.empty(emb.weight.shape[1], rank_emb))         # (d, r)

        # initialize
        nn.init.xavier_uniform_(self.A_rows)
        nn.init.xavier_uniform_(self.B)

    def apply(self, ids: torch.Tensor) -> torch.Tensor:
        base = self.emb(ids)
        # Map ids to positions in idx; for simplicity use a boolean match
        if self.idx.numel() == 0 or ids.numel() == 0:
            return base
        ids_exp = ids.view(-1, 1)
        idx_exp = self.idx.view(1, -1)
        matches = (ids_exp == idx_exp)  # (B, m)
        if not matches.any():
            return base
        coeffs = matches.float()                 # (B, m)
        A_sel = coeffs @ self.A_rows            # (B, r)
        delta = A_sel @ self.B.T                # (B, d)
        return base + delta

def zero_like_param_dict(model: nn.Module) -> Dict[nn.Parameter, torch.Tensor]: 
    return {p: torch.zeros_like(p) for p in model.parameters() if p.requires_grad}

@torch.no_grad()
def collect_touched_ids(S_loader: DataLoader) -> Tuple[Set[int], Set[int]]:
    users, items = set(), set()
    for u, i, y in S_loader:
        users.update(u.tolist())
        items.update(i.tolist())
    return users, items


def grad_snapshot(model: nn.Module) -> Dict[nn.Parameter, torch.Tensor]: 
    out = {} 
    for p in model.parameters(): 
        if p.requires_grad: out[p] = (torch.zeros_like(p) if p.grad is None else p.grad.detach().clone()) 
    return out



# Intent: Compute the average gradient contribution of the deleted samples S
# Final output: gS = gradient of deleted data w.r.t. each parameter

#-----------
# Using BCE
#-----------

# def compute_deleted_gradient(
#     model: nn.Module,
#     S_loader: DataLoader,
#     device: torch.device
# ) -> Dict[nn.Parameter, torch.Tensor]:

#     # Put model in evaluation mode and move to target device
#     model.eval().to(device)

#     # Initialize zero gradient accumulator for every trainable parameter
#     gS = zero_like_param_dict(model)

#     # Iterate over deleted samples
#     for u, i, y in S_loader:
#         u = u.to(device)
#         i = i.to(device)

#         # Forward pass on deleted user-item pairs
#         logits = model(u, i)

#         # Since these are deleted positive interactions,
#         # we optimize against target=1 here to estimate their contribution
#         loss = F.binary_cross_entropy_with_logits(
#             logits,
#             torch.ones_like(logits)
#         )

#         for p in model.parameters():
#             if p.grad is not None:
#                 p.grad.zero_()

#         loss.backward()

#         # Take snapshot of current gradients
#         g_batch = grad_snapshot(model)

#         # Add batch gradients into total deleted-data gradient
#         for p in gS:
#             gS[p] += g_batch[p]

#     return gS


# Using BPR triplets exact from noisey training
def compute_deleted_gradient(
    model: nn.Module,
    S_loader: DataLoader,
    device: torch.device
) -> Dict[nn.Parameter, torch.Tensor]:

    # Put model in eval mode and move to device
    model.eval().to(device)

    # Initialize zero gradient accumulator
    gS = {
        p: torch.zeros_like(p, device=p.device)
        for p in model.parameters()
        if p.requires_grad
    }

    for u, i_pos, i_neg in S_loader:
        u = u.to(device)
        i_pos = i_pos.to(device)
        i_neg = i_neg.to(device)

        model.zero_grad(set_to_none=True)

        # Forward pass
        pos_scores = model(u, i_pos)
        neg_scores = model(u, i_neg)

        # BPR loss: SUM, not mean
        loss = -F.logsigmoid(pos_scores - neg_scores).sum()

        loss.backward()

        for p in gS:
            if p.grad is not None:
                gS[p] += p.grad.detach().clone()

    return gS


# Intent: Approximate H^{-1} gS using Adam's second-moment statistics
# This gives a cheap approximation of the inverse-Hessian step.

def precondition_Hinv(gS, model, optimizer, eps=1e-8):

    # Store parameter updates
    delta = {}

    for p in model.parameters():

        if not p.requires_grad:
            continue
        g = gS[p]
        v = optimizer.state.get(p, {}).get('exp_avg_sq', None)

        # If unavailable, approximate using squared gradient
        if v is None:
            v = g.pow(2) + 1e-6

        # Approximate inverse-Hessian update:
        delta[p] = -g / (v.sqrt() + eps)

    return delta

def build_LUA_adapters_for_embeddings(base_model, optimizer, S_loader, device, rank_emb=8, trust_rel=1.5e-1):
    base_model.eval().to(device)
    gS      = compute_deleted_gradient(base_model, S_loader, device)   
    delta0  = precondition_Hinv(gS, base_model, optimizer)             

    # pick only the emb weights we’ll edit
    params_to_edit = []
    if hasattr(base_model, 'user_emb'): params_to_edit.append(base_model.user_emb.weight)
    if hasattr(base_model, 'item_emb'): params_to_edit.append(base_model.item_emb.weight)


    touched_users, touched_items = collect_touched_ids(S_loader)
    touched_users = torch.tensor(sorted(list(touched_users)), device=device, dtype=torch.long)
    touched_items = torch.tensor(sorted(list(touched_items)), device=device, dtype=torch.long)

    adapters = {}
    if hasattr(base_model, 'user_emb') and touched_users.numel():
        dE_rows = delta0[base_model.user_emb.weight].index_select(0, touched_users)
        A_rows, B = truncated_svd(dE_rows, rank_emb)
        adapters['user'] = EmbeddingRowAdapter(base_model.user_emb, touched_users, A_rows, B)

    if hasattr(base_model, 'item_emb') and touched_items.numel():
        dE_rows = delta0[base_model.item_emb.weight].index_select(0, touched_items)
        A_rows, B = truncated_svd(dE_rows, rank_emb)
        adapters['item'] = EmbeddingRowAdapter(base_model.item_emb, touched_items, A_rows, B)

    return adapters

class AdapterAugmentedMF(nn.Module):
    def __init__(self, base: MF,
                 user_adapter: Optional[EmbeddingRowAdapter],
                 item_adapter: Optional[EmbeddingRowAdapter]):
        super().__init__()
        self.base = base
        self.user_adapter = user_adapter
        self.item_adapter = item_adapter
        # freeze base
        for p in self.base.parameters():
            p.requires_grad_(False)

    def teacher_logits(self, u, i):
        with torch.no_grad():
            return self.base(u, i)

    def forward(self, u, i):
        # Embeddings + row adapters
        if self.user_adapter is not None:
            u_vec = self.user_adapter.apply(u)
        else:
            u_vec = self.base.user_emb(u)
        if self.item_adapter is not None:
            i_vec = self.item_adapter.apply(i)
        else:
            i_vec = self.base.item_emb(i)
        return self.base.forward_logits_from_embeddings(u_vec, i_vec)


    @torch.no_grad()
    def predict_for_rank(self, users: List[int]):
        self.eval()
        device = next(self.base.parameters()).device
        u = torch.tensor([int(x) for x in users], device=device).long()

        # user vectors with adapter
        if self.user_adapter is not None:
            u_vec = self.user_adapter.apply(u)
        else:
            u_vec = self.base.user_emb(u)

        # all item vectors with adapter applied to touched rows
        all_items = torch.arange(self.base.item_emb.num_embeddings, device=device).long()
        if self.item_adapter is not None:
            i_mat = self.item_adapter.apply(all_items)          # (n_items, d)
        else:
            i_mat = self.base.item_emb.weight                   # (n_items, d)

        scores =  torch.sigmoid(torch.matmul(u_vec, i_mat.t()))             # (B, n_items) logits
        return scores.detach().cpu().numpy()


def unlearn_pointwise_bce(model: nn.Module,
                          S_batch: Tuple[torch.Tensor, torch.Tensor],
                          Neg_batch: Tuple[torch.Tensor, torch.Tensor]):
    """Deleted positives -> 0 ; sampled negatives -> 1."""
    u_s, i_s = S_batch
    u_n, i_n = Neg_batch
    z_pos = model(u_s, i_s)
    z_neg = model(u_n, i_n)
    loss_pos = F.binary_cross_entropy_with_logits(z_pos, torch.zeros_like(z_pos))
    loss_neg = F.binary_cross_entropy_with_logits(z_neg, torch.ones_like(z_neg))
    return loss_pos + loss_neg

def unlearn_bpr(model, S_batch, Neg_batch):
    """
    S_batch: (u_s, i_s) deleted positives
    Neg_batch: (u_n, j_n) sampled negatives
    """
    u_s, i_s = S_batch
    u_n, j_n = Neg_batch
    
    # Deleted interactions (S)
    z_s = model(u_s, i_s)   # f(u, i) for deleted positives
    # Sampled negatives
    z_n = model(u_n, j_n)   # f(u, j) for negatives
    
    # We want: z_n > z_s
    diff = z_n - z_s
    loss = -F.logsigmoid(diff).mean()
    return loss


def unlearn_margin(model, S_batch, Neg_batch, margin=0.5):
    u_s, i_s = S_batch
    u_n, j_n = Neg_batch
    
    s_del = model(u_s, i_s)
    s_neg = model(u_n, j_n)
    
    loss = F.softplus(s_del - s_neg + margin).mean()
    return loss


def unlearn_margin_k(model, S_batch, Neg_batch, margin=0.5):
    u_pos, i_pos = S_batch
    u_neg, j_neg = Neg_batch
    s_pos = model(u_pos, i_pos)                 # (B,)
    s_neg = model(u_neg, j_neg)                 # (B*k,)
    k = s_neg.numel() // max(1, s_pos.numel())
    s_pos = s_pos.repeat_interleave(max(1, k))  # (B*k,)
    return F.softplus(s_pos - s_neg + margin).mean()


def distill_mse(model: nn.Module, teacher_callable, R_batch: Tuple[torch.Tensor, torch.Tensor]):
    u_r, i_r = R_batch
    z_new = model(u_r, i_r)
    with torch.no_grad():
        z_old = teacher_callable(u_r, i_r)
    return F.mse_loss(z_new, z_old)

def distill_prob_mse(model, teacher, R_batch, tau=2.0):
    u, i = R_batch
    z_new = model(u, i) / tau
    with torch.no_grad():
        z_old = teacher(u, i) / tau
    # p_new = torch.sigmoid(z_new)
    # p_old = torch.sigmoid(z_old)
    return F.mse_loss(z_new, z_old)

def adapter_reg(adapters: List[nn.Module]) -> torch.Tensor:
    reg = torch.tensor(0.0, device=next(adapters[0].parameters()).device) if adapters else torch.tensor(0.0)
    for ad in adapters:
        for p in ad.parameters():
            reg = reg + p.norm(p=2)
    return reg


class NegativeSampler:
    def __init__(self, n_items: int, user_pos: Dict[int, Set[int]]):
        self.n_items = n_items
        self.user_pos = user_pos

    def __call__(self, users: torch.Tensor, k: int = 50) -> Tuple[torch.Tensor, torch.Tensor]:
        u_list, i_list = [], []
        for u in users.tolist():
            banned = self.user_pos.get(u, set())
            cnt = 0
            while cnt < k:
                j = random.randrange(self.n_items)
                if j not in banned:
                    u_list.append(u); i_list.append(j); cnt += 1
        device = users.device
        return torch.tensor(u_list, device=device), torch.tensor(i_list, device=device)

def build_neg_lookup(triples_pickle_path):
    data = pickle.load(open(triples_pickle_path, "rb"))

    users = data["users"]
    pos_items = data["pos_items"]
    neg_items = data["neg_items"]

    neg_lookup = defaultdict(list)
    for u, p, n in zip(users, pos_items, neg_items):
        neg_lookup[(int(u), int(p))].append(int(n))

    return neg_lookup


def get_batch_negatives(u_batch, pos_batch, neg_lookup, device=None):
    """
    u_batch:   tensor [B]
    pos_batch: tensor [B]
    neg_lookup: dict[(u, pos)] -> list of negs

    returns:
        neg_batch: tensor [B]
    """
    negs = []

    u_list = u_batch.detach().cpu().tolist()
    p_list = pos_batch.detach().cpu().tolist()

    for u, p in zip(u_list, p_list):
        cand_negs = neg_lookup[(u, p)]
        if len(cand_negs) == 0:
            raise ValueError(f"No negative found for (user={u}, pos={p})")
        negs.append(random.choice(cand_negs))   # pick one

    neg_batch = torch.tensor(negs, dtype=torch.long)

    if device is not None:
        neg_batch = neg_batch.to(device)

    return neg_batch

def train_LAC(adapter_model: nn.Module,
              adapters_to_train: List[nn.Module],
              S_loader: DataLoader,
              R_local_loader: DataLoader,
              neg_sampler: NegativeSampler,
              n_steps: int = 200,
              lambda_loc: float = 1.0,
              lambda_reg: float = 1e-4,
              lr: float = 1e-3,
              device: Optional[torch.device] = None,
              verbose: bool = True):
    """
    Train only adapters (A,B) with joint loss on witness set.
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    adapter_model.to(device).train()
    for p in adapter_model.parameters():
        p.requires_grad_(False)
    # Unfreeze adapters
    params = []
    for ad in adapters_to_train:
        for p in ad.parameters():
            p.requires_grad_(True)
            params.append(p)
    opt = torch.optim.Adam(params, lr=lr)

    neg_lookup = build_neg_lookup("deleted_triples.pkl")

    S_iter = iter(S_loader)
    Rloc_iter = iter(R_local_loader)

    step = 0
    while step < n_steps:
        try:
            u_s, i_s, _ = next(S_iter)
        except StopIteration:
            S_iter = iter(S_loader)
            u_s, i_s, _ = next(S_iter)

        try:
            u_rl, i_rl, _ = next(Rloc_iter)
        except StopIteration:
            Rloc_iter = iter(R_local_loader)
            u_rl, i_rl, _ = next(Rloc_iter)

        u_s, i_s = u_s.to(device), i_s.to(device)
        u_rl, i_rl = u_rl.to(device), i_rl.to(device)

        # Sample negatives (one per positive; call sampler k times externally for more)
        # u_n, i_n = neg_sampler(u_s, k=200)
        i_n = get_batch_negatives(u_s, i_s, neg_lookup, device=u.device)

        L_un = unlearn_bpr(adapter_model, (u_s, i_s), (u_s, i_n), margin=0.5)
        # L_un = unlearn_margin_k(adapter_model, (u_s, i_s), (u_n, i_n), margin=0.5)
        # L_loc = distill_mse(adapter_model, adapter_model.teacher_logits, (u_rl, i_rl))
        L_loc  = distill_prob_mse(adapter_model, adapter_model.teacher_logits, (u_rl, i_rl), tau=2.0)
        L_reg = adapter_reg(adapters_to_train)

        L = L_un + lambda_loc * L_loc + lambda_reg * L_reg

        opt.zero_grad()
        L.backward()
        opt.step()

        if verbose and (step % 20 == 0 or step == n_steps - 1):
            print(f"[LAC] step {step:04d}  L={L.item():.4f}  (un={L_un.item():.4f}, loc={L_loc.item():.4f} reg={L_reg.item():.4f})")

        step += 1

def avg_scores(model, pairs):
    u = torch.tensor([u for u,i,_ in pairs]).long().cuda()
    i = torch.tensor([i for u,i,_ in pairs]).long().cuda()
    return model(u,i).mean().item()

@torch.no_grad()
def demotion_rate_streaming(
    model,
    pairs,                      # list of (u,i,*) for the deleted set S
    neg_sampler,
    k: int = 100,               # total negatives per user
    batch_users: int = 2048,    # users per outer batch (lower if you still OOM)
    k_chunk: int = 25,          # negatives per user per inner chunk (k must be multiple-ish)
    device: str = "cuda",       # "cuda" or "cpu"
    use_autocast: bool = False  # set True for fp16/bf16 if your GPU supports it
):
    """
    Returns Pr[ s(u,i_del) < s(u,j_neg) ] estimated over S with k negatives/user.
    Memory-safe: processes users and negatives in chunks; no repeat_interleave.
    """
    model.eval()
    total, better = 0, 0

    # pre-pack S as tensors once (on CPU) and slice
    u_all = torch.tensor([u for u, i, *_ in pairs], dtype=torch.long)
    i_all = torch.tensor([i for u, i, *_ in pairs], dtype=torch.long)

    num = u_all.numel()
    autocast_ctx = (
        torch.cuda.amp.autocast() if (use_autocast and device == "cuda") else contextlib.nullcontext()
    )

    for start in range(0, num, batch_users):
        end = min(start + batch_users, num)
        u = u_all[start:end].to(device, non_blocking=True)
        i = i_all[start:end].to(device, non_blocking=True)

        with autocast_ctx:
            s_pos = model(u, i)                       # (B,)
        B = u.size(0)

        # split k negatives per user into chunks of size k_chunk
        remain = k
        while remain > 0:
            kc = min(k_chunk, remain)
            u_neg, j_neg = neg_sampler(u, k=kc)       # (B*kc,)
            with autocast_ctx:
                s_neg = model(u_neg, j_neg)           # (B*kc,)
            # reshape to (B, kc) in the same user-major order produced by the sampler
            s_neg = s_neg.view(B, kc)
            # compare without repeat_interleave
            cmp = (s_pos.view(B, 1) < s_neg)          # (B, kc) boolean
            better += int(cmp.sum().item())
            total  += B * kc
            # free temps promptly
            del u_neg, j_neg, s_neg, cmp
            remain -= kc

        del u, i, s_pos
        if device == "cuda":
            torch.cuda.empty_cache()

    return better / max(1, total)


def demotion_rate(model, pairs, neg_sampler, k=100):
    u = torch.tensor([u for u,i,_ in pairs]).long().cuda()
    i = torch.tensor([i for u,i,_ in pairs]).long().cuda()
    s_pos = model(u,i)
    u_neg, j_neg = neg_sampler(u, k=k)
    s_neg = model(u_neg, j_neg)
    s_pos = s_pos.repeat_interleave(k)
    return (s_pos < s_neg).float().mean().item()



if __name__ == "__main__":

    set_seed(42)

    n_users = 6040
    n_items = 3706

    config = {
    "epochs": 50,
    "emb_dim": 100,
    "n_users": n_users,
    "n_items": n_items,
    "weight_decay": 0.01,
    "learning_rate": 0.01,
    "batch_size": 2048,
    "model_saving_path": None,
    "save_model": False,
    "verbose": False,
    "test_at": False}

    start_time = time.time()
    model_saving_path = '/home/ubuntu/unlearning_rec_exps/dataset_prep/ml-1m_models'
    checkpoint = torch.load(model_saving_path + '/best_model_noisy_bpr_02.pth', weights_only=False)
    base = MF(n_users, n_items, config.emb_dim)
    
    base.load_state_dict(checkpoint['model'])
    base = base.cuda()

    opt_base = torch.optim.Adam(
        base.parameters(),
        lr=config['learning_rate'],
        weight_decay=config['weight_decay'] / config['batch_size']
    )
    opt_base.load_state_dict(checkpoint['optimizer'])

    dir_path = "/home/ubuntu/unlearning_rec_exps/dataset_prep/ml-1m"
    D_Set=pickle.load(open(f"{dir_path}/02_modelled_noisy_data_full.p", "rb")) #Complete Data with add interactions needed to be removed
    test_df=pickle.load(open(f"{dir_path}/test_df.p", "rb"))

    user_item_dict = D_Set.groupby('user_id')['item_id'].apply(list).to_dict()

    print('Trained on D Set:')
    get_ranks(base, test_df, user_item_dict)

    

    # Doing Unlearning

    print("Doing Unlearning...")

    S_set = D_Set[D_Set['noise_added'] == 1]

    clean = D_Set[D_Set["noise_added"] == 0]
    noisy = D_Set[D_Set["noise_added"] == 1]

    # Users with noisy interactions
    noisy_users = noisy["user_id"].unique()

    # --- 1. Sample from noisy users ---
    clean_noisy_users = clean[clean["user_id"].isin(noisy_users)]

    def per_user_sample(g):
        n = len(g)
        if n == 0:
            return g.iloc[[]]  # empty
        k = min(n, max(1, math.ceil(1 * n)))  # at least 1, not more than n
        return g.sample(n=k, random_state=42)

    Retain_Set_local = (
        clean_noisy_users.groupby("user_id", group_keys=False)
                        .apply(per_user_sample)
                        .reset_index(drop=True)
    )


    S = [(int(u), int(i), 1) for u, i in zip(S_set["user_id"], S_set["item_id"])]       
    R_local = [(int(u), int(i), 1) for u, i in zip(Retain_Set_local["user_id"], Retain_Set_local["item_id"])]         
    full_data = [(int(u), int(i), 1) for u, i in zip(D_Set["user_id"], D_Set["item_id"])] 

    # file pre created for set 
    deleted_dataset = DeletedTripletDataset("deleted_triples.pkl")
    S_loader = DataLoader(deleted_dataset, batch_size=2048, shuffle=False)

    user_pos = defaultdict(set)
    for (u,i,y) in full_data:
        if y == 1:
            user_pos[u].add(i)

    neg_sampler = NegativeSampler(n_items, user_pos)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    emb_adapters = build_LUA_adapters_for_embeddings(base, opt_base, S_loader, device, rank_emb=4)

    emb_adapters['item'] = None
    model = AdapterAugmentedMF(base, emb_adapters.get('user'), emb_adapters.get('item'))


    # Stage 2 (LAC): train only adapters with joint loss
    adapters_to_train = [a for a in [emb_adapters.get('user'), emb_adapters.get('item')] if a is not None]

    # Why not doing triplets, as main Idea of stage is giving 
    # the calibration which should improve generilzation not specific to triplets using in training.

    S_loader = DataLoader(InteractionDataset(S), batch_size=512, shuffle=True)
    Rloc_loader = DataLoader(InteractionDataset(R_local), batch_size=512, shuffle=True)

    rate = demotion_rate_streaming(
    model,
    S,                    
    neg_sampler,
    k=100,               
    batch_users=2048,    
    k_chunk=25,          
    device="cuda",       
    use_autocast=False    
    )
    print(f"Pr[neg > del] before LAC:= {rate:.4f}")


    train_LAC(model, adapters_to_train, S_loader, Rloc_loader, neg_sampler,
            n_steps=160, lambda_loc=1.0, lambda_reg=1e-4, lr=1e-3, device = device, verbose=True)

    rate = demotion_rate_streaming(
    model,
    S,                    
    neg_sampler,
    k=100,              
    batch_users=2048,     
    k_chunk=25,           
    device="cuda",      
    use_autocast=False    
    )
    print(f"Pr[neg > del] after LAC:= {rate:.4f}")
    end_time = time.time()
    total_time = end_time - start_time

    print(f"\nTotal execution time: {total_time:.2f} seconds")
    print(f"Total execution time: {total_time/60:.2f} minutes")
        
    print("== Unlearned model ==")
    get_ranks(model, test_df, user_item_dict)

