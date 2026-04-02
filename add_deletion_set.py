import numpy as np
import pandas as pd
import pickle

dir_path = "./amazon_data"
train_df = pickle.load(open(f"{dir_path}/train_df.p", "rb"))

train_df['feedback'] = 1

with open("amazon_data_user_reverse_preference.pkl", "rb") as f:
    sorted_item_ids_per_user = pickle.load(f)


# Noise configuration
sample = 42
np.random.seed(sample)
train_df['noise_added'] = 0
noise_percentage = 0.2
interaction_percentage = 0.8

# Select users to modify
unique_users = train_df['user_id'].unique()
num_users_to_modify = int(noise_percentage * len(unique_users))
selected_users = np.random.choice(unique_users, num_users_to_modify, replace=False)


noise_rows = []

for user in selected_users:
    user_interactions = train_df[train_df['user_id'] == user]
    n = len(user_interactions)
    num_noise = int(interaction_percentage * n)
    
    if num_noise == 0:
        continue

    # Items already interacted with by this user
    user_items = set(user_interactions['item_id'])

    # Items sorted by least score for this user (ascending)
    sorted_items = sorted_item_ids_per_user[user]
    
    # Exclude already interacted items
    filtered_items = [item for item in sorted_items if item not in user_items]
    selected_items = list(filtered_items[:num_noise])
    
    # Build new noisy interactions
    new_rows = pd.DataFrame({
        'user_id': [user] * len(selected_items),
        'item_id': selected_items,
        'feedback': [1] * len(selected_items),   # set feedback label as you want
        'noise_added': [1] * len(selected_items),
    })
    noise_rows.append(new_rows)

# Combine original and noisy data
if noise_rows:
    noisy_data = pd.concat([train_df] + noise_rows, ignore_index=True)
else:
    noisy_data = train_df.copy()


noisy_data.to_pickle('./amazon_data/02_modelled_noisy_data_full.p')