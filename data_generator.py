import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# --- Configuration ---
NUM_USERS = 1000
NUM_TRIPS = 10000
FRAUD_RATE = 0.02 # 2% of trips are fraudulent

# --- Generate Data ---
users_data = {
    'user_id': range(NUM_USERS),
    'signup_date': [datetime(2024, 1, 1) + timedelta(days=np.random.randint(0, 365)) for _ in range(NUM_USERS)],
    'device_id': [f'device_{np.random.randint(0, NUM_USERS * 0.8)}' for _ in range(NUM_USERS)] # Some devices are shared
}
users_df = pd.DataFrame(users_data)

trips_data = {
    'trip_id': range(NUM_TRIPS),
    'user_id': np.random.randint(0, NUM_USERS, NUM_TRIPS),
    'trip_timestamp': [datetime(2025, 1, 1) + timedelta(minutes=np.random.randint(0, 30*24*60)) for _ in range(NUM_TRIPS)],
    'fare': np.round(np.random.uniform(5, 100, NUM_TRIPS), 2),
    'payment_method_hash': [f'card_{np.random.randint(0, NUM_USERS * 0.9)}' for _ in range(NUM_TRIPS)] # Some cards are shared
}
trips_df = pd.DataFrame(trips_data)

# --- Inject Fraud Patterns ---
trips_df['is_fraud'] = 0
fraud_indices = trips_df.sample(frac=FRAUD_RATE, random_state=42).index
trips_df.loc[fraud_indices, 'is_fraud'] = 1

# Make fraud patterns more obvious for the model to learn
# Pattern 1: High fare trips from new users
new_user_ids = users_df.sort_values('signup_date', ascending=False).head(int(NUM_USERS * 0.1))['user_id']
high_fare_fraud_indices = trips_df[trips_df['user_id'].isin(new_user_ids) & (trips_df['is_fraud'] == 1)].index
trips_df.loc[high_fare_fraud_indices, 'fare'] = trips_df.loc[high_fare_fraud_indices, 'fare'] * 3

# Pattern 2: Rapid succession of trips from the same device
trips_df = trips_df.sort_values(['user_id', 'trip_timestamp']).reset_index(drop=True)
for user_id in trips_df['user_id'].unique():
    user_trips = trips_df[trips_df['user_id'] == user_id]
    if len(user_trips) > 1 and user_trips['is_fraud'].sum() > 0:
        fraud_trip_index = user_trips[user_trips['is_fraud'] == 1].index[0]
        if fraud_trip_index > 0:
            prev_trip_time = trips_df.loc[fraud_trip_index - 1, 'trip_timestamp']
            trips_df.loc[fraud_trip_index, 'trip_timestamp'] = prev_trip_time + timedelta(minutes=np.random.randint(1, 5))

# Merge dataframes to create a single analysis table
df = pd.merge(trips_df, users_df, on='user_id')

# Save to a file
df.to_csv('rideshare_fraud_data.csv', index=False)

print("Generated rideshare_fraud_data.csv successfully!")
print(f"\nFraud Rate: {df['is_fraud'].mean():.2%}")