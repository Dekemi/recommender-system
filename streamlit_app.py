import pandas as pd
import numpy as np
import streamlit as st
from sklearn.metrics.pairwise import cosine_similarity

# -----------------------------
# STEP 1: Create Mock Data
# -----------------------------
data = {
    "user": ["A", "A", "A", "B", "B", "C", "C", "D"],
    "product": ["Laptop", "Phone", "Tablet", "Laptop", "Tablet", "Phone", "Headphones", "Laptop"],
    "rating": [5, 3, 4, 4, 5, 2, 5, 3]
}

df = pd.DataFrame(data)

# -----------------------------
# STEP 2: Create User-Item Matrix
# -----------------------------
user_item = df.pivot_table(index="user", columns="product", values="rating").fillna(0)

# -----------------------------
# STEP 3: Compute Similarity
# -----------------------------
similarity = cosine_similarity(user_item)
similarity_df = pd.DataFrame(similarity, index=user_item.index, columns=user_item.index)

# -----------------------------
# STEP 4: Recommendation Function
# -----------------------------
def precision_at_k(user, k=2):
    user_data = df[df["user"] == user]
    
    if len(user_data) < 2:
        return 0.0
    
    # Hide one product (simulate test)
    test_item = user_data.sample(1)["product"].values[0]
    
    train_data = user_data[user_data["product"] != test_item]
    
    # Rebuild matrix WITHOUT test item
    temp_df = df[~((df["user"] == user) & (df["product"] == test_item))]
    
    user_item_temp = temp_df.pivot_table(index="user", columns="product", values="rating").fillna(0)
    
    similarity_temp = cosine_similarity(user_item_temp)
    similarity_df_temp = pd.DataFrame(similarity_temp, index=user_item_temp.index, columns=user_item_temp.index)
    
    # Recommend
    similar_users = similarity_df_temp[user].sort_values(ascending=False)[1:]
    scores = np.dot(similar_users.values, user_item_temp.loc[similar_users.index])
    recs = pd.Series(scores, index=user_item_temp.columns).sort_values(ascending=False)
    
    top_k = recs.head(k).index
    
    return 1.0 if test_item in top_k else 0.0

def recommend(user, top_n=2):
    similar_users = similarity_df[user].sort_values(ascending=False)[1:]
    
    weighted_scores = np.dot(similar_users.values, user_item.loc[similar_users.index])
    recommendations = pd.Series(weighted_scores, index=user_item.columns)
    
    already_rated = user_item.loc[user]
    recommendations = recommendations[already_rated == 0]
    
    return recommendations.sort_values(ascending=False).head(top_n)

# -----------------------------
# STREAMLIT UI
# -----------------------------
st.title("🛍️ Simple Recommendation System")

selected_user = st.selectbox("Select User", user_item.index)

if st.button("Get Recommendations"):
    recs = recommend(selected_user)
    st.write("### Recommended Products:")
    st.write(recs)

if st.button("Evaluate Precision@K"):
    score = precision_at_k(selected_user)
    st.write(f"Precision@K: {score:.2f}")