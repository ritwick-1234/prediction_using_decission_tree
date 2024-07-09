import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt


data=pd.read_csv('diabetes.csv')

st.sidebar.title("Hyperparameters")
criterion = st.sidebar.selectbox("Criterion", options=["gini", "entropy", "log_loss"], index=0)
splitter = st.sidebar.selectbox("Splitter", options=["best", "random"], index=0)
max_depth = st.sidebar.slider("Max Depth", min_value=1, max_value=20, value=20)
min_samples_split = st.sidebar.slider("Min Samples Split", min_value=2, max_value=500, value=2)
min_samples_leaf = st.sidebar.slider("Min Samples Leaf", min_value=1, max_value=500, value=1)
max_features = st.sidebar.slider("Max Features", min_value=1, max_value=len(data.columns)-1, value=len(data.columns)-1)
max_leaf_nodes = st.sidebar.slider("Max Leaf Nodes", min_value=2, max_value=200, value=20)
min_impurity_decrease = st.sidebar.slider("Min Impurity Decrease", min_value=0.0, max_value=0.5, value=0.0, step=0.01)

if st.sidebar.button("Submit"):
    # Prepare data
    X = data.drop(columns=["Outcome"])
    y = data["Outcome"]

   
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

  
    model = DecisionTreeClassifier(
        criterion=criterion,
        splitter=splitter,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        max_features=max_features,
        max_leaf_nodes=max_leaf_nodes,
        min_impurity_decrease=min_impurity_decrease
    )
    model.fit(X_train, y_train)

   
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)


    st.title("Decision Tree Classifier")
    st.write(f"Accuracy: {accuracy*100:.2f}")

 
    fig, ax = plt.subplots(figsize=(10, 6))
    plot_tree(model, filled=True, feature_names=X.columns, class_names=["No Diabetes", "Diabetes"], ax=ax)
    st.pyplot(fig)
