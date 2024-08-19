# -*- coding: utf-8 -*-

# unsupervised learning - K-means clustering
# dataset: Mall

# streamlit run D:/gitbuild/clustering/mall.py

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score
# import warnings
# warnings.filterwarnings("ignore")

# ============
# page design
# ============
st.set_page_config(layout='wide')

if "malldata" not in st.session_state:
    st.session_state["malldata"] = None


def createclusters():
    
    data = st.session_state["malldata"]
    
    # annualincome vs score
    X=np.array(data[['annualincome','score']].values)
    kmeans=KMeans(n_clusters=5, init='k-means++',max_iter=300, n_init=10, random_state=0)
    clusters = kmeans.fit_predict(X)

    data['cluster'] = clusters

    clrs = ['red','green','blue','magenta','cyan']
    fig = plt.figure(figsize=(12,6))
    ax = fig.add_subplot(1,1,1)

    # annual income & score
    for i in range(5):
        cl = "Cluster " + str(i+1)
        plt.scatter(data.annualincome[data.cluster==i], data.score[data.cluster==i],s=50,c=clrs[i],label=cl)
        
    plt.title("Mall Customers - Clustering")
    plt.xlabel("Annual Income")
    plt.ylabel("Spending Score")
    
    st.write(fig)

# ================================================================================================

def homepage():
    st.header("Demo : Clustering")
    st.divider()
    st.subheader("Customer Segmentation using Mall Customer Data")

def dataset():
    st.header("Dataset")
    file = "mall.csv"
    data = pd.read_csv(file)
    
    data=data.rename(columns={'CustomerID':'custid',
                              'Annual Income (k$)':'annualincome',
                              'Spending Score (1-100)':'score'})
    st.dataframe(data)

    tot = len(data)
    cols = len(data.columns) - 1
    
    st.success("Total Records = " + str(tot))
    st.success("Total Features = " + str(cols))
    
    st.session_state["malldata"] = data

# ==============================================
# calling each function based on the click value
# ==============================================
# main menu settings
options=[":house:",":memo:",":lower_left_fountain_pen:"]
captions=['Home','Dataset',"Customer Clustering"]
nav = st.sidebar.radio("Select Option",options,captions=captions)
ndx = options.index(nav)

if (ndx==0):
    homepage()

if (ndx==1):
    dataset()
    
if (ndx==2):
    createclusters()
