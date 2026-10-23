# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import graphtools as gt
import seaborn as sns
import phate
import magic
import scprep
import meld
import sklearn
import scipy
import scanpy as sc
import os


import matplotlib.pyplot as plt
plt.rc('font', size=14)
np.random.seed(42)
np.random.seed(0)



data_path_1= "/gpfs/gibbs/project/ishizuka/wt284/MELD/merged_all/Input/ns_vs_IFNg_raw_counts_for_MELD_harmony_integrated_tumors.csv"
data_path_2 = "/gpfs/gibbs/project/ishizuka/wt284/MELD/merged_all/Input/ns_vs_IFNg_metadata_for_MELD_harmony_integrated_tumors.csv"


data1 = pd.read_csv(data_path_1,header=None, skiprows=1,index_col=0).T
data1.iloc[:, : 10].head()


#I actually want my columns here to be the metadata title and not the cells
metadata = pd.DataFrame.transpose(pd.read_csv(data_path_2,header=0).T)
#pd.set_option('display.max_columns', None)
metadata.iloc[:,:32].head()

metadata['is_stimulated'] = metadata['condition'] == 'IFNg'

#Creating index
#The original data has only numbered indices, but this is easy to mix up using Pandas.
#To make sanity checks easier, we'll use a concatenation of the cell barcode and the condition

data1.index = ['.'.join([c,co]) for i, (c, co) in metadata[['Unnamed: 0', 'condition']].iterrows()]
metadata.index = data1.index



#This does the multiplication by 10,000 which is what is also done in Seurat pipeline
data1,libsize = scprep.normalize.library_size_normalize(data1,return_library_size=True)

metadata['library_size'] = libsize

#Instead of doing sqrt, we want to do log
#data1 = scprep.transform.sqrt(data1)
data1 = scprep.transform.log(data1)

#Running PCA on this
data1_pca = scprep.reduce.pca(data1)

#Embedding data using PHATE
phate_op = phate.PHATE(knn=10, decay=10, n_jobs=-1)
data1_phate = phate_op.fit_transform(data1_pca)

meld_op = meld.MELD()
sample_densities = meld_op.fit_transform(data1, metadata['condition'])
sample_densities.index = data1.index

metadata['stimulated_likelihood'] = meld.utils.normalize_densities(sample_densities)['IFNg']

metadata.iloc[:,:30].head(10)
metadata.to_csv("/gpfs/gibbs/project/ishizuka/wt284/Merged_All/MELD_output/new_metadata_IFNg_harmony_integrated_tumors.csv")


import sklearn.mixture

#Fit the mixture model

mixture_model = sklearn.mixture.GaussianMixture(n_components=3)
classes = mixture_model.fit_predict(metadata['stimulated_likelihood'].values.reshape(-1,1))

#Reorder the class labels so that 0 has the lowest average likelihood and 2 has the highest.
#Note this doesn't change the assignments, just the labels.

classes = scprep.utils.sort_clusters_by_values(classes, metadata['stimulated_likelihood'])

#Plot the distribution of likelihood values for each class

fig, ax = plt.subplots(1, figsize=(5,4))
for c in np.unique(classes):
  ax.hist(metadata['stimulated_likelihood'][classes == c], range=(0.01,.9), bins=100,
          color=meld.utils.get_meld_cmap()([0,255//2, 255])[c])
fig.tight_layout()
plt.show()

classes
df = pd.DataFrame(classes)
df


df.to_csv("/gpfs/gibbs/project/ishizuka/wt284/Merged_All/MELD_output/ns_vs_IFNg_classes.csv")


#Commenting out the VFC part to just run the first part that returns the stimulated likelihood scores


# # Get cluster indicides and number of cells per cluster
clusters, counts = np.unique(metadata['broad_cell_type'], return_counts=True)
# # Keep cluster labels with at least 1% of the data
clusters = clusters[counts > data1.shape[0] * 0.01]

print(clusters)



#First, we will start by visualizing each cluster using PHATE
#Here we want to visualize the heterogeneity within each cluster using PHATE run on each dataset. If the PHATE embedding shows multiple groups of cells with different likelihood values, we can reason that the cluster might need further subclustering using VFC.

data1_cluster_phate = {}
for cluster in clusters:
    curr_data = data1_pca.loc[metadata['broad_cell_type'] == cluster]
    data1_cluster_phate[cluster] = phate.PHATE(verbose=0).fit_transform(curr_data)

fig,axes= plt.subplots(5,4, figsize=(4*3, 5*3))

for i , ax in enumerate(axes.flatten()):
    if not i < len(clusters):
        ax.axis('off')
        continue
    curr_cluster = clusters[i]
    curr_phate = data1_cluster_phate[curr_cluster]
    scprep.plot.scatter2d(curr_phate,
                      c=metadata['stimulated_likelihood'].loc[metadata['broad_cell_type'] == curr_cluster],
                      cmap=meld.get_meld_cmap(), vmin=0, vmax=1,
                      ax=ax, ticks=False,
                      title='Cluster {} ({})'.format(curr_cluster, curr_phate.shape[0]),
                      legend=False, fontsize=10)
plt.savefig("/gpfs/gibbs/project/ishizuka/wt284/Merged_All/MELD_output/ns_vs_IFNg_plot1.pdf")

#In this next part, we will be building a VFC operator for each cluster
#We want to examine the differences between the results of VFC clustering at different numbers of clusters. The goal here is to make sure that the results of the clustering align with the grouping of cells we observed in the PHATE plots and the distribution of likelihood values.

np.random.seed(0)
vfc_op_per_cluster = {}

for cluster in clusters:
    curr_G = gt.Graph(data1_pca.loc[metadata['broad_cell_type'] == cluster], use_pygsp=True)
    curr_G.compute_fourier_basis()
    curr_sample_labels = metadata['condition_2'].loc[metadata['broad_cell_type'] == cluster]
    curr_likelihood = metadata['stimulated_likelihood'].loc[metadata['broad_cell_type'] == cluster]
    curr_vfc = meld.VertexFrequencyCluster(n_clusters = 3, random_state=None)
    curr_vfc.fit_transform(curr_G, curr_sample_labels, curr_likelihood)
    vfc_op_per_cluster[cluster] = curr_vfc

#clusters = np.delete(clusters,4)
#print(clusters)

subclustering_results = {}
for cluster in clusters:
    print('starting subclustering for',cluster)
    curr_vfc = vfc_op_per_cluster[cluster]
    clusters_by_n = {}
    for n in [2,3,4,5]:
        clusters_by_n[n] = curr_vfc.predict(n)
    subclustering_results[cluster] = clusters_by_n
    print('completed subclustering for cluster',cluster)

#Now we're going to plot the results of all of the VFC clustering results. This produces a very long figure, but it's worth looking at each of the plots individually. We don't claim that the cluster values we selected are "correct" because there's no clear objective we're targeting here. To dive more deeply, one could perform differential expression testing between the 5 clusters to see if the heterogeneity in the visualization is biologically relevant.

fig, axes= plt.subplots(10,5, figsize=(15, 10*3))

for i, r_ax in enumerate(axes):
    cluster = clusters[i]
    curr_phate = data1_cluster_phate[cluster]
    for i, n in enumerate([0,2,3,4,5]):
        if i == 0:
            cvec = metadata['stimulated_likelihood'].loc[metadata['broad_cell_type'] == cluster]
            ylabel= str(cluster) + ' ({})'.format(len(cvec))
            cmap= meld.get_meld_cmap()
            vmin=0
            vmax=1
        else:
            cvec = subclustering_results[cluster][n]
            cmap = ylabel = vmin = vmax = None
        if np.sum(metadata['broad_cell_type'] == cluster) < (metadata.shape[0] * 0.01):
            plt.setp(r_ax[i].spines.values(), color='lightgrey', linewidth=4)
        scprep.plot.scatter2d(curr_phate, c=cvec, cmap=cmap, vmin=vmin, vmax=vmax,
                      ax=r_ax[i], ticks=False, ylabel=ylabel, fontsize=8, legend=False)
    plt.savefig('/gpfs/gibbs/project/ishizuka/wt284/Merged_All/MELD_output/IFNg_PHATE_by_VFC.pdf')
    
##############



#I've already looked through the clusters and determined which number of clusters look best to me. We store the results in the following dictionary.
#picked_clusters = {0:3,1:2,3:3,4:2,5:2,6:2,7:3,9:2,10:2,12:2,14:2}

picked_clusters = {"B cells":3, "CD4+ T cells (Tfh?)":1, "CD8+ T cell-effector memory":3, "Merkel Cell Carcinoma tumor cells":1, "NK cells":2, "Tregs":4, "melanoma tumor cells":1, "memory CD4+ T cell- high ISG":2, "memory CD4+ T cell- low ISG": 2, "monocytes (inflammatory M1)":1}


#Now let's actually break the clusters using the VFC results and create a new jitter plot.
#metadata['VFC'] = metadata['clusterID'].copy()
#for curr_cluster in clusters:
    #if curr_cluster not in picked_clusters:
        #continue
    #new_clusters = subclustering_results[curr_cluster][picked_clusters[curr_cluster]]
    #metadata.loc[metadata['clusterID'] == curr_cluster, 'VFC'] = new_clusters + 1 + np.max(metadata['VFC'])
#metadata['VFC'] = scprep.utils.sort_clusters_by_values(metadata['VFC'], metadata['stimulated_likelihood'])
#metadata.to_csv("/gpfs/ycga/project/ishizuka/wt284/MELD/Output/new_metadata_IFNb_merged_tumors_with_VFC.csv")




#Now let's actually break the clusters using the VFC results and create a new jitter plot.
metadata['VFC'] = metadata['broad_cell_type'].copy()
#metadata['VFC'] = 0
metadata = metadata.assign(VFC= 0)

metadata.loc[metadata['broad_cell_type'] == "memory CD4+ T cell- low ISG", "VFC"] = 1
metadata.loc[metadata['broad_cell_type'] == "memory CD4+ T cell- high ISG", "VFC"] = 2
metadata.loc[metadata['broad_cell_type'] == "CD8+ T cell-effector memory", "VFC"] = 3
metadata.loc[metadata['broad_cell_type'] == "NK cells", "VFC"] = 4
metadata.loc[metadata['broad_cell_type'] == "CD4+ T cells (Tfh?)", "VFC"] = 5
metadata.loc[metadata['broad_cell_type'] == "Merkel Cell Carcinoma tumor cells", "VFC"] = 6
metadata.loc[metadata['broad_cell_type'] == "B cells", "VFC"] = 7
metadata.loc[metadata['broad_cell_type'] == "Tregs", "VFC"] = 8

metadata.loc[metadata['broad_cell_type'] == "monocytes (inflammatory M1)", "VFC"] = 9
metadata.loc[metadata['broad_cell_type'] == "melanoma tumor cells", "VFC"] = 10
metadata.loc[metadata['broad_cell_type'] == "proliferating MCC tumor cells", "VFC"] = 11
metadata.loc[metadata['broad_cell_type'] == "MCC tumor cells", "VFC"] = 12
metadata.loc[metadata['broad_cell_type'] == "DCs", "VFC"] = 13
metadata.loc[metadata['broad_cell_type'] == "macrophages (M2)", "VFC"] = 14



metadata['New_Cluster_ID'] = metadata['VFC'].copy()

metadata['VFC_2'] = metadata['broad_cell_type'].copy()


#metadata['VFC'] = metadata['broad_cell_type'].copy()
for curr_cluster in clusters:
    if curr_cluster not in picked_clusters:
        continue
    #new_clusters = subclustering_results[curr_cluster][picked_clusters[curr_cluster]]
    new_clusters = (subclustering_results[curr_cluster][picked_clusters[curr_cluster]]).copy()
    metadata.loc[metadata['broad_cell_type'] == curr_cluster, 'VFC'] = new_clusters + 1 + np.max(metadata['VFC'])
    num = str(new_clusters)
    metadata.loc[metadata['broad_cell_type'] == curr_cluster, 'VFC_2'] = curr_cluster + "." + num
    
metadata['VFC'] = scprep.utils.sort_clusters_by_values(metadata['VFC'], metadata['stimulated_likelihood'])

metadata.to_csv("/gpfs/gibbs/project/ishizuka/wt284/Merged_All/MELD_output/new_metadata/new_metadata_IFNg_merged_tumors_with_VFC.csv")

ax = scprep.plot.jitter(metadata['VFC'], metadata['stimulated_likelihood'], metadata['broad_cell_type'],
cmap=meld.get_meld_cmap(), figsize=(18,7), legend=False)
ax.set_ylim(0,1)
ax.axhline(0.5, linestyle = '--', color='grey', zorder=0)

ax = scprep.plot.jitter(metadata['VFC_2'], metadata['stimulated_likelihood'], metadata['broad_cell_type'],
cmap=meld.get_meld_cmap(), figsize=(18,7), legend=False)
ax.set_ylim(0,1)
ax.axhline(0.5, linestyle = '--', color='grey', zorder=0)

plt.savefig('/gpfs/gibbs/project/ishizuka/wt284/Merged_All/MELD_output/jitterplot/IFNg_VFC_jitterplot.pdf')



########
