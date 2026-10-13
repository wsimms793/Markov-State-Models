%pip install MDAnalysisData
%pip install MDAnalysis
%pip install torch.nn
%pip install tqdm.notebook

import numpy as np
import matplotlib.pyplot as plt
from MDAnalysisData import datasets
import deeptime
import mdtraj as md
import MDAnalysis as mda
import torch
import torch.nn as nn
from tqdm.notebook import tqdm
from deeptime.util.data import TrajectoryDataset, TrajectoriesDataset
from MDAnalysis.analysis import rms
from deeptime.decomposition.deep import VAMPNet
from torch.utils.data import DataLoader
from deeptime.decomposition import VAMP
from deeptime.util.validation import implied_timescales, ck_test
from deeptime.plots import plot_implied_timescales, plot_ck_test
from deeptime.clustering import KMeans
from deeptime.markov import TransitionCountEstimator
from deeptime.markov.msm import BayesianMSM
from deeptime.plots import plot_markov_model
from MDAnalysis.analysis import align
from deeptime.markov.msm import MaximumLikelihoodMSM
from MDAnalysis.lib.distances import self_distance_array
from deeptime.clustering import KMeans
from MDAnalysis.analysis import align
import networkx as nx
from sklearn.decomposition import PCA

if torch.cuda.is_available():
    device = torch.device("cuda")
    torch.backends.cudnn.benchmark = True
else:
    device = torch.device("cpu")
torch.set_num_threads(12)

print(f"Using device {device}")

files = datasets.fetch_adk_equilibrium()

pdb = files.topology
traj = files.trajectory
u = mda.Universe(pdb, traj)

ref = mda.Universe(pdb, traj)
ref.trajectory[0]

align.AlignTraj(
    u,
    ref,
    select="protein and name CA C N O CB",
    in_memory=True
).run()

backbone = u.select_atoms("protein and name CA C N O CB")

X = []
for t in u.trajectory:
    X.append(backbone.positions.copy())

X = np.array(X)

X = X.reshape(X.shape[0], -1)

kmeans = KMeans(n_clusters=100)

model = kmeans.fit(X).fetch_model()

dtraj = model.transform(X)


msm = MaximumLikelihoodMSM(
    lagtime=1,
    reversible=True
).fit(dtraj).fetch_model()

lags = [1, 2, 5, 10, 20]

models = []

for lag in lags:

    counts = TransitionCountEstimator(
        lagtime=lag,
        count_mode="sliding"
    ).fit(dtraj).fetch_model()

    msm = MaximumLikelihoodMSM(
        reversible=True
    ).fit(counts).fetch_model()

    models.append(msm)

its = implied_timescales(models)

plot_implied_timescales(its, n_its=5)

plt.xlabel(r"Lag time, $\tau$")
plt.ylabel(r"Implied timescale, $t_i$")
plt.yscale("log")
msm = MaximumLikelihoodMSM(
    lagtime=9,
    reversible=True
).fit(dtraj).fetch_model()


fig, ax = plt.subplots(1, 1, figsize=(10, 10))

threshold = 0.025

title = f"Transition matrix with connectivity threshold {threshold:.3f}"

G = nx.DiGraph()
ax.set_title(title)
for i in range(msm.n_states):
    G.add_node(i, title=f"{i+1}")
for i in range(msm.n_states):
    for j in range(msm.n_states):
        if msm.transition_matrix[i, j] > threshold:
            G.add_edge(i, j, title=f"{msm.transition_matrix[i, j]:.3e}")

edge_labels = nx.get_edge_attributes(G, 'title')
pos = nx.fruchterman_reingold_layout(G)
nx.draw_networkx_nodes(G, pos, ax=ax)
nx.draw_networkx_labels(G, pos, ax=ax, labels=nx.get_node_attributes(G, 'title'));
nx.draw_networkx_edges(G, pos, ax=ax, arrowstyle='-|>',
                       connectionstyle='arc3, rad=0.3');

pcca = msm.pcca(n_metastable_sets=3)

fig, ax = plt.subplots(1, 1, figsize=(10, 10))

threshold = 0.025


G = nx.DiGraph()
for i in range(3):
    G.add_node(i, title=f"{i+1}")
for i in range(3):
    for j in range(3):
        if pcca.coarse_grained_transition_matrix[i, j] > threshold:
            G.add_edge(i, j, title=f"{pcca.coarse_grained_transition_matrix[i, j]:.3e}")

edge_labels = nx.get_edge_attributes(G, 'title')
pos = nx.fruchterman_reingold_layout(G)
nx.draw_networkx_nodes(G, pos, ax=ax)
nx.draw_networkx_labels(G, pos, ax=ax, labels=nx.get_node_attributes(G, 'title'));
nx.draw_networkx_edges(G, pos, ax=ax, arrowstyle='-|>',
                       connectionstyle='arc3, rad=0.3');

fig, ax = plt.subplots(1, 1, figsize=(10, 10))

threshold = 0.025

pi = pcca.coarse_grained_stationary_probability

node_sizes = pi * 5000

G = nx.DiGraph()

for i in range(3):
    G.add_node(i, title=f"{i+1}")

for i in range(3):
    for j in range(3):
        if pcca.coarse_grained_transition_matrix[i, j] > threshold:
            G.add_edge(
                i, j,
                title=f"{pcca.coarse_grained_transition_matrix[i, j]:.3e}"
            )

edge_labels = nx.get_edge_attributes(G, 'title')

pos = nx.fruchterman_reingold_layout(G)

nx.draw_networkx_nodes(
    G, pos, ax=ax,
    node_size=node_sizes
)

nx.draw_networkx_labels(
    G, pos, ax=ax,
    labels=nx.get_node_attributes(G, 'title')
)

nx.draw_networkx_edges(
    G, pos, ax=ax,
    arrowstyle='-|>',
    connectionstyle='arc3, rad=0.3'
)

plt.show()

pcca.coarse_grained_transition_matrix

pcca.coarse_grained_stationary_probability

ix = np.argsort(pcca.assignments)
plt.figure(figsize=(30, 5))
plt.plot(np.arange(msm.n_states)+1, pcca.metastable_distributions[0][ix], 'o', label="Metastable set 1")
plt.plot(np.arange(msm.n_states)+1, pcca.metastable_distributions[1][ix], 'x', label="Metastable set 2")
plt.plot(np.arange(msm.n_states)+1, pcca.metastable_distributions[2][ix], 'x', label="Metastable set 2")
plt.plot(np.arange(msm.n_states)+1, pcca.metastable_distributions[3][ix], 'x', label="Metastable set 2")

plt.xticks(np.arange(msm.n_states)+1, [f"{i}" for i in ix])
plt.title("Metastable distributions")
plt.legend()
plt.xlabel('micro state')
plt.ylabel('probability');


