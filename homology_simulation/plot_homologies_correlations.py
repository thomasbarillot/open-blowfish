# %%
%load_ext autoreload
%autoreload 2
import itertools
import json
import math
import sys
from pathlib import Path

import boto3
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from common_tools.connections import get_boto_session
from scipy.optimize import curve_fit
from scipy.stats import gaussian_kde

from config import AWS_PROFILE_NAME, S3_EXPERIMENT_RETRIEVAL_AMBIGUITY
from . import plotting
from .utils import (
    load_scenario_data_from_s3,
    load_scenario_data_from_s3_projection,
)

# %%

# %%
# Exponential decay model: a*exp(-b*x) + c

s1 = []
conds = [[64,16,8], [128,32,16], [256,64,32]]
for cond in conds:
    s1.append(load_scenario_data_from_s3(scenario_num=1, n_parent_features=cond[2], embedding_dim=cond[0], max_features=cond[1]))
s1 = pd.concat(s1)

s2 = []
for cond in conds:
    s2.append(load_scenario_data_from_s3(scenario_num=2, n_parent_features=cond[2], embedding_dim=cond[0], max_features=cond[1]))
s2 = pd.concat(s2)

# %%
s3 = []
for cond in conds:
    s3.append(load_scenario_data_from_s3(scenario_num=3, n_parent_features=cond[2], embedding_dim=cond[0], max_features=cond[1]))
s3 = pd.concat(s3)
s3["scenario"] = s3["scenario"].apply(lambda x: "parent_to_children_mixed")

s4 = []
for cond in conds:
    s4.append(load_scenario_data_from_s3(scenario_num=4, n_parent_features=cond[2], embedding_dim=cond[0], max_features=cond[1]))
s4 = pd.concat(s4)
s4["scenario"] = s4["scenario"].apply(lambda x: "parent_to_children_mixed")

# r3_pivoted = res3_filtered[["depsilon", "w1_h0"]].groupby(["depsilon"]).mean().reset_index().pivot(index="depsilon", columns="direct_children_count")
# r3_pivoted_std = res3_filtered[["depsilon", "w1_h0"]].groupby(["depsilon"]).std().reset_index().pivot(index="depsilon", columns="direct_children_count")
# %%

fig_256_embed = plotting.plot_homology_histograms_2x2(results_s1=s1[s1.embedding_dim==256], results_s2=s2[s2.embedding_dim==256], depsilon_filter=0.4,noise_filter=0.1)
plt.savefig("homologies_simulation_256d.png", dpi=150, bbox_inches="tight")
plt.close(fig_256_embed)

fig_all_embed = plotting.plot_homology_histograms_2x2(results_s1=s1, results_s2=s2, depsilon_filter=0.4,noise_filter=0.1)
plt.savefig("homologies_simulation_all_d.png", dpi=150, bbox_inches="tight")
plt.close(fig_256_embed)
# %%
# fig_256_embed = plotting.plot_homology_histograms_2x2(results_s1=s4[s4.embedding_dim==256], results_s2=s1[s1.embedding_dim==256], depsilon_filter=0.4,noise_filter=0.1)
# plt.savefig("homologies_simulation_256d.png", dpi=150, bbox_inches="tight")
# plt.close(fig_256_embed)

# fig_all_embed = plotting.plot_homology_histograms_2x2(results_s1=s3, results_s2=s2, depsilon_filter=0.4,noise_filter=0.1)
# plt.savefig("homologies_simulation_all_d.png", dpi=150, bbox_inches="tight")
# plt.close(fig_256_embed)
# # %%

# %%
#plotting.plot_homology_histogram_with_scale(results_s1=s4[s4.embedding_dim==256], results_s2=s1[s1.embedding_dim==256], depsilon_filter=0.4,noise_filter=0.15, metric="w1_h0")
#plotting.plot_homology_histogram_with_scale(results_s1=s4[s4.embedding_dim==256], results_s2=s1[s1.embedding_dim==256], depsilon_filter=0.4,noise_filter=0.2, metric="ltmax_h1")
plotting.plot_homology_scales(results_s1=s1[s1.embedding_dim==256], results_s2=s4[s4.embedding_dim==256], depsilon_filter=0.4,noise_filter=0.2,palette={ "parent_to_children": "red","parent_to_children_mixed": "#B4A7D6" })
plt.savefig("homologies_simulation_scale_256d_mixed.png", dpi=150, bbox_inches="tight")
plt.close(fig_256_embed)
#plt.close(fig_256_embed)
# %%


s1p = []
conds = [[64,16,8], [128,32,16], [256,64,32]]
for cond in conds:
    s1p.append(load_scenario_data_from_s3_projection(scenario_num=1))
s1p = pd.concat(s1p)

s2p = []
for cond in conds:
    s2p.append(load_scenario_data_from_s3_projection(scenario_num=2))
s2p = pd.concat(s2p)

# %%
fig_projection = plotting.plot_projection_comparison_2x2(s1p=s1p, s2p=s2p, depsilon_filter=0.4, noise_filter=0.01)
plt.savefig("homology_projection_comparison_kde.png", dpi=150, bbox_inches="tight")
plt.close(fig_projection)
# %%
