import fastai
import numpy as np
from fastai.tabular.all import *


dls = TabularDataLoaders.from_csv('Dataset/glass_data-4_lev.csv', 'Dataset')


learn = tabular_learner(dls, metrics=accuracy)
learn.fit_one_cycle(2)