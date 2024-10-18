#external imports
import numpy as np

#internal imports
from data_reader import create_datasets

fpath = 'Instructions/Dataset_C_MD_outcome2.xlsx'

training, testing = create_datasets(fpath)