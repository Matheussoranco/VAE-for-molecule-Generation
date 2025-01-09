#!pip -q install rdkit-pypi==2021.9.4

import os

os.environ["KERAS_BACKEND"] = "tensorflow"

import ast

import pandas as pd
import numpy as np

import tensorflow as tf
import keras
from keras import layers
from keras import ops

import matplotlib.pyplot as plt
from rdkit import Chem, RDLogger
from rdkit.Chem import BondType
from rdkit.Chem.Draw import MolsToGridImage

RDLogger.DisableLog("rdApp.*")
