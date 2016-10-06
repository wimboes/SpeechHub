### authors : Wim Boes & Robbe Van Rompaey
### date: 5-10-2016 

### imports and definition of paths

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
import sys
import time
import numpy as np
import tensorflow as tf
import reader

python_path = os.path.abspath(os.getcwd())
general_path = os.path.split(python_path)[0]
input_path = os.path.join(general_path,'Input')
output_path = os.path.join(general_path,'Output')
config_path = os.path.join(general_path,'Configurations')
global_path = os.path.join(os.path.split(general_path)[0],'Global')

sys.path.append(config_path)
sys.path.append(global_path)
from config0 import *

### creating, training, testing, saving model

execfile('ptb_word_lm.py')

