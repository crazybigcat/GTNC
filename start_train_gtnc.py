import numpy as np
from library import MPSMLclass


A = MPSMLclass.GTNC()
A.training_gtn()  # if the GTN are not trained
acc = A.calculate_accuracy('test')