from library import MPSMLclass
from library import Parameters

para = Parameters.gtn()
GTN = MPSMLclass.GTN(para=para, device='cpu') # change device='cuda' to use GPU
GTN.start_learning()

