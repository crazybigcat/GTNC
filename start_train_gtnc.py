from library import MPSMLclass
from library import Parameters

para = Parameters.gtnc()
A = MPSMLclass.GTNC(para=para, device='cpu')  # change device='cuda' to use GPU
A.training_gtn()  # if the GTN are not trained
acc = A.calculate_accuracy('test')