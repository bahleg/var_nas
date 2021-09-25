import torch
import configobj
import sys
sys.path.append('.')
from models.cnn.search_cnn import SearchCNNController
import json 
import numpy as np 

def calc_param_number(model, g_reduce, g_normal):
    penalty = 0
    for id, cell in enumerate(model.net.cells):
            # можно не пробегать несколько раз, т.к. клетки одинаковы (С точностью до normal и reduce)                        
            weights = g_reduce if cell.reduction else g_normal
            
            for edges, w_list in zip(cell.dag, weights):
                for mixed_op, weight in zip(edges, w_list):
                    op = mixed_op._ops[weight]

                    for param in op.parameters():
                        penalty += np.prod(param.shape) 
    return penalty    


if __name__=='__main__':
	print ('args: <path to config> <path to checkpoint> <mode>  <path to save>')
	config = configobj.ConfigObj(sys.argv[1])
	config['device'] = 'cuda:0'
	model = SearchCNNController(**config)
	model.load_state_dict(torch.load(sys.argv[2]))
	red, norm = model.genotype( mode=sys.argv[3])
	print ('param num', calc_param_number(model, red, norm))
	with open(sys.argv[-1], 'w') as out:
		out.write(json.dumps([red,norm]))		
