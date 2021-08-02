import os
for seed in [0,50,100]:
	for l in [0.0, 0.5, 1.0]:
		cmd = 'python3 analysis/hyper/make_genotype.py  ./configs/fmnist_hyper_final/fmnist_hyper.cfg  ./searchs/fmnist_darts_hyper/checkpoint_{0}_49.ckp DARTS {1}  ./searchs/fmnist_darts_hyper/genotype_{0}_{2}.json'.format(seed, l, str(l).replace('.', '_'))
		print (cmd)
		os.system(cmd)

