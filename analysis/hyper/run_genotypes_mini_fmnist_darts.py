import os
for seed in [150,200]:
		cmd = 'python3 analysis/hyper/make_genotype_darts.py  ./configs/mini_fmnist_hyper_final/fmnist.cfg  ./searchs/mini_fmnist_darts/checkpoint_{0}_49.ckp simple    ./searchs/mini_fmnist_darts_hyper/darts_genotype_{0}.json'.format(seed)
		print (cmd)
		os.system(cmd)

