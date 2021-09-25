import os
for seed in [0,50,100, 150, 200]:
		cmd = 'python3 analysis/hyper/make_genotype_darts.py  ./configs/mini_cifar_hyper_final/cifar.cfg  ./searchs/mini_cifar_darts/checkpoint_{0}_49.ckp simple    ./searchs/mini_cifar_darts_hyper/darts_genotype_{0}.json'.format(seed)
		print (cmd)
		os.system(cmd)

