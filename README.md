
# RRRR

Python implementation of Reduced Rank Ridge Regression introduced in the work: 

Mukherjee, A., & Zhu, J. (2011). Reduced rank ridge regression and its kernel extensions. *Statistical analysis and data mining: the ASA data science journal*, 4(6), 612-622.

And used in the paper:  

[Transfer learning of deep neural network representations for fMRI decoding](https://google.com)  
Michele Svanera, Mattia Savardi, Sergio Benini, Alberto Signoroni, Gal Raz, Talma Hendler, Lars Muckli, Rainer Goebel, Giancarlo Valente, 
Biorvix, 2019. 


## Description

The code import fMRI data and CNN data, applies a pre-processing, and runs an optimisation process to obtain the best values, for `rank` (number of hidden components) and `reg` (regularisation term), to reconstruct `fc7` from brain data. The search space for these parameters is defined by the variable:

~~~
	space = [Integer(1, 50),
			  Real(1, 1e+12, "log-uniform")]
~~~

A log file is saved with any useful information.


## Requirements

Not particular requirement are needed, except common python packages (Numpy, Scipy, sklearn, skopt).
It works with Python2 and Python3.


## Training and testing

To train (and test) the method:

~~~~
python ./training.py \
		--n_calls=500 \
		--correlation_measure=pearsonr \
		--selected_layer='fc7_R' \
		--n_random_starts=100 \
		--log_name='./results_RRRR/my_log.log' 
~~~~

## Demo

Please see `demo.py` to see an example on how to use the code.

## Authors

[Michele Svanera](https://github.com/rockNroll87q)

[Mattia Savardi](https://github.com/Metunibs)


## Citation

If you find this code useful in your research, please, consider citing our paper:
```
TO ADD
```

And cite the original work:
```
@article{MZ11,
	title={Reduced rank ridge regression and its kernel extensions},
	author={Mukherjee, Ashin and Zhu, Ji},
	journal={Statistical analysis and data mining: the ASA data science journal},
	volume={4},
	number={6},
	pages={612--622},
	year={2011},
	publisher={Wiley Online Library}
}
```





