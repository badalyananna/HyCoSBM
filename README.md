<h1 align="center">
HyCoSBM <br/>  
<i>Hypergraph Covariate Stochastic Block Model</i>
</h1>

<p align="center">
<i>Probabilistic model on hypergraphs able to incorporate the information about node covariates. </i>
</p>

<p align="center">
<a href="https://github.com/badalyananna/HyCoSBM/blob/main/LICENSE" target="_blank">
<img alt="License: MIT" src="https://img.shields.io/github/license/badalyananna/HyCoSBM">
</a>

<a href="https://www.python.org/" target="_blank">
<img alt="Made with Python" src="https://img.shields.io/badge/made%20with-python-1f425f.svg">
</a>

<a href="https://github.com/psf/black" target="_blank">
<img alt="Code style: black" src="https://img.shields.io/badge/code%20style-black-000000.svg">
</a>

<a href="http://arxiv.org/abs/2311.03857" target="_blank">
<img alt="ARXIV: 2311.03857" src="https://img.shields.io/badge/arXiv-2311.03857-red.svg">
</a>

</p>

This repository contains the implementation of the <i>HyCoSBM</i> model presented in:

&nbsp;&nbsp; 
[1] <i> Hypergraphs with node attributes: structure and inference. </i><br/>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; 
Anna Badalyan, Nicolò Ruggeri, and Caterina De Bacco<br/>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; 
[
    <a href="http://arxiv.org/abs/2311.03857" target="_blank">ArXiv</a>
]

<i>HyCoSBM</i> is a stochastic block model for higher-order interactions that can 
incorporate node covariates for improved inference. <br/> 
This code is made available for the public, if you make use of it please cite our work 
in the form of the references above.
The implementation is based on the <a href="https://github.com/nickruggeri/Hy-MMSBM"> Hy-MMSBM </a> model.


<h2>Code installation</h2>

The code was developed utilizing <b>Python 3.9</b>, and can be downloaded and used locally as-is. <br>
To install the necessary packages, run the following command

`pip install -r requirements.txt`


<h2>Inference of community structure</h2>

The inference of the affinity matrix <i>w</i> and community assignments <i>u</i> is 
performed by running the code in `main_inference.py`. 

The most basic run only needs a hypergraph, the number of communities <i>K</i>, and a path to store the results. <br/>
For example, to perform inference on the High  School dataset with <i>K=2</i> 
communities, one can run the following command:

```
python main_inference.py 
--K 2 --out_dir ./out_inference --pickle_file data/examples/high_school_dataset/hypergraph.pkl
```

The basic run, however, doesn't use the attributes. To add the attributes we need to specify the link to a csv file containing attributes with `--attribute_file` parameter and the names of the columns to be used as attributes in `--attribute_names`. By default, `gamma = 0.0`, we can also change this parameter by using `--gamma 0.8` command. The following command runs inference on High School dataset using attributes class and sex with `K = 2` and `gamma = 0.8`.
``` 
python main_inference.py 
--K 2 
--gamma 0.8
--out_dir ./out_inference 
--pickle_file data/examples/high_school_dataset/hypergraph.pkl
--attribute_file data/examples/high_school_dataset/attributes.csv
--attribute_names class sex
```

<h3>Input dataset format</h3>

It is possible to provide the input dataset in two formats.

__1. Text format__<br/> 
A hypergraph can be provided as input via two *.txt* files,
containing the list of hyperedges, and the relative weights. 
This allows the user to provide arbitrary datasets as inputs. 
To perform inference on a dataset specified in text format, provide the path to the two 
files as 
```
python main_inference.py 
--K 2 
--out_dir ./out_inference 
--hyperedge_file data/examples/high_school_dataset/hyperedges.txt 
--weight_file data/examples/high_school_dataset/weights.txt
```

__2. Pickle format__<br/>
Alternatively, one can provide a `Hypergraph` instance, which is the main representation 
utilized internally in the code (see `src.data.representation`), serialized via the 
<a href="https://docs.python.org/3/library/pickle.html">pickle</a> Python library. <br/>
An example equivalent to the above is
```
python main_inference.py 
--K 2 
--out_dir ./out_inference 
--pickle_file data/examples/high_school_dataset/hypergraph.pkl
```
Similarly to the text format, this allows to provide arbitrary hypergraphs as input.

<h3>Additional options</h3>

Additional options can be specified, the full documentation is shown by running
        
`python main_inference.py --help`

Among the important ones we list:
- `--assortative` whether to run inference with a diagonal affinity matrix <i>w</i>.  
- `--max_hye_size` to keep only hyperedges up to a given size for inference. If `None`, all hyperedges are utilized.
- `--w_prior` and `--u_prior` the rates for the exponential priors on the parameters. A value of zero is equivalent to no prior, any positive value is utilized for MAP inference. <br/>
For non-uniform priors, the path to a file containing a NumPy array can be specified, which will be loaded via `numpy.load`.
- `--em_rounds` number of EM steps during optimization. It is sometimes useful when the model doesn't converge rapidly.
- `--training_rounds` the number of models to train with different random initializations. The one with the highest log-likelihood is returned and saved.
- `--seed` integer random seed.

<h2>Data release</h2>

All synthetically generated attributes and hypergraphs used in the experiments are available in `data/generated` folder.

All real datasets used in the experiments are publically available.