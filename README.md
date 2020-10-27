# Neural Datalog Through Time
Source code for [Neural Datalog Through Time: Informed Temporal Modeling via Logical Specification (ICML 2020)](https://arxiv.org/abs/2006.16723). 

## Reference
If you use this code as part of any published research, please acknowledge the following paper (it encourages researchers who publish their code!):

```
@inproceedings{mei-2020-smoothing,
  author =      {Hongyuan Mei and Guanghui Qin and Minjie Xu and Jason Eisner},
  title =       {Neural Datalog Through Time: Informed Temporal Modeling via Logical Specification},
  booktitle =   {Proceedings of the International Conference on Machine Learning},
  year =        {2020}
}
```

## Instructions
Here are the instructions to use the code base.

### Dependencies and Installation
This code is written in Python 3, and I recommend you to install:
* [Anaconda](https://www.continuum.io/) that provides almost all the Python-related dependencies;

Run the command line below to install the package (add `-e` option if you need an editable installation):
```
pip install .
```
It will automatically install the following important dependencies: 
* [PyTorch 1.1.0](https://pytorch.org/) that handles auto-differentiation.
* [pyDatalog](https://sites.google.com/site/pydatalog/) that handles back-end Datalog deductible database.

### Write Datalog Programs
Write down the Datalog programs. Some examples used in our experiments can be downloaded from this [Google Drive directory](https://drive.google.com/drive/folders/17vtQdx3d1wR-SADSMamt4E2mqHfEOu9q?usp=sharing). 

To replicate our experiments, download our datasets from the same Google drive directory. 

Organize your Datalog programs and datasets like:
```
domains/YOUR_DOMAIN/YOUR_PROGRAMS_AND_DATA
```

### Build Dynamic Databases
Go to the `ndtt/run` directory. 

To build the dynamic databases for your data, try the command line below for detailed guide: 
```
python build.py --help
```

The generated dynamic model architectures (represented by database facts) are stored in this directory: 
```
domains/YOUR_DOMAIN/YOUR_PROGRAMS_AND_DATA/tdbcache
```

### Train Models
To train the model specified by your Datalog probram, try the command line below for detailed guide:
```
python train.py --help
```

The training log and model parameters are stored in this directory: 
```
domains/YOUR_DOMAIN/YOUR_PROGRAMS_AND_DATA/Logs
```

### Test Models
To test the trained model, use the command line below for detailed guide: 
```
python test.py --help
```

To evaluate the model predictions, use the command line below for detailed guide: 
```
python eval.py --help
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
