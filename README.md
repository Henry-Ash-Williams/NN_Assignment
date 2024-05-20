# Neural Networks 2024 Assignment 

This repository contains my source code for the Neural Networks 2024 Assignment. The brief for this assignment can be found in `pdfs/neural_networks_assignment_2024.pdf`. 

## Structure 

```
.
├── 233561.ipynb                                # The Report 
├── README.md                                   
├── count_jupyter_nb_words.py
├── models                                      # Contains trained models, and plots
│   ├── experiment-1                                
│   │    ├── lr-0.0001
│   │    │   ├── 0000
│   │    │   ├── 0001
│   │    │   ├── 0002
│   │    │   ├── 0003
│   │    │   └── 0004
│   │    ├── lr-0.0026
│   │    ├── lr-0.0051
│   │    ├── lr-0.0077
│   │    └── lr-0.01
│   ├── experiment-2                                
│   └── experiment-3                                
├── pdfs
│   └── neural_networks_assignment_2024.pdf     # Assignment brief 
├── profiles                                    # Profiling information 
│   ├── no-multithreading.prof
│   └── multithreading.prof
├── references.bib                              # Citations 
├── requirements.txt                            # Dependencies
└── src                                         # Source code
    ├── experiment-1
    │   ├── averaged_data.csv
    │   ├── raw_data.csv
    │   └── main.py                             # The Experiment itself
    ├── experiment-2
    │   └── experiment2.py
    ├── experiment-3
    ├── model.py                                # The model 
    └── train.py                                # Train the model
```

## Installing 

```sh
$ git clone git@github.com/Henry-Ash-Williams/NN_Assignment
$ cd NN_Assignment 
$ python3.11 -m venv .venv 
$ source .venv/bin/activate
$ pip install -r requirements.txt 
```

## Running the Experiments 

In order to run these experiments, make sure you are in the root directory of this reposititory. Your current working directory should be something like the following: 

```sh
$ pwd 
/path/to/NN_Assignment 
```

### Experiment 1 

On my machine, an M2 Pro Macbook with 16GB of RAM, this took around 1 hour to run the 25 training runs. However, modifying the hyperparameters could impact this. 

```sh
$ python src/experiment-1/main.py 
```

### Experiment 2 

TODO 

### Experiment 3 

TODO 