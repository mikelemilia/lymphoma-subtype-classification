# Project 6 - Lymphoma Subtype Classification

## Project structure

```
.
├── data/                 
│   ├── CLL/               # CLL class 
│   ├── FL/                # FL class 
│   └── MCL/               # MCL class 
├── src/      
│   ├── models/
│   │   ├── init.py        # File used to export Neural Network models    
│   │   ├── CNN.py         # CNN with 2 block (baseline)
│   │   ├── CNNv1.py       # CNN with 3 block     
│   │   ├── CNNv2.py       # CNN with 3 block + 2 skip connections     
│   │   ├── CNNv3.py       # CNN with 3 block + 2 skip connections + GAP    
│   │   ├── CNNv3.py       # CNN with 4 block + 3 skip connections + GAP       
│   │   └── Network.py     # Class from which all the other were derived    
│   ├── Dataset/           # Preprocessing and dataset generation happens here
│   └── main.py            # Core
├── output/  
├── plot/  
└── README.md                     # project overview
```

The project should be structured in four main folders: `data`, `src`, `output` and `plot`.

Specifically, `data` should contain all the input files related to the classification task (images), `src` contains all
the implemented code, `output` will contain the trained `.h5` models and `plot` will contain both confusion matrix and
curves of trained models.

## Help usage

In order to easily run the code an ArgumentParser was implemented, in order to see all the input parameters the code can
be run with flag `-h` which will provide the following help.

```
usage: main.py [-h] [-f FOLDER] [-a ARCH] [-m MODE] [-s SIZE] [-c COLOR]
               [-e EXTRA] [--train]

3-CLASS LYMPHOMA SUBTYPE CLASSIFICATION

optional arguments:
  -h, --help            show this help message and exit
  -f FOLDER, --folder FOLDER
                        dataset folder path
  -a ARCH, --arch ARCH  architecture, it can be CNN, CNNv1, CNNv2, CNNv3,
                        CNNv4
  -m MODE, --mode MODE  preprocessing mode, it can be FULL or PATCH
  -s SIZE, --size SIZE  preprocessing mode, it can be 128
  -c COLOR, --color COLOR
                        color space used, it can be RGB, GRAY, HSV
  -e EXTRA, --extra EXTRA
                        extracted features, it can be CANNY, WAV
  --train               use it only if you need to train
```

# How to run

Code can be run by default parameters, or in the following way:

```shell
python $DIRECTORY/src/main.py -f $DATA -a CNNv4 -c RGB -m PATCH --train
```

Where DIRECTORY is the location of the project in your machine, and DATA the location of the DATASET (that can be
different from the `data` folder specified above inside the project folder). In this way it should be easy to run even
in Cluster DEI.