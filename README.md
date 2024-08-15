# capstoneppl
SIADS 699 repo

### how to run our code
1. two major executables, to start: a) on a linux machine, where node.js is istalled, [run](https://github.com/stefanbund/capstoneppl/blob/main/MLOPS/START_SAMPLER.mjs) 'node START_SAMPLER.mjs', which will start the data sampling, live. b) use python to [begin the predictor](https://github.com/stefanbund/capstoneppl/blob/main/MLOPS/Predictor.py), 'python3 Predictor.py'. If you are [running linux, use the](https://github.com/stefanbund/capstoneppl/blob/main/MLOPS/run_pause_predictor.sh) 'run_pause_predictor.sh', which causes the Predictor to run continuously, without end, './run*' works fine.
2. [run the Modeler.py](https://github.com/stefanbund/capstoneppl/blob/main/MLOPS/Modeler.py), to see it discover best accuracy models.
3. [run Preprocessor.py](https://github.com/stefanbund/capstoneppl/blob/main/MLOPS/Preprocessor.py) to see it do data mining on surges and precursors.
4. [run analyzePredictions.py](https://github.com/stefanbund/capstoneppl/blob/main/MLOPS/analyzePredictions.py) to see it estimate time durations
5. running into issues? It was built for Ubuntu 22.4, Python 3.10.12, but a video will help illustrate our work, in addition to [our results video](https://www.loom.com/share/c2079b7a9aee40e7b8cc203c1da98b0d?sid=0356b3b7-5a6f-4b53-b053-eaf6ec9ad537).

### some videos
1. [introduction to the project](https://www.loom.com/share/dd66d1e0db974329bd64bebd8c3a97a2?sid=7cc453d4-5d2f-45cf-81aa-c066427a2425)
2. [Team HFT Discussion 2: Drilling Down into Prior Work (M2)](https://www.loom.com/share/a809130b687b45bdb117eb3375ab4a61?sid=24d14dba-70ad-470e-8db9-9949f8c6511a)

### helpful for code
1.  [G collab, how to replicate the MLOPS in your notebook](https://colab.research.google.com/drive/1VWcb37M-XYESfQaOXFaF5zjzi984wzw4?usp=sharing)
2. [identify and anlyze ideal trades, by completion time](https://colab.research.google.com/drive/1DDCHRaHRBGfl59Xw7sq8-4uTa7iVPcVn?usp=sharing)

### what we did for M2
1. [step 1 notebook: preprocessing sample data, for one symbol (AVAX-USD)](https://github.com/stefanbund/capstoneppl/blob/main/M2%20assets/step%201%2C%20data%20cleaning%2C%20binary_multi_pipelines.ipynb)
2. [binary binned pipeline csv, how the preprocessed data looks, after (1)](https://github.com/stefanbund/capstoneppl/blob/main/M2%20assets/binary_binned_pipeline.csv)
3. [model discovery, best models we found](https://github.com/stefanbund/capstoneppl/blob/main/M2%20assets/BIN%20BAL%20BINNED.ipynb)
