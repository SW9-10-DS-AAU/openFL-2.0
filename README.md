# OpenFL: Decentralized Federated Learning on Public Blockchain Systems

```
//   ___                   _____ _     
//  / _ \ _ __   ___ _ __ |  ___| |    
// | | | | '_ \ / _ \ '_ \| |_  | |    
// | |_| | |_) |  __/ | | |  _| | |___ 
//  \___/| .__/ \___|_| |_|_|   |_____|
//       |_|                           
//
```

# Getting started
## 1. Ganache
- Download Ganache
- Set up a workspace (Not quickstart)
- Set gas limit much higher than default, same with balance
- Set accounts to 8

## 2. Environment Variables
The project contains a .env file located in the .env folder, but supports easy replacement of this environmnent.
The project runs with the .env.ganache .env file by default. If another .env is preferred run the program with the 
``ENV=<env_file_identifier>`` prefix. Providing no ENV prefix and providing ``ENV=ganache`` is therefore equivalent.

In your Environment, you must have the following variables set:
```
RPC_URL="<RPC_URL from ganache or sepolia, including port>"
PRIVATE_KEYS="<Private keys from your accounts colon separated (for non-locally forked blockchain). If you have fork=true (using Ganache), there is no need to set private keys. Then just keep this variable empty>"
```

## 3. Requirements
- Only tested with Python3.10
- Run ``pip install -e .[dev]``

## 4. Running an Experiment
The Experiment folder contains files for running experiments on different datasets.
To change the experiment setup, modify the experiment_configuration.py file.
To change the dataset, modify the experiments.py file.

The file experiments.py runs one such experiment and can be run with:
``ENV=ganache python ./experiment/experiment_runner.py``

