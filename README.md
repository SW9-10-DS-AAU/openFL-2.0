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
- Copy RCP Server URL into smartcontracts.py (This line: ``w3 = Web3(Web3.HTTPProvider("HTTP://127.0.0.1:7545"))``)
  - This value should be changed when moving to an acutal testnet.

## 2. Requirements
- Make sure you use Python3.10
- Run ``pip install -r requirements.txt``
- Go to mnist_ropsten_experiment.ipynb and run jupyter notebook