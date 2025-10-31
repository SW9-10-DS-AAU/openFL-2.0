from pytorch_model import PytorchModel
from smartcontracts import FLManager, FLChallenge

#DATASET = "cifar-10"
DATASET = "mnist"

with open("rpc_endpoint.txt", "r") as file:
    RPC_ENDPOINT = file.read()

NUMBER_OF_GOOD_CONTRIBUTORS = 4
NUMBER_OF_BAD_CONTRIBUTORS = 1
NUMBER_OF_FREERIDER_CONTRIBUTORS = 1
NUMBER_OF_INACTIVE_CONTRIBUTORS = 0

REWARD = int(1e18) # 1 ETH
MINIMUM_ROUNDS = 3
MIN_BUY_IN = int(1e18) # 1 ETH
MAX_BUY_IN = int(1.8e18) # 1.8 ETH
STANDARD_BUY_IN = int(1e18) # 1 ETH
EPOCHES = 1 #25
BATCH_SIZE = 32 #128
PUNISHFACTOR = 3
FIRST_ROUND_FEE = 50 # 50% OF MIN DEPOSIT

FORK = True # Fork Chain or communicate directly with RPC

NUMBER_OF_CONTRIBUTERS = NUMBER_OF_GOOD_CONTRIBUTORS      + \
                         NUMBER_OF_BAD_CONTRIBUTORS       + \
                         NUMBER_OF_FREERIDER_CONTRIBUTORS + \
                         NUMBER_OF_INACTIVE_CONTRIBUTORS


# Only for the real-net simulation
# In order to use a non-locally forked blockchain,
# private keys are required to unlock accounts
if FORK == False:
    from web3 import Web3
    w3 = Web3(Web3.HTTPProvider(RPC_ENDPOINT))
    PRIVKEYS = []
    with open("private_keys.txt", "r") as file:
        for f in file:
            PRIVKEYS.append(f.split(":")[0])

    PRIVKEYS = [w3.eth.account.privateKeyToAccount(i) for i in PRIVKEYS]
else:
    PRIVKEYS = None

pytorch_model = PytorchModel(DATASET,
                             NUMBER_OF_GOOD_CONTRIBUTORS,
                             NUMBER_OF_CONTRIBUTERS,
                             EPOCHES,
                             BATCH_SIZE,
                             STANDARD_BUY_IN,
                             MAX_BUY_IN)

for i in range(NUMBER_OF_BAD_CONTRIBUTORS):
    pytorch_model.add_participant("bad", 3)

for i in range(NUMBER_OF_FREERIDER_CONTRIBUTORS):
    pytorch_model.add_participant("freerider", 1)

for i in range(NUMBER_OF_INACTIVE_CONTRIBUTORS):
    pytorch_model.add_participant("inactive", 1)



manager = FLManager(pytorch_model, True).init(NUMBER_OF_GOOD_CONTRIBUTORS,
                                              NUMBER_OF_BAD_CONTRIBUTORS,
                                              NUMBER_OF_FREERIDER_CONTRIBUTORS,
                                              NUMBER_OF_INACTIVE_CONTRIBUTORS,
                                              MINIMUM_ROUNDS,
                                              RPC_ENDPOINT,
                                              FORK,
                                              PRIVKEYS)

manager.buildContract()

configs = manager.deployChallengeContract(MIN_BUY_IN,
                                          MAX_BUY_IN,
                                          REWARD,
                                          MINIMUM_ROUNDS,
                                          PUNISHFACTOR,
                                          FIRST_ROUND_FEE)

model = FLChallenge(manager,
                    configs,
                    pytorch_model)


model.simulate(rounds=MINIMUM_ROUNDS)
