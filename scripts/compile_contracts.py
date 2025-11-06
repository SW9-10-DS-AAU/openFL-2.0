from solcx import install_solc, set_solc_version, compile_standard
from pathlib import Path
import json

# 1) Ensure exact compiler
install_solc("0.8.9")
set_solc_version("0.8.9")

# 2) Load sources
root = Path(__file__).parents[1]
contracts_dir = root / "contracts"
sources = {
    "OpenFLManager.sol": {"content": (contracts_dir / "OpenFLManager.sol").read_text(encoding="utf-8")},
    "OpenFLModel.sol":   {"content": (contracts_dir / "OpenFLModel.sol").read_text(encoding="utf-8")},
}

# 3) Compile
compiled = compile_standard({
    "language": "Solidity",
    "sources": sources,
    "settings": {
        "outputSelection": {"*": {"*": ["abi","evm.bytecode.object"]}}
    }
})

# 4) Extract artifacts
mgr = compiled["contracts"]["OpenFLManager.sol"]["OpenFLManager"]
mdl = compiled["contracts"]["OpenFLModel.sol"]["OpenFLModel"]

build = root / "artifacts" / "bytecode"
build.mkdir(exist_ok=True)

# IMPORTANT: abi.txt should be JSON, because Python should json.load it later
(Path(build / "abi.txt")).write_text(json.dumps(mgr["abi"], separators=(",",":")), encoding="utf-8")
(Path(build / "bytecode.txt")).write_text(mgr["evm"]["bytecode"]["object"], encoding="utf-8")

(Path(build / "abi_model.txt")).write_text(json.dumps(mdl["abi"], separators=(",",":")), encoding="utf-8")
(Path(build / "bytecode_model.txt")).write_text(mdl["evm"]["bytecode"]["object"], encoding="utf-8")

print("Artifacts written to build/: abi.txt, bytecode.txt, abi_model.txt, bytecode_model.txt")
