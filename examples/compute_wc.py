from ovito.io import import_file

import WarrenCowleyParameters as wc

pipeline = import_file("fcc.dump")
mod = wc.WarrenCowleyParameters(nneigh=[0, 12, 18], only_selected=False)
pipeline.modifiers.append(mod)
data = pipeline.compute()

wc_for_shells = data.attributes["Warren-Cowley parameters"]
print(f"1NN Warren-Cowley parameters: \n {wc_for_shells[0]}")
print(f"2NN Warren-Cowley parameters: \n {wc_for_shells[1]}")

# Alternatively, can see it as a dictionarry
print(data.attributes["Warren-Cowley parameters by particle name"])

# The per-particle Warren-Cowley parameter are accessible as well
print(
    "Per-particle 1NN Warren-Cowley parameters:\n",
    data.particles["Warren-Cowley parameter (shell=1)"][...],
)
print(
    "Per-particle 2NN Warren-Cowley parameters:\n",
    data.particles["Warren-Cowley parameter (shell=2)"][...],
)

