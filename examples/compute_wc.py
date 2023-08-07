from ovito.io import import_file

import WarrenCowleyParameters as wc

pipeline = import_file("fcc.dump")
mod = wc.WarrenCowleyParameters(nneigh=[0, 12, 18])
pipeline.modifiers.append(mod)
data = pipeline.compute()

wc_for_shells = data.attributes["Warren-Cowley parameters"]
print(f"1NN Warren-Cowley parameters: {wc_for_shells[0]}")
print(f"2NN Warren-Cowley parameters: {wc_for_shells[1]}")

breakpoint()
