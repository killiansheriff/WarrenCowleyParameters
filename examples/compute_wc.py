from ovito.io import import_file
from ovito.modifiers import WarrenCowleyParameters

pipeline = import_file("fcc.dump")
mod = WarrenCowleyParameters(nneigh=[0, 12, 18])
pipeline.modifiers.append(mod)
data = pipeline.compute()

wc_for_shells = data.attributes["Warren-Cowley parameters"]
print(f"1NN Warren-Cowley parameters: \n {wc_for_shells[0]}")
print(f"2NN Warren-Cowley parameters: \n {wc_for_shells[1]}")

In this corrected version of the code, the WarrenCowleyParameters module is imported from ovito.modifiers. This should resolve the import error and allow you to use the WarrenCowleyParameters modifier in your OVITO pipeline correctly.
