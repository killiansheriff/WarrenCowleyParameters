import sys

sys.path.append("/home/ksheriff/PACKAGES/WarrenCowleyParameters/src")
from ovito.io import import_file

from WarrenCowleyParameters import WarrenCowleyParameters

pipeline = import_file("fcc.dump")
mod = WarrenCowleyParameters(nneigh=[0, 12, 18])
pipeline.modifiers.append(mod)
pipeline.compute()
