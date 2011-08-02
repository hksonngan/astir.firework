# Python
from pet import *

# C code
try:
    from pet_c import *
except:
    print '[Warning] impossible to load the module pet_c!'

# CUDA code
try:
    from pet_cuda import *
except:
    print '[Warning] impossible to load the module pet_cuda!'
