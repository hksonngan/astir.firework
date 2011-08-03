# Python
from render import *

# C code
try:
    from render_c import *
except:
    print '[Warning] impossible to load the module render_c!'
