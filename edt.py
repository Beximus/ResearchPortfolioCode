import math
import numpy as np
import pandas as pd
import os

a = np.array((0.027956882,-0.020826107,-0.04603296,-0.01250429,-0.01250429))
b = np.array((-0.006607811,-0.017397948,-0.04597349,-0.0038895006,0.014797262))
c = np.array((0.029920286,-0.008981087,-0.033041533,-0.02164808,0.015607642))

distab = np.linalg.norm(a-b)
distac = np.linalg.norm(a-c)
distbc = np.linalg.norm(b-c)

print("distance between a & b: "+str(distab))

print("distance between b & c: "+str(distbc))
print("distance between c & a: "+str(distac))