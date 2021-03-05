import numpy as np

def ASK_2_Slicer(received):
    return (received > 0) + -1*(received < 0)
