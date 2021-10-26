import os
import numpy as np
import matplotlib.pyplot as plt

def average_sliding_window(logdir, communicative_successes):
    window_width = 100
    cumsum_vec = np.cumsum(np.insert(communicative_successes, 0, 0)) 
    ma_vec = (cumsum_vec[window_width:] - cumsum_vec[:-window_width]) / window_width
    firsts = [0] * window_width
    firsts.extend(ma_vec)
    ma_vec = firsts

    plt.style.use('seaborn-whitegrid')

    x = [i for i in range(0, len(ma_vec))]
    y = ma_vec

    plt.plot(x, y, '-k')
    figdir = os.path.join(logdir, "figures")
    os.makedirs(figdir, exist_ok=True)
    plt.savefig(os.path.join(figdir, 'communicative_success.png'))

def write_measure(monitor, fname):
    """Writes data out to a specified file with a s-expression format required by Babel."""
    out = ""
    for serie in monitor: 
        data = str([int(i) for i in serie]) # change booleans to integers (e.g. comm. success)
        data = data.replace(',','').replace('[','').replace(']','') # remove commas and square brackets
        data = "(" + data + ")" # add brackets around list
        out += data # concatenate strings
    out = "((" + out + "))" # add final round brackets

    with open(f"{fname}.lisp", "w") as file:
        file.write(out)