import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def convert_data(monitor):
    converted_data = []
    for trial in monitor:
        if isinstance(trial[0], bool) or isinstance(trial[0], int):
            data = [int(i) for i in trial]
        else:
            data = [round(i, ndigits=2) for i in trial]
        converted_data.append(data)
    return converted_data

def plot_monitors(monitors):
    comm_success = pd.DataFrame(convert_data(monitors['communicative-success']))
    lex_size = pd.DataFrame(convert_data(monitors['lexicon-size']))
    lex_coh = pd.DataFrame(convert_data(monitors['lexicon-coherence']))

    timesteps = [i for i in range(len(comm_success.mean().tolist()))]
    data_cs = comm_success.mean().rolling(100, min_periods=1).mean().tolist()
    data_ls = lex_size.mean().rolling(100, min_periods=1).mean().tolist()
    data_lc = lex_coh.mean().rolling(100, min_periods=1).mean().tolist()

    fig, ax1 = plt.subplots()
    fig.set_figheight(4)
    fig.set_figwidth(6)

    ax1.set_xlabel('number of games')
    ax1.set_ylabel('communicative success / lexical coherence')
    cs = ax1.plot(timesteps, data_cs, linestyle='-', color='blue', label='communicative success')
    lc = ax1.plot(timesteps, data_lc, linestyle=':', color='red', label='lexical coherence')
    ax1.tick_params(axis='y')
    ax1.set_xlim([0, 5000])
    ax1.set_xticks(np.arange(0, 5001, 500))
    ax1.set_ylim([0, 1])
    ax1.grid(axis='y')

    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
    ax2.set_ylabel('lexicon size')  # we already handled the x-label with ax1
    ls = ax2.plot(timesteps, data_ls, linestyle='--', color='y', label='lexicon size')
    ax2.tick_params(axis='y')
    ax2.set_ylim([0, 35])

    lns = cs+lc+ls
    labs = [l.get_label() for l in lns]
    ax1.legend(lns, labs, loc=0)
    
    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    plt.show()
