import matplotlib.pyplot as plt
import json

with open('percentiles_esrf_data_zf13_0.3um_65keV.json', 'r') as f:
    d = json.load(f)
    sample = 'zf13_0.3um_65keV'
    plt.hist(d['0.39% percentile'], bins=100)
    print('0.39% percentile', min(d['0.39% percentile']))
    plt.savefig(sample + '_0_39% percentile.png')
    plt.hist(d['99.61% percentile'], bins=100)
    print('99.61% percentile', max(d['99.61% percentile']))
    plt.savefig(sample + '_99_61% percentile.png')