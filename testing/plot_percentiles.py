import matplotlib.pyplot as plt
import json

with open('percentiles_esrf_data_zf14.json', 'r') as f:
    d = json.load(f)
    sample = 'zf14_s2_hr'
    plt.hist(d[sample]['0.39% percentile'], bins=100)
    print('0.39% percentile', min(d[sample]['0.39% percentile']))
    plt.savefig(sample + '_0_39% percentile.png')
    plt.hist(d[sample]['99.61% percentile'], bins=100)
    print('99.61% percentile', max(d[sample]['99.61% percentile']))
    plt.savefig(sample + '_99_61% percentile.png')