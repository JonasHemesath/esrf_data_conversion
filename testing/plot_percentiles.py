import matplotlib.pyplot as plt
import json

with open('percentiles_esrf_data_zf11_hr.json', 'r') as f:
    d = json.load(f)
    sample = 'zf11_hr1'
    plt.hist(d['0.39% percentile'], bins=100)
    print('0.39% percentile', min(d['0.39% percentile']))
    plt.savefig(sample + '_0_39% percentile.png')
    plt.hist(d['99.61% percentile'], bins=100)
    print('99.61% percentile', max(d['99.61% percentile']))
    plt.savefig(sample + '_99_61% percentile.png')