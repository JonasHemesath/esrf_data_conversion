import matplotlib.pyplot as plt
import json

with open('percentiles_esrf_data.json', 'r') as f:
    plt.hist(f['zf13_hr2']['0.39% percentile'], title='0.39% percentile')
    print('0.39% percentile', min(f['zf13_hr2']['0.39% percentile']))
    plt.savefig('zf13_hr2_0_39% percentile.png')
    plt.hist(f['zf13_hr2']['99.61% percentile'], title='99.61% percentile')
    print('99.61% percentile', max(f['zf13_hr2']['99.61% percentile']))
    plt.savefig('zf13_hr2_99_61% percentile.png')