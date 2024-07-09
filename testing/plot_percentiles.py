import matplotlib.pyplot as plt
import json

with open('percentiles_esrf_data_zf11.json', 'r') as f:
    d = json.load(f)
    
    plt.hist(d['zf11_hr']['0.39% percentile'], bins=100)
    print('0.39% percentile', min(d['zf11_hr']['0.39% percentile']))
    plt.savefig('zf11_hr_0_39% percentile.png')
    plt.hist(d['zf11_hr']['99.61% percentile'], bins=100)
    print('99.61% percentile', max(d['zf11_hr']['99.61% percentile']))
    plt.savefig('zf11_hr_99_61% percentile.png')