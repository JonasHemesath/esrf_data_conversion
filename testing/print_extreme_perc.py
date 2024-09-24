import json


r = 30

sample = 'zf11_hr'

with open('percentiles_esrf_data_' + sample + '.json', 'r') as f:
    data = json.load(f)

list1, list2 = zip(*sorted(zip(data['0.39% percentile'], data['file'])))
print('low 0.39% percentile')
for i in range(0,r):
    print(list2[i], list1[i])

list1, list2 = zip(*sorted(zip(data['99.61% percentile'], data['file'])))
print('high 99.61% percentile')
n = len(list1)
for i in range(n-r,n):
    print(list2[i], list1[i])


