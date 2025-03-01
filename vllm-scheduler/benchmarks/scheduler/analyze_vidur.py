import sys
import json


f = open(sys.argv[1])
item = json.load(f)

batch_start_time = item['batch_start_time']
model_forward_time = item['model_forward_time'][1:]


#print(len(batch_start_time), len(model_forward_time))

total_time = batch_start_time[-1] - batch_start_time[0]
model_forward_time = sum(model_forward_time)

print(f'{sys.argv[1]},{total_time},{model_forward_time}')