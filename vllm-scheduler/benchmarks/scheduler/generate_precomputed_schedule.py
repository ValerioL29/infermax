import pickle
import sys 
import json
import math
def read_pkl_file(file_path):
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    return data


def generate_precomputed_schedule(fpath, k=1):
    scheduler, requests = read_pkl_file(fpath)
    schedule = {
        'c': [{} for i in range(scheduler.batch_id)],
        'p': [set([]) for i in range(scheduler.batch_id)],
        'evicted': [set([]) for i in range(scheduler.batch_id)],
        'requests': [],
        'chunked_prefill': scheduler.chunked_prefill,
        'first_generation_batch_id': [],
    }
    log = {
        'c': [{} for i in range(scheduler.batch_id)],
        'p': [[] for i in range(scheduler.batch_id)],
        'evicted': [[] for i in range(scheduler.batch_id)],
        'requests': [],
        'chunked_prefill': scheduler.chunked_prefill,
        'first_generation_batch_id': [],
    }
    request_id = 0
    
    for request in requests:
        for t in range(k):
            for batch_id, c in request._num_processed_tokens_per_batch.items():
                schedule['c'][batch_id][request_id] = c
                log['c'][batch_id][request_id] = c
            for batch_id in request._p:
                schedule['p'][batch_id].add(request_id)
                log['p'][batch_id].append(request_id)
            for batch_id in request._evicted:
                schedule['evicted'][batch_id].add(request_id)
                log['evicted'][batch_id].append(request_id)
            schedule['requests'].append([request._I, request._O])
            request_id += 1
    with open(fpath + '.schedule.json', "w") as f:
        json.dump(log, f, indent=2)
    return schedule


# I : 2, O: 2

def validate(item, request_id, max_batch_id, schedule, I, O):
    num_processed_tokens = 0

    # if request_id == 24:
    #     c_list = []
    #     e_list = []
    #     for batch_id in range(max_batch_id):
    #         c = schedule['c'][batch_id].get(request_id, 0)
    #         if  c > 0:
    #             c_list.append([batch_id, c])
            
    #         if request_id in schedule['evicted'][batch_id]:
    #             assert item['e'][batch_id][request_id] > 0
    #             e_list.append(batch_id)


    #     print(c_list)
    #     print(e_list)

    for batch_id in range(max_batch_id):
        
        if request_id in schedule['evicted'][batch_id]:
            num_processed_tokens = 0
            #if (request_id == 24): print('evict')
            continue

        num_processed_tokens += schedule['c'][batch_id].get(request_id, 0)

        if num_processed_tokens == (I + O -1):
            break
    
    assert num_processed_tokens == (I + O - 1), (batch_id, request_id, num_processed_tokens, I + O - 1)

def generate_precomputed_schedule_from_vidur(fpath):
    print(fpath)
    item = read_pkl_file(fpath)
    max_batch_id = max(list(item['c'].keys())) + 1
    schedule = {
        'c': [{} for i in range(max_batch_id)],
        'p': [set([]) for i in range(max_batch_id)],
        'evicted': [set([]) for i in range(max_batch_id)],
        'requests': [],
    }
    request_id = 0
    for batch_id, dic in item['c'].items():
        schedule['c'][batch_id] = dic

    for batch_id, dic in item['e'].items():
        for k, v in dic.items():
            if v != 0: schedule['evicted'][batch_id].add(k)
    
    for batch_id, dic in item['p'].items():
        for k, v in dic.items():
            if v != 0: schedule['p'][batch_id].add(k)    
    # for i in range(B):
    #     validate(item, i, max_batch_id, schedule, I, O)
    #     schedule['requests'].append([I,O])

    f = open(fpath.replace('schedule.pkl', 'request_metrics.csv'))
    lines = f.readlines()
    for i, line in enumerate(lines[1:]):
        es = line.strip().split(',')
        I_,O_ = int(es[-4]),int(es[-3])
        validate(item, i, max_batch_id, schedule, I_, O_)
        schedule['requests'].append([I_,O_])
    
    
    print(max_batch_id)
    print(max(list(item['last_batch_id'].values())))
    print(len(schedule['c']))    

    return schedule


def generate_synthetic_schedule(I, O, B, chunk_size):

    max_batch_id = math.ceil(I / chunk_size) + O - 1
    schedule = {
        'c': [{} for i in range(max_batch_id)],
        'p': [set([]) for i in range(max_batch_id)],
        'evicted': [set([]) for i in range(max_batch_id)],
        'requests': [],
    }
    for request_id in range(B):
        C = 0
        batch_id = 0
        while C < I:
            schedule['c'][batch_id][request_id] = min(chunk_size, I - C)
            schedule['p'][batch_id].add(request_id)
            batch_id += 1
            C += chunk_size
        C = 1
        while C < O:
            schedule['c'][batch_id][request_id] = 1
            C += 1
            batch_id += 1

    for i in range(B):
        validate(None, i, max_batch_id, schedule, I, O)
        schedule['requests'].append([I, O])
    # print(schedule)

    #schedule = {'c': [{0: 1}, {0: 1, 1: 1, 2: 1}, {1: 1, 2: 1}], 'p': [set([0]), set([0, 1, 2]), set([1,2])], 'evicted': [set(), set(), set()], 'requests': [[2, 1], [2, 1], [2,1]]}
    schedule['chunked_prefill'] = chunk_size < I     


    return schedule
    


# 2024-10-30_00-51-48-270584
if __name__ == '__main__':
    #pass

    import pandas as pd
    df = pd.read_csv('./vidur/schedule_backup_20241030_075959.csv')
    for index, row in df.iterrows():
        I = row['I']
        O = row['O']
        B = row['B']
        directory = row['directory']
        schedule = generate_precomputed_schedule_from_vidur(f'./vidur/{directory}/schedule.pkl', I, O, B)

    #schedule = generate_precomputed_schedule_from_vidur('./vidur/2024-10-30_00-53-19-813657/schedule.pkl', 8, 32, 1024)


    #print(schedule)
    #print(schedule)