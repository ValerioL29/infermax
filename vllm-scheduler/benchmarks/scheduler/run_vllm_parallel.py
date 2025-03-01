from multiprocessing import Process, Queue
import subprocess
import sys

# 작업을 큐에 추가하는 함수
def add_jobs(queue, commands, num_workers):
    for command in commands:
        queue.put(command)
    # 모든 작업을 추가한 후, None을 큐에 넣어 종료 신호를 보냅니다.
    for _ in range(num_workers):
        queue.put(None)

# 프로세스가 큐에서 작업을 가져와 실행하는 함수
def worker(queue, worker_id):
    while True:
        command = queue.get()
        if command is None:  # 종료 신호 확인
            break
        try:
            # 우분투 커맨드를 실행
            result = subprocess.run(f'CUDA_VISIBLE_DEVICES={worker_id} ' + command, shell=True, capture_output=True, text=True)
            print(f"[Worker {worker_id}] Command: {command}")
            print(f"[Worker {worker_id}] Output: {result.stdout}")
            print(f"[Worker {worker_id}] Error: {result.stderr}")
        except Exception as e:
            print(f"[Worker {worker_id}] Failed to execute {command}: {e}")


def run_vllm_parallel(commands, num_workers):

    num_workers = int(num_workers)
    queue = Queue()
    add_jobs(queue, commands, num_workers)
    
    # 작업을 큐에 추가하는 프로세스 시작    
    # 워커 프로세스 시작
    workers = [Process(target=worker, args=(queue, worker_id)) for worker_id in range(num_workers)]
    for w in workers:
        w.start()
    
    # 작업 추가 프로세스가 완료될 때까지 대기    
    # 워커 프로세스가 완료될 때까지 대기
    for w in workers:
        w.join()
    
    print("All jobs have been processed.")
