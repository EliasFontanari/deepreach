import multiprocessing
import time

# Create a Manager to handle shared data
manager = multiprocessing.Manager()

# Global counter variable managed by the Manager
counter = manager.Value('i', 0)  # Shared integer ('i' means integer type)
counter_lock = multiprocessing.Lock()  # Lock to protect the counter

def worker(dummy_arg):
    """Worker function to increment a global counter."""
    with counter_lock:  # Acquire the lock before modifying the global counter
        print(f"Worker {dummy_arg} starting.")
        time.sleep(1)  # Simulate some work
        counter.value += 1  # Modify the global counter
        print(f"Worker {dummy_arg} finished. Current counter: {counter.value}")

def main():
    num_processes = 4  # Number of processes to run in parallel
    num_tasks = 100    # Total number of times to execute the worker function
    
    # Create a pool of worker processes
    with multiprocessing.Pool(processes=num_processes) as pool:
        # Use pool.map to run the worker function 100 times
        pool.map(worker, range(num_tasks))  # `range(num_tasks)` generates dummy arguments

    print(f"Final counter value: {counter.value}")

if __name__ == '__main__':
    main()