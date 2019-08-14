from dask.distributed import Client
client = Client(n_workers=1, threads_per_worker=4, processes=False, memory_limit='4GB')
client