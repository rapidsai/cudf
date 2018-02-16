import numpy as np
import pygdf
import dask_gdf
import dask
from distributed import Client


def main():
    client = Client('localhost:8786')
    dask.set_options(get=client.get)

    def do_init():
        import pygdf.gpu_ipc_broker
        pygdf.gpu_ipc_broker.enable_ipc()

    client.run(do_init)

    pygdf.gpu_ipc_broker.enable_ipc()

    print(client)
    df = pygdf.DataFrame()
    nelem = 10000000
    df['a'] = np.random.randint(0, 6, size=nelem)
    df['b'] = np.random.random(nelem)
    df = dask_gdf.from_pygdf(df, npartitions=4)

    out = df.groupby('a').apply(lambda df:df).compute().to_pandas()
    # out = df.compute().to_pandas()
    print(out)
    print("DONE")

    # client.restart()

if __name__ == '__main__':
    main()
