from pyarrow import cuda as pac
import cudf
import multiprocessing
import pyarrow as pa
import pandas as pd


def export_ipc(df: cudf.DataFrame):
    ctx = pac.Context()
    table = df.to_arrow()
    batches = table.to_batches()
    results = []
    for batch in batches:
        schema: pa.Schema = batch.schema
        print("schema:", schema, type(schema))
        s: pa.Buffer = schema.serialize()
        print("schema:", s, type(s))
        cbuf = pac.serialize_record_batch(batch, ctx)
        ipc_handle = cbuf.export_for_ipc()
        handle_buffer = ipc_handle.serialize()
        results.append((handle_buffer, schema))
    return results


def import_ipc(batches):
    ctx = pac.Context()
    roundtrip = []
    references = []
    for batch in batches:
        handle_buffer = batch[0]
        ipc_handle = ipc_handle = pa.cuda.IpcMemHandle.from_buffer(handle_buffer)
        schema = batch[1]
        cbuf = ctx.open_ipc_buffer(ipc_handle)

        read = pac.read_record_batch(cbuf, schema)
        roundtrip.append(read)
        references.append(cbuf)

    result = pa.Table.from_batches(roundtrip)
    df = cudf.DataFrame.from_arrow(result)

    df: pd.DataFrame = df.to_pandas()
    df.to_csv("result.csv")


if __name__ == "__main__":
    df = pd.DataFrame({"a": [1, 2, 3, 4], "b": [2, 3, 4, 5]})
    # table = pa.Table.from_pandas(df)
    df = cudf.from_pandas(df)
    ipc_handles = export_ipc(df)

    mctx = multiprocessing.get_context("spawn")
    # subprocess.check_call(["python", "./run-test-ipc.py"])

    p = mctx.Process(target=import_ipc, args=(ipc_handles, ))
    p.start()
    p.join()
    print("exit", p.exitcode)
    assert p.exitcode == 0
    # table = import_ipc(ipc_handles)
    # print(table)
