import cudf
import torch
import torch.utils.dlpack
import gc

from librmm_cffi import librmm_config as rmm_cfg

rmm_cfg.use_pool_allocator = False
rmm_cfg.use_managed_memory = False # default is false
rmm_cfg.enable_logging = True      # default is False -- has perf overhead

from librmm_cffi import librmm as rmm

for i in range(5):
  x = cudf.DataFrame([('a', list(range(34000000))),
                      ('b', list(reversed(range(34000000)))),
                      ('c', list(range(34000000)))])
  y = x.to_dlpack()
  b = torch.utils.dlpack.from_dlpack(y)
  a = b.to(device='cpu')
  del x,y,b
  torch.cuda.empty_cache()
  gc.collect()

log = rmm.csv_log()

with open("log.csv", "w") as text_file:
  print(log, file=text_file)