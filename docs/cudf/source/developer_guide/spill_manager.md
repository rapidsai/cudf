# Spill Manager

``cudf`` supports automatic spilling (and "unspilling") of objects
from device to host to enable out-of-memory computation,
i.e., computing on objects that occupy more memory than is available
on the GPU.
