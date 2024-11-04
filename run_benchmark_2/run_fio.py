#!/usr/bin/env python3

import subprocess
import enum
import os
import copy

class FileSize:
    def __init__(self, fileSize: int | str):
        self._KiB = 1024
        self._MiB = 1024 * self._KiB
        self._GiB = 1024 * self._MiB
        self._TiB = 1024 * self._GiB
        self._PiB = 1024 * self._TiB
        self._EiB = 1024 * self._PiB

        self._B: int = 0
        if type(fileSize) is int:
            self._B = fileSize
        else:
            fileSize = fileSize.replace(" ", "").lower()

            if fileSize.isdigit():
                self._B = int(fileSize)
            else:
                if fileSize.endswith("kib"):
                    self._B = int(fileSize[0:-3]) * self._KiB
                elif fileSize.endswith("mib"):
                    self._B = int(fileSize[0:-3]) * self._MiB
                elif fileSize.endswith("gib"):
                    self._B = int(fileSize[0:-3]) * self._GiB
                elif fileSize.endswith("tib"):
                    self._B = int(fileSize[0:-3]) * self._TiB
                elif fileSize.endswith("pib"):
                    self._B = int(fileSize[0:-3]) * self._PiB
                elif fileSize.endswith("b"):
                    self._B = int(fileSize[0:-1])
                else:
                    raise Exception("Invalid value for FileSize.")

        if self._B < 0:
            raise Exception("Invalid value for FileSize.")

    def toB(self) -> int:
        return self._B

    def toKiB(self) -> float:
        return self._B / self._KiB

    def toMiB(self) -> float:
        return self._B / self._MiB

    def toGiB(self) -> float:
        return self._B / self._GiB

    def toTiB(self) -> float:
        return self._B / self._TiB

    def toPiB(self) -> float:
        return self._B / self._PiB

    def toCompactTuple(self) -> tuple[float, str]:
        if self._B >= 0 and self._B < self._KiB:
            return (float(self._B), "B")
        elif self._B >= self._KiB and self._B < self._MiB:
            return (self.toKiB(), "KiB")
        elif self._B >= self._MiB and self._B < self._GiB:
            return (self.toMiB(), "MiB")
        elif self._B >= self._GiB and self._B < self._TiB:
            return (self.toGiB(), "GiB")
        elif self._B >= self._TiB and self._B < self._PiB:
            return (self.toTiB(), "TiB")
        elif self._B >= self._PiB and self._B < self._EiB:
            return (self.toPiB(), "PiB")
        else:
            raise Exception("Invalid value for FileSize.")

    def toCompactStr(self) -> str:
        value, unit = self.toCompactTuple()
        return f"{value}{unit}"


class FileOp(enum.Enum):
    READ = 1
    WRITE = 2


class Analyzer:
    def __init__(self, config: dict):
        self.color_green = "\x1b[1;32m"
        self.color_magenta = "\x1b[1;35m"
        self.color_end = "\x1b[0m"

        self.fioBin: str = config.get("fio_bin", "fio")
        self.directory: str = config.get("directory", ".")
        self.benchName: str = config.get("benchname", "placeholder_name")
        self.engine: str = config["engine"]
        self.blockSize: FileSize = config.get("blocksize", FileSize("4 KiB"))
        self.ioDepth: int = config.get("iodepth", 1)
        self.maxRuntime: int = config.get("max_runtime", 600)
        self.fileSize: FileSize = config.get("filesize", FileSize("1 MiB"))
        self.numThreads: int = config.get("num_threads", 4)
        self.useDirect: int = config.get("use_direct", 0)
        self.debug: bool = config.get("debug", False)

        if self.engine == "libcufile":
            self.cudaIO: str = config.get("cuda_io", "cufile")

        self.fileSizeIncrementPerThread: FileSize = FileSize(
            int(self.fileSize.toB() / self.numThreads))

    def _showInfo(self):
        print("--> fio version")
        fullCmd = "fio --version"
        cmdList = fullCmd.split()
        subprocess.run(cmdList)

        print("--> disk info")
        fullCmd = "lsblk -o +MODEL -e 7"
        cmdList = fullCmd.split()
        subprocess.run(cmdList)

    def _benchSeqOp(self, fileOp: FileOp):
        """
        Emulate the workload in cuDF/KvikIO where multiple threads read from the same file, starting at different offset.
        """

        fileOpStr = ""
        if fileOp == FileOp.READ:
            fileOpStr = "read"
        else:
            fileOpStr = "write"

        fullCmd = (
            "{} --directory={} --name={} ".format(
                self.fioBin, self.directory, self.benchName)
            + "--readwrite={} --ioengine={} --direct={} ".format(
                fileOpStr, self.engine, self.useDirect)
            + "--filename=seq_{}_file --group_reporting --numjobs={} --offset_increment={} --thread ".format(
                fileOpStr, self.numThreads, self.fileSizeIncrementPerThread.toB()
            )
            + "--iodepth={} --blocksize={} --runtime={} ".format(
                self.ioDepth, self.blockSize.toB(), self.maxRuntime)
            + "--filesize={}".format(self.fileSize.toB())
        )

        if self.engine == "libcufile":
            fullCmd += f" --cuda_io={self.cudaIO}"

        if self.debug:
            fullCmd += " --debug=file,io --number_ios=100"

        print("{}--> Full command: {}{}".format(self.color_magenta,
              fullCmd, self.color_end))
        cmdList = fullCmd.split()

        my_env = None
        if self.engine == "libcufile":
            my_env = dict()
            my_env["LD_LIBRARY_PATH"] = "/usr/local/cuda/targets/sbsa-linux/lib"
        subprocess.run(cmdList, env=my_env)

    def run(self):
        print("\n\n\n{}--> Engine: {}{}".format(self.color_green,
              self.engine, self.color_end))
        # self._showInfo()
        self._benchSeqOp(FileOp.READ)
        self._benchSeqOp(FileOp.WRITE)

def testAll(configBase):
    config = copy.deepcopy(configBase)

    engineList = ["psync", "sync", "io_uring", "libaio", "posixaio", "mmap"]
    for engine in engineList:
        config["engine"] = engine
        config["iodepth"] = 1
        az = Analyzer(config)
        az.run()

def testCufile(configBase):
    config = copy.deepcopy(configBase)
    config["engine"] = "libcufile"

    # Use GDS to copy from file to GPU
    config["cuda_io"] = "cufile"
    az = Analyzer(config)
    az.run()

    # Use POSIX to copy from file to host, then to GPU
    config["cuda_io"] = "posix"
    az = Analyzer(config)
    az.run()



if __name__ == "__main__":
    # Build fio from source:
    # Reference: https://git.kernel.dk/cgit/fio/commit/?id=10756b2c95ef275501d4dbda060caac072cf6973
    # CFLAGS must specify the location of CUDA and cuFile headers.
    # e.g. MY_CFLAGS="-I/usr/local/cuda/include -I/usr/local/cuda/targets/sbsa-linux/include"
    # LDFLAGS must specify the location of CUDA and cuFile libraries.
    # e.g. MY_LDFLAGS="-L/usr/local/cuda/lib64"
    # CFLAGS=$MY_CFLAGS LDFLAGS=$MY_LDFLAGS ./configure --prefix=/mnt/fio/install --enable-libcufile

    configBase = {
        "fio_bin": "/mnt/fio/install/bin/fio",
        "benchname": "seq_read_and_write",
        "directory": ".",
        "filesize": FileSize("1 GiB"),
        "blocksize": FileSize("4 MiB"),
        "num_threads": 4,
        "iodepth": 1,
        "use_direct": 1,
    }

    testAll(configBase)

    testCufile(configBase)



