import sys
import nvstrings
import pickle

if(len(sys.argv) < 2):
    print("require parameter: 'server' or 'client'")
else:
    if(str(sys.argv[1]) == 'client'):
        filehandler = open("/tmp/ipctest", 'rb')
        ipc_data = pickle.load(filehandler)

        new_strs = nvstrings.create_from_ipc(ipc_data)
    elif(str(sys.argv[1]) == 'server'):
        strs = nvstrings.to_device(
            ["abc", "defghi", None, "jkl", "mno",
             "pqr", "stu", "dog and cat", "accénted", ""])

        ipc_data = strs.get_ipc_data()

        with open("/tmp/ipctest", 'wb') as filehandler:
            pickle.dump(ipc_data, filehandler)

        input("Server ready. Press enter to terminate.")
