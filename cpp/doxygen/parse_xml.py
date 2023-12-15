# Copyright (c) 2023, NVIDIA CORPORATION.

import xml.etree.ElementTree as ET
import glob


types_to_remove = ("CUDF_ENABLE_IF", "std::enable_if")
defvals_to_remove = ("nullptr")
def remove_param_with_cudf_enable_if(root):
    for parent in root.findall(".//param/.."):
        for param in parent.findall(".//param"):
            for type_ in param.findall("./type"):
                if any(r in ET.tostring(type_).decode() for r in types_to_remove):
                    parent.remove(param)
                    break
        for param in parent.findall(".//param"):
            for type_ in param.findall("./defval"):
                if any(r in ET.tostring(type_).decode() for r in defvals_to_remove):
                    parent.remove(param)
                    break

    strings_to_remove = ("__forceinline__", "CUDF_HOST_DEVICE", "decltype(auto)")
    for field in (".//type", ".//definition"):
        for type_ in root.findall(field):
            if type_.text is not None:
                for string in strings_to_remove:
                    type_.text = type_.text.replace(string, "")
                if field == ".//type":
                    # Due to https://github.com/breathe-doc/breathe/issues/916
                    # friend is inserted twice, once in the type and once in the
                    # definition. We choose to remove from the definition.
                    type_.text = type_.text.replace("friend", "")



# Parse the XML file using xml.etree.ElementTree
for fn in glob.glob("xml/*.xml"):
    tree = ET.parse(fn)
    root = tree.getroot()

    # Remove <param> nodes with <type> containing 'CUDF_ENABLE_IF'
    remove_param_with_cudf_enable_if(root)

    # Save the modified XML to a new file
    tree.write(fn)
