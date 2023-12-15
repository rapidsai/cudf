# Copyright (c) 2023, NVIDIA CORPORATION.

import xml.etree.ElementTree as ET
import glob


def clean_definitions(root):
    # Breathe can't handle SFINAE properly:
    # https://github.com/breathe-doc/breathe/issues/624
    seen_ids = set()
    for sectiondef in root.findall(".//sectiondef"):
        for memberdef in sectiondef.findall("./memberdef"):
            id_ = memberdef.get("id")
            for tparamlist in memberdef.findall("./templateparamlist"):
                for param in tparamlist.findall("./param"):
                    for type_ in param.findall("./type"):
                        if "enable_if" in ET.tostring(type_).decode().lower():
                            if id_ not in seen_ids:
                                # If this is the first time we're seeing this function,
                                # just remove the template parameter.
                                seen_ids.add(id_)
                                tparamlist.remove(param)
                            else:
                                # Otherwise, remove the overload altogether and just
                                # rely on documenting one of the SFINAE overloads.
                                sectiondef.remove(memberdef)
                            break

                    # If the id is in seen_ids we've already either removed the param or
                    # the entire memberdef.
                    if id_ not in seen_ids:
                        # In addition to enable_if, check for overloads set up by
                        # ...*=nullptr.
                        for type_ in param.findall("./defval"):
                            if "nullptr" in ET.tostring(type_).decode():
                                tparamlist.remove(param)
                                break


    # All of these in type declarations cause Breathe to choke.
    # For friend, see https://github.com/breathe-doc/breathe/issues/916
    strings_to_remove = ("__forceinline__", "CUDF_HOST_DEVICE", "decltype(auto)", "friend")
    for field in (".//type", ".//definition"):
        for type_ in root.findall(field):
            if type_.text is not None:
                for string in strings_to_remove:
                    type_.text = type_.text.replace(string, "")



for fn in glob.glob("xml/*.xml"):
    tree = ET.parse(fn)
    clean_definitions(tree.getroot())
    tree.write(fn)
