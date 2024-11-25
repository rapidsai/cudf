# Copyright (c) 2024, NVIDIA CORPORATION.

import enum
import inspect
import itertools
import sys
import types
import warnings
from collections.abc import Callable, Generator, Mapping
from pathlib import Path

import libcst as cst
import libcst.matchers as m


class StubParser(cst.CSTVisitor):
    def __init__(self):
        self.classes: list[cst.ClassDef] = []
        self.functions: list[cst.FunctionDef] = []
        self.misc: list[cst.AnnAssign] = []

    @classmethod
    def definitions(
        cls, file: Path
    ) -> tuple[list[cst.ClassDef], list[cst.FunctionDef], list[cst.AnnAssign]]:
        """Extract definitions from a type stub file.

        Parameters
        ----------
        file
            File to extract definitions from

        Returns
        -------
        list[ClassDef], list[FunctionDef], list[AnnAssign]
            Tuple of concrete syntax tree nodes, class definitions,
            function definitions, annotated assignments.
        """
        if file.name == "__init__.py":
            raise RuntimeError("Not expecting __init__.py here")
        mod = cst.parse_module(file.read_text())
        collector = cls()
        mod.visit(collector)
        return collector.classes, collector.functions, collector.misc

    def visit_ClassDef(self, node: cst.ClassDef) -> bool:
        self.classes.append(node)
        return False

    def visit_FunctionDef(self, node: cst.FunctionDef) -> bool:
        self.functions.append(node)
        return False

    def visit_AnnAssign(self, node: cst.AnnAssign) -> bool:
        self.misc.append(node)
        return False


def traverse_modules(
    root: types.ModuleType,
) -> Generator[types.ModuleType, None, None]:
    """
    Yield unique modules referenced from a root

    Parameters
    ----------
    root
        Root module to start at

    Yields
    ------
    Modules referred to by the root module.

    Notes
    -----
    This only yields modules named in `__all__`.
    """
    lifo = [root]
    seen = set()
    while lifo:
        mod = lifo.pop()
        members = dict(inspect.getmembers(mod))
        if "__all__" not in members:
            raise RuntimeError(f"Module {mod.__name__} must provide __all__")
        yield mod
        seen.add(mod)
        for member in members["__all__"]:
            obj = members[member]
            if isinstance(obj, types.ModuleType) and obj not in seen:
                lifo.append(obj)


def check_module(
    mod: types.ModuleType, installed_location: Path, source_location: Path
):
    """
    Check if a module has valid type stubs.

    Parameters
    ----------
    mod
        Module to check.
    installed_location
        Path to installed location.
    source_location
        Path to source location.

    Warns
    -----
    UserWarning
        If invalid stubs are found.

    Notes
    -----
    This uses textual heuristics and runtime import information to
    check correctness of type stubs. It is far from foolproof.
    """
    if mod.__file__ is None:
        warnings.warn(f"Module '{mod.__name__}' doesn't have a matching file")
        return
    filename = Path(mod.__file__)
    if filename.suffix == ".py":
        # Nothing to do, mypy handles it
        return
    # Peel suffixes
    for _ in filename.suffixes:
        filename = filename.with_suffix("")
    pyi_file = source_location / filename.relative_to(
        installed_location
    ).with_suffix(".pyi")
    if not pyi_file.exists():
        warnings.warn(
            f"Module '{mod.__name__}' does not have a matching type stub"
        )
        return

    # Definitions in the type stub
    classes, functions, misc = StubParser.definitions(pyi_file)
    # Runtime exported members
    members = dict(inspect.getmembers(mod))
    exported_members = members["__all__"]
    exported_functions = {}
    exported_classes = {}
    exported_misc = {}
    for name in exported_members:
        cls = members[name]
        if isinstance(cls, types.ModuleType):
            # Handled by traversal
            continue
        elif inspect.isroutine(cls):
            exported_functions[name] = cls
        elif inspect.isclass(cls):
            exported_classes[name] = cls
        else:
            exported_misc[name] = cls

    seen_classes = set()
    for classdef in classes:
        name = classdef.name.value
        if name not in exported_classes:
            warnings.warn(
                f"Type stub '{pyi_file.name}' advertises class '{name}' "
                f"but module '{mod.__name__}' does not export a class of that name."
            )
            continue
        seen_classes.add(name)
        cls = exported_classes[classdef.name.value]
        if issubclass(cls, enum.IntEnum):
            EnumChecker.check(cls, classdef.body, mod)
        else:
            ClassChecker.check(cls, classdef.body, mod)
    missing = set(exported_classes) - seen_classes
    if missing:
        missing_classes = ", ".join(sorted(missing))
        warnings.warn(
            f"Module '{mod.__name__}' exports classes: '{missing_classes}' "
            f"but type stub '{pyi_file.name}' does not define them."
        )
    seen_functions = set()
    for functiondef in functions:
        name = functiondef.name.value
        if name not in exported_functions:
            warnings.warn(
                f"Type stub '{pyi_file.name}' advertises function '{name}' "
                f"but module '{mod.__name__}' does not export a function of that name."
            )
            continue
        seen_functions.add(name)
        fn = exported_functions[name]
        check_function(fn, functiondef, f"{mod.__name__}.name")
    missing = set(exported_functions) - seen_functions
    if missing:
        missing_functions = ", ".join(sorted(missing))
        warnings.warn(
            f"Module '{mod.__name__}' exports functions: '{missing_functions}' "
            f"but type stub '{pyi_file.name}' does not define them."
        )


class ClassChecker(cst.CSTVisitor):
    def __init__(self, cls: type, obj_name: str, *, is_dataclass: bool):
        self.cls = cls
        if is_dataclass:
            self.members = {
                name: inspect.Attribute(name, "data", cls, None)
                for name in cls.__dataclass_fields__
            }
        else:
            self.members = {
                attr.name: attr
                for attr in inspect.classify_class_attrs(cls)
                if not (
                    attr.name.startswith("_")
                    and attr.name
                    not in {"__init__", "__contains__", "__getitem__"}
                )
            }
        self.seen = set()
        self.obj_name = obj_name

    @classmethod
    def check(
        cls, typ: type, info: cst.BaseSuite, mod: types.ModuleType
    ) -> None:
        """
        Check a class definition in a type stub.

        Parameters
        ----------
        cls
            Runtime class to check against.
        info
            CST node representing the body of the class.
        mod
            Module class is defined in, for error reporting.

        Warns
        -----
        UserWarning
            If the checker finds any discrepancies.
        """
        assert not issubclass(cls, enum.IntEnum)
        checker = cls(
            typ,
            f"{mod.__name__}.{typ.__name__}",
            is_dataclass=hasattr(typ, "__dataclass_fields__"),
        )
        _ = info.visit(checker)
        checker.validate()

    def validate(self):
        unseen = sorted(set(self.members) - self.seen)
        if unseen:
            missing = ", ".join(unseen)
            warnings.warn(
                f"{self.obj_name}: Type stub did not define the "
                f"following class attributes: '{missing}'"
            )

    def visit_ClassDef(self, node: cst.ClassDef) -> bool:
        warnings.warn(
            f"{self.obj_name}: Not expected nested class {node.name.value}"
        )
        return False

    def visit_FunctionDef(self, node: cst.FunctionDef) -> bool:
        name = node.name.value
        if name not in self.members:
            warnings.warn(
                f"{self.obj_name}: Type stub defines function '{name}' "
                "but class object does not."
            )
            return False
        is_overload = False
        is_classmethod = False
        is_staticmethod = False
        if len(node.decorators) == 1:
            (dec,) = node.decorators
            if isinstance(dec.decorator, cst.Name):
                is_overload = dec.decorator.value == "overload"
                is_classmethod = dec.decorator.value == "classmethod"
                is_staticmethod = dec.decorator.value == "staticmethod"

        self.seen.add(name)
        attr = self.members[name]
        isgetset = inspect.isgetsetdescriptor(attr.object)
        if isgetset:
            if len(node.decorators) != 1:
                warnings.warn(
                    f"{self.obj_name}: Attribute '{name}' is a property, "
                    "but type stub doesn't define it as such"
                )
            return False
        elif node.decorators:
            if len(node.decorators) and not (
                is_classmethod or is_staticmethod or is_overload
            ):
                warnings.warn(
                    f"{self.obj_name}: Type stub defines '{name}' with decorator, "
                    "but it is not a property"
                )
        if not inspect.isroutine(attr.object):
            warnings.warn(
                f"{self.obj_name}: Type stub defines '{name}' as a function, "
                "but it is not a function in the class object"
            )
            return False
        # TODO: Need to handle __init__ which doesn't have a useful
        # type signature that we can inspect.
        if name == "__init__":
            return False
        check_function(attr.object, node, f"{self.obj_name}.{name}")
        return False

    def _check_assignment(self, name: str) -> None:
        if name not in self.members:
            warnings.warn(
                f"{self.obj_name}: Type stub defines '{name}' but class does not"
            )
            return
        attr = self.members[name]
        self.seen.add(name)
        if attr.kind != "data":
            warnings.warn(
                f"{self.obj_name}: Type stub defines '{name}' as attribute, but it is not data"
            )

    def visit_Assign(self, node: cst.Assign) -> bool:
        names = m.findall(node, m.Name())
        if len(names) != 1:
            warnings.warn(
                f"{self.obj_name}: Expected only a single name in assignment"
            )
            return False
        (name,) = names
        assert isinstance(name, cst.Name)
        self._check_assignment(name.value)
        return False

    def visit_AnnAssign(self, node: cst.AnnAssign) -> bool:
        assert isinstance(node.target, cst.Name)
        self._check_assignment(node.target.value)
        return False


def check_function(fn: Callable, info: cst.FunctionDef, fn_name: str):
    """
    Check a function in a type stub.

    Parameters
    ----------
    fn
        Runtime function object to check against.
    info
        CST node of the function definition
    fn_name
        Fully-qualified name, for error reporting.

    Warns
    -----
    UserWarning
        If the checker finds any discrepancies.

    Notes
    -----
    This tries to match up function signature names, but cannot do so
    for overloaded functions with fused cython types, or classes with
    `__cinit__`.
    """
    # Not handling overloaded functions right now
    if len(info.decorators) == 1:
        (dec,) = info.decorators
        if (
            isinstance(dec.decorator, cst.Name)
            and dec.decorator.value == "overload"
        ):
            return False
    sig = inspect.signature(fn)
    all_params = list(
        itertools.chain(
            info.params.posonly_params,
            info.params.params,
            info.params.kwonly_params,
        )
    )
    if (expect := len(sig.parameters)) != (actual := len(all_params)):
        warnings.warn(
            f"{fn_name}: Type stub has '{actual}' parameter(s), expected '{expect}'"
        )
    for (pname, _), param in zip(
        sig.parameters.items(), all_params, strict=False
    ):
        if pname != param.name.value:
            warnings.warn(
                f"{fn_name}: Parameter '{pname}' is called '{param.name.value}' in type stub"
            )


class EnumChecker(cst.CSTVisitor):
    def __init__(self, cls: type[enum.IntEnum], obj_name: str):
        self.cls = cls
        self.members = set(cls.__members__)
        self.seen = set()
        self.obj_name = obj_name

    @classmethod
    def check(
        cls,
        typ: type[enum.IntEnum],
        info: cst.BaseSuite,
        mod: types.ModuleType,
    ) -> None:
        """
        Check an enum definition in a type stub.

        Parameters
        ----------
        cls
            Runtime enum to check against.
        info
            CST node representing the body of the class.
        mod
            Module class is defined in, for error reporting.

        Warns
        -----
        UserWarning
            If the checker finds any discrepancies.
        """
        assert issubclass(typ, enum.IntEnum)
        checker = cls(typ, f"{mod.__name__}.{typ.__name__}")
        _ = info.visit(checker)
        checker.validate()

    def validate(self):
        unseen = sorted(self.members - self.seen)
        if unseen:
            missing = ", ".join(unseen)
            warnings.warn(
                f"{self.obj_name}: Type stub did not define the "
                f"following enum attributes: '{missing}'"
            )

    def visit_Assign(self, node: "cst.Assign") -> bool:
        attr: Mapping[str, cst.Name] = m.extract(
            node,
            m.Assign(
                targets=[
                    m.AssignTarget(target=m.SaveMatchedNode(m.Name(), "name"))
                ],
                value=m.Ellipsis(),
            ),
        )
        if not attr:
            warnings.warn(
                f"{self.obj_name}: Expected enum attribute with pattern 'NAME = ...'"
            )
            return False

        attr_name = attr["name"].value
        if attr_name not in self.members:
            warnings.warn(
                f"{self.obj_name}: Type stub defines '{attr_name}' but class does not"
            )
            return False
        self.seen.add(attr_name)
        return False


if __name__ == "__main__":
    import pylibcudf

    installed_location = Path(pylibcudf.aggregation.__file__).parent
    source_location = Path(pylibcudf.__file__).parent

    with warnings.catch_warnings(record=True) as w:
        for module in traverse_modules(pylibcudf):
            check_module(module, installed_location, source_location)
    for msg in w:
        warnings.showwarning(
            msg.message, msg.category, msg.filename, msg.lineno
        )
    if w:
        sys.exit(1)
