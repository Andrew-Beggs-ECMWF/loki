r"""
Collection of classes to represent basic and complex types. The key ideas are:
 * Expression symbols (eg. `Scalar`, `Array` or `Literal` has a
   `SymbolType` accessible via the `.type` attribute. This encodes the
   type definition in the local scope.
 * Each symbol type can represent either a `BasicType`, which may be `DEFERRED`, or
   a `DerivedType`. This is generally accessible as `symbol.type.dtype`.
   TODO: A `ProcedureType` may be added soon.
 * Each 'scope' object (eg. `Subroutine` or `Module` uses `TypeTable` objects to
   map symbol instances to types and derived type definitions.

           symbols.Variable  ---                      Subroutine | Module | TypeDef
                                 \                            \      |      /
                                  \                            \     |     /
                                   SymbolType    ------------   SymbolTable
                                 /     |      \
                                /      |       \
                       BasicType   DerivedType  ProcedureType

A note on scoping:
==================
When importing `TypeDef` objects into a local scope, a `DerivedType` object
will act as a wrapper in the `symbol.type.dtype` attribute. Importantly, when
variable instances based on this get created, the `DerivedType` object will
re-create all member variable of the object in the local scope, which are then
accessible via `symbol.type.dtype.variables`. If the original member declaration
variables are required, these can be accessed via `symbol.type.dtype.typedef.variables`.
"""

import weakref
from enum import IntEnum
from collections import OrderedDict
from loki.tools import flatten, as_tuple


__all__ = ['BasicType', 'DerivedType', 'ProcedureType', 'SymbolType', 'TypeTable', 'Scope']


class BasicType(IntEnum):
    """
    Representation of intrinsic data types, names taken from the FORTRAN convention.

    Currently, there are
    - `LOGICAL`
    - `INTEGER`
    - `REAL`
    - `CHARACTER`
    - `COMPLEX`
    and, to mark symbols without a known type, `DEFERRED` (e.g., for members of an externally
    defined derived type on use).

    For convenience, string representations of FORTRAN and C99 types can be
    heuristically converted.
    """

    DEFERRED = -1
    LOGICAL = 1
    INTEGER = 2
    REAL = 3
    CHARACTER = 4
    COMPLEX = 5

    @classmethod
    def from_str(cls, value):
        """
        Try to convert the given string using any of the `from_*_type` methods.
        """
        lookup_methods = (cls.from_fortran_type, cls.from_c99_type)
        for meth in lookup_methods:
            try:
                return meth(value)
            except KeyError:
                pass
        return ValueError('Unknown data type: %s' % value)

    @classmethod
    def from_fortran_type(cls, value):
        """
        Convert the given string representation of a FORTRAN type.
        """
        type_map = {'logical': cls.LOGICAL, 'integer': cls.INTEGER, 'real': cls.REAL,
                    'double precision': cls.REAL, 'double complex': cls.COMPLEX,
                    'character': cls.CHARACTER, 'complex': cls.COMPLEX}
        return type_map[value.lower()]

    @classmethod
    def from_c99_type(cls, value):
        """
        Convert the given string representation of a C99 type.
        """
        logical_types = ['bool', '_Bool']
        integer_types = ['short', 'int', 'long', 'long long']
        integer_types += flatten([('signed %s' % t, 'unsigned %s' % t) for t in integer_types])
        real_types = ['float', 'double', 'long double']
        character_types = ['char']
        complex_types = ['float _Complex', 'double _Complex', 'long double _Complex']

        type_map = {t: cls.LOGICAL for t in logical_types}
        type_map.update({t: cls.INTEGER for t in integer_types})
        type_map.update({t: cls.REAL for t in real_types})
        type_map.update({t: cls.CHARACTER for t in character_types})
        type_map.update({t: cls.COMPLEX for t in complex_types})

        return type_map[value]


class DerivedType:
    """
    Representation of a complex derived data types that may have an associated `TypeDef`.

    Please note that the typedef attribute may be of `ir.TypeDef` or `BasicType.DEFERRED`,
    depending on the scope of the derived type declaration.
    """

    def __init__(self, name=None, typedef=None):
        assert name or typedef
        self._name = name
        self.typedef = typedef if typedef is not None else BasicType.DEFERRED

        # This is intentionally left blank, as the parent variable
        # generation will populate this, if the typedef is known.
        self.variables = tuple()

    @property
    def name(self):
        return self._name if self.typedef is BasicType.DEFERRED else self.typedef.name

    @property
    def variable_map(self):
        return OrderedDict([(v.basename, v) for v in self.variables])

    def __str__(self):
        return self.name


class ProcedureType:
    """
    Representation of a function or subroutine type definition.
    """

    def __init__(self, name=None, is_function=False, procedure=None):
        assert name or procedure
        self._name = name
        self._is_function = is_function
        self.procedure = procedure if procedure is not None else BasicType.DEFERRED

    @property
    def name(self):
        return self._name if self.procedure is BasicType.DEFERRED else self.procedure.name

    @property
    def parameters(self):
        if self.procedure is BasicType.DEFERRED:
            return tuple()
        return self.procedure.arguments

    @property
    def is_function(self):
        if self.procedure is BasicType.DEFERRED:
            return self._is_function
        return self.procedure.is_function

    def __str__(self):
        return self.name


class SymbolType:
    """
    Representation of a symbols type.

    It has a fixed class:``BasicType`` associated, available as the property `BasicType.dtype`.

    Any other properties can be attached on-the-fly, thus allowing to store arbitrary metadata
    for a symbol, e.g., declaration attributes such as `POINTER`, `ALLOCATABLE` or structural
    information, e.g., whether a variable is a loop index, argument, etc.

    There is no need to check for the presence of attributes, undefined attributes can be queried
    and default to `None`.
    """

    def __init__(self, dtype, **kwargs):
        if isinstance(dtype, (BasicType, DerivedType, ProcedureType)):
            self.dtype = dtype
        else:
            self.dtype = BasicType.from_str(dtype)

        for k, v in kwargs.items():
            if v is not None:
                self.__setattr__(k, v)

    def __setattr__(self, name, value):
        if value is None and name in dir(self):
            delattr(self, name)
        else:
            object.__setattr__(self, name, value)

    def __getattr__(self, name):
        if name not in dir(self):
            return None
        return object.__getattribute__(self, name)

    def __delattr__(self, name):
        object.__delattr__(self, name)

    def __repr__(self):
        parameters = [str(self.dtype)]
        for k, v in self.__dict__.items():
            if k in ['dtype', 'source']:
                continue
            if isinstance(v, bool):
                if v:
                    parameters += [str(k)]
            elif k == 'parent' and v is not None:
                parameters += ['parent=%s(%s)' % ('Type' if isinstance(v, SymbolType)
                                                  else 'Variable', v.name)]
            else:
                parameters += ['%s=%s' % (k, str(v))]
        return '<Type %s>' % ', '.join(parameters)

    def __getinitargs__(self):
        args = [self.dtype]
        for k, v in self.__dict__.items():
            if k in ['dtype', 'source']:
                continue
            args += [(k, v)]
        return tuple(args)

    def clone(self, **kwargs):
        args = self.__dict__.copy()
        args.update(kwargs)
        dtype = args.pop('dtype')
        return self.__class__(dtype, **args)

    def compare(self, other, ignore=None):
        """
        Compare `SymbolType` objects while ignoring a set of select attributes.
        """
        ignore_attrs = as_tuple(ignore)
        keys = set(as_tuple(self.__dict__.keys()) + as_tuple(self.__dict__.keys()))
        return all(self.__dict__[k] == other.__dict__[k]
                   for k in keys if k not in ignore_attrs)


class TypeTable(dict):
    """
    Lookup table for types that essentially behaves like a class:``dict``.

    Used to store types for symbols or derived types within a scope.
    For derived types, no separate entries for the declared variables within a type
    are added. Instead, lookup methods (such as ``get``, ``__getitem__``, ``lookup`` etc.)
    disect the name and take care of chasing the information chain automagically.

    :param parent: class:``TypeTable`` instance of the parent scope to allow
                   for recursive lookups.
    :param case_sensitive: Treat names of variables to be case sensitive.
    """

    def __init__(self, parent=None, case_sensitive=False, **kwargs):
        super().__init__(**kwargs)
        self._parent = weakref.ref(parent) if parent is not None else None
        self._case_sensitive = case_sensitive

    @property
    def parent(self):
        return self._parent() if self._parent is not None else None

    def format_lookup_name(self, name):
        if not self._case_sensitive:
            name = name.lower()
        name = name.partition('(')[0]  # Remove any dimension parameters
        return name

    def _lookup(self, name, recursive):
        """
        Recursively look for a symbol in the table.
        """
        value = super().get(name, None)
        if value is None and recursive and self.parent is not None:
            return self.parent._lookup(name, recursive)
        return value

    def lookup(self, name, recursive=True):
        """
        Lookup a symbol in the type table and return the type or `None` if not found.

        :param name: Name of the type or symbol.
        :param recursive: If no entry by that name is found, try to find it in the
                          table of the parent scope.
        """
        name_parts = self.format_lookup_name(name)
        value = self._lookup(name_parts, recursive)
        return value

    def __contains__(self, key):
        return super().__contains__(self.format_lookup_name(key))

    def __getitem__(self, key):
        value = self.lookup(key, recursive=False)
        if value is None:
            raise KeyError(key)
        return value

    def get(self, key, default=None):
        value = self.lookup(key, recursive=False)
        return value if value is not None else default

    def __setitem__(self, key, value):
        name_parts = self.format_lookup_name(key)
        super().__setitem__(name_parts, value)

    def __hash__(self):
        return hash(tuple(self.keys()))

    def __repr__(self):
        return '<loki.types.TypeTable object at %s>' % hex(id(self))

    def setdefault(self, key, default=None):
        super().setdefault(self.format_lookup_name(key), default)


class Scope:
    """
    Scoping object that manages type caching and derivation for typed symbols.

    The ``Scope`` provides two key tables:
     * ``scope.symbols`` uniquely maps each variables name to a ``SymbolType``
     * ``scope.types`` uniquely maps derived and procedure type names to
       their respective data type objects and definitions.

    Note that derived and procedure type definitions may be markes as
    ``BasicType.DEFERRED``, in which case the ``Scope`` may be able to
    map them to concrete definitions at a later stage.
    """

    def __init__(self, parent=None):
        self._parent = weakref.ref(parent) if parent is not None else None

        parent_symbols = self.parent.symbols if self.parent is not None else None
        self.symbols = TypeTable(parent=parent_symbols)

        parent_types = self.parent.types if self.parent is not None else None
        self.types = TypeTable(parent=parent_types)

        # Potential link-back to the owner that can be used to
        # traverse the dependency chain upwards.
        self._defined_by = None

    @property
    def parent(self):
        """
        Access the enclosing scope.
        """
        return self._parent() if self._parent is not None else None

    @property
    def defined_by(self):
        """
        Object that owns this `Scope` and defines the types and symbols it connects
        """
        return self._defined_by() if self._defined_by is not None else None

    @defined_by.setter
    def defined_by(self, value):
        """
        Ensure we only ever store a weakref to the defining object.
        """
        self._defined_by = weakref.ref(value)
