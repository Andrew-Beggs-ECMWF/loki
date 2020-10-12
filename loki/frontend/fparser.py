from collections import OrderedDict
import re

from fparser.two.parser import ParserFactory
from fparser.two.utils import get_child, BlockBase
try:
    from fparser.two.utils import walk
except ImportError:
    from fparser.two.utils import walk_ast as walk
from fparser.two import Fortran2003
from fparser.common.readfortran import FortranStringReader

from loki.visitors import GenericVisitor
from loki.frontend.source import Source
from loki.frontend.preprocessing import blacklist
from loki.frontend.util import (
    inline_comments, cluster_comments, inline_pragmas,
    process_dimension_pragmas, read_file, import_external_symbols, FP
)
import loki.ir as ir
import loki.expression.symbol_types as sym
from loki.expression.operations import (
    StringConcat, ParenthesisedAdd, ParenthesisedMul, ParenthesisedPow)
from loki.expression import ExpressionDimensionsMapper
from loki.logging import DEBUG
from loki.tools import timeit, as_tuple, flatten, CaseInsensitiveDict
from loki.types import DataType, DerivedType, SymbolType


__all__ = ['FParser2IR', 'parse_fparser_file', 'parse_fparser_source', 'parse_fparser_ast']


@timeit(log_level=DEBUG)
def parse_fparser_file(filename):
    """
    Generate an internal IR from file via the fparser AST.
    """
    fcode = read_file(filename)
    return parse_fparser_source(source=fcode)


@timeit(log_level=DEBUG)
def parse_fparser_source(source):
    reader = FortranStringReader(source, ignore_comments=False)
    f2008_parser = ParserFactory().create(std='f2008')

    return f2008_parser(reader)


@timeit(log_level=DEBUG)
def parse_fparser_ast(ast, raw_source, pp_info=None, definitions=None, scope=None):
    """
    Generate an internal IR from file via the fparser AST.
    """

    # Parse the raw FParser language AST into our internal IR
    _ir = FParser2IR(raw_source=raw_source, definitions=definitions, scope=scope).visit(ast)

    # Apply postprocessing rules to re-insert information lost during preprocessing
    if pp_info is not None:
        for r_name, rule in blacklist[FP].items():
            info = pp_info.get(r_name, None)
            _ir = rule.postprocess(_ir, info)

    # Perform some minor sanitation tasks
    _ir = inline_comments(_ir)
    _ir = inline_pragmas(_ir)
    _ir = process_dimension_pragmas(_ir)
    _ir = cluster_comments(_ir)

    return _ir


def node_sublist(nodelist, starttype, endtype):
    """
    Extract a subset of nodes from a list that sits between marked
    start and end nodes.
    """
    sublist = []
    active = False
    for node in nodelist:
        if isinstance(node, endtype):
            active = False

        if active:
            sublist += [node]

        if isinstance(node, starttype):
            active = True
    return sublist


def rget_child(node, node_type):
    """
    Searches for the last, immediate child of the supplied node that is of
    the specified type.

    :param node: the node whose children will be searched.
    :type node: :py:class:`fparser.two.utils.Base`
    :param node_type: the class(es) of child node to search for.
    :type node_type: type or tuple of type

    :returns: the last child node of type node_type that is encountered or None.
    :rtype: py:class:`fparser.two.utils.Base`

    """
    for child in reversed(node.children):
        if isinstance(child, node_type):
            return child
    return None


def extract_fparser_source(node, raw_source):
    """
    Extract the py:class:`Source` object for any py:class:`fparser.two.utils.BlockBase`
    from the raw source string.
    """
    assert isinstance(node, BlockBase)
    if node.item is not None:
        lines = node.item.span
    else:
        start_type = getattr(Fortran2003, node.use_names[0], None)
        if start_type is None:
            # If we don't have any starting point we have to bail out
            return None
        start_node = get_child(node, start_type)
        end_node = node.children[-1]
        if any(i is None or i.item is None for i in [start_node, end_node]):
            # If we don't have source information for start/end we have to bail out
            return None
        lines = (start_node.item.span[0], end_node.item.span[1])
    string = None
    if raw_source is not None:
        string = ''.join(raw_source.splitlines(keepends=True)[lines[0]-1:lines[1]])
    return Source(lines, string=string)


class FParser2IR(GenericVisitor):
    # pylint: disable=no-self-use  # Stop warnings about visitor methods that could do without self
    # pylint: disable=unused-argument  # Stop warnings about unused arguments

    def __init__(self, raw_source, definitions=None, scope=None):
        super(FParser2IR, self).__init__()
        self.raw_source = raw_source.splitlines(keepends=True)
        self.definitions = CaseInsensitiveDict((d.name, d) for d in as_tuple(definitions))
        self.scope = scope

    def get_source(self, o, source):
        """
        Helper method that builds the source object for the node.
        """
        if not isinstance(o, str) and o.item is not None:
            lines = (o.item.span[0], o.item.span[1])
            string = ''.join(self.raw_source[lines[0] - 1:lines[1]]).strip('\n')
            source = Source(lines=lines, string=string)
        return source

    def get_label(self, o):
        """
        Helper method that returns the label of the node.
        """
        if not isinstance(o, str) and o.item is not None:
            return getattr(o.item, 'label', None)
        return None

    def visit(self, o, **kwargs):  # pylint: disable=arguments-differ
        """
        Generic dispatch method that tries to generate meta-data from source.
        """
        kwargs['source'] = self.get_source(o, kwargs.get('source'))
        kwargs['label'] = self.get_label(o)
        return super(FParser2IR, self).visit(o, **kwargs)

    def visit_Base(self, o, **kwargs):
        """
        Universal default for ``Base`` FParser-AST nodes
        """
        children = tuple(self.visit(c, **kwargs) for c in o.items if c is not None)
        if len(children) == 1:
            return children[0]  # Flatten hierarchy if possible
        return children if len(children) > 0 else None

    def visit_BlockBase(self, o, **kwargs):
        """
        Universal default for ``BlockBase`` FParser-AST nodes
        """
        children = tuple(self.visit(c, **kwargs) for c in o.content)
        children = tuple(c for c in children if c is not None)
        if len(children) == 1:
            return children[0]  # Flatten hierarchy if possible
        return children if len(children) > 0 else None

    def visit_List(self, o, **kwargs):
        """
        Universal routine for auto-generated *_List types in fparser.
        """
        return as_tuple(flatten(self.visit(i, **kwargs) for i in o.items))

    visit_Attr_Spec_List = visit_List
    visit_Component_Attr_Spec_List = visit_List
    visit_Entity_Decl_List = visit_List
    visit_Component_Decl_list = visit_List
    visit_Explicit_Shape_Spec_List = visit_List
    visit_Assumed_Shape_Spec_List = visit_List
    visit_Deferred_Shape_Spec_List = visit_List
    visit_Allocate_Shape_Spec_List = visit_List
    visit_Ac_Value_List = visit_List
    visit_Section_Subscript_List = visit_List

    def visit_Actual_Arg_Spec_List(self, o, **kwargs):
        """
        Needs special treatment to avoid flattening key-value-pair tuples.
        """
        return as_tuple(self.visit(i, **kwargs) for i in o.items)

    def visit_Name(self, o, **kwargs):
        # This one is evil, as it is used flat in expressions,
        # forcing us to generate ``Variable`` objects, and in
        # declarations, where none of the metadata is available
        # at this low level!
        vname = o.tostr()

        # Careful! Mind the many ways in which this can get called with
        # outside information (either in kwargs or maps stored on self).
        dimensions = kwargs.get('dimensions')
        external = kwargs.get('external')
        dtype = kwargs.get('dtype')
        parent = kwargs.get('parent')
        shape = kwargs.get('shape')
        initial = kwargs.get('initial')
        scope = kwargs.get('scope', self.scope)

        source = kwargs.get('source')
        if source:
            source = source.clone_with_string(o.string)

        if parent is not None:
            basename = vname
            vname = '%s%%%s' % (parent.name, vname)

        # Try to find the symbol in the symbol tables
        if dtype is None and scope is not None:
            dtype = self.scope.symbols.lookup(vname, recursive=True)

        # If a parent variable is given, try to infer type from the
        # derived type definition
        if parent is not None and dtype is None:
            if parent.type is not None and isinstance(parent.type.dtype, DerivedType):
                if parent.type.variables is not None and \
                        basename in parent.type.variables:
                    dtype = parent.type.variables[basename].type

        if shape is not None and dtype is not None and dtype.shape != shape:
            dtype = dtype.clone(shape=shape)

        if dimensions:
            dimensions = sym.ArraySubscript(dimensions)

        if external:
            if dtype is None:
                dtype = SymbolType(DataType.DEFERRED)
            dtype.external = external

        return sym.Variable(name=vname, dimensions=dimensions, type=dtype, scope=scope.symbols,
                            parent=parent, initial=initial, source=source)

    def visit_literal(self, o, _type, kind=None, **kwargs):
        source = kwargs.get('source')
        if source:
            source = source.clone_with_string(str(o.items[0]))
            val = source.string
        else:
            val = o.items[0]
        if kind is not None:
            return sym.Literal(value=val, type=_type, kind=kind, source=source)
        return sym.Literal(value=val, type=_type, source=source)

    def visit_Char_Literal_Constant(self, o, **kwargs):
        return self.visit_literal(o, DataType.CHARACTER, **kwargs)

    def visit_Int_Literal_Constant(self, o, **kwargs):
        kind = o.items[1] if o.items[1] is not None else None
        return self.visit_literal(o, DataType.INTEGER, kind=kind, **kwargs)

    visit_Signed_Int_Literal_Constant = visit_Int_Literal_Constant

    def visit_Real_Literal_Constant(self, o, **kwargs):
        kind = o.items[1] if o.items[1] is not None else None
        return self.visit_literal(o, DataType.REAL, kind=kind, **kwargs)

    visit_Signed_Real_Literal_Constant = visit_Real_Literal_Constant

    def visit_Logical_Literal_Constant(self, o, **kwargs):
        return self.visit_literal(o, DataType.LOGICAL, **kwargs)

    def visit_Complex_Literal_Constant(self, o, **kwargs):
        source = kwargs.get('source')
        if source:
            source = source.clone_with_string(o.string)
            val = source.string
        else:
            val = o.string
        return sym.IntrinsicLiteral(value=val, source=source)

    visit_Binary_Constant = visit_Complex_Literal_Constant
    visit_Octal_Constant = visit_Complex_Literal_Constant
    visit_Hex_Constant = visit_Complex_Literal_Constant

    def visit_Dimension_Attr_Spec(self, o, **kwargs):
        return self.visit(o.items[1], **kwargs)

    def visit_Component_Attr_Spec(self, o, **kwargs):
        return o.tostr()

    def visit_Intent_Attr_Spec(self, o, **kwargs):
        return o.tostr()

    def visit_Attr_Spec(self, o, **kwargs):
        return o.tostr()

    def visit_Specification_Part(self, o, **kwargs):
        children = tuple(self.visit(c, **kwargs) for c in o.content)
        children = tuple(c for c in children if c is not None)
        return list(children)

    def visit_Use_Stmt(self, o, **kwargs):
        name = o.items[2].tostr()
        only_list = get_child(o, Fortran2003.Only_List)  # pylint: disable=no-member
        symbols = None
        if only_list:
            symbol_names = tuple(item.tostr() for item in only_list.items)
            module = self.definitions.get(name, None)
            symbols = import_external_symbols(module=module, symbol_names=symbol_names, scope=self.scope)
        return ir.Import(module=name, symbols=symbols, source=kwargs.get('source'),
                         label=kwargs.get('label'))

    def visit_Include_Stmt(self, o, **kwargs):
        fname = o.items[0].tostr()
        return ir.Import(module=fname, f_include=True, source=kwargs.get('source'),
                         label=kwargs.get('label'))

    def visit_Implicit_Stmt(self, o, **kwargs):
        return ir.Intrinsic(text='IMPLICIT %s' % o.items[0], source=kwargs.get('source'),
                            label=kwargs.get('label'))

    def visit_Print_Stmt(self, o, **kwargs):
        return ir.Intrinsic(text='PRINT %s' % (', '.join(str(i) for i in o.items)),
                            source=kwargs.get('source'), label=kwargs.get('label'))

    # TODO: Deal with line-continuation pragmas!
    _re_pragma = re.compile(r'\!\$(?P<keyword>\w+)\s+(?P<content>.*)', re.IGNORECASE)

    def visit_Comment(self, o, **kwargs):
        source = kwargs.get('source', None)
        match_pragma = self._re_pragma.search(o.tostr())
        if match_pragma:
            # Found pragma, generate this instead
            gd = match_pragma.groupdict()
            return ir.Pragma(keyword=gd['keyword'], content=gd['content'], source=source)
        return ir.Comment(text=o.tostr(), source=source)

    def visit_Entity_Decl(self, o, **kwargs):
        # pylint: disable=no-member  # *_List are autogenerated and not found by pylint
        dims = get_child(o, Fortran2003.Explicit_Shape_Spec_List)
        dims = get_child(o, Fortran2003.Assumed_Shape_Spec_List) if dims is None else dims
        if dims is not None:
            kwargs['dimensions'] = self.visit(dims)

        init = get_child(o, Fortran2003.Initialization)
        if init is not None:
            kwargs['initial'] = self.visit(init)

        # We know that this is a declaration, so the ``dimensions``
        # here also define the shape of the variable symbol within the
        # currently cached context.
        kwargs['shape'] = kwargs.get('dimensions', None)

        return self.visit(o.items[0], **kwargs)

    def visit_Component_Decl(self, o, **kwargs):
        # pylint: disable=no-member  # *_List are autogenerated and not found by pylint
        dims = get_child(o, Fortran2003.Explicit_Shape_Spec_List)
        dims = get_child(o, Fortran2003.Assumed_Shape_Spec_List) if dims is None else dims
        dims = get_child(o, Fortran2003.Deferred_Shape_Spec_List) if dims is None else dims
        if dims is not None:
            dims = self.visit(dims)
            # We know that this is a declaration, so the ``dimensions``
            # here also define the shape of the variable symbol within the
            # currently cached context.
            kwargs['dimensions'] = dims
            kwargs['shape'] = dims

        return self.visit(o.items[0], **kwargs)

    def visit_Subscript_Triplet(self, o, **kwargs):
        children = tuple(self.visit(i, **kwargs) if i is not None else None for i in o.items)
        source = kwargs.get('source')
        if source:
            source = source.clone_with_string(o.string)
        return sym.RangeIndex(children, source=source)

    visit_Assumed_Shape_Spec = visit_Subscript_Triplet
    visit_Deferred_Shape_Spec = visit_Subscript_Triplet

    def visit_Explicit_Shape_Spec(self, o, **kwargs):
        children = tuple(self.visit(i, **kwargs) if i is not None else None for i in o.items)
        if children[0] is None:
            return children[1]
        source = kwargs.get('source')
        if source:
            source = source.clone_with_string(o.string)
        return sym.RangeIndex(children, source=source)

    visit_Allocate_Shape_Spec = visit_Explicit_Shape_Spec

    def visit_Allocation(self, o, **kwargs):
        dimensions = self.visit(o.items[1])
        kwargs['dimensions'] = dimensions
        kwargs['shape'] = dimensions
        return self.visit(o.items[0], **kwargs)

    def visit_Allocate_Stmt(self, o, **kwargs):
        # pylint: disable=no-member  # *_List are autogenerated and not found by pylint
        kw_args = {arg.items[0].lower(): self.visit(arg.items[1], **kwargs)
                   for arg in walk(o, Fortran2003.Alloc_Opt)}
        allocations = get_child(o, Fortran2003.Allocation_List)
        variables = tuple(self.visit(a, **kwargs) for a in allocations.items)
        return ir.Allocation(variables=variables, source=kwargs.get('source'),
                             data_source=kw_args.get('source'), label=kwargs.get('label'))

    def visit_Deallocate_Stmt(self, o, **kwargs):
        # pylint: disable=no-member  # *_List are autogenerated and not found by pylint
        deallocations = get_child(o, Fortran2003.Allocate_Object_List)
        variables = tuple(self.visit(a, **kwargs) for a in deallocations.items)
        return ir.Deallocation(variables=variables, source=kwargs.get('source'),
                               label=kwargs.get('label'))

    def visit_Intrinsic_Type_Spec(self, o, **kwargs):
        dtype = o.items[0]
        kind = get_child(o, Fortran2003.Kind_Selector)
        if kind is not None:
            if kind.items[1].tostr().isnumeric():
                kind = sym.Literal(value=kind.items[1].tostr())
            else:
                kind = sym.Variable(name=kind.items[1].tostr(), scope=self.scope.symbols)
        length = get_child(o, Fortran2003.Length_Selector)
        if length is not None:
            length = length.items[1].tostr()
        return dtype, kind, length

    def visit_Intrinsic_Name(self, o, **kwargs):
        return o.tostr()

    def visit_Initialization(self, o, **kwargs):
        return self.visit(o.items[1], **kwargs)

    def visit_Array_Constructor(self, o, **kwargs):
        values = self.visit(o.items[1], **kwargs)
        source = kwargs.get('source')
        if source:
            source = source.clone_with_string(o.string)
        return sym.LiteralList(values=values, source=source)

    def visit_Ac_Implied_Do(self, o, **kwargs):
        # TODO: Implement this properly!
        return o.tostr()

    def visit_Intrinsic_Function_Reference(self, o, **kwargs):
        # pylint: disable=no-member  # *_List are autogenerated and not found by pylint
        # Do not recurse here to avoid treating function names as variables
        name = o.items[0].tostr()  # self.visit(o.items[0], **kwargs)
        source = kwargs.get('source')
        if source:
            source = source.clone_with_string(o.string)

        if name.upper() in ('REAL', 'INT'):
            args = walk(o.items, (Fortran2003.Actual_Arg_Spec_List,))[0]
            expr = self.visit(args.items[0])
            if len(args.items) > 1:
                # Do not recurse here to avoid treating kind names as variables
                kind = walk(o.items, (Fortran2003.Actual_Arg_Spec,))
                # If kind is not specified as named argument, simply take the second
                # argument and convert it to a string
                kind = kind[0].items[1].tostr() if kind else args.items[1].tostr()
            else:
                kind = None
            return sym.Cast(name, expr, kind=kind, source=source)

        args = self.visit(o.items[1], **kwargs) if o.items[1] else None
        if args:
            kwarguments = {a[0]: a[1] for a in args if isinstance(a, tuple)}
            arguments = tuple(a for a in args if not isinstance(a, tuple))
        else:
            arguments = None
            kwarguments = None
        return sym.InlineCall(name, parameters=arguments, kw_parameters=kwarguments, source=source)

    visit_Function_Reference = visit_Intrinsic_Function_Reference

    def visit_Actual_Arg_Spec(self, o, **kwargs):
        key = o.items[0].tostr()
        value = self.visit(o.items[1], **kwargs)
        return (key, value)

    def visit_Data_Ref(self, o, **kwargs):
        v = self.visit(o.items[0], source=kwargs.get('source'))
        for i in o.items[1:-1]:
            # Careful not to propagate type or dims here
            v = self.visit(i, parent=v, source=kwargs.get('source'))
        # Attach types and dims to final leaf variable
        return self.visit(o.items[-1], parent=v, **kwargs)

    def visit_Data_Pointer_Object(self, o, **kwargs):
        v = self.visit(o.items[0], source=kwargs.get('source'))
        for i in o.items[1:-1]:
            if i == '%':
                continue
            # Careful not to propagate type or dims here
            v = self.visit(i, parent=v, source=kwargs.get('source'))
        # Attach types and dims to final leaf variable
        return self.visit(o.items[-1], parent=v, **kwargs)

    def visit_Part_Ref(self, o, **kwargs):
        # WARNING: Due to fparser's lack of a symbol table, it is not always possible to
        # distinguish between array subscript and function call. This employs a heuristic
        # identifying only intrinsic function calls and calls with keyword parameters as
        # a function call.
        name = o.items[0].tostr()
        parent = kwargs.get('parent', None)
        if parent:
            name = '%s%%%s' % (parent, name)
        args = as_tuple(self.visit(o.items[1])) if o.items[1] else None
        if args:
            kwarguments = {a[0]: a[1] for a in args if isinstance(a, tuple)}
            arguments = as_tuple(a for a in args if not isinstance(a, tuple))
        else:
            arguments = None
            kwarguments = None

        if name.lower() in Fortran2003.Intrinsic_Name.function_names or kwarguments:
            source = kwargs.get('source')
            if source:
                source = source.clone_with_string(o.string)
            # This is (presumably) a function call
            return sym.InlineCall(name, parameters=arguments, kw_parameters=kwarguments,
                                  source=source)

        # This is an array access and the arguments define the dimension.
        kwargs['dimensions'] = args
        # Recurse down to visit_Name
        return self.visit(o.items[0], **kwargs)

    def visit_Structure_Constructor(self, o, **kwargs):
        # TODO: fparser wrongfully parses calls to functions without arguments as this type.
        # This means this routine also produces inline calls for actual inline calls...
        name = get_child(o, Fortran2003.Type_Name).tostr()
        component_specs = get_child(o, Fortran2003.Component_Spec_List)  # pylint: disable=no-member
        if component_specs:
            args = as_tuple(self.visit(component_specs, **kwargs))
            kwarguments = {a[0]: a[1] for a in args if isinstance(a, tuple)}
            arguments = as_tuple(a for a in args if not isinstance(a, tuple))
        else:
            arguments = None
            kwarguments = None
        return sym.InlineCall(name, parameters=arguments, kw_parameters=kwarguments,
                              source=kwargs.get('source'))

    def visit_Proc_Component_Ref(self, o, **kwargs):
        '''This is the compound object for accessing procedure components of a variable.'''
        pname = o.items[0].tostr().lower()
        v = sym.Variable(name=pname, scope=self.scope.symbols)
        for i in o.items[1:-1]:
            if i != '%':
                v = self.visit(i, parent=v, source=kwargs.get('source'))
        return self.visit(o.items[-1], parent=v, **kwargs)

    def visit_Array_Section(self, o, **kwargs):
        kwargs['dimensions'] = as_tuple(self.visit(o.items[1]))
        return self.visit(o.items[0], **kwargs)

    visit_Substring_Range = visit_Subscript_Triplet

    def visit_Type_Declaration_Stmt(self, o, **kwargs):
        """
        Declaration statement in the spec of a module/routine. This function is also called
        for declarations of members of a derived type.
        """
        # Super-hacky, this fecking DIMENSION keyword will be my undoing one day!
        dimensions = [self.visit(a, **kwargs)
                      for a in walk(o.items, (Fortran2003.Dimension_Component_Attr_Spec,
                                              Fortran2003.Dimension_Attr_Spec))]
        if dimensions:
            if isinstance(o, Fortran2003.Data_Component_Def_Stmt):
                dimensions = dimensions[0][1]
            else:
                dimensions = dimensions[0]
        else:
            dimensions = None

        # First, pick out parameters, including explicit DIMENSIONs and EXTERNAL
        attrs = as_tuple(str(self.visit(a)).lower().strip()
                         for a in walk(o.items, (
                             Fortran2003.Attr_Spec, Fortran2003.Component_Attr_Spec,
                             Fortran2003.Intent_Attr_Spec)))
        intent = None
        if 'intent(in)' in attrs:
            intent = 'in'
        elif 'intent(inout)' in attrs:
            intent = 'inout'
        elif 'intent(out)' in attrs:
            intent = 'out'

        external = 'external' in attrs

        # Next, figure out the type we're declaring
        dtype = None
        basetype_ast = get_child(o, Fortran2003.Intrinsic_Type_Spec)
        if basetype_ast is not None:
            dtype, kind, length = self.visit(basetype_ast)
            dtype = SymbolType(DataType.from_fortran_type(dtype), kind=kind, intent=intent,
                               parameter='parameter' in attrs, optional='optional' in attrs,
                               allocatable='allocatable' in attrs, pointer='pointer' in attrs,
                               contiguous='contiguous' in attrs, target='target' in attrs,
                               shape=dimensions, length=length)

        derived_type_ast = get_child(o, Fortran2003.Declaration_Type_Spec)
        if derived_type_ast is not None:
            typename = derived_type_ast.items[1].tostr().lower()
            dtype = self.scope.types.lookup(typename, recursive=True)
            if dtype is None:
                typedef = self.scope.symbols.lookup(typename, recursive=True)
                typedef = typedef if typedef is DataType.DEFERRED else typedef.typedef
                dtype = SymbolType(DerivedType(name=typename, typedef=typedef),
                                   intent=intent, allocatable='allocatable' in attrs,
                                   pointer='pointer' in attrs, optional='optional' in attrs,
                                   parameter='parameter' in attrs, target='target' in attrs,
                                   contiguous='contiguous' in attrs, shape=dimensions)
            else:
                # Ensure we inherit declaration attributes via a local clone
                dtype = dtype.clone(intent=intent, allocatable='allocatable' in attrs,
                                   pointer='pointer' in attrs, optional='optional' in attrs,
                                   parameter='parameter' in attrs, target='target' in attrs,
                                   contiguous='contiguous' in attrs, shape=dimensions)

        # Now create the actual variables declared in this statement
        # (and provide them with the type and dimension information)
        kwargs['dimensions'] = dimensions
        kwargs['external'] = external
        kwargs['dtype'] = dtype
        variables = as_tuple(self.visit(o.items[2], **kwargs))
        return ir.Declaration(variables=variables, dimensions=dimensions, external=external,
                              source=kwargs.get('source'), label=kwargs.get('label'))

    def visit_External_Stmt(self, o, **kwargs):
        # pylint: disable=no-member
        kwargs['external'] = True
        variables = as_tuple(self.visit(get_child(o, Fortran2003.External_Name_List), **kwargs))
        return ir.Declaration(variables=variables, external=True, source=kwargs.get('source'),
                              label=kwargs.get('label'))

    def visit_Derived_Type_Def(self, o, **kwargs):
        name = get_child(o, Fortran2003.Derived_Type_Stmt).items[1].tostr().lower()
        source = kwargs.get('source')
        # Create the typedef with all the information we have so far (we need its symbol table
        # for the next step)
        typedef = ir.TypeDef(name=name, body=[], source=source, label=kwargs.get('label'))
        # Create declarations and update the parent typedef
        component_nodes = (Fortran2003.Component_Part, Fortran2003.Comment)
        body = flatten([self.visit(i, scope=typedef, **kwargs)
                        for i in walk(o.content, component_nodes)])
        # Infer any additional shape information from `!$loki dimension` pragmas
        # Note that this needs to be done before we create `dtype` below, to allow
        # propagation of type info through multiple typedefs in the same module.
        body = inline_pragmas(body)
        body = process_dimension_pragmas(body)
        typedef._update(body=body, symbols=typedef.symbols)

        # Now create a SymbolType instance to make the typedef known in its scope's type table
        self.scope.types[name] = SymbolType(DerivedType(name=name, typedef=typedef))

        return typedef

    def visit_Component_Part(self, o, **kwargs):
        return as_tuple(flatten(self.visit(a, **kwargs) for a in o.content))

    # Declaration of members of a derived type (i.e., part of the definition of the derived type.
    visit_Data_Component_Def_Stmt = visit_Type_Declaration_Stmt

    def visit_Block_Nonlabel_Do_Construct(self, o, **kwargs):
        do_stmt_types = (Fortran2003.Nonlabel_Do_Stmt, Fortran2003.Label_Do_Stmt)
        # In the banter before the loop, Pragmas are hidden...
        banter = []
        for ch in o.content:
            if isinstance(ch, do_stmt_types):
                do_stmt = ch
                break
            banter += [self.visit(ch, **kwargs)]
        else:
            do_stmt = get_child(o, do_stmt_types)
        # Extract source by looking at everything between DO and END DO statements
        end_do_stmt = rget_child(o, Fortran2003.End_Do_Stmt)
        has_end_do = True
        if end_do_stmt is None:
            # We may have a labeled loop with an explicit CONTINUE statement
            has_end_do = False
            end_do_stmt = rget_child(o, Fortran2003.Continue_Stmt)
            assert str(end_do_stmt.item.label) == do_stmt.label.string
        lines = (do_stmt.item.span[0], end_do_stmt.item.span[1])
        string = ''.join(self.raw_source[lines[0]-1:lines[1]]).strip('\n')
        source = Source(lines=lines, string=string)
        label = self.get_label(do_stmt)
        construct_name = do_stmt.item.name
        # Extract loop header and get stepping info
        variable, bounds = self.visit(do_stmt, **kwargs)
        # Extract and process the loop body
        body_nodes = node_sublist(o.content, do_stmt.__class__, Fortran2003.End_Do_Stmt)
        body = as_tuple(flatten(self.visit(node, **kwargs) for node in body_nodes))
        # Loop label for labeled do constructs
        loop_label = str(do_stmt.items[1]) if isinstance(do_stmt, Fortran2003.Label_Do_Stmt) else None
        # Select loop type
        if bounds:
            obj = ir.Loop(variable=variable, body=body, bounds=bounds, loop_label=loop_label,
                          label=label, name=construct_name, has_end_do=has_end_do, source=source)
        else:
            obj = ir.WhileLoop(condition=variable, body=body, loop_label=loop_label,
                               label=label, name=construct_name, has_end_do=has_end_do, source=source)
        return (*banter, obj, )

    visit_Block_Label_Do_Construct = visit_Block_Nonlabel_Do_Construct

    def visit_Nonlabel_Do_Stmt(self, o, **kwargs):
        variable, bounds = None, None
        loop_control = get_child(o, Fortran2003.Loop_Control)
        if loop_control:
            variable, bounds = self.visit(loop_control, **kwargs)
        return variable, bounds

    visit_Label_Do_Stmt = visit_Nonlabel_Do_Stmt

    def visit_If_Construct(self, o, **kwargs):
        # The banter before the construct...
        banter = []
        for ch in o.content:
            if isinstance(ch, Fortran2003.If_Then_Stmt):
                if_then_stmt = ch
                break
            banter += [self.visit(ch, **kwargs)]
        else:
            if_then_stmt = get_child(o, Fortran2003.If_Then_Stmt)
        # Extract source by looking at everything between IF and END IF statements
        end_if_stmt = rget_child(o, Fortran2003.End_If_Stmt)
        lines = (if_then_stmt.item.span[0], end_if_stmt.item.span[1])
        string = ''.join(self.raw_source[lines[0]-1:lines[1]]).strip('\n')
        source = Source(lines=lines, string=string)
        construct_name = if_then_stmt.item.name
        label = self.get_label(if_then_stmt)
        # Start with the condition that is always there
        conditions = [self.visit(if_then_stmt, **kwargs)]
        # Walk throught the if construct and collect statements for the if branch
        # Pick up any ELSE IF along the way and collect their statements as well
        bodies = []
        body = []
        for child in node_sublist(o.content, Fortran2003.If_Then_Stmt, Fortran2003.Else_Stmt):
            if isinstance(child, Fortran2003.End_If_Stmt):
                # Skip this explicitly in case it has a construct name
                continue
            node = self.visit(child, **kwargs)
            if isinstance(child, Fortran2003.Else_If_Stmt):
                bodies.append(as_tuple(flatten(body)))
                body = []
                conditions.append(node)
            else:
                body.append(node)
        bodies.append(as_tuple(flatten(body)))
        assert len(conditions) == len(bodies)
        else_ast = node_sublist(o.content, Fortran2003.Else_Stmt, Fortran2003.End_If_Stmt)
        else_body = as_tuple(flatten(self.visit(a, **kwargs) for a in as_tuple(else_ast)))
        return (*banter, ir.Conditional(conditions=conditions, bodies=bodies, else_body=else_body,
                                        inline=False, label=label, name=construct_name, source=source))

    def visit_If_Then_Stmt(self, o, **kwargs):
        return self.visit(o.items[0], **kwargs)

    visit_Else_If_Stmt = visit_If_Then_Stmt

    def visit_If_Stmt(self, o, **kwargs):
        cond = as_tuple(self.visit(o.items[0], **kwargs))
        body = as_tuple(self.visit(o.items[1], **kwargs))
        return ir.Conditional(conditions=cond, bodies=body, else_body=(), inline=True,
                              label=kwargs.get('label'), source=kwargs.get('source'))

    def visit_Call_Stmt(self, o, **kwargs):
        name = o.items[0].tostr()
        args = self.visit(o.items[1], **kwargs) if o.items[1] else None
        if args:
            kw_args = tuple(arg for arg in args if isinstance(arg, tuple))
            args = tuple(arg for arg in args if not isinstance(arg, tuple))
        else:
            args = ()
            kw_args = ()
        return ir.CallStatement(name=name, arguments=args, kwarguments=kw_args,
                                label=kwargs.get('label'), source=kwargs.get('source'))

    def visit_Loop_Control(self, o, **kwargs):
        if o.items[0]:
            # Scalar logical expression
            return self.visit(o.items[0], **kwargs), None
        variable = self.visit(o.items[1][0], **kwargs)
        bounds = as_tuple(flatten(self.visit(a, **kwargs) for a in as_tuple(o.items[1][1])))
        source = kwargs.get('source')
        if source:
            variable_source = source.clone_with_string(o.string[:o.string.find('=')])
            variable = variable.clone(source=variable_source)
            source = source.clone_with_string(o.string[o.string.find('=')+1:])
        return variable, sym.LoopRange(bounds, source=source)

    def visit_Assignment_Stmt(self, o, **kwargs):
        ptr = isinstance(o, Fortran2003.Pointer_Assignment_Stmt)
        target = self.visit(o.items[0], **kwargs)
        expr = self.visit(o.items[2], **kwargs)
        return ir.Statement(target=target, expr=expr, ptr=ptr,
                            label=kwargs.get('label'), source=kwargs.get('source'))

    visit_Pointer_Assignment_Stmt = visit_Assignment_Stmt

    def create_operation(self, op, exprs, source):
        """
        Construct expressions from individual operations.
        """
        exprs = as_tuple(exprs)
        if op == '*':
            return sym.Product(exprs, source=source)
        if op == '/':
            return sym.Quotient(numerator=exprs[0], denominator=exprs[1], source=source)
        if op == '+':
            return sym.Sum(exprs, source=source)
        if op == '-':
            if len(exprs) > 1:
                # Binary minus
                return sym.Sum((exprs[0], sym.Product((-1, exprs[1]))), source=source)
            # Unary minus
            return sym.Product((-1, exprs[0]), source=source)
        if op == '**':
            return sym.Power(base=exprs[0], exponent=exprs[1], source=source)
        if op.lower() == '.and.':
            return sym.LogicalAnd(exprs, source=source)
        if op.lower() == '.or.':
            return sym.LogicalOr(exprs, source=source)
        if op.lower() in ('==', '.eq.'):
            return sym.Comparison(exprs[0], '==', exprs[1], source=source)
        if op.lower() in ('/=', '.ne.'):
            return sym.Comparison(exprs[0], '!=', exprs[1], source=source)
        if op.lower() in ('>', '.gt.'):
            return sym.Comparison(exprs[0], '>', exprs[1], source=source)
        if op.lower() in ('<', '.lt.'):
            return sym.Comparison(exprs[0], '<', exprs[1], source=source)
        if op.lower() in ('>=', '.ge.'):
            return sym.Comparison(exprs[0], '>=', exprs[1], source=source)
        if op.lower() in ('<=', '.le.'):
            return sym.Comparison(exprs[0], '<=', exprs[1], source=source)
        if op.lower() == '.not.':
            return sym.LogicalNot(exprs[0], source=source)
        if op.lower() == '.eqv.':
            return sym.LogicalOr((sym.LogicalAnd(exprs, source=source),
                                  sym.LogicalNot(sym.LogicalOr(exprs, source=source))), source=source)
        if op.lower() == '.neqv.':
            return sym.LogicalAnd((sym.LogicalNot(sym.LogicalAnd(exprs, source=source)),
                                   sym.LogicalOr(exprs, source=source)), source=source)
        if op == '//':
            return StringConcat(exprs, source=source)
        raise RuntimeError('FParser: Error parsing generic expression')

    def visit_Add_Operand(self, o, **kwargs):
        source = kwargs.get('source')
        if source:
            source = source.clone_with_string(o.string)
        if len(o.items) > 2:
            # Binary operand
            exprs = [self.visit(o.items[0], **kwargs)]
            exprs += [self.visit(o.items[2], **kwargs)]
            return self.create_operation(op=o.items[1], exprs=exprs, source=source)
        # Unary operand
        exprs = [self.visit(o.items[1], **kwargs)]
        return self.create_operation(op=o.items[0], exprs=exprs, source=source)

    visit_Mult_Operand = visit_Add_Operand
    visit_And_Operand = visit_Add_Operand
    visit_Or_Operand = visit_Add_Operand
    visit_Equiv_Operand = visit_Add_Operand

    def visit_Level_2_Expr(self, o, **kwargs):
        source = kwargs.get('source')
        if source:
            source = source.clone_with_string(o.string)
        e1 = self.visit(o.items[0], **kwargs)
        e2 = self.visit(o.items[2], **kwargs)
        return self.create_operation(op=o.items[1], exprs=(e1, e2), source=source)

    def visit_Level_2_Unary_Expr(self, o, **kwargs):
        source = kwargs.get('source')
        if source:
            source = source.clone_with_string(o.string)
        exprs = as_tuple(self.visit(o.items[1], **kwargs))
        return self.create_operation(op=o.items[0], exprs=exprs, source=source)

    visit_Level_3_Expr = visit_Level_2_Expr
    visit_Level_4_Expr = visit_Level_2_Expr
    visit_Level_5_Expr = visit_Level_2_Expr

    def visit_Parenthesis(self, o, **kwargs):
        source = kwargs.get('source')
        expression = self.visit(o.items[1], **kwargs)
        if source:
            source = source.clone_with_string(o.string)
        if isinstance(expression, sym.Sum):
            expression = ParenthesisedAdd(expression.children, source=source)
        if isinstance(expression, sym.Product):
            expression = ParenthesisedMul(expression.children, source=source)
        if isinstance(expression, sym.Power):
            expression = ParenthesisedPow(expression.base, expression.exponent, source=source)
        return expression

    def visit_Associate_Construct(self, o, **kwargs):
        children = [self.visit(c, **kwargs) for c in o.content]
        children = as_tuple(flatten(c for c in children if c is not None))
        # Search for the ASSOCIATE statement and add all following items as its body
        assoc_index = [isinstance(ch, ir.Scope) for ch in children].index(True)
        # Extract source for the entire scope
        lines = (children[assoc_index].source.lines[0], children[-1].source.lines[1])
        string = ''.join(self.raw_source[lines[0]-1:lines[1]]).strip('\n')
        children[assoc_index]._update(body=children[assoc_index + 1:],
                                      source=Source(lines=lines, string=string))
        return children[:assoc_index + 1]

    def visit_Associate_Stmt(self, o, **kwargs):
        associations = OrderedDict()
        for assoc in get_child(o, Fortran2003.Association_List).items:  # pylint: disable=no-member
            var = self.visit(assoc.items[2], **kwargs)
            if isinstance(var, sym.Array):
                shape = ExpressionDimensionsMapper()(var)
            else:
                shape = None
            dtype = var.type.clone(name=None, parent=None, shape=shape)
            associations[var] = self.visit(assoc.items[0], dtype=dtype, **kwargs)
        return ir.Scope(associations=associations, label=kwargs.get('label'),
                        source=kwargs.get('source'))

    def visit_Intrinsic_Stmt(self, o, **kwargs):
        return ir.Intrinsic(text=o.tostr(), label=kwargs.get('label'), source=kwargs.get('source'))

    visit_Format_Stmt = visit_Intrinsic_Stmt
    visit_Write_Stmt = visit_Intrinsic_Stmt
    visit_Goto_Stmt = visit_Intrinsic_Stmt
    visit_Return_Stmt = visit_Intrinsic_Stmt
    visit_Continue_Stmt = visit_Intrinsic_Stmt
    visit_Cycle_Stmt = visit_Intrinsic_Stmt
    visit_Exit_Stmt = visit_Intrinsic_Stmt
    visit_Save_Stmt = visit_Intrinsic_Stmt
    visit_Read_Stmt = visit_Intrinsic_Stmt
    visit_Open_Stmt = visit_Intrinsic_Stmt
    visit_Close_Stmt = visit_Intrinsic_Stmt
    visit_Inquire_Stmt = visit_Intrinsic_Stmt
    visit_Access_Stmt = visit_Intrinsic_Stmt
    visit_Namelist_Stmt = visit_Intrinsic_Stmt
    visit_Parameter_Stmt = visit_Intrinsic_Stmt
    visit_Dimension_Stmt = visit_Intrinsic_Stmt
    visit_Final_Binding = visit_Intrinsic_Stmt
    visit_Procedure_Stmt = visit_Intrinsic_Stmt
    visit_Equivalence_Stmt = visit_Intrinsic_Stmt
    visit_Common_Stmt = visit_Intrinsic_Stmt
    visit_Stop_Stmt = visit_Intrinsic_Stmt
    visit_Backspace_Stmt = visit_Intrinsic_Stmt
    visit_Rewind_Stmt = visit_Intrinsic_Stmt

    def visit_Cpp_If_Stmt(self, o, **kwargs):
        return ir.PreprocessorDirective(text=o.tostr(), source=kwargs.get('source'))

    visit_Cpp_Elif_Stmt = visit_Cpp_If_Stmt
    visit_Cpp_Else_Stmt = visit_Cpp_If_Stmt
    visit_Cpp_Endif_Stmt = visit_Cpp_If_Stmt
    visit_Cpp_Macro_Stmt = visit_Cpp_If_Stmt
    visit_Cpp_Undef_Stmt = visit_Cpp_If_Stmt
    visit_Cpp_Line_Stmt = visit_Cpp_If_Stmt
    visit_Cpp_Warning_Stmt = visit_Cpp_If_Stmt
    visit_Cpp_Error_Stmt = visit_Cpp_If_Stmt
    visit_Cpp_Null_Stmt = visit_Cpp_If_Stmt

    def visit_Cpp_Include_Stmt(self, o, **kwargs):
        fname = o.items[0].tostr()
        return ir.Import(module=fname, c_import=True, source=kwargs.get('source'))

    def visit_Where_Construct(self, o, **kwargs):
        # The banter before the construct...
        banter = []
        for ch in o.content:
            if isinstance(ch, Fortran2003.Where_Construct_Stmt):
                break
            banter += [self.visit(ch, **kwargs)]
        # The mask condition
        condition = self.visit(get_child(o, Fortran2003.Where_Construct_Stmt), **kwargs)
        default_ast = node_sublist(o.children, Fortran2003.Elsewhere_Stmt,
                                   Fortran2003.End_Where_Stmt)
        if default_ast:
            body_ast = node_sublist(o.children, Fortran2003.Where_Construct_Stmt,
                                    Fortran2003.Elsewhere_Stmt)
        else:
            body_ast = node_sublist(o.children, Fortran2003.Where_Construct_Stmt,
                                    Fortran2003.End_Where_Stmt)
        body = as_tuple(self.visit(ch, **kwargs) for ch in body_ast)
        default = as_tuple(self.visit(ch, **kwargs) for ch in default_ast)
        return (*banter, ir.MaskedStatement(condition, body, default, label=kwargs.get('label'),
                                            source=kwargs.get('source')))

    def visit_Where_Construct_Stmt(self, o, **kwargs):
        return self.visit(o.items[0], **kwargs)

    def visit_Where_Stmt(self, o, **kwargs):
        condition = self.visit(o.items[0], **kwargs)
        body = as_tuple(self.visit(o.items[1], **kwargs))
        default = ()
        return ir.MaskedStatement(condition, body, default, label=kwargs.get('label'),
                                  source=kwargs.get('source'))

    def visit_Case_Construct(self, o, **kwargs):
        # The banter before the construct...
        banter = []
        for ch in o.content:
            if isinstance(ch, Fortran2003.Select_Case_Stmt):
                select_stmt = ch
                break
            banter += [self.visit(ch, **kwargs)]
        else:
            select_stmt = get_child(o, Fortran2003.Select_Case_Stmt)
        # Extract source by looking at everything between SELECT and END SELECT
        end_select_stmt = rget_child(o, Fortran2003.End_Select_Stmt)
        lines = (select_stmt.item.span[0], end_select_stmt.item.span[1])
        string = ''.join(self.raw_source[lines[0]-1:lines[1]]).strip('\n')
        source = Source(lines=lines, string=string)
        label = self.get_label(select_stmt)
        construct_name = select_stmt.item.name

        # The SELECT argument
        expr = self.visit(select_stmt, **kwargs)
        body_ast = node_sublist(o.children, Fortran2003.Select_Case_Stmt,
                                Fortran2003.End_Select_Stmt)
        values = []
        bodies = []
        body = []
        is_else_body = False
        else_body = ()
        for child in body_ast:
            node = self.visit(child, **kwargs)
            if isinstance(child, Fortran2003.Case_Stmt):
                if is_else_body:
                    else_body = as_tuple(body)
                    is_else_body = False
                elif values:  # Avoid appending empty body before first Case_Stmt
                    bodies.append(as_tuple(body))
                body = []
                if node is None:  # default case
                    is_else_body = True
                else:
                    values.append(node)
            else:
                body.append(node)
        if is_else_body:
            else_body = body
        else:
            bodies.append(as_tuple(body))
        assert len(values) == len(bodies)
        return (*banter, ir.MultiConditional(expr, values, bodies, else_body, label=label,
                                             name=construct_name, source=source))

    def visit_Select_Case_Stmt(self, o, **kwargs):
        return self.visit(o.items[0], **kwargs)

    def visit_Case_Stmt(self, o, **kwargs):
        return self.visit(o.items[0], **kwargs)

    visit_Case_Value_Range = visit_Subscript_Triplet

    def visit_Select_Type_Construct(self, o, **kwargs):
        # The banter before the construct...
        banter = []
        for ch in o.content:
            if isinstance(ch, Fortran2003.Select_Type_Stmt):
                select_stmt = ch
                break
            banter += [self.visit(ch, **kwargs)]
        else:
            select_stmt = get_child(o, Fortran2003.Select_Type_Stmt)
        # Extract source by looking at everything between SELECT and END SELECT
        end_select_stmt = rget_child(o, Fortran2003.End_Select_Type_Stmt)
        lines = (select_stmt.item.span[0], end_select_stmt.item.span[1])
        string = ''.join(self.raw_source[lines[0]-1:lines[1]]).strip('\n')
        source = Source(lines=lines, string=string)
        label = self.get_label(select_stmt)
        # TODO: Treat this with a dedicated IR node (LOKI-33)
        return (*banter, ir.Intrinsic(text=string, label=label, source=source))

    def visit_Data_Stmt_Set(self, o, **kwargs):
        # TODO: actually parse the statements
        # pylint: disable=no-member
        variable = self.visit(get_child(o, Fortran2003.Data_Stmt_Object_List), **kwargs)
        values = as_tuple(self.visit(get_child(o, Fortran2003.Data_Stmt_Value_List), **kwargs))
        return ir.DataDeclaration(variable=variable, values=values, label=kwargs.get('label'),
                                  source=kwargs.get('source'))

    def visit_Data_Stmt_Value(self, o, **kwargs):
        exprs = as_tuple(flatten(self.visit(c) for c in o.items))
        return self.create_operation('*', exprs, source=kwargs.get('source'))

    def visit_Nullify_Stmt(self, o, **kwargs):
        if not o.items[1]:
            return ()
        variables = as_tuple(flatten(self.visit(v, **kwargs) for v in o.items[1].items))
        return ir.Nullify(variables=variables, label=kwargs.get('label'),
                          source=kwargs.get('source'))

    def visit_Interface_Block(self, o, **kwargs):
        spec = get_child(o, Fortran2003.Interface_Stmt).items[0]
        if spec:
            spec = spec if isinstance(spec, str) else spec.tostr()
        body_ast = node_sublist(o.children, Fortran2003.Interface_Stmt,
                                Fortran2003.End_Interface_Stmt)
        body = as_tuple(flatten(self.visit(ch, **kwargs) for ch in body_ast))
        return ir.Interface(spec=spec, body=body, label=kwargs.get('label'),
                            source=kwargs.get('source'))

    def visit_Function_Stmt(self, o, **kwargs):
        # TODO: not implemented
        return ir.Intrinsic(text=o.tostr(), label=kwargs.get('label'), source=kwargs.get('source'))

    visit_Subroutine_Stmt = visit_Function_Stmt
