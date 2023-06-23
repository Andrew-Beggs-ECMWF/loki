# (C) Copyright 2018- ECMWF.
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""
Single-Column-Coalesced CUDA Fortran (SCC-CUF) transformation.
"""

from loki.expression import symbols as sym
from loki.transform import resolve_associates, single_variable_declaration, HoistVariablesTransformation
from loki import ir
from loki import (
    Transformation, FindNodes, FindVariables, Transformer,
    SubstituteExpressions, SymbolAttributes, pragmas_attached,
    CaseInsensitiveDict, as_tuple, flatten, types, SubroutineItem
)

from transformations.single_column_coalesced import SCCBaseTransformation
from transformations.single_column_coalesced_vector import SCCDevectorTransformation

__all__ = ['SccCufTransformation', 'HoistTemporaryArraysDeviceAllocatableTransformation']


class HoistTemporaryArraysDeviceAllocatableTransformation(HoistVariablesTransformation):
    """
    Synthesis part for variable/array hoisting for CUDA Fortran (CUF) (transformation).
    """

    def driver_variable_declaration(self, routine, var):
        """
        CUDA Fortran (CUF) Variable/Array device declaration including
        allocation and de-allocation.

        Parameters
        ----------
        routine: :any:`Subroutine`
            The subroutine to add the variable declaration
        var: :any:`Variable`
            The variable to be declared
        """
        vtype = var.type.clone(device=True, allocatable=True)
        routine.variables += tuple([var.clone(scope=routine, dimensions=as_tuple(
            [sym.RangeIndex((None, None))] * (len(var.dimensions))), type=vtype)])

        allocations = FindNodes(ir.Allocation).visit(routine.body)
        if allocations:
            insert_index = routine.body.body.index(allocations[-1])
            routine.body.insert(insert_index + 1, ir.Allocation((var.clone(),)))
        else:
            routine.body.prepend(ir.Allocation((var.clone(),)))
        de_allocations = FindNodes(ir.Deallocation).visit(routine.body)
        if de_allocations:
            insert_index = routine.body.body.index(de_allocations[-1])
            routine.body.insert(insert_index + 1, ir.Deallocation((var.clone(dimensions=None),)))
        else:
            routine.body.append(ir.Deallocation((var.clone(dimensions=None),)))


def dynamic_local_arrays(routine, vertical):
    """
    Declaring local arrays with the ``vertical`` :any:`Dimension` to be
    dynamically allocated.

    .. warning :: depends on single/unique variable declarations

    Parameters
    ----------
    routine: :any:`Subroutine`
        The subroutine to dynamically allocate the local arrays
    vertical: :any:`Dimension`
        The dimension specifying the horizontal vector dimension
    """
    local_arrays = []
    argnames = routine.arguments # [name.upper() for name in routine.argnames]
    decl_map = {}
    for decl in FindNodes(ir.VariableDeclaration).visit(routine.spec):
        if any(isinstance(smbl, sym.Array) for smbl in decl.symbols) and not \
                any(smbl in argnames for smbl in decl.symbols) and \
                any(vertical.size in list(FindVariables().visit(smbl.shape)) for smbl in decl.symbols):
            local_arrays.extend(decl.symbols)
            dimensions = [sym.RangeIndex((None, None))] * len(decl.symbols[0].dimensions)
            symbols = [smbl.clone(type=smbl.type.clone(device=True, allocatable=True),
                                  dimensions=as_tuple(dimensions)) for smbl in decl.symbols]
            decl_map[decl] = decl.clone(symbols=as_tuple(symbols))
    routine.spec = Transformer(decl_map).visit(routine.spec)

    allocations = FindNodes(ir.Allocation).visit(routine.body)
    if allocations:
        insert_index = routine.body.body.index(allocations[-1]) + 1
        for local_array in local_arrays:
            routine.body.insert(insert_index, ir.Allocation((local_array,)))
    else:
        for local_array in reversed(local_arrays):
            routine.body.prepend(ir.Allocation((local_array,)))

    for local_array in local_arrays:
        routine.body.append(ir.Deallocation((local_array.clone(dimensions=None),)))


def increase_heap_size(routine):
    """
    Increase the heap size via call to `cudaDeviceSetLimit` needed for version with dynamic
    memory allocation on the device.

    .. note :: `cudaDeviceSetLimit` need to be called before the first kernel call!

    Parameters
    ----------
    routine: :any:`Subroutine`
        The subroutine (e.g. the driver) to increase the heap size
    """
    vtype = SymbolAttributes(types.BasicType.INTEGER, kind=sym.Variable(name="cuda_count_kind"))
    routine.spec.append(ir.VariableDeclaration((sym.Variable(name="cudaHeapSize", type=vtype),)))

    assignment_lhs = routine.variable_map["istat"]
    assignment_rhs = sym.InlineCall(function=sym.ProcedureSymbol(name="cudaDeviceSetLimit", scope=routine),
                                    parameters=(sym.Variable(name="cudaLimitMallocHeapSize"),
                                                routine.variable_map["cudaHeapSize"]))

    routine.body.prepend(ir.Assignment(lhs=assignment_lhs, rhs=assignment_rhs))
    routine.body.prepend(ir.Comment(''))

    # TODO: heap size, to be calculated?
    routine.body.prepend(
        ir.Assignment(lhs=routine.variable_map["cudaHeapSize"], rhs=sym.Product((10, 1024, 1024, 1024))))


def remove_pragmas(routine):
    """
    Remove all pragmas.

    Parameters
    ----------
    routine: :any:`Subroutine`
        The subroutine in which to remove all pragmas
    """
    pragmas = [p for p in FindNodes(ir.Pragma).visit(routine.body) if not p.keyword.lower() == 'loki']
    pragma_map = {p: None for p in pragmas}
    routine.body = Transformer(pragma_map).visit(routine.body)


# TODO: correct "definition" of elemental/pure routines ...
def is_elemental(routine):
    """
    Check whether :any:`Subroutine` ``routine`` is an elemental routine.

    Need for distinguishing elemental and non-elemental function to transform
    those in a different way.

    Parameters
    ----------
    routine: :any:`Subroutine`
        The subroutine to check whether elemental
    """
    for prefix in routine.prefix:
        if prefix.lower() == 'elemental':
            return True
    return False


def update_successor_driver_vars(call, driver_vars, successors, key):
    if successors:
        successor = [s for s in successors if isinstance(s, SubroutineItem)]
        successor = [s for s in successor if s.routine.name == call.routine.name][0]
        successor.trafo_data = {key: {'driver_vars': set()}}
        for rarg, carg in call.arg_iter():
            if isinstance(rarg, sym.Array) and carg.name.lower() in driver_vars:
                successor.trafo_data[key]['driver_vars'].add(rarg.name.lower())

def kernel_cuf(routine, horizontal, vertical, block_dim, transformation_type,
               depth, derived_type_variables, key, successors, targets=None, item=None):
    """
    For CUDA Fortran (CUF) kernels and device functions: thread mapping, array dimension transformation,
    transforming (call) arguments, ...

    Parameters
    ----------
    routine: :any:`Subroutine`
        The subroutine (kernel/device subroutine)
    horizontal: :any:`Dimension`
        The dimension object specifying the horizontal vector dimension
    vertical: :any:`Dimension`
        The dimension object specifying the vertical loop dimension
    block_dim: :any:`Dimension`
        The dimension object specifying the block loop dimension
    transformation_type: int
        Type of SCC-CUF transformation
    depth: int
        Depth of routine (within the call graph) to distinguish between kernels (`global` subroutines)
        and device functions (`device` subroutines)
    derived_type_variables: tuple
        Tuple of derived types within the routine
    targets : tuple of str
        Tuple of subroutine call names that are processed in this traversal
    """

    if is_elemental(routine) or SCCBaseTransformation.check_routine_pragmas(routine, None):
        # TODO: correct "definition" of elemental/pure routines and corresponding removing
        #  of subroutine prefix(es)/specifier(s)
        routine.prefix = as_tuple([prefix for prefix in routine.prefix if prefix not in ["ELEMENTAL"]])
        return

    kernel_demote_private_locals(routine, horizontal, vertical)

    if depth > 1:
        single_variable_declaration(routine, variables=(horizontal.index, block_dim.index))

    # Map variables to declarations
    decl_map = dict((v, decl) for decl in routine.declarations for v in decl.symbols)

    # Keep optional arguments last; a workaround for the fact that keyword arguments are not supported
    # in device code
    arg_pos = [routine.arguments.index(arg) for arg in routine.arguments if arg.type.optional]

    vtype = routine.variable_map[horizontal.index].type.clone()

    if depth == 1:
        jblk_var = routine.variable_map[horizontal.index].clone(name=block_dim.index, type=vtype)
        routine.spec.append(ir.VariableDeclaration((jblk_var,)))

        # This adds argument and variable declaration !
        vtype = routine.variable_map[horizontal.size].type.clone(intent='in', value=True)
        new_argument = as_tuple(routine.variable_map[horizontal.size].clone(name=block_dim.size, type=vtype))
        if arg_pos:
            routine.arguments = routine.arguments[:arg_pos[0]] + new_argument + routine.arguments[arg_pos[0]:]
        else:
            routine.arguments += new_argument

        # CUDA thread mapping
        var_thread_idx = sym.Variable(name="THREADIDX")
        var_x = sym.Variable(name="X", parent=var_thread_idx)
        jl_assignment = ir.Assignment(lhs=routine.variable_map[horizontal.index], rhs=var_x)

        var_thread_idx = sym.Variable(name="BLOCKIDX")
        var_x = sym.Variable(name="Z", parent=var_thread_idx)
        jblk_assignment = ir.Assignment(lhs=routine.variable_map[block_dim.index], rhs=var_x)

        condition = sym.LogicalAnd((sym.Comparison(routine.variable_map[block_dim.index], '<=',
                                                   routine.variable_map[block_dim.size]),
                                    sym.Comparison(routine.variable_map[horizontal.index], '<=',
                                                   routine.variable_map[horizontal.size])))

        routine.body = ir.Section((jl_assignment, jblk_assignment, ir.Comment(''),
                        ir.Conditional(condition=condition, body=routine.body.body, else_body=())))

    elif depth > 1:
        vtype = routine.variable_map[horizontal.size].type.clone(intent='in', value=True)
        new_arguments = as_tuple(routine.variable_map[horizontal.index].clone(type=vtype))
        if arg_pos:
            routine.arguments = routine.arguments[:arg_pos[0]] + new_arguments + routine.arguments[arg_pos[0]:]
        else:
            routine.arguments += new_arguments

    for call in FindNodes(ir.CallStatement).visit(routine.body):
        if call.name not in as_tuple(targets):
            continue

        if not (is_elemental(call.routine) or SCCBaseTransformation.check_routine_pragmas(call.routine, None)):

            # Keep optional arguments last; a workaround for the fact that keyword arguments are not supported
            # in device code
            arg_pos = [call.routine.arguments.index(arg) for arg in call.routine.arguments if arg.type.optional]
            if arg_pos:
                arguments = call.arguments[:arg_pos[0]]
                arguments += as_tuple(routine.variable_map[horizontal.index])
                arguments += call.arguments[arg_pos[0]:]
            else:
                arguments = call.arguments
                arguments += as_tuple(routine.variable_map[horizontal.index])

            call._update(arguments=arguments)

            # pass driver_vars to successors
            if item:
                update_successor_driver_vars(call, item.trafo_data[key]['driver_vars'], successors, key)

    variables = routine.variables
    arguments = routine.arguments

    # Filter out variables that we will pass down the call tree
    calls = FindNodes(ir.CallStatement).visit(routine.body)
    call_args = [arg for call in calls for arg in call.arguments]

    relevant_local_arrays = []
    var_map = {}
    for var in variables:
        if isinstance(var, sym.Scalar) and var.name != block_dim.size and var not in derived_type_variables and \
        str(var.type.intent).lower() == 'in' and var in arguments:
            var_map[var] = var.clone(type=var.type.clone(value=True))
        elif var.name.lower() in getattr(item, 'trafo_data', {}).get(key, {}).get('driver_vars', set()) or \
        (not item and var.name in arguments):
            dimensions = list(var.dimensions)
            shape = list(var.shape)
            if depth == 1:
                dimensions += [routine.variable_map[block_dim.size]]
                shape += [routine.variable_map[block_dim.size]]

            vtype = var.type.clone(shape=as_tuple(shape), device=True)
            var_map[var] = var.clone(dimensions=as_tuple(dimensions), type=vtype)
            if decl_map[var].dimensions and block_dim.size.lower() not in \
                                            [str(d).lower() for d in decl_map[var].dimensions]:
                dimensions = decl_map[var].dimensions
                if depth == 1:
                    dimensions += as_tuple(routine.variable_map[block_dim.size])
                decl_map[var]._update(dimensions=dimensions)
        else:
            if isinstance(var, sym.Array):
                dimensions = list(var.dimensions)
                if horizontal.size in list(FindVariables().visit(var.dimensions)):
                    if transformation_type == 'hoist':
                        dimensions += [routine.variable_map[block_dim.size]]
                        shape = list(var.shape) + [routine.variable_map[block_dim.size]]
                        vtype = var.type.clone(shape=as_tuple(shape))
                        relevant_local_arrays.append(var.name)
                        var_map[var] = var.clone(dimensions=as_tuple(dimensions), type=vtype)
                    else:
                        dimensions.remove(horizontal.size)
                        if decl_map[var].dimensions:
                            if any(s in decl_map[var].dimensions for s in horizontal.size_expressions):
                                dimensions = decl_map[var].dimensions[1:]
                                decl_map[var]._update(dimensions=dimensions)

                        relevant_local_arrays.append(var.name)
                        vtype = var.type.clone(device=True)
                        var_map[var] = var.clone(dimensions=as_tuple(dimensions), type=vtype)

    routine.spec = SubstituteExpressions(var_map).visit(routine.spec)

    var_map = {}
    arguments_name = [var.name for var in arguments]
    for var in FindVariables().visit(routine.body):
        if var.name.lower() in item.trafo_data[key]['driver_vars']:
            if isinstance(var, sym.Array):
                dimensions = list(var.dimensions)
                if depth == 1:
                    dimensions.append(routine.variable_map[block_dim.index])
                var_map[var] = var.clone(dimensions=as_tuple(dimensions),
                                         type=var.type.clone(shape=as_tuple(dimensions)))
        else:
            if transformation_type == 'hoist':
                if var.name in relevant_local_arrays:
                    dimensions = var.dimensions
                    if depth == 1:
                        dimensions += as_tuple(routine.variable_map[block_dim.index])
                    var_map[var] = var.clone(dimensions=dimensions)
            else:
                if var.name in relevant_local_arrays:
                    dimensions = list(var.dimensions)
                    var_map[var] = var.clone(dimensions=as_tuple(dimensions[1:]))

    routine.body = SubstituteExpressions(var_map).visit(routine.body)

    for call in FindNodes(ir.CallStatement).visit(routine.body):
        if call.name not in as_tuple(targets):
            continue

        if not (is_elemental(call.routine) or SCCBaseTransformation.check_routine_pragmas(call.routine, None)):
            arguments = []
            for arg in call.arguments:
                if isinstance(arg, sym.Array):
                    arguments.append(arg.clone(dimensions=None))
                else:
                    arguments.append(arg)

            if depth == 1:
                vmap = {}
                for arg in arguments:
                    if not isinstance(arg, sym._Literal):
                        if routine.variable_map[arg.name] in routine.arguments and isinstance(routine.variable_map[arg.name], sym.Array):
                            dims = as_tuple(sym.IntrinsicLiteral(':') for d in routine.variable_map[arg.name].dimensions[:-1])
                            dims += as_tuple(routine.variable_map[block_dim.index])

                            vmap[arg] = arg.clone(dimensions=dims)

                arguments = SubstituteExpressions(vmap).visit(arguments)

            call._update(arguments=as_tuple(arguments))

    with pragmas_attached(routine, ir.CallStatement):
        for call in FindNodes(ir.CallStatement).visit(routine.body):
            if call.pragma:
                pragma = call.pragma[0]
                if pragma.keyword.lower() == 'loki' and 'inline' in pragma.content.lower():
                    vmap = {}
                    for arg in call.arguments:
                        if arg in routine.arguments and isinstance(arg, sym.Array):
                           dims = as_tuple(routine.variable_map[horizontal.index])
                           vmap[arg] = arg.clone(dimensions=dims)
                    arguments = SubstituteExpressions(vmap).visit(call.arguments)
                    call._update(arguments=as_tuple(arguments))


def kernel_demote_private_locals(routine, horizontal, vertical):
    """
    Demotes all local variables.

    Array variables whose dimensions include only the vector dimension
    or known (short) constant dimensions (eg. local vector or matrix arrays)
    can be privatized without requiring shared GPU memory. Array variables
    with unknown (at compile time) dimensions (eg. the vertical dimension)
    cannot be privatized at the vector loop level and should therefore not
    be demoted here.

    Parameters
    ----------
    routine: :any:`Subroutine`
        The subroutine to demote the private locals
    horizontal: :any:`Dimension`
        The dimension object specifying the horizontal vector dimension
    vertical: :any:`Dimension`
        The dimension object specifying the vertical loop dimension
    """

    # Map variables to declarations
    decl_map = dict((v, decl) for decl in routine.declarations for v in decl.symbols)

    # Establish the new dimensions and shapes first, before cloning the variables
    # The reason for this is that shapes of all variable instances are linked
    # via caching, meaning we can easily void the shape of an unprocessed variable.
    variables = list(routine.variables)
    variables += list(FindVariables(unique=False).visit(routine.body))

    # Filter out purely local array variables
    argument_map = CaseInsensitiveDict({a.name: a for a in routine.arguments})
    variables = [v for v in variables if not v.name in argument_map]
    variables = [v for v in variables if isinstance(v, sym.Array)]

    # Find all arrays with shapes that do not include the vertical
    # dimension and can thus be privatized.
    variables = [v for v in variables if v.shape is not None]
    variables = [v for v in variables if not any(vertical.size in d for d in v.shape)]

    # Filter out variables that we will pass down the call tree
    calls = FindNodes(ir.CallStatement).visit(routine.body)
    call_args = flatten(call.arguments for call in calls)
    call_args += flatten(list(dict(call.kwarguments).values()) for call in calls)
    variables = [v for v in variables if v.name not in call_args]

    shape_map = CaseInsensitiveDict({v.name: v.shape for v in variables})
    vmap = {}
    for v in variables:
        old_shape = shape_map[v.name]
        # TODO: "s for s in old_shape if s not in expressions" sufficient?
        new_shape = as_tuple(s for s in old_shape if s not in horizontal.size_expressions)

        if old_shape and old_shape[0] in horizontal.size_expressions:
            new_type = v.type.clone(shape=new_shape or None)
            new_dims = v.dimensions[1:] or None
            vmap[v] = v.clone(dimensions=new_dims, type=new_type)
            if v in decl_map:
                if any(s in decl_map[v].dimensions for s in horizontal.size_expressions):
                    dimensions = decl_map[v].dimensions[1:]
                    decl_map[v]._update(dimensions=dimensions)

    routine.body = SubstituteExpressions(vmap).visit(routine.body)
    routine.spec = SubstituteExpressions(vmap).visit(routine.spec)


def driver_device_variables(routine, key, successors, targets=None, data_offload=False, item=None):
    """
    Driver device variable versions including

    * variable declaration
    * allocation
    * host-device synchronisation
    * de-allocation

    Parameters
    ----------
    routine: :any:`Subroutine`
        The subroutine (driver) to handle the device variables
    targets : tuple of str
        Tuple of subroutine call names that are processed in this traversal
    """

    # istat: status of CUDA runtime function (e.g. for cudaDeviceSynchronize(), cudaMalloc(), cudaFree(), ...)
    i_type = SymbolAttributes(types.BasicType.INTEGER)
    routine.spec.append(ir.VariableDeclaration(symbols=(sym.Variable(name="istat", type=i_type),)))

    relevant_arrays = []
    calls = tuple(
        call for call in FindNodes(ir.CallStatement).visit(routine.body)
        if call.name in as_tuple(targets)
    )
    for call in calls:
        relevant_arrays.extend([arg for arg in call.arguments if isinstance(arg, sym.Array)])

    relevant_arrays = list(dict.fromkeys(relevant_arrays))

    if data_offload:
        # Declaration
        routine.spec.append(ir.Comment(''))
        routine.spec.append(ir.Comment('! Device arrays'))
        for array in relevant_arrays:
            vtype = array.type.clone(device=True, allocatable=True, intent=None, shape=None)
            vdimensions = [sym.RangeIndex((None, None))] * len(array.dimensions)
            var = array.clone(name=f"{array.name}_d", type=vtype, dimensions=as_tuple(vdimensions))
            routine.spec.append(ir.VariableDeclaration(symbols=as_tuple(var)))

        # Allocation
        for array in reversed(relevant_arrays):
            vtype = array.type.clone(device=True, allocatable=True, intent=None, shape=None)
            routine.body.prepend(ir.Allocation((array.clone(name=f"{array.name}_d", type=vtype,
                                                            dimensions=routine.variable_map[array.name].dimensions),)))
        routine.body.prepend(ir.Comment('! Device array allocation'))
        routine.body.prepend(ir.Comment(''))

        allocations = FindNodes(ir.Allocation).visit(routine.body)
        if allocations:
            insert_index = routine.body.body.index(allocations[-1]) + 1
        else:
            insert_index = None
        # or: insert_index = routine.body.body.index(calls[0])
        # Copy host to device
        for array in reversed(relevant_arrays):
            vtype = array.type.clone(device=True, allocatable=True, intent=None, shape=None)
            lhs = array.clone(name=f"{array.name}_d", type=vtype, dimensions=None)
            rhs = array.clone(dimensions=None)
            if insert_index is not None:
                routine.body.insert(insert_index, ir.Assignment(lhs=lhs, rhs=rhs))
            else:
                routine.body.prepend(ir.Assignment(lhs=lhs, rhs=rhs))
        routine.body.insert(insert_index, ir.Comment('! Copy host to device'))
        routine.body.insert(insert_index, ir.Comment(''))

        # TODO: this just assumes that host-device-synchronisation is only needed at the beginning and end
        # Copy device to host
        insert_index = None
        for call in FindNodes(ir.CallStatement).visit(routine.body):
            if "THREAD_END" in str(call.name):  # TODO: fix/check: very specific to CLOUDSC
                insert_index = routine.body.body.index(call) + 1

        if insert_index is None:
            routine.body.append(ir.Comment(''))
            routine.body.append(ir.Comment('! Copy device to host'))
        for v in reversed(relevant_arrays):
            if v.type.intent != "in":
                lhs = v.clone(dimensions=None)
                vtype = v.type.clone(device=True, allocatable=True, intent=None, shape=None)
                rhs = v.clone(name=f"{v.name}_d", type=vtype, dimensions=None)
                if insert_index is None:
                    routine.body.append(ir.Assignment(lhs=lhs, rhs=rhs))
                else:
                    routine.body.insert(insert_index, ir.Assignment(lhs=lhs, rhs=rhs))
        if insert_index is not None:
            routine.body.insert(insert_index, ir.Comment('! Copy device to host'))

        # De-allocation
        routine.body.append(ir.Comment(''))
        routine.body.append(ir.Comment('! De-allocation'))
        for array in relevant_arrays:
            routine.body.append(ir.Deallocation((array.clone(name=f"{array.name}_d", dimensions=None),)))

    call_map = {}
    if item:
        item.trafo_data = {key: {'driver_vars': set()}}
    for call in calls:
        arguments = []
        for arg in call.arguments:
            if arg in relevant_arrays:
                if data_offload:
                    vtype = arg.type.clone(device=True, allocatable=True, shape=None, intent=None)
                    arguments.append(arg.clone(name=f"{arg.name}_d", type=vtype, dimensions=None))
                else:
                    arguments.append(arg.clone(dimensions=None))

            else:
                arguments.append(arg)

        if item:
            # build set of arrays passed down via driver
            for rarg, carg in call.arg_iter():
                if isinstance(rarg, sym.Array):
                    item.trafo_data[key]['driver_vars'].add(carg.name.lower())
            update_successor_driver_vars(call, item.trafo_data[key]['driver_vars'], successors, key)
        call_map[call] = call.clone(arguments=as_tuple(arguments))
    routine.body = Transformer(call_map).visit(routine.body)


def driver_launch_configuration(routine, block_dim, targets=None):
    """
    Launch configuration for kernel calls within the driver with the
    CUDA Fortran (CUF) specific chevron syntax `<<<griddim, blockdim>>>`.

    Parameters
    ----------
    routine: :any:`Subroutine`
        The subroutine to specify the launch configurations for kernel calls.
    block_dim: :any:`Dimension`
        The dimension object specifying the block loop dimension
    targets : tuple of str
        Tuple of subroutine call names that are processed in this traversal
    """

    d_type = SymbolAttributes(types.DerivedType("DIM3"))
    routine.spec.append(ir.VariableDeclaration(symbols=(sym.Variable(name="GRIDDIM", type=d_type),
                                                        sym.Variable(name="BLOCKDIM", type=d_type))))

    mapper = {}
    for loop in FindNodes(ir.Loop).visit(routine.body):
        # TODO: fix/check: do not use _aliases
        if loop.variable == block_dim.index or loop.variable in block_dim._aliases:
            mapper[loop] = loop.body
            kernel_within = False
            for call in FindNodes(ir.CallStatement).visit(routine.body):
                if call.name not in as_tuple(targets):
                    continue

                kernel_within = True

                assignment_lhs = routine.variable_map["istat"]
                assignment_rhs = sym.InlineCall(
                    function=sym.ProcedureSymbol(name="cudaDeviceSynchronize", scope=routine),
                    parameters=())

                call_map = {call: (call.clone(chevron=(routine.variable_map["GRIDDIM"],
                                                routine.variable_map["BLOCKDIM"]),
                                           arguments=call.arguments + (routine.symbol_map[block_dim.size],)),
                                ir.Assignment(lhs=assignment_lhs, rhs=assignment_rhs))}
                mapper[loop] = Transformer(call_map).visit(mapper[loop])

            if kernel_within:
                upper = routine.symbol_map[loop.bounds.children[1].name]
                if loop.bounds.children[2]:
                    step = routine.variable_map[loop.bounds.children[2].name]
                else:
                    step = sym.IntLiteral(1)

                func_dim3 = sym.ProcedureSymbol(name="DIM3", scope=routine)
                func_ceiling = sym.ProcedureSymbol(name="CEILING", scope=routine)

                # BLOCKDIM
                lhs = routine.variable_map["blockdim"]
                rhs = sym.InlineCall(function=func_dim3, parameters=(step, sym.IntLiteral(1), sym.IntLiteral(1)))
                blockdim_assignment = ir.Assignment(lhs=lhs, rhs=rhs)

                # GRIDDIM
                lhs = routine.variable_map["griddim"]
                rhs = sym.InlineCall(function=func_dim3, parameters=(sym.IntLiteral(1), sym.IntLiteral(1),
                                                                     sym.InlineCall(function=func_ceiling,
                                                                                    parameters=as_tuple(
                                                                                        sym.Cast(name="REAL",
                                                                                                 expression=upper) /
                                                                                        sym.Cast(name="REAL",
                                                                                                 expression=step)))))
                griddim_assignment = ir.Assignment(lhs=lhs, rhs=rhs)
                mapper[loop] = as_tuple(griddim_assignment) + as_tuple(blockdim_assignment) + mapper[loop]

    routine.body = Transformer(mapper=mapper).visit(routine.body)


def device_derived_types(routine, derived_types, targets=None):
    """
    Create device versions of variables of specific derived types including
    host-device-synchronisation.

    Parameters
    ----------
    routine: :any:`Subroutine`
        The subroutine to create device versions of the specified derived type variables.
    derived_types: tuple
        Tuple of derived types within the routine
    targets : tuple of str
        Tuple of subroutine call names that are processed in this traversal
    """
    # used_members = [v for v in FindVariables().visit(routine.ir) if v.parent]
    # variables = [v for v in used_members if v.parent.type.dtype.name.upper() in derived_types]

    _variables = list(FindVariables().visit(routine.ir))
    variables = []
    for var in _variables:
        for derived_type in derived_types:
            if derived_type in str(var.type):
                variables.append(var)

    var_map = {}
    for var in variables:
        new_var = var.clone(name=f"{var.name}_d", type=var.type.clone(intent=None, imported=None,
                                                                      allocatable=None, device=True,
                                                                      module=None))
        var_map[var] = new_var
        routine.spec.append(ir.VariableDeclaration((new_var,)))
        routine.body.prepend(ir.Assignment(lhs=new_var, rhs=var))

    for call in FindNodes(ir.CallStatement).visit(routine.body):
        if call.name not in as_tuple(targets):
            continue
        arguments = tuple(var_map.get(arg, arg) for arg in call.arguments)
        call._update(arguments=arguments)
    return variables


def device_subroutine_prefix(routine, depth):
    """
    Add prefix/specifier `ATTRIBUTES(GLOBAL)` for kernel subroutines and
    `ATTRIBUTES(DEVICE)` for device subroutines.

    Parameters
    ----------
    routine: :any:`Subroutine`
        The subroutine (kernel/device subroutine) to add a prefix/specifier
    depth: int
        The subroutines depth
    """
    if depth == 1:
        routine.prefix += ("ATTRIBUTES(GLOBAL)",)
    elif depth > 1:
        routine.prefix += ("ATTRIBUTES(DEVICE)",)


class SccCufTransformation(Transformation):
    """
    Single Column Coalesced CUDA Fortran - SCC-CUF: Direct CPU-to-GPU
    transformation for block-indexed gridpoint routines.

    This transformation will remove individiual CPU-style
    vectorization loops from "kernel" routines and distributes the
    work for GPU threads according to the CUDA programming model using
    CUDA Fortran (CUF) syntax.

    .. note::
       This requires preprocessing with the :any:`DerivedTypeArgumentsTransformation`.

    .. note::
       In dependence of the transformation type ``transformation_type``, further
       transformations are necessary:

       * ``transformation_type = 'parametrise'`` requires a subsequent
         :any:`ParametriseTransformation` transformation with the necessary information
         to parametrise (at least) the ``vertical`` `size`
       * ``transformation_type = 'hoist'`` requires subsequent :any:`HoistVariablesAnalysis`
         and :class:`HoistVariablesTransformation` transformations (e.g.
         :any:`HoistTemporaryArraysAnalysis` for analysis and
         :any:`HoistTemporaryArraysTransformationDeviceAllocatable` for synthesis)
       * ``transformation_type = 'dynamic'`` does not require a subsequent transformation

    Parameters
    ----------
    horizontal : :any:`Dimension`
        :any:`Dimension` object describing the variable conventions used in code
        to define the horizontal data dimension and iteration space.
    vertical : :any:`Dimension`
        :any:`Dimension` object describing the variable conventions used in code
        to define the vertical dimension, as needed to decide array privatization.
    block_dim : :any:`Dimension`
        :any:`Dimension` object to define the blocking dimension
        to use for hoisted column arrays if hoisting is enabled.
    transformation_type : str
        Kind of SCC-CUF transformation, as automatic arrays currently not supported. Thus
        automatic arrays need to transformed by either

        - `parametrise`: parametrising the array dimensions to make the vertical dimension
          a compile-time constant
        - `hoist`: host side hoisting of (relevant) arrays
        - `dynamic`: dynamic memory allocation on the device (not recommended for performance reasons)

    """

    _key = 'SccCufTransformation'

    def __init__(self, horizontal, vertical, block_dim, transformation_type='parametrise',
                 derived_types=None, data_offload=True, key=None):
        self.horizontal = horizontal
        self.vertical = vertical
        self.block_dim = block_dim

        self.transformation_type = transformation_type
        # `parametrise` : parametrising the array dimensions
        # `hoist`: host side hoisting
        # `dynamic`: dynamic memory allocation on the device
        assert self.transformation_type in ['parametrise', 'hoist', 'dynamic']
        self.transformation_description = {'parametrise': 'parametrised array dimensions of local arrays',
                                           'hoist': 'host side hoisted local arrays',
                                           'dynamic': 'dynamic memory allocation on the device'}

        if derived_types is None:
            self.derived_types = ()
        else:
            self.derived_types = [_.upper() for _ in derived_types]
        self.derived_type_variables = ()
        self.data_offload = data_offload

        if key:
            self._key = key

    def transform_module(self, module, **kwargs):

        role = kwargs.get('role')

        if role == 'driver':
            module.spec.prepend(ir.Import(module="cudafor"))

    def transform_subroutine(self, routine, **kwargs):

        item = kwargs.get('item', None)
        role = kwargs.get('role')
        depths = kwargs.get('depths', None)
        targets = kwargs.get('targets', None)
        if depths is None:
            if role == 'driver':
                depth = 0
            elif role == 'kernel':
                depth = 1
        else:
            depth = depths[item]

        remove_pragmas(routine)
        single_variable_declaration(routine=routine, group_by_shape=True)
        device_subroutine_prefix(routine, depth)
        successors = kwargs.get('successors', ())

        if depth > 0:
            routine.spec.prepend(ir.Import(module="cudafor"))

        if role == 'driver':
            self.process_routine_driver(routine, targets=targets, item=item, successors=successors)
        if role == 'kernel':
            self.process_routine_kernel(routine, depth=depth, targets=targets, item=item, successors=successors)

    def process_routine_kernel(self, routine, successors, depth=1, targets=None, item=None):
        """
        Kernel/Device subroutine specific changes/transformations.

        Parameters
        ----------
        routine: :any:`Subroutine`
            The subroutine (kernel/device subroutine) to process
        depth: int
            The subroutines depth
        """

        v_index = SCCBaseTransformation.get_integer_variable(routine, name=self.horizontal.index)
        resolve_associates(routine)
        SCCBaseTransformation.resolve_masked_stmts(routine, loop_variable=v_index)
        SCCBaseTransformation.resolve_vector_dimension(routine, loop_variable=v_index, bounds=self.horizontal.bounds)
        SCCDevectorTransformation.kernel_remove_vector_loops(routine, self.horizontal)

        kernel_cuf(
            routine, self.horizontal, self.vertical, self.block_dim,
            self.transformation_type, depth=depth,
            derived_type_variables=self.derived_type_variables, targets=targets, item=item, key=self._key,
            successors=successors
        )

        # dynamic memory allocation of local arrays (only for version with dynamic memory allocation on device)
        if self.transformation_type == 'dynamic':
            dynamic_local_arrays(routine, self.vertical)

    def process_routine_driver(self, routine, successors, targets=None, item=None):
        """
        Driver subroutine specific changes/transformations.

        Parameters
        ----------
        routine: :any:`Subroutine`
            The subroutine (driver) to process
        """

        self.derived_type_variables = device_derived_types(
            routine=routine, derived_types=self.derived_types, targets=targets
        )
        # create variables needed for the device execution, especially generate device versions of arrays
        driver_device_variables(routine=routine, targets=targets, data_offload=self.data_offload, item=item,
                                key=self._key, successors=successors)
        # remove block loop and generate launch configuration for CUF kernels
        driver_launch_configuration(routine=routine, block_dim=self.block_dim, targets=targets)

        # increase heap size (only for version with dynamic memory allocation on device)
        if self.transformation_type == 'dynamic':
            increase_heap_size(routine)

        routine.body.prepend(ir.Comment(f"!@cuf print *, 'executing SCC-CUF type: {self.transformation_type} - "
                                        f"{self.transformation_description[self.transformation_type]}'"))
        routine.body.prepend(ir.Comment(""))
