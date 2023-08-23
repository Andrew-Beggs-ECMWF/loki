# (C) Copyright 2018- ECMWF.
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""
Utility transformations to update or remove calls to DR_HOOK and other
utility routines
"""

from loki import (
    FindNodes, Transformer, Transformation, CallStatement,
    Conditional, as_tuple, Literal, Intrinsic, Import
)


__all__ = ['DrHookTransformation', 'RemoveCallsTransformation']

def remove_unused_drhook_import(routine):
    """
    Remove unsed DRHOOK imports and corresponding handle.

    Parameters
    ----------
    routine : :any:`Subroutine`
        The subroutine from which to remove DRHOOK import/handle.
    """

    mapper = {}
    for imp in FindNodes(Import).visit(routine.spec):
        if imp.module.lower() == 'yomhook':
            mapper[imp] = None

    if mapper:
        routine.spec = Transformer(mapper).visit(routine.spec)

    #Remove unused zhook_handle
    routine.variables = as_tuple(v for v in routine.variables if v != 'zhook_handle')

class DrHookTransformation(Transformation):
    """
    Re-write or remove the DrHook label markers in transformed
    kernel routines

    Parameters
    ----------
    remove : bool
        Remove calls to ``DR_HOOK``
    mode : str
        Transformation mode to insert into DrHook labels
    """
    def __init__(self, remove=False, mode=None, **kwargs):
        self.remove = remove
        self.mode = mode
        super().__init__(**kwargs)

    def transform_subroutine(self, routine, **kwargs):
        """
        Apply transformation to subroutine object
        """
        role = kwargs['item'].role

        # Leave DR_HOOK annotations in driver routine
        if role == 'driver':
            return

        mapper = {}
        for call in FindNodes(CallStatement).visit(routine.body):
            # Lazily changing the DrHook label in-place
            if call.name == 'DR_HOOK':
                if self.remove:
                    mapper[call] = None
                else:
                    new_label = f'{call.arguments[0].value.upper()}_{str(self.mode).upper()}'
                    new_args = (Literal(value=new_label),) + call.arguments[1:]
                    mapper[call] = call.clone(arguments=new_args)

        if self.remove:
            for cond in FindNodes(Conditional).visit(routine.body):
                if cond.inline and 'LHOOK' in as_tuple(cond.condition):
                    mapper[cond] = None

        routine.body = Transformer(mapper).visit(routine.body)

        #Get rid of unused import and variable
        if self.remove:
            remove_unused_drhook_import(routine)


class RemoveCallsTransformation(Transformation):
    """
    Removes specified :any:`CallStatement` objects from any :any:`Subroutine`.

    In addition, this transformation will also remove inline conditionals that
    guard the respective utility calls, in order to preserve consistent code.

    Parameters
    ----------
    routines : list of str
        List of subroutine names to remove
    include_intrinsics : bool
        Option to extend searches to :any:`Intrinsic` nodes to
        capture print/write statements
    kernel_only : bool
        Option to only remove calls in routines marked as "kernel"; default: ``False``
    """
    def __init__(self, routines, include_intrinsics=False, kernel_only=False, **kwargs):
        self.routines = as_tuple(routines)
        self.include_intrinsics = include_intrinsics
        self.kernel_only = kernel_only
        super().__init__(**kwargs)

    def transform_subroutine(self, routine, **kwargs):
        """
        Apply transformation to subroutine object
        """

        # Skip driver layer if requested
        role = kwargs.get('role', None)
        if role and role == 'driver' and self.kernel_only:
            return

        mapper = {}

        # First remove inline conditionals with calls to specified routines or intrinsics
        # This check happesn first, as we would leave empty-bodies conditionals otherwise
        inline_conditionals = tuple(
            cond for cond in FindNodes(Conditional).visit(routine.body) if cond.inline
        )
        for cond in inline_conditionals:
            if len(cond.body) == 1 and isinstance(cond.body[0], CallStatement):
                if cond.body[0].name in self.routines:
                    mapper[cond] = None

            if self.include_intrinsics:
                if len(cond.body) == 1 and isinstance(cond.body[0], Intrinsic):
                    if any(str(r).lower() in cond.body[0].text.lower() for r in self.routines):
                        mapper[cond] = None

        # Find direct calls to specified routines
        for call in FindNodes(CallStatement).visit(routine.body):
            if call.name in self.routines:
                mapper[call] = None

        # Include intrinsics that match the routine name partially
        if self.include_intrinsics:
            for intr in FindNodes(Intrinsic).visit(routine.body):
                if any(str(r).lower() in intr.text.lower() for r in self.routines):
                    mapper[intr] = None

        routine.body = Transformer(mapper).visit(routine.body)

        #Get rid of unused DRHOOK import and handle
        if 'dr_hook' in [r.lower() for r in self.routines]:
            remove_unused_drhook_import(routine)
