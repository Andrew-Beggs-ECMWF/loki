# (C) Copyright 2018- ECMWF.
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

from more_itertools import split_at

from loki.expression import symbols as sym
from loki import (
     Transformation, FindNodes, ir, FindScopes, as_tuple, flatten, Transformer,
     NestedTransformer, FindVariables, demote_variables, is_dimension_constant,
     is_loki_pragma, dataflow_analysis_attached, BasicType, fgen
)
from transformations.single_column_coalesced import SCCBaseTransformation

__all__ = ['SCCDevectorTransformation', 'SCCRevectorTransformation', 'SCCDemoteTransformation']


class SCCDevectorTransformation(Transformation):
    """
    A set of utilities that can be used to strip vector loops from a :any:`Subroutine`
    and determine the regions of the IR to be placed within thread-parallel loop directives.

    Parameters
    ----------
    horizontal : :any:`Dimension`
        :any:`Dimension` object describing the variable conventions used in code
        to define the horizontal data dimension and iteration space.
    trim_vector_sections : bool
        Flag to trigger trimming of extracted vector sections to remove
        nodes that are not assignments involving vector parallel arrays.
    """

    def __init__(self, horizontal, trim_vector_sections=False):
        self.horizontal = horizontal
        self.trim_vector_sections = trim_vector_sections

    @classmethod
    def kernel_remove_vector_loops(cls, routine, horizontal):
        """
        Remove all vector loops over the specified dimension.

        Parameters
        ----------
        routine : :any:`Subroutine`
            The subroutine in the vector loops should be removed.
        horizontal : :any:`Dimension`
            The dimension specifying the horizontal vector dimension
        """
        loop_map = {}
        for loop in FindNodes(ir.Loop).visit(routine.body):
            if loop.variable == horizontal.index:
                loop_map[loop] = loop.body
        routine.body = Transformer(loop_map).visit(routine.body)

    @classmethod
    def extract_vector_sections(cls, section, horizontal):
        """
        Extract a contiguous sections of nodes that contains vector-level
        computations and are not interrupted by recursive subroutine calls
        or nested control-flow structures.

        Parameters
        ----------
        section : tuple of :any:`Node`
            A section of nodes from which to extract vector-level sub-sections
        horizontal: :any:`Dimension`
            The dimension specifying the horizontal vector dimension
        """

        _scope_node_types = (ir.Loop, ir.Conditional, ir.MultiConditional)

        # Identify outer "scopes" (loops/conditionals) constrained by recursive routine calls
        separator_nodes = []
        calls = FindNodes(ir.CallStatement).visit(section)
        _pragmas = FindNodes(ir.Pragma).visit(section)
        pragmas = []
        for _pragma in _pragmas:
            if _pragma.keyword.lower() == "loki" and _pragma.content.lower() == "seperator":
                pragmas.append(_pragma)
        for pragma in pragmas:
            separator_nodes.append(pragma)
        for call in calls:

            # check if calls have been enriched
            if not call.routine is BasicType.DEFERRED:
                # check if called routine is marked as sequential
                if SCCBaseTransformation.check_routine_pragmas(routine=call.routine, directive=None):
                    continue

            if call in section:
                # If the call is at the current section's level, it's a separator
                # print(f"seperator call: {call.name}")
                separator_nodes.append(call)

            else:
                # If the call is deeper in the IR tree, it's highest ancestor is used
                ancestors = flatten(FindScopes(call).visit(section))
                ancestor_scopes = [a for a in ancestors if isinstance(a, _scope_node_types)]
                if len(ancestor_scopes) > 0 and ancestor_scopes[0] not in separator_nodes:
                    separator_nodes.append(ancestor_scopes[0])

        for pragma in FindNodes(ir.Pragma).visit(section):
            # Reductions over thread-parallel regions should be marked as a separator node
            if (is_loki_pragma(pragma, starts_with='vector-reduction') or
                is_loki_pragma(pragma, starts_with='end vector-reduction')):
                separator_nodes.append(pragma)

        # Extract contiguous node sections between separator nodes
        assert all(n in section for n in separator_nodes)
        subsections = [as_tuple(s) for s in split_at(section, lambda n: n in separator_nodes)]

        # Filter sub-sections that do not use the horizontal loop index variable
        subsections = [s for s in subsections if horizontal.index in list(FindVariables().visit(s))]

        # Recurse on all separator nodes that might contain further vector sections
        for separator in separator_nodes:

            if isinstance(separator, ir.Loop):
                subsec_body = cls.extract_vector_sections(separator.body, horizontal)
                if subsec_body:
                    subsections += subsec_body

            if isinstance(separator, ir.Conditional):
                subsec_body = cls.extract_vector_sections(separator.body, horizontal)
                if subsec_body:
                    subsections += subsec_body
                subsec_else = cls.extract_vector_sections(separator.else_body, horizontal)
                if subsec_else:
                    subsections += subsec_else

            if isinstance(separator, ir.MultiConditional):
                for body in separator.bodies:
                    subsec_body = cls.extract_vector_sections(body, horizontal)
                    if subsec_body:
                        subsections += subsec_body
                subsec_else = cls.extract_vector_sections(separator.else_body, horizontal)
                if subsec_else:
                    subsections += subsec_else

        return subsections

    @classmethod
    def get_trimmed_sections(cls, routine, horizontal, sections):
        """
        Trim extracted vector sections to remove nodes that are not assignments
        involving vector parallel arrays.
        """

        trimmed_sections = ()
        with dataflow_analysis_attached(routine):
            for sec in sections:
                vec_nodes = [node for node in sec if horizontal.index.lower() in node.uses_symbols]
                start = sec.index(vec_nodes[0])
                end = sec.index(vec_nodes[-1])

                trimmed_sections += (sec[start:end+1],)

        return trimmed_sections

    def transform_subroutine(self, routine, **kwargs):
        """
        Apply SCCDevector utilities to a :any:`Subroutine`.

        Parameters
        ----------
        routine : :any:`Subroutine`
            Subroutine to apply this transformation to.
        role : string
            Role of the subroutine in the call tree; should be ``"kernel"``
        """
        role = kwargs['role']
        item = kwargs.get('item', None)
        targets = kwargs.get('targets', None)

        if role == 'kernel':
            self.process_kernel(routine)
        else:
            self.process_driver(routine, targets=targets, item=item)

    def process_kernel(self, routine):
        """
        Applies the SCCDevector utilities to a "kernel". This consists simply
        of stripping vector loops and determing which sections of the IR can be
        placed within thread-parallel loops.

        Parameters
        ----------
        routine : :any:`Subroutine`
            Subroutine to apply this transformation to.
        """

        # Remove all vector loops over the specified dimension
        self.kernel_remove_vector_loops(routine, self.horizontal)

        # Extract vector-level compute sections from the kernel
        sections = self.extract_vector_sections(routine.body.body, self.horizontal)

        if self.trim_vector_sections:
            sections = self.get_trimmed_sections(routine, self.horizontal, sections)

        # Replace sections with marked Section node
        section_mapper = {s: ir.Section(body=s, label='vector_section') for s in sections}
        routine.body = NestedTransformer(section_mapper).visit(routine.body)

    def process_driver(self, routine, targets=None, item=None):

        driver_loop_map = {}
        driver_loops = []
        new_driver_loops = []
        for call in FindNodes(ir.CallStatement).visit(routine.body):
            if not call.name in targets:
                continue

            # Find the driver loop by checking the call's heritage
            ancestors = flatten(FindScopes(call).visit(routine.body))
            loops = [a for a in ancestors if isinstance(a, ir.Loop)]
            if not loops:
                # Skip if there are no driver loops
                continue
            driver_loop = loops[0]
            kernel_loop = [l for l in loops if l.variable == self.horizontal.index]
            if kernel_loop:
                kernel_loop = kernel_loop[0]

            assert not driver_loop == kernel_loop
            driver_loops.append(driver_loop)

            # self.kernel_remove_vector_loops(driver_loop, self.horizontal)
            ##
            loop_map = {}
            for loop in FindNodes(ir.Loop).visit(driver_loop.body):
                if loop.variable == self.horizontal.index:
                    loop_map[loop] = loop.body
            # driver_loop.body = Transformer(loop_map).visit(driver_loop.body)
            new_driver_loop = Transformer(loop_map).visit(driver_loop.body)
            new_driver_loop = driver_loop.clone(body=new_driver_loop)
            new_driver_loops.append(new_driver_loop)
            # print("new driver loop -------")
            # print(fgen(new_driver_loop))
            # print("-------")
            driver_loop_map[driver_loop] = new_driver_loop
            ##

        # print(f"driver_loop_map: {driver_loop_map}")
        routine.body = Transformer(driver_loop_map).visit(routine.body)

        driver_loop_map = {}
        for driver_loop in new_driver_loops: #driver_loops:
            # Extract vector-level compute sections from the kernel
            sections = self.extract_vector_sections(driver_loop.body, self.horizontal)
            # print(f"sections: {sections}")

            # Replace sections with marked Section node
            section_mapper = {s: ir.Section(body=s, label='vector_section') for s in sections}
            # print(f"section_mapper: {section_mapper}")
            new_driver_loop = NestedTransformer(section_mapper).visit(driver_loop)
            driver_loop_map[driver_loop] = new_driver_loop
        # routine.body = NestedTransformer(section_mapper).visit(routine.body)
        # print(f"here driver_loop_map: {driver_loop_map}")
        routine.body = Transformer(driver_loop_map).visit(routine.body)



class SCCRevectorTransformation(Transformation):
    """
    A transformation to wrap thread-parallel IR sections within a horizontal loop.
    This transformation relies on markers placed by :any:`SCCDevectorTransformation`.

    Parameters
    ----------
    horizontal : :any:`Dimension`
        :any:`Dimension` object describing the variable conventions used in code
        to define the horizontal data dimension and iteration space.
    """

    def __init__(self, horizontal):
        self.horizontal = horizontal

    @classmethod
    def wrap_vector_section(cls, section, routine, horizontal):
        """
        Wrap a section of nodes in a vector-level loop across the horizontal.

        Parameters
        ----------
        section : tuple of :any:`Node`
            A section of nodes to be wrapped in a vector-level loop
        routine : :any:`Subroutine`
            The subroutine in the vector loops should be removed.
        horizontal: :any:`Dimension`
            The dimension specifying the horizontal vector dimension
        """

        # Create a single loop around the horizontal from a given body
        v_start = routine.variable_map[horizontal.bounds[0]]
        v_end = routine.variable_map[horizontal.bounds[1]]
        index = SCCBaseTransformation.get_integer_variable(routine, horizontal.index)
        bounds = sym.LoopRange((v_start, v_end))

        # Ensure we clone all body nodes, to avoid recursion issues
        vector_loop = ir.Loop(variable=index, bounds=bounds, body=Transformer().visit(section))

        # Add a comment before and after the pragma-annotated loop to ensure
        # we do not overlap with neighbouring pragmas
        return (ir.Comment(''), vector_loop, ir.Comment(''))

    def transform_subroutine(self, routine, **kwargs):
        """
        Apply SCCRevector utilities to a :any:`Subroutine`.

        Parameters
        ----------
        routine : :any:`Subroutine`
            Subroutine to apply this transformation to.
        role : string
            Role of the subroutine in the call tree; should be ``"kernel"``
        """
        role = kwargs['role']
        item = kwargs.get('item', None)
        targets = kwargs.get('targets', None)

        # if role == 'kernel':
        self.process_kernel(routine)
        # else:
        #     self.process_driver(routine, targets=targets, item=item)

    def process_kernel(self, routine):
        """
        Applies the SCCRevector utilities to a "kernel" and wraps all thread-parallel sections within
        a horizontal loop. The markers placed by :any:`SCCDevectorTransformation` are removed.

        Parameters
        ----------
        routine : :any:`Subroutine`
            Subroutine to apply this transformation to.
        """

        # Promote vector loops to be the outermost loop dimension in the kernel
        mapper = {s.body: self.wrap_vector_section(s.body, routine, self.horizontal)
                          for s in FindNodes(ir.Section).visit(routine.body)
                          if s.label == 'vector_section'}
        routine.body = NestedTransformer(mapper).visit(routine.body)

    def process_driver(self, routine, targets=None, item=None):
        for call in FindNodes(ir.CallStatement).visit(routine.body):
            if not call.name in targets:
                continue

            # Find the driver loop by checking the call's heritage
            ancestors = flatten(FindScopes(call).visit(routine.body))
            loops = [a for a in ancestors if isinstance(a, ir.Loop)]
            if not loops:
                # Skip if there are no driver loops
                continue
            driver_loop = loops[0]
            kernel_loop = [l for l in loops if l.variable == self.horizontal.index]
            if kernel_loop:
                kernel_loop = kernel_loop[0]

            assert not driver_loop == kernel_loop

            mapper = {s.body: self.wrap_vector_section(s.body, routine, self.horizontal)
                      for s in FindNodes(ir.Section).visit(driver_loop.body)
                      if s.label == 'vector_section'}
            # print(f"mapper: {mapper}")
            # driver_loop.body = NestedTransformer(mapper).visit(driver_loop.body)
            routine.body = NestedTransformer(mapper).visit(routine.body)

class SCCDemoteTransformation(Transformation):
    """
    A set of utilities to determine which local arrays can be safely demoted in a
    :any:`Subroutine` as part of a transformation pass.

    Parameters
    ----------
    horizontal : :any:`Dimension`
        :any:`Dimension` object describing the variable conventions used in code
        to define the horizontal data dimension and iteration space.
    """

    def __init__(self, horizontal, demote_local_arrays=True):
        self.horizontal = horizontal

        self.demote_local_arrays = demote_local_arrays

    @classmethod
    def kernel_get_locals_to_demote(cls, routine, sections, horizontal):

        argument_names = [v.name for v in routine.arguments]

        def _get_local_arrays(section):
            """
            Filters out local argument arrays that solely buffer the
            horizontal vector dimension
            """
            arrays = FindVariables(unique=False).visit(section)
            # Only demote local arrays with the horizontal as fast dimension
            arrays = [v for v in arrays if isinstance(v, sym.Array)]
            arrays = [v for v in arrays if v.name not in argument_names]
            arrays = [v for v in arrays if v.shape and v.shape[0] == horizontal.size]

            # Also demote arrays whose remaning dimensions are known constants
            arrays = [v for v in arrays if all(is_dimension_constant(d) for d in v.shape[1:])]
            return arrays

        # Create a list of all local horizontal temporary arrays
        candidates = _get_local_arrays(routine.body)

        # Create an index into all variable uses per vector-level section
        vars_per_section = {s: set(v.name.lower() for v in _get_local_arrays(s)) for s in sections}

        # Count in how many sections each temporary is used
        counts = {}
        for arr in candidates:
            counts[arr] = sum(1 if arr.name.lower() in v else 0 for v in vars_per_section.values())

        # Mark temporaries that are only used in one section for demotion
        to_demote = [k for k, v in counts.items() if v == 1]

        # Filter out variables that we will pass down the call tree
        calls = FindNodes(ir.CallStatement).visit(routine.body)
        call_args = flatten(call.arguments for call in calls)
        call_args += flatten(list(dict(call.kwarguments).values()) for call in calls)
        to_demote = [v for v in to_demote if v.name not in call_args]

        return set(to_demote)

    def transform_subroutine(self, routine, **kwargs):
        """
        Apply SCCDemote utilities to a :any:`Subroutine`.

        Parameters
        ----------
        routine : :any:`Subroutine`
            Subroutine to apply this transformation to.
        role : string
            Role of the subroutine in the call tree; should be ``"kernel"``
        """
        role = kwargs['role']
        item = kwargs.get('item', None)

        if role == 'kernel':
            demote_locals = self.demote_local_arrays
            if item:
                demote_locals = item.config.get('demote_locals', self.demote_local_arrays)
            self.process_kernel(routine, demote_locals=demote_locals)

    def process_kernel(self, routine, demote_locals=True):
        """
        Applies the SCCDemote utilities to a "kernel" and demotes all suitable local arrays.

        Parameters
        ----------
        routine : :any:`Subroutine`
            Subroutine to apply this transformation to.
        """

        # Find vector sections marked in the SCCDevectorTransformation
        sections = [s for s in FindNodes(ir.Section).visit(routine.body) if s.label == 'vector_section']

        # Extract the local variables to demote after we wrap the sections in vector loops.
        # We do this, because need the section blocks to determine which local arrays
        # may carry buffered values between them, so that we may not demote those!
        to_demote = self.kernel_get_locals_to_demote(routine, sections, self.horizontal)

        # Demote all private local variables that do not buffer values between sections
        if demote_locals:
            variables = tuple(v.name for v in to_demote)
            if variables:
                demote_variables(routine, variable_names=variables, dimensions=self.horizontal.size)
