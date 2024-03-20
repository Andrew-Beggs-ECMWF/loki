# (C) Copyright 2018- ECMWF.
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

from functools import partial

from loki.transform import Pipeline
from transformations.single_column_base import SCCBaseTransformation
from transformations.single_column_annotate import SCCAnnotateTransformation
from transformations.single_column_coalesced_vector import (
    SCCDevectorTransformation, SCCDemoteTransformation, SCCRevectorTransformation
)


__all__ = ['SCCVectorPipeline']


"""
The basic Single Column Coalesced (SCC) transformation with
vector-level kernel parallelism.

This tranformation will convert kernels with innermost vectorisation
along a common horizontal dimension to a GPU-friendly loop-layout via
loop inversion and local array variable demotion. The resulting kernel
remains "vector-parallel", but with the ``hosrizontal`` loop as the
outermost iteration dimension (as far as data dependencies
allow). This allows local temporary arrays to be demoted to scalars,
where possible.

The outer "driver" loop over blocks is used as the secondary dimension
of parallelism, where the outher data indexing dimension
(``block_dim``) is resolved in the first call to a "kernel"
routine. This is equivalent to a so-called "gang-vector" parallisation
scheme.

This :any:`Pipeline` applies the following :any:`Transformation`
classes in sequence:
1. :any:`SCCBaseTransformation` - Ensure utility variables and resolve
   problematic code constructs.
2. :any:`SCCDevectorTransformation` - Remove horizontal vector loops.
3. :any:`SCCDemoteTransformation` - Demote local temporary array
   variables where appropriate.
4. :any:`SCCRevectorTransformation` - Re-insert the vecotr loops outermost,
   according to identified vector sections.
5. :any:`SCCAnnotateTransformation` - Annotate loops according to
   programming model (``directive``).

Parameters
----------
horizontal : :any:`Dimension`
    :any:`Dimension` object describing the variable conventions used in code
    to define the horizontal data dimension and iteration space.
vertical : :any:`Dimension`
    :any:`Dimension` object describing the variable conventions used in code
    to define the vertical dimension, as needed to decide array privatization.
block_dim : :any:`Dimension`
    Optional ``Dimension`` object to define the blocking dimension
    to use for hoisted column arrays if hoisting is enabled.
directive : string or None
    Directives flavour to use for parallelism annotations; either
    ``'openacc'`` or ``None``.
trim_vector_sections : bool
    Flag to trigger trimming of extracted vector sections to remove
    nodes that are not assignments involving vector parallel arrays.
demote_local_arrays : bool
    Flag to trigger local array demotion to scalar variables where possible
"""
SCCVectorPipeline = partial(
    Pipeline, classes=(
        SCCBaseTransformation,
        SCCDevectorTransformation,
        SCCDemoteTransformation,
        SCCRevectorTransformation,
        SCCAnnotateTransformation
    )
)
