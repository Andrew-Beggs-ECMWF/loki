# (C) Copyright 2018- ECMWF.
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import pytest

from loki import Subroutine, Dimension, FindNodes, Loop
from loki.expression import symbols as sym
from loki.frontend import available_frontends
from loki.types import BasicType


@pytest.mark.parametrize('frontend', available_frontends())
def test_dimension_size(frontend):
    """
    Test that :any:`Dimension` objects match size expressions.
    """
    fcode = """
subroutine test_dimension_size(nlon, start, end, arr)
  integer, intent(in) :: NLON, START, END
  real, intent(inout) :: arr(nlon)
  real :: local_arr(1:nlon)
  real :: range_arr(end-start+1)

  arr(start:end) = 1.
end subroutine test_dimension_size
"""
    routine = Subroutine.from_source(fcode, frontend=frontend)

    # Create the dimension object and make sure we match all array sizes
    dim = Dimension(name='test_dim', size='nlon', bounds=('start', 'end'))
    assert routine.variable_map['nlon'] == dim.size
    assert routine.variable_map['arr'].dimensions[0] == dim.size

    # Ensure that aliased size expressions laos trigger right
    assert routine.variable_map['nlon'] in dim.size_expressions
    assert routine.variable_map['local_arr'].dimensions[0] in dim.size_expressions
    assert routine.variable_map['range_arr'].dimensions[0] in dim.size_expressions


@pytest.mark.parametrize('frontend', available_frontends())
def test_dimension_index_range(frontend):
    """
    Test that :any:`Dimension` objects match index and range expressions.
    """
    fcode = """
subroutine test_dimension_index(nlon, start, end, arr)
  integer, intent(in) :: NLON, START, END
  real, intent(inout) :: arr(nlon)
  integer :: I

  do i=start, end
    arr(I) = 1.
  end do
end subroutine test_dimension_index
"""
    routine = Subroutine.from_source(fcode, frontend=frontend)

    # Create the dimension object and make sure we match all array sizes
    dim = Dimension(name='test_dim', index='i', bounds=('start', 'end'))
    assert routine.variable_map['i'] == dim.index

    assert FindNodes(Loop).visit(routine.body)[0].bounds == dim.range
    assert FindNodes(Loop).visit(routine.body)[0].bounds.lower == dim.bounds[0]
    assert FindNodes(Loop).visit(routine.body)[0].bounds.upper == dim.bounds[1]

    # Test the correct creation of horizontal dim with aliased bounds vars
    _ = Dimension('test_dim_alias', bounds_aliases=('bnds%start', 'bnds%end'))
    with pytest.raises(RuntimeError):
        _ = Dimension('test_dim_alias', bounds_aliases=('bnds%start',))
    with pytest.raises(RuntimeError):
        _ = Dimension('test_dim_alias', bounds_aliases=('bnds%start', 'some_other_bnds%end'))


@pytest.mark.parametrize('frontend', available_frontends())
def test_dimension_get_loop_bounds(frontend):
    """
    Test that :any:`Dimension` objects find the correct loop bound expressions.
    """
    fcode = """
subroutine test_dimension_loop_bounds(nlon, start, end, arr)
  integer, intent(in) :: NLON, START, END
  real, intent(inout) :: arr(nlon)
  real :: local_arr(1:nlon)
  real :: range_arr(end-start+1)

  arr(start:end) = 1.
end subroutine test_dimension_loop_bounds
"""
    routine = Subroutine.from_source(fcode, frontend=frontend)

    # Create the dimension object and make sure we match all array sizes
    dim = Dimension(name='test_dim', size='nlon', bounds=('start', 'end'))
    start, end = dim.get_loop_bounds(routine)
    assert isinstance(start, sym.Scalar)
    assert isinstance(end, sym.Scalar)
    assert start, end == ('start', 'end')

    # Test correct error handling
    dim2 = Dimension(name='dim_nope', size='k', bounds=('a', 'b'))
    with pytest.raises(RuntimeError):
        _ = dim2.get_loop_bounds(routine)


@pytest.mark.parametrize('frontend', available_frontends())
def test_dimension_get_index_variable(frontend):
    """
    Test that :any:`Dimension` objects find or create the index variable.
    """
    fcode = """
subroutine test_dimension_loop_bounds(nlon, start, end, arr)
  integer, intent(in) :: NLON, START, END
  real, intent(inout) :: arr(nlon)
  real :: local_arr(1:nlon)
  real :: range_arr(end-start+1)
  integer :: i

  arr(start:end) = 1.
end subroutine test_dimension_loop_bounds
"""
    routine = Subroutine.from_source(fcode, frontend=frontend)

    # Create the dimension object and make sure we match all array sizes
    dim_i = Dimension(name='test_dim', size='nlon', index='I', bounds=('start', 'end'))
    i = dim_i.get_index_variable(routine)
    assert isinstance(i, sym.Scalar)
    assert i == 'i'
    assert i.type.dtype == BasicType.INTEGER

    i_decl = routine.spec.body[-1]
    assert i_decl.symbols == ('i',)
    assert i_decl.symbols[0].type.dtype == BasicType.INTEGER

    # Test addition of declaration for unknown index variable
    dim_j = Dimension(name='test_dim2', size='nlon', index='j')
    j = dim_j.get_index_variable(routine)
    assert isinstance(j, sym.Scalar)
    assert j == 'j'
    assert j.type.dtype == BasicType.INTEGER

    j_decl = routine.spec.body[-1]
    assert j_decl.symbols == ('j',)
    assert j_decl.symbols[0].type.dtype == BasicType.INTEGER
