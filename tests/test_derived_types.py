from pathlib import Path
import pytest
import numpy as np

from conftest import jit_compile, clean_test
from loki import (
    OFP, OMNI, FP, Module, Subroutine, FindVariables, IntLiteral,
    RangeIndex, Scalar, BasicType
)


@pytest.fixture(scope='module', name='here')
def fixture_here():
    return Path(__file__).parent


@pytest.mark.parametrize('frontend', [OFP, OMNI, FP])
def test_simple_loops(here, frontend):
    """
    Test simple vector/matrix arithmetic with a derived type
    """

    fcode = """
module derived_types_mod
  integer, parameter :: jprb = selected_real_kind(13,300)

  type explicit
    real(kind=jprb) :: scalar, vector(3), matrix(3, 3)
    real(kind=jprb) :: red_herring
  end type explicit
contains

  subroutine simple_loops(item)
    type(explicit), intent(inout) :: item
    integer :: i, j, n

    n = 3
    do i=1, n
       item%vector(i) = item%vector(i) + item%scalar
    end do

    do j=1, n
       do i=1, n
          item%matrix(i, j) = item%matrix(i, j) + item%scalar
       end do
    end do
  end subroutine simple_loops
end module
"""
    module = Module.from_source(fcode, frontend=frontend)
    filepath = here/('derived_types_simple_loops_%s.f90' % frontend)
    mod = jit_compile(module, filepath=filepath, objname='derived_types_mod')

    item = mod.explicit()
    item.scalar = 2.
    item.vector[:] = 5.
    item.matrix[:, :] = 4.
    mod.simple_loops(item)
    assert (item.vector == 7.).all() and (item.matrix == 6.).all()

    clean_test(filepath)


@pytest.mark.parametrize('frontend', [OFP, OMNI, FP])
def test_array_indexing_explicit(here, frontend):
    """
    Test simple vector/matrix arithmetic with a derived type
    """

    fcode = """
module derived_types_mod
  integer, parameter :: jprb = selected_real_kind(13,300)

  type explicit
    real(kind=jprb) :: scalar, vector(3), matrix(3, 3)
    real(kind=jprb) :: red_herring
  end type explicit
contains

  subroutine array_indexing_explicit(item)
    type(explicit), intent(inout) :: item
    real(kind=jprb) :: vals(3) = (/ 1., 2., 3. /)
    integer :: i

    item%vector(:) = 666.
    do i=1, 3
       item%matrix(:, i) = vals(i)
    end do
  end subroutine array_indexing_explicit
end module
"""
    module = Module.from_source(fcode, frontend=frontend)
    filepath = here/('derived_types_array_indexing_explicit_%s.f90' % frontend)
    mod = jit_compile(module, filepath=filepath, objname='derived_types_mod')

    item = mod.explicit()
    mod.array_indexing_explicit(item)
    assert (item.vector == 666.).all()
    assert (item.matrix == np.array([[1., 2., 3.], [1., 2., 3.], [1., 2., 3.]])).all()

    clean_test(filepath)


@pytest.mark.parametrize('frontend', [OFP, OMNI, FP])
def test_array_indexing_deferred(here, frontend):
    """
    Test simple vector/matrix arithmetic with a derived type
    with dynamically allocated arrays.
    """

    fcode = """
module derived_types_mod
  integer, parameter :: jprb = selected_real_kind(13,300)

  type deferred
    real(kind=jprb), allocatable :: scalar, vector(:), matrix(:, :)
    real(kind=jprb), allocatable :: red_herring
  end type deferred
contains

  subroutine alloc_deferred(item)
    type(deferred), intent(inout) :: item
    allocate(item%vector(3))
    allocate(item%matrix(3, 3))
  end subroutine alloc_deferred

  subroutine free_deferred(item)
    type(deferred), intent(inout) :: item
    deallocate(item%vector)
    deallocate(item%matrix)
  end subroutine free_deferred

  subroutine array_indexing_deferred(item)
    type(deferred), intent(inout) :: item
    real(kind=jprb) :: vals(3) = (/ 1., 2., 3. /)
    integer :: i

    item%vector(:) = 666.

    do i=1, 3
       item%matrix(:, i) = vals(i)
    end do
  end subroutine array_indexing_deferred
end module
"""
    module = Module.from_source(fcode, frontend=frontend)
    filepath = here/('derived_types_array_indexing_deferred_%s.f90' % frontend)
    mod = jit_compile(module, filepath=filepath, objname='derived_types_mod')

    item = mod.deferred()
    mod.alloc_deferred(item)
    mod.array_indexing_deferred(item)
    assert (item.vector == 666.).all()
    assert (item.matrix == np.array([[1., 2., 3.], [1., 2., 3.], [1., 2., 3.]])).all()
    mod.free_deferred(item)

    clean_test(filepath)


@pytest.mark.parametrize('frontend', [OFP, OMNI, FP])
def test_array_indexing_nested(here, frontend):
    """
    Test simple vector/matrix arithmetic with a nested derived type
    """

    fcode = """
module derived_types_mod
  integer, parameter :: jprb = selected_real_kind(13,300)

  type explicit
    real(kind=jprb) :: scalar, vector(3), matrix(3, 3)
    real(kind=jprb) :: red_herring
  end type explicit

  type nested
    real(kind=jprb) :: a_scalar, a_vector(3)
    type(explicit) :: another_item
  end type nested
contains

  subroutine array_indexing_nested(item)
    type(nested), intent(inout) :: item
    real(kind=jprb) :: vals(3) = (/ 1., 2., 3. /)
    integer :: i

    item%a_vector(:) = 666.
    item%another_item%vector(:) = 999.

    do i=1, 3
       item%another_item%matrix(:, i) = vals(i)
    end do
  end subroutine array_indexing_nested
end module
"""
    module = Module.from_source(fcode, frontend=frontend)
    filepath = here/('derived_types_array_indexing_nested_%s.f90' % frontend)
    mod = jit_compile(module, filepath=filepath, objname='derived_types_mod')

    item = mod.nested()
    mod.array_indexing_nested(item)
    assert (item.a_vector == 666.).all()
    assert (item.another_item.vector == 999.).all()
    assert (item.another_item.matrix == np.array([[1., 2., 3.],
                                                  [1., 2., 3.],
                                                  [1., 2., 3.]])).all()

    clean_test(filepath)


@pytest.mark.parametrize('frontend', [OFP, OMNI, FP])
def test_deferred_array(here, frontend):
    """
    Test simple vector/matrix with an array of derived types
    """

    fcode = """
module derived_types_mod
  integer, parameter :: jprb = selected_real_kind(13,300)

  type deferred
    real(kind=jprb), allocatable :: scalar, vector(:), matrix(:, :)
    real(kind=jprb), allocatable :: red_herring
  end type deferred
contains

  subroutine alloc_deferred(item)
    type(deferred), intent(inout) :: item
    allocate(item%vector(3))
    allocate(item%matrix(3, 3))
  end subroutine alloc_deferred

  subroutine free_deferred(item)
    type(deferred), intent(inout) :: item
    deallocate(item%vector)
    deallocate(item%matrix)
  end subroutine free_deferred

  subroutine deferred_array(item)
    type(deferred), intent(inout) :: item
    type(deferred), allocatable :: item2(:)
    real(kind=jprb) :: vals(3) = (/ 1., 2., 3. /)
    integer :: i, j

    allocate(item2(4))

    do j=1, 4
      call alloc_deferred(item2(j))

      item2(j)%vector(:) = 666.

      do i=1, 3
        item2(j)%matrix(:, i) = vals(i)
      end do
    end do

    item%vector(:) = 0.
    item%matrix(:,:) = 0.

    do j=1, 4
      item%vector(:) = item%vector(:) + item2(j)%vector(:)

      do i=1, 3
          item%matrix(:,i) = item%matrix(:,i) + item2(j)%matrix(:,i)
      end do

      call free_deferred(item2(j))
    end do

    deallocate(item2)
  end subroutine deferred_array
end module
"""
    module = Module.from_source(fcode, frontend=frontend)
    filepath = here/('derived_types_deferred_array_%s.f90' % frontend)
    mod = jit_compile(module, filepath=filepath, objname='derived_types_mod')

    item = mod.deferred()
    mod.alloc_deferred(item)
    mod.deferred_array(item)
    assert (item.vector == 4 * 666.).all()
    assert (item.matrix == 4 * np.array([[1., 2., 3.], [1., 2., 3.], [1., 2., 3.]])).all()
    mod.free_deferred(item)

    clean_test(filepath)


@pytest.mark.parametrize('frontend', [OFP, OMNI, FP])
def test_derived_type_caller(here, frontend):
    """
    Test a simple call to another routine specifying a derived type as argument
    """

    fcode = """
module derived_types_mod
  integer, parameter :: jprb = selected_real_kind(13,300)

  type explicit
    real(kind=jprb) :: scalar, vector(3), matrix(3, 3)
    real(kind=jprb) :: red_herring
  end type explicit
contains

  subroutine simple_loops(item)
    type(explicit), intent(inout) :: item
    integer :: i, j, n

    n = 3
    do i=1, n
       item%vector(i) = item%vector(i) + item%scalar
    end do

    do j=1, n
       do i=1, n
          item%matrix(i, j) = item%matrix(i, j) + item%scalar
       end do
    end do
  end subroutine simple_loops

  subroutine derived_type_caller(item)
    ! simple call to another routine specifying a derived type as argument
    type(explicit), intent(inout) :: item

    item%red_herring = 42.
    call simple_loops(item)
  end subroutine derived_type_caller

end module
"""
    module = Module.from_source(fcode, frontend=frontend)
    filepath = here/('derived_types_derived_type_caller_%s.f90' % frontend)
    mod = jit_compile(module, filepath=filepath, objname='derived_types_mod')

    # Test the generated identity
    item = mod.explicit()
    item.scalar = 2.
    item.vector[:] = 5.
    item.matrix[:, :] = 4.
    item.red_herring = -1.
    mod.derived_type_caller(item)
    assert (item.vector == 7.).all() and (item.matrix == 6.).all() and item.red_herring == 42.

    clean_test(filepath)


@pytest.mark.parametrize('frontend', [OFP, OMNI, FP])
def test_associates(here, frontend):
    """
    Test the use of associate to access and modify other items
    """

    fcode = """
module derived_types_mod
  integer, parameter :: jprb = selected_real_kind(13,300)

  type explicit
    real(kind=jprb) :: scalar, vector(3), matrix(3, 3)
    real(kind=jprb) :: red_herring
  end type explicit

  type deferred
    real(kind=jprb), allocatable :: scalar, vector(:), matrix(:, :)
    real(kind=jprb), allocatable :: red_herring
  end type deferred
contains

  subroutine alloc_deferred(item)
    type(deferred), intent(inout) :: item
    allocate(item%vector(3))
    allocate(item%matrix(3, 3))
  end subroutine alloc_deferred

  subroutine free_deferred(item)
    type(deferred), intent(inout) :: item
    deallocate(item%vector)
    deallocate(item%matrix)
  end subroutine free_deferred

  subroutine associates(item)
    type(explicit), intent(inout) :: item
    type(deferred) :: item2

    item%scalar = 17.0

    associate(vector2=>item%matrix(:,1))
        vector2(:) = 3.
        item%matrix(:,3) = vector2(:)
    end associate

    associate(vector=>item%vector)
        item%vector(2) = vector(1)
        vector(3) = item%vector(1) + vector(2)
        vector(1) = 1.
    end associate

    call alloc_deferred(item2)

    associate(vec=>item2%vector(2))
        vec = 1.
    end associate

    call free_deferred(item2)
  end subroutine associates
end module
"""
    # Test the internals
    module = Module.from_source(fcode, frontend=frontend)
    routine = module['associates']
    variables = FindVariables().visit(routine.body)
    if frontend == OMNI:
        assert all([v.shape == (RangeIndex((IntLiteral(1), IntLiteral(3))),)
                    for v in variables if v.name in ['vector', 'vector2']])
    else:
        assert all([v.shape == (IntLiteral(3),)
                    for v in variables if v.name in ['vector', 'vector2']])

    # Test the generated module
    filepath = here/('derived_types_associates_%s.f90' % frontend)
    mod = jit_compile(module, filepath=filepath, objname='derived_types_mod')

    item = mod.explicit()
    item.scalar = 0.
    item.vector[0] = 5.
    item.vector[1:2] = 0.
    mod.associates(item)
    assert item.scalar == 17.0 and (item.vector == [1., 5., 10.]).all()

    clean_test(filepath)


@pytest.mark.parametrize('frontend', [
    OFP,
    pytest.param(OMNI, marks=pytest.mark.xfail(reason='OMNI fails to read without full module')),
    FP
])
def test_associates_deferred(frontend):
    """
    Verify that reading in subroutines with deferred external type definitions
    and associates working on that are supported.
    """

    fcode = """
SUBROUTINE ASSOCIATES_DEFERRED(ITEM, IDX)
USE SOME_MOD, ONLY: SOME_TYPE
IMPLICIT NONE
TYPE(SOME_TYPE), INTENT(IN) :: ITEM
INTEGER, INTENT(IN) :: IDX
ASSOCIATE(SOME_VAR=>ITEM%SOME_VAR(IDX))
SOME_VAR = 5
END ASSOCIATE
END SUBROUTINE
    """
    routine = Subroutine.from_source(fcode, frontend=frontend)
    variables = {v.name: v for v in FindVariables().visit(routine.body)}
    assert len(variables) == 3
    some_var = variables['SOME_VAR']
    assert isinstance(some_var, Scalar)
    assert some_var.name.upper() == 'SOME_VAR'
    assert some_var.type.dtype == BasicType.DEFERRED


@pytest.mark.parametrize('frontend', [OFP, OMNI, FP])
def test_case_sensitivity(here, frontend):
    """
    Some abuse of the case agnostic behaviour of Fortran
    """

    fcode = """
module derived_types_mod
  integer, parameter :: jprb = selected_real_kind(13,300)

  type case_sensitive
    real(kind=jprb) :: u, v, T
    real(kind=jprb) :: q, A
  end type case_sensitive
contains

  subroutine check_case(item)
    type(case_sensitive), intent(inout) :: item

    item%u = 1.0
    item%v = 2.0
    item%t = 3.0
    item%q = -1.0
    item%A = -5.0
  end subroutine check_case
end module
"""
    module = Module.from_source(fcode, frontend=frontend)
    filepath = here/('derived_types_case_sensitivity_%s.f90' % frontend)
    mod = jit_compile(module, filepath=filepath, objname='derived_types_mod')

    item = mod.case_sensitive()
    item.u = 0.
    item.v = 0.
    item.t = 0.
    item.q = 0.
    item.a = 0.
    mod.check_case(item)
    assert item.u == 1.0 and item.v == 2.0 and item.t == 3.0
    assert item.q == -1.0 and item.a == -5.0

    clean_test(filepath)


@pytest.mark.parametrize('frontend', [OFP, OMNI, FP])
def test_check_alloc_source(here, frontend):
    """
    Test the use of SOURCE in allocate
    """

    fcode = """
module derived_types_mod
  integer, parameter :: jprb = selected_real_kind(13,300)

  type explicit
    real(kind=jprb) :: scalar, vector(3), matrix(3, 3)
    real(kind=jprb) :: red_herring
  end type explicit

  type deferred
    real(kind=jprb), allocatable :: scalar, vector(:), matrix(:, :)
    real(kind=jprb), allocatable :: red_herring
  end type deferred
contains

  subroutine alloc_deferred(item)
    type(deferred), intent(inout) :: item
    allocate(item%vector(3))
    allocate(item%matrix(3, 3))
  end subroutine alloc_deferred

  subroutine free_deferred(item)
    type(deferred), intent(inout) :: item
    deallocate(item%vector)
    deallocate(item%matrix)
  end subroutine free_deferred

  subroutine check_alloc_source(item, item2)
    type(explicit), intent(inout) :: item
    type(deferred), intent(inout) :: item2
    real(kind=jprb), allocatable :: vector(:), vector2(:)

    allocate(vector, source=item%vector)
    vector(:) = vector(:) + item%scalar
    item%vector(:) = vector(:)

    allocate(vector2, source=item2%vector)  ! Try mold here when supported by fparser
    vector2(:) = item2%scalar
    item2%vector(:) = vector2(:)
  end subroutine check_alloc_source
end module
"""

    module = Module.from_source(fcode, frontend=frontend)
    filepath = here/('derived_types_check_alloc_source_%s.f90' % frontend)
    mod = jit_compile(module, filepath=filepath, objname='derived_types_mod')

    item = mod.explicit()
    item.scalar = 1.
    item.vector[:] = 1.

    item2 = mod.deferred()
    mod.alloc_deferred(item2)
    item2.scalar = 2.
    item2.vector[:] = -1.

    mod.check_alloc_source(item, item2)
    assert (item.vector == 2.).all()
    assert (item2.vector == 2.).all()
    mod.free_deferred(item2)

    clean_test(filepath)
