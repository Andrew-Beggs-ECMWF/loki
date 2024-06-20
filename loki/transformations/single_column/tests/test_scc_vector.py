# (C) Copyright 2018- ECMWF.
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import pytest

from loki import Subroutine, Sourcefile, Dimension, fgen
from loki.batch import Pipeline
from loki.frontend import available_frontends
from loki.ir import (
    FindNodes, Assignment, CallStatement, Conditional, Comment,
    VariableDeclaration, Loop, Pragma, Section
)
from loki.transformations.single_column import (
    SCCDevectorTransformation, SCCRevectorTransformation,
    SCCRevectorDriverTransformation, SCCVectorPipeline
)


@pytest.fixture(scope='module', name='horizontal')
def fixture_horizontal():
    return Dimension(
        name='horizontal', size='nlon', index='jl',
        bounds=('start', 'end'), aliases=('nproma',)
    )

@pytest.fixture(scope='module', name='horizontal_bounds_aliases')
def fixture_horizontal_bounds_aliases():
    return Dimension(
        name='horizontal_bounds_aliases', size='nlon', index='jl',
        bounds=('start', 'end'), aliases=('nproma',),
        bounds_aliases=('bnds%start', 'bnds%end')
    )

@pytest.fixture(scope='module', name='vertical')
def fixture_vertical():
    return Dimension(name='vertical', size='nz', index='jk', aliases=('nlev',))

@pytest.fixture(scope='module', name='blocking')
def fixture_blocking():
    return Dimension(name='blocking', size='nb', index='b')


@pytest.mark.parametrize('frontend', available_frontends())
def test_scc_revector_transformation(frontend, horizontal):
    """
    Test removal of vector loops in kernel and re-insertion of a single
    hoisted horizontal loop in the kernel.
    """

    fcode_driver = """
  SUBROUTINE column_driver(nlon, nz, q, t, nb)
    INTEGER, INTENT(IN)   :: nlon, nz, nb  ! Size of the horizontal and vertical
    REAL, INTENT(INOUT)   :: t(nlon,nz,nb)
    REAL, INTENT(INOUT)   :: q(nlon,nz,nb)
    INTEGER :: b, start, end

    start = 1
    end = nlon
    do b=1, nb
      call compute_column(start, end, nlon, nz, q(:,:,b), t(:,:,b))
    end do
  END SUBROUTINE column_driver
"""

    fcode_kernel = """
  SUBROUTINE compute_column(start, end, nlon, nz, q, t)
    INTEGER, INTENT(IN) :: start, end  ! Iteration indices
    INTEGER, INTENT(IN) :: nlon, nz    ! Size of the horizontal and vertical
    REAL, INTENT(INOUT) :: t(nlon,nz)
    REAL, INTENT(INOUT) :: q(nlon,nz)
    INTEGER :: jl, jk
    REAL :: c

    c = 5.345
    DO jk = 2, nz
      DO jl = start, end
        t(jl, jk) = c * jk
        q(jl, jk) = q(jl, jk-1) + t(jl, jk) * c
      END DO
    END DO

    ! The scaling is purposefully upper-cased
    DO JL = START, END
      Q(JL, NZ) = Q(JL, NZ) * C
    END DO
  END SUBROUTINE compute_column
"""
    kernel = Subroutine.from_source(fcode_kernel, frontend=frontend)
    driver = Subroutine.from_source(fcode_driver, frontend=frontend)

    # Ensure we have three loops in the kernel prior to transformation
    kernel_loops = FindNodes(Loop).visit(kernel.body)
    assert len(kernel_loops) == 3

    scc_transform = (SCCDevectorTransformation(horizontal=horizontal),)
    scc_transform += (SCCRevectorTransformation(horizontal=horizontal),)
    for transform in scc_transform:
        transform.apply(driver, role='driver')
        transform.apply(kernel, role='kernel')

    # Ensure we have two nested loops in the kernel
    # (the hoisted horizontal and the native vertical)
    kernel_loops = FindNodes(Loop).visit(kernel.body)
    assert len(kernel_loops) == 2
    assert kernel_loops[1] in FindNodes(Loop).visit(kernel_loops[0].body)
    assert kernel_loops[0].variable == 'jl'
    assert kernel_loops[0].bounds == 'start:end'
    assert kernel_loops[1].variable == 'jk'
    assert kernel_loops[1].bounds == '2:nz'

    # Ensure all expressions and array indices are unchanged
    assigns = FindNodes(Assignment).visit(kernel.body)
    assert fgen(assigns[1]).lower() == 't(jl, jk) = c*jk'
    assert fgen(assigns[2]).lower() == 'q(jl, jk) = q(jl, jk - 1) + t(jl, jk)*c'
    assert fgen(assigns[3]).lower() == 'q(jl, nz) = q(jl, nz)*c'

    # Ensure driver remains unaffected
    driver_loops = FindNodes(Loop).visit(driver.body)
    assert len(driver_loops) == 1
    assert driver_loops[0].variable == 'b'
    assert driver_loops[0].bounds == '1:nb'

    kernel_calls = FindNodes(CallStatement).visit(driver_loops[0])
    assert len(kernel_calls) == 1
    assert kernel_calls[0].name == 'compute_column'


@pytest.mark.parametrize('frontend', available_frontends())
def test_scc_revector_transformation_aliased_bounds(frontend, horizontal_bounds_aliases):
    """
    Test removal of vector loops in kernel and re-insertion of a single
    hoisted horizontal loop in the kernel with aliased loop bounds.
    """

    fcode_bnds_type_mod = """
    module bnds_type_mod
    implicit none
       type bnds_type
          integer :: start
          integer :: end
       end type bnds_type
    end module bnds_type_mod
"""

    fcode_driver = """
  SUBROUTINE column_driver(nlon, nz, q, t, nb)
    USE bnds_type_mod, only : bnds_type
    INTEGER, INTENT(IN)   :: nlon, nz, nb  ! Size of the horizontal and vertical
    REAL, INTENT(INOUT)   :: t(nlon,nz,nb)
    REAL, INTENT(INOUT)   :: q(nlon,nz,nb)
    INTEGER :: b, start, end
    TYPE(bnds_type) :: bnds

    bnds%start = 1
    bnds%end = nlon
    do b=1, nb
      call compute_column(bnds, nlon, nz, q(:,:,b), t(:,:,b))
    end do
  END SUBROUTINE column_driver
"""

    fcode_kernel = """
  SUBROUTINE compute_column(bnds, nlon, nz, q, t)
    USE bnds_type_mod, only : bnds_type
    TYPE(bnds_type), INTENT(IN) :: bnds
    INTEGER, INTENT(IN) :: nlon, nz    ! Size of the horizontal and vertical
    REAL, INTENT(INOUT) :: t(nlon,nz)
    REAL, INTENT(INOUT) :: q(nlon,nz)
    INTEGER :: jl, jk
    REAL :: c

    c = 5.345
    DO jk = 2, nz
      DO jl = bnds%start, bnds%end
        t(jl, jk) = c * jk
        q(jl, jk) = q(jl, jk-1) + t(jl, jk) * c
      END DO
    END DO

    ! The scaling is purposefully upper-cased
    DO JL = BNDS%START, BNDS%END
      Q(JL, NZ) = Q(JL, NZ) * C
    END DO
  END SUBROUTINE compute_column
"""
    bnds_type_mod = Sourcefile.from_source(fcode_bnds_type_mod, frontend=frontend)
    kernel = Sourcefile.from_source(fcode_kernel, frontend=frontend,
                                    definitions=bnds_type_mod.definitions).subroutines[0]
    driver = Sourcefile.from_source(fcode_driver, frontend=frontend,
                                    definitions=bnds_type_mod.definitions).subroutines[0]

    # Ensure we have three loops in the kernel prior to transformation
    kernel_loops = FindNodes(Loop).visit(kernel.body)
    assert len(kernel_loops) == 3

    scc_transform = (SCCDevectorTransformation(horizontal=horizontal_bounds_aliases),)
    scc_transform += (SCCRevectorTransformation(horizontal=horizontal_bounds_aliases),)
    for transform in scc_transform:
        transform.apply(driver, role='driver')
        transform.apply(kernel, role='kernel')

    # Ensure we have two nested loops in the kernel
    # (the hoisted horizontal and the native vertical)
    kernel_loops = FindNodes(Loop).visit(kernel.body)
    assert len(kernel_loops) == 2
    assert kernel_loops[1] in FindNodes(Loop).visit(kernel_loops[0].body)
    assert kernel_loops[0].variable == 'jl'
    assert kernel_loops[0].bounds == 'bnds%start:bnds%end'
    assert kernel_loops[1].variable == 'jk'
    assert kernel_loops[1].bounds == '2:nz'

    # Ensure all expressions and array indices are unchanged
    assigns = FindNodes(Assignment).visit(kernel.body)
    assert fgen(assigns[1]).lower() == 't(jl, jk) = c*jk'
    assert fgen(assigns[2]).lower() == 'q(jl, jk) = q(jl, jk - 1) + t(jl, jk)*c'
    assert fgen(assigns[3]).lower() == 'q(jl, nz) = q(jl, nz)*c'

    # Ensure driver remains unaffected
    driver_loops = FindNodes(Loop).visit(driver.body)
    assert len(driver_loops) == 1
    assert driver_loops[0].variable == 'b'
    assert driver_loops[0].bounds == '1:nb'

    kernel_calls = FindNodes(CallStatement).visit(driver_loops[0])
    assert len(kernel_calls) == 1
    assert kernel_calls[0].name == 'compute_column'


@pytest.mark.parametrize('frontend', available_frontends())
def test_scc_devector_transformation(frontend, horizontal):
    """
    Test the correct identification of vector sections and removal of vector loops.
    """

    fcode_kernel = """
  SUBROUTINE compute_column(start, end, nlon, nz, q)
    INTEGER, INTENT(IN) :: start, end  ! Iteration indices
    INTEGER, INTENT(IN) :: nlon, nz    ! Size of the horizontal and vertical
    REAL, INTENT(INOUT) :: q(nlon,nz)
    INTEGER :: jl, jk, niter
    LOGICAL :: maybe
    REAL :: c

    if (maybe)  call logger()

    c = 5.345
    DO JL = START, END
      Q(JL, NZ) = Q(JL, NZ) + 3.0
    END DO

    DO niter = 1, 3

      DO JL = START, END
        Q(JL, NZ) = Q(JL, NZ) + 1.0
      END DO

      call update_q(start, end, nlon, nz, q, c)

    END DO

    DO JL = START, END
      Q(JL, NZ) = Q(JL, NZ) * C
    END DO

    IF (.not. maybe) THEN
      call update_q(start, end, nlon, nz, q, c)
    END IF

    DO JL = START, END
      Q(JL, NZ) = Q(JL, NZ) + C * 3.
    END DO

    IF (maybe)  call logger()

  END SUBROUTINE compute_column
"""
    kernel = Subroutine.from_source(fcode_kernel, frontend=frontend)

    # Check number of horizontal loops prior to transformation
    loops = [l for l in FindNodes(Loop).visit(kernel.body) if l.variable == 'jl']
    assert len(loops) == 4

    # Test SCCDevector transform for kernel with scope-splitting outer loop
    scc_transform = SCCDevectorTransformation(horizontal=horizontal)
    scc_transform.apply(kernel, role='kernel')

    # Check removal of horizontal loops
    loops = [l for l in FindNodes(Loop).visit(kernel.body) if l.variable == 'jl']
    assert not loops

    # Check number and content of vector sections
    sections = [s for s in FindNodes(Section).visit(kernel.body) if s.label == 'vector_section']
    assert len(sections) == 4

    assigns = FindNodes(Assignment).visit(sections[0])
    assert len(assigns) == 2
    assigns = FindNodes(Assignment).visit(sections[1])
    assert len(assigns) == 1
    assigns = FindNodes(Assignment).visit(sections[2])
    assert len(assigns) == 1
    assigns = FindNodes(Assignment).visit(sections[3])
    assert len(assigns) == 1


@pytest.mark.parametrize('frontend', available_frontends())
def test_single_column_coalesced_vector_inlined_call(frontend, horizontal):
    """
    Test that calls targeted for inlining inside a vector region are not treated as separators
    """

    fcode = """
    subroutine some_inlined_kernel(work)
    !$loki routine seq
       real, intent(inout) :: work

       work = work*2.
    end subroutine some_inlined_kernel

    subroutine some_kernel(start, end, nlon, work, cond)
       logical, intent(in) :: cond
       integer, intent(in) :: nlon, start, end
       real, dimension(nlon), intent(inout) :: work

       integer :: jl

       do jl=start,end
          if(cond)then
             call some_inlined_kernel(work(jl))
          endif
          work(jl) = work(jl) + 1.
       enddo

       call some_other_kernel()

    end subroutine some_kernel
    """

    source = Sourcefile.from_source(fcode, frontend=frontend)
    routine = source['some_kernel']
    inlined_routine = source['some_inlined_kernel']
    routine.enrich((inlined_routine,))

    scc_transform = (SCCDevectorTransformation(horizontal=horizontal),)
    scc_transform += (SCCRevectorTransformation(horizontal=horizontal),)
    for transform in scc_transform:
        transform.apply(routine, role='kernel', targets=['some_kernel', 'some_inlined_kernel'])

    # Check loki pragma has been removed
    assert not FindNodes(Pragma).visit(routine.body)

    # Check that 'some_inlined_kernel' remains within vector-parallel region
    loops = FindNodes(Loop).visit(routine.body)
    assert len(loops) == 1
    calls = FindNodes(CallStatement).visit(loops[0].body)
    assert len(calls) == 1
    calls = FindNodes(CallStatement).visit(routine.body)
    assert len(calls) == 2


@pytest.mark.parametrize('frontend', available_frontends())
@pytest.mark.parametrize('trim_vector_sections', [False, True])
def test_single_column_coalesced_vector_section_trim_simple(frontend, horizontal, trim_vector_sections):
    """
    Test the trimming of vector-sections to exclude scalar assignments.
    """

    fcode_kernel = """
    subroutine some_kernel(start, end, nlon)
       implicit none

       integer, intent(in) :: nlon, start, end
       logical :: flag0
       real, dimension(nlon) :: work
       integer :: jl

       flag0 = .true.

       do jl=start,end
          work(jl) = 1.
       enddo
       ! random comment
    end subroutine some_kernel
    """

    routine = Subroutine.from_source(fcode_kernel, frontend=frontend)

    scc_transform = (SCCDevectorTransformation(horizontal=horizontal, trim_vector_sections=trim_vector_sections),)
    scc_transform += (SCCRevectorTransformation(horizontal=horizontal),)

    for transform in scc_transform:
        transform.apply(routine, role='kernel', targets=['some_kernel',])

    assign = FindNodes(Assignment).visit(routine.body)[0]
    loop = FindNodes(Loop).visit(routine.body)[0]
    comment = [c for c in FindNodes(Comment).visit(routine.body) if c.text == '! random comment'][0]

    # check we found the right assignment
    assert assign.lhs.name.lower() == 'flag0'

    # check we found the right comment
    assert comment.text == '! random comment'

    if trim_vector_sections:
        assert assign not in loop.body
        assert assign in routine.body.body

        assert comment not in loop.body
        assert comment in routine.body.body
    else:
        assert assign in loop.body
        assert assign not in routine.body.body

        assert comment in loop.body
        assert comment not in routine.body.body


@pytest.mark.parametrize('frontend', available_frontends())
@pytest.mark.parametrize('trim_vector_sections', [False, True])
def test_single_column_coalesced_vector_section_trim_nested(frontend, horizontal, trim_vector_sections):
    """
    Test the trimming of vector-sections to exclude nested scalar assignments.
    """

    fcode_kernel = """
    subroutine some_kernel(start, end, nlon, flag0)
       implicit none

       integer, intent(in) :: nlon, start, end
       logical, intent(in) :: flag0
       logical :: flag1, flag2
       real, dimension(nlon) :: work

       integer :: jl

       if(flag0)then
         flag1 = .true.
         flag2 = .false.
       else
         flag1 = .false.
         flag2 = .true.
       endif

       do jl=start,end
          work(jl) = 1.
       enddo
    end subroutine some_kernel
    """

    routine = Subroutine.from_source(fcode_kernel, frontend=frontend)

    scc_transform = (SCCDevectorTransformation(horizontal=horizontal, trim_vector_sections=trim_vector_sections),)
    scc_transform += (SCCRevectorTransformation(horizontal=horizontal),)

    for transform in scc_transform:
        transform.apply(routine, role='kernel', targets=['some_kernel',])

    cond = FindNodes(Conditional).visit(routine.body)[0]
    loop = FindNodes(Loop).visit(routine.body)[0]

    if trim_vector_sections:
        assert cond not in loop.body
        assert cond in routine.body.body
    else:
        assert cond in loop.body
        assert cond not in routine.body.body


@pytest.mark.parametrize('frontend', available_frontends())
@pytest.mark.parametrize('trim_vector_sections', [False, True])
def test_single_column_coalesced_vector_section_trim_complex(
        frontend, horizontal, vertical, blocking, trim_vector_sections
):
    """
    Test to highlight the limitations of vector-section trimming.
    """

    fcode_kernel = """
    subroutine some_kernel(start, end, nlon, flag0)
       implicit none

       integer, intent(in) :: nlon, start, end
       logical, intent(in) :: flag0
       logical :: flag1, flag2
       real, dimension(nlon) :: work, work1

       integer :: jl

       flag1 = .true.
       if(flag0)then
         flag2 = .false.
       else
         work1(start:end) = 1.
       endif

       do jl=start,end
          work(jl) = 1.
       enddo
    end subroutine some_kernel
    """

    routine = Subroutine.from_source(fcode_kernel, frontend=frontend)

    scc_pipeline = SCCVectorPipeline(
        horizontal=horizontal, vertical=vertical, block_dim=blocking,
        directive='openacc', trim_vector_sections=trim_vector_sections
    )
    scc_pipeline.apply(routine, role='kernel', targets=['some_kernel',])

    assign = FindNodes(Assignment).visit(routine.body)[0]

    # check we found the right assignment
    assert assign.lhs.name.lower() == 'flag1'

    cond = FindNodes(Conditional).visit(routine.body)[0]
    loop = FindNodes(Loop).visit(routine.body)[0]

    assert cond in loop.body
    assert cond not in routine.body.body
    if trim_vector_sections:
        assert assign not in loop.body
        assert(len(FindNodes(Assignment).visit(loop.body)) == 3)
    else:
        assert assign in loop.body
        assert(len(FindNodes(Assignment).visit(loop.body)) == 4)


@pytest.mark.parametrize('frontend', available_frontends())
def test_scc_revector_driver(frontend, horizontal, blocking):
    """
    Test revectorisation at the driver level.
    """

    fcode_driver = """
  SUBROUTINE column_driver(nlon, nz, q, t, nb)
    INTEGER, INTENT(IN)   :: nlon, nz, nb  ! Size of the horizontal and vertical
    REAL, INTENT(INOUT)   :: t(nlon,nz,nb)
    REAL, INTENT(INOUT)   :: q(nlon,nz,nb)
    INTEGER :: b, start, end

    start = 1
    end = nlon
    do b=1, nb
      call compute_column(start, end, nlon, nz, q(:,:,b), t(:,:,b))
    end do
  END SUBROUTINE column_driver
"""

    fcode_kernel = """
  SUBROUTINE compute_column(start, end, nlon, nz, q, t)
    INTEGER, INTENT(IN) :: start, end  ! Iteration indices
    INTEGER, INTENT(IN) :: nlon, nz    ! Size of the horizontal and vertical
    REAL, INTENT(INOUT) :: t(nlon,nz)
    REAL, INTENT(INOUT) :: q(nlon,nz)
    INTEGER :: jl, jk
    REAL :: c

    c = 5.345
    DO jk = 2, nz
      DO jl = start, end
        t(jl, jk) = c * jk
        q(jl, jk) = q(jl, jk-1) + t(jl, jk) * c
      END DO
    END DO

    ! The scaling is purposefully upper-cased
    DO JL = START, END
      Q(JL, NZ) = Q(JL, NZ) * C
    END DO
  END SUBROUTINE compute_column
"""
    kernel = Subroutine.from_source(fcode_kernel, frontend=frontend)
    driver = Subroutine.from_source(fcode_driver, frontend=frontend)

    # Ensure we have three loops in the kernel prior to transformation
    kernel_loops = FindNodes(Loop).visit(kernel.body)
    assert len(kernel_loops) == 3

    pipeline = Pipeline(
        classes=(SCCDevectorTransformation, SCCRevectorDriverTransformation),
        horizontal=horizontal, block_dim=blocking
    )
    pipeline.apply(driver, role='driver', targets=('compute_column',))
    pipeline.apply(kernel, role='kernel')

    # Check that vector loop exists in driver
    driver_loops = FindNodes(Loop).visit(driver.body)
    assert len(driver_loops) == 2
    assert driver_loops[0].variable == 'b'
    assert driver_loops[0].bounds == '1:nb'
    assert driver_loops[1].variable == 'jl'
    assert driver_loops[1].bounds == 'start:end'

    # Check that the vector index is passed down via keyword-args
    driver_calls = FindNodes(CallStatement).visit(driver.body)
    assert len(driver_calls) == 1
    assert ('jl', 'jl') in driver_calls[0].kwarguments

    # Now check loops and arguments in kernel
    kernel_loops = FindNodes(Loop).visit(kernel.body)
    assert len(kernel_loops) == 1
    assert kernel_loops[0].variable == 'jk'
    assert kernel_loops[0].bounds == '2:nz'

    # Check that vector index variable is declared correctly
    assert 'jl' in kernel.arguments
    decls = tuple(
        decl for decl in FindNodes(VariableDeclaration).visit(kernel.spec)
        if 'jl' in decl.symbols
    )
    assert len(decls) == 1
    assert len(decls[0].symbols) == 1
    assert decls[0].symbols[0].type.intent == 'in'

    # Check that the vector label has been remove
    assert 'vector_section' not in kernel.to_fortran()
