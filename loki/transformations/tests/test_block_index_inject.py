# (C) Copyright 2018- ECMWF.
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

from shutil import rmtree
import pytest

from loki import (
    Dimension, gettempdir, Scheduler, OMNI, FindNodes, Assignment, FindVariables, CallStatement, Subroutine,
    Item, available_frontends, Module, fgen, symbols as sym, ir
)
from loki.transformations import (
        BlockViewToFieldViewTransformation, InjectBlockIndexTransformation,
        LowerBlockIndexTransformation, LowerBlockLoopTransformation
)

@pytest.fixture(scope='module', name='horizontal')
def fixture_horizontal():
    return Dimension(name='horizontal', size='nlon', index='jl', bounds=('start', 'end'),
                     aliases=('nproma',), bounds_aliases=('bnds%start', 'bnds%end'))


@pytest.fixture(scope='module', name='blocking')
def fixture_blocking():
    return Dimension(name='blocking', size='nb', index='ibl', index_aliases='bnds%kbl')


@pytest.fixture(scope='function', name='config')
def fixture_config():
    """
    Default configuration dict with basic options.
    """
    return {
        'default': {
            'mode': 'idem',
            'role': 'kernel',
            'expand': True,
            'strict': True,
            'enable_imports': True,
            'disable': ['*%init', '*%final', 'abor1'],
        },
    }


@pytest.fixture(scope='module', name='blockview_to_fieldview_code', params=[True, False])
def fixture_blockview_to_fieldview_code(request):
    fcode = {
        #-------------
        'variable_mod': (
        #-------------
"""
module variable_mod
  implicit none

  type variable_3d
      real, pointer :: p(:,:) => null()
      real, pointer :: p_field(:,:,:) => null()
  end type variable_3d

  type variable_3d_ptr
      integer :: comp
      type(variable_3d), pointer :: ptr => null()
  end type variable_3d_ptr

end module variable_mod
"""
        ).strip(),
        #--------------------
        'field_variables_mod': (
        #--------------------
"""
module field_variables_mod
  use variable_mod, only: variable_3d, variable_3d_ptr
  implicit none

  type field_variables
      type(variable_3d_ptr), allocatable :: gfl_ptr_g(:)
      type(variable_3d_ptr), pointer :: gfl_ptr(:) => null()
      type(variable_3d) :: var
  end type field_variables

end module field_variables_mod
"""
        ).strip(),
        #-------------------
        'container_type_mod': (
        #-------------------
"""
module container_type_mod
  implicit none

  type container_3d_var
    real, pointer :: p(:,:) => null()
    real, pointer :: p_field(:,:,:) => null()
  end type container_3d_var

  type container_type
    type(container_3d_var), allocatable :: vars(:)
  end type container_type

end module container_type_mod
"""
        ).strip(),
        #--------------
        'dims_type_mod': (
        #--------------
"""
module dims_type_mod
   type dims_type
      integer :: start, end, kbl, nb
   end type dims_type
end module dims_type_mod
"""
        ).strip(),
        #-------
        'driver': (
        #-------
f"""
subroutine driver(data, ydvars, container, nlon, nlev, {'start, end, nb' if request.param else 'bnds'})
   use field_array_module, only: field_3rb_array
   use container_type_mod, only: container_type
   use field_variables_mod, only: field_variables
   {'use dims_type_mod, only: dims_type' if not request.param else ''}
   implicit none

   #include "kernel.intfb.h"

   real, intent(inout) :: data(:,:,:)
   integer, intent(in) :: nlon, nlev
   type(field_variables), intent(inout) :: ydvars
   type(container_type), intent(inout) :: container
   {'integer, intent(in) :: start, end, nb' if request.param else 'type(dims_type), intent(in) :: bnds'}

   integer :: ibl
   type(field_3rb_array) :: yla_data

   call yla_data%init(data)

   do ibl=1,{'nb' if request.param else 'bnds%nb'}
      {'bnds%kbl = ibl' if not request.param else ''}
      call kernel(nlon, nlev, {'start, end, ibl' if request.param else 'bnds'}, ydvars, container, yla_data)
   enddo

   call yla_data%final()

end subroutine driver
"""
        ).strip(),
        #-------
        'kernel': (
        #-------
f"""
subroutine kernel(nlon, nlev, {'start, end, ibl' if request.param else 'bnds'}, ydvars, container, yla_data)
   use field_array_module, only: field_3rb_array
   use container_type_mod, only: container_type
   use field_variables_mod, only: field_variables
   {'use dims_type_mod, only: dims_type' if not request.param else ''}
   implicit none

#include "another_kernel.intfb.h"
#include "abor1.intfb.h"

   integer, intent(in) :: nlon, nlev
   type(field_variables), intent(inout) :: ydvars
   type(container_type), intent(inout) :: container
   {'integer, intent(in) :: start, end, ibl' if request.param else 'type(dims_type), intent(in) :: bnds'}
   type(field_3rb_array), intent(inout) :: yda_data

   integer :: jl, jfld
   {'associate(start=>bnds%start, end=>bnds%end, ibl=>bnds%kbl)' if not request.param else ''}

   if(nlon < 0) call abor1('kernel')

   ydvars%var%p_field(:,:) = 0. !... this should only get the block-index
   ydvars%var%p_field(:,:,ibl) = 0. !... this should be untouched

   yda_data%p(start:end,:) = 1
   ydvars%var%p(start:end,:) = 1

   do jfld=1,size(ydvars%gfl_ptr)
      do jl=start,end
         ydvars%gfl_ptr(jfld)%ptr%p(jl,:) = yda_data%p(jl,:)
         container%vars(ydvars%gfl_ptr(jfld)%comp)%p(jl,:) = 0.
      enddo
   enddo

   call another_kernel(nlon, nlev, data=yda_data%p)

   {'end associate' if not request.param else ''}
end subroutine kernel
"""
        ).strip(),
        #-------
        'another_kernel': (
        #-------
"""
subroutine another_kernel(nproma, nlev, data)
   implicit none
   !... not a sequential routine but still labelling it as one to test the
   !... bail-out mechanism
   !$loki routine seq
   integer, intent(in) :: nproma, nlev
   real, intent(inout) :: data(nproma, nlev)
end subroutine another_kernel
"""
        ).strip()
    }

    workdir = gettempdir()/'test_blockview_to_fieldview'
    if workdir.exists():
        rmtree(workdir)
    workdir.mkdir()
    for name, code in fcode.items():
        (workdir/f'{name}.F90').write_text(code)

    yield workdir, request.param

    rmtree(workdir)


@pytest.mark.parametrize('frontend', available_frontends(xfail=[(OMNI,
                         'OMNI fails to import undefined module.')]))
def test_blockview_to_fieldview_pipeline(horizontal, blocking, config, frontend, blockview_to_fieldview_code, tmp_path):

    config['routines'] = {
        'driver': {'role': 'driver'}
    }

    scheduler = Scheduler(
        paths=(blockview_to_fieldview_code[0],), config=config, seed_routines='driver', frontend=frontend,
        xmods=[tmp_path]
    )
    scheduler.process(BlockViewToFieldViewTransformation(horizontal, global_gfl_ptr=True))
    scheduler.process(InjectBlockIndexTransformation(blocking))

    kernel = scheduler['#kernel'].ir
    aliased_bounds = not blockview_to_fieldview_code[1]
    ibl_expr = blocking.index
    if aliased_bounds:
        ibl_expr = blocking.index_expressions[1]

    assigns = FindNodes(Assignment).visit(kernel.body)

    # check that access pointers for arrays without horizontal index in dimensions were not updated
    assert assigns[0].lhs == f'ydvars%var%p_field(:,:,{ibl_expr})'
    assert assigns[1].lhs == f'ydvars%var%p_field(:,:,{ibl_expr})'

    # check that vector notation was resolved correctly
    assert assigns[2].lhs == f'yda_data%p_field(jl, :, {ibl_expr})'
    assert assigns[3].lhs == f'ydvars%var%p_field(jl, :, {ibl_expr})'

    # check thread-local ydvars%gfl_ptr was replaced with its global equivalent
    gfl_ptr_vars = {v for v in FindVariables().visit(kernel.body) if 'ydvars%gfl_ptr' in v.name.lower()}
    gfl_ptr_g_vars = {v for v in FindVariables().visit(kernel.body) if 'ydvars%gfl_ptr_g' in v.name.lower()}
    assert gfl_ptr_g_vars
    assert not gfl_ptr_g_vars - gfl_ptr_vars

    assert assigns[4].lhs == f'ydvars%gfl_ptr_g(jfld)%ptr%p_field(jl,:,{ibl_expr})'
    assert assigns[4].rhs == f'yda_data%p_field(jl,:,{ibl_expr})'
    assert assigns[5].lhs == f'container%vars(ydvars%gfl_ptr_g(jfld)%comp)%p_field(jl,:,{ibl_expr})'

    # check callstatement was updated correctly
    calls = FindNodes(CallStatement).visit(kernel.body)
    assert f'yda_data%p_field(:,:,{ibl_expr})' in calls[1].arg_map.values()


@pytest.mark.parametrize('frontend', available_frontends(xfail=[(OMNI,
                         'OMNI fails to import undefined module.')]))
@pytest.mark.parametrize('global_gfl_ptr', [False, True])
def test_blockview_to_fieldview_only(horizontal, blocking, config, frontend, blockview_to_fieldview_code,
                                     global_gfl_ptr, tmp_path):

    config['routines'] = {
        'driver': {'role': 'driver'}
    }

    scheduler = Scheduler(
        paths=(blockview_to_fieldview_code[0],), config=config, seed_routines='driver', frontend=frontend,
        xmods=[tmp_path]
    )
    scheduler.process(BlockViewToFieldViewTransformation(horizontal, global_gfl_ptr=global_gfl_ptr))

    kernel = scheduler['#kernel'].ir
    aliased_bounds = not blockview_to_fieldview_code[1]
    ibl_expr = blocking.index
    if aliased_bounds:
        ibl_expr = blocking.index_expressions[1]

    assigns = FindNodes(Assignment).visit(kernel.body)

    # check that access pointers for arrays without horizontal index in dimensions were not updated
    assert assigns[0].lhs == 'ydvars%var%p_field(:,:)'
    assert assigns[1].lhs == f'ydvars%var%p_field(:,:,{ibl_expr})'

    # check that vector notation was resolved correctly
    assert assigns[2].lhs == 'yda_data%p_field(jl, :)'
    assert assigns[3].lhs == 'ydvars%var%p_field(jl, :)'

    # check thread-local ydvars%gfl_ptr was replaced with its global equivalent
    if global_gfl_ptr:
        gfl_ptr_vars = {v for v in FindVariables().visit(kernel.body) if 'ydvars%gfl_ptr' in v.name.lower()}
        gfl_ptr_g_vars = {v for v in FindVariables().visit(kernel.body) if 'ydvars%gfl_ptr_g' in v.name.lower()}
        assert gfl_ptr_g_vars
        assert not gfl_ptr_g_vars - gfl_ptr_vars
    else:
        assert not {v for v in FindVariables().visit(kernel.body) if 'ydvars%gfl_ptr_g' in v.name.lower()}

    assert assigns[4].rhs == 'yda_data%p_field(jl,:)'
    if global_gfl_ptr:
        assert assigns[4].lhs == 'ydvars%gfl_ptr_g(jfld)%ptr%p_field(jl,:)'
        assert assigns[5].lhs == 'container%vars(ydvars%gfl_ptr_g(jfld)%comp)%p_field(jl,:)'
    else:
        assert assigns[4].lhs == 'ydvars%gfl_ptr(jfld)%ptr%p_field(jl,:)'
        assert assigns[5].lhs == 'container%vars(ydvars%gfl_ptr(jfld)%comp)%p_field(jl,:)'

    # check callstatement was updated correctly
    calls = FindNodes(CallStatement).visit(kernel.body)
    assert 'yda_data%p_field' in calls[1].arg_map.values()


@pytest.mark.parametrize('frontend', available_frontends(xfail=[(OMNI,
                         'OMNI correctly complains about rank mismatch in assignment.')]))
def test_simple_blockindex_inject(blocking, frontend):
    fcode = """
subroutine kernel(nlon,nlev,nb,var)
  implicit none

  interface
    subroutine compute(nlon,nlev,var)
      implicit none
      integer, intent(in) :: nlon,nlev
      real, intent(inout) :: var(nlon,nlev)
    end subroutine compute
  end interface

  integer, intent(in) :: nlon,nlev,nb
  real, intent(inout) :: var(nlon,nlev,4,nb) !... this dummy arg was potentially promoted by a previous transformation

  integer :: ibl

  do ibl=1,nb !... this loop was potentially lowered by a previous transformation
     var(:,:,:) = 0.
     call compute(nlon,nlev,var(:,:,1))
  enddo

end subroutine kernel
"""

    kernel = Subroutine.from_source(fcode, frontend=frontend)
    InjectBlockIndexTransformation(blocking).apply(kernel, role='kernel', targets=('compute',))

    assigns = FindNodes(Assignment).visit(kernel.body)
    assert assigns[0].lhs == 'var(:,:,:,ibl)'

    calls = FindNodes(CallStatement).visit(kernel.body)
    assert 'var(:,:,1,ibl)' in calls[0].arguments


@pytest.mark.parametrize('frontend', available_frontends(xfail=[(OMNI,
                         'OMNI complains about undefined type.')]))
def test_blockview_to_fieldview_exception(frontend, horizontal):
    fcode = """
subroutine kernel(nlon,nlev,start,end,var)
  implicit none

  interface
    subroutine compute(nlon,nlev,var)
      implicit none
      integer, intent(in) :: nlon,nlev
      real, intent(inout) :: var(nlon,nlev)
    end subroutine compute
  end interface

  integer, intent(in) :: nlon,nlev,start,end
  type(wrapped_field) :: var

  call compute(nlon,nlev,var%p)

end subroutine kernel
"""

    kernel = Subroutine.from_source(fcode, frontend=frontend)
    item = Item(name='#kernel', source=kernel)
    item.trafo_data['BlockViewToFieldViewTransformation'] = {'definitions': []}
    with pytest.raises(KeyError):
        BlockViewToFieldViewTransformation(horizontal).apply(kernel, item=item, role='kernel',
                                           targets=('compute',))

    with pytest.raises(RuntimeError):
        BlockViewToFieldViewTransformation(horizontal).apply(kernel, role='kernel',
                                           targets=('compute',))


@pytest.mark.parametrize('frontend', available_frontends(xfail=[(OMNI,
                         'OMNI correctly complains about rank mismatch in assignment.')]))
@pytest.mark.parametrize('block_dim_arg', (False, True))
@pytest.mark.parametrize('recurse_to_kernels', (False, True))
def test_simple_lower_loop(blocking, frontend, block_dim_arg, recurse_to_kernels):

    fcode_driver = f"""
subroutine driver(nlon,nlev,nb,var)
  implicit none
  use kernel_mod, only: kernel
  integer, intent(in) :: nlon,nlev,nb
  real, intent(inout) :: var(nlon,nlev,nb)
  real :: some_var(nlon,nlev,nb)
  integer :: ibl
  integer :: offset
  integer :: some_val
  integer :: loop_start, loop_end
  loop_start = 2
  loop_end = nb
  some_val = 0
  offset = 1
  !$omp test
  do ibl=loop_start, loop_end
    ibl = ibl - offset + some_val
    call kernel(nlon,nlev,var(:,:,ibl), some_var(:,:,ibl),offset, loop_start, loop_end{', ibl, nb' if block_dim_arg else ''})
  enddo
end subroutine driver
"""

    fcode_kernel = f"""
module kernel_mod
implicit none
contains
subroutine kernel(nlon,nlev,var,another_var,icend,lstart,lend{', ibl, nb' if block_dim_arg else ''})
  implicit none
  use compute_mod, only: compute
  integer, intent(in) :: nlon,nlev,icend,lstart,lend
  real, intent(inout) :: var(nlon,nlev)
  real, intent(inout) :: another_var(nlon, nlev)
  {'integer, intent(in) :: ibl, nb' if block_dim_arg else ''}
  integer :: jk, jl
  var(:,:) = 0.
  do jk = 1,nlev
    do jl = 1, nlon
      var(jl, jk) = 0.
    end do
  end do
  call compute(nlon,nlev,var)
  call compute(nlon,nlev,another_var)
end subroutine kernel
end module kernel_mod
"""

    fcode_nested_kernel = """
module compute_mod
implicit none
contains
subroutine compute(nlon,nlev,var)
  implicit none
  integer, intent(in) :: nlon,nlev
  real, intent(inout) :: var(nlon,nlev)
  var(:,:) = 0.
end subroutine compute
end module compute_mod
"""

    # recurse_to_kernels = True # False
    # kernel = Subroutine.from_source(fcode, frontend=frontend)
    nested_kernel_mod = Module.from_source(fcode_nested_kernel, frontend=frontend)
    kernel_mod = Module.from_source(fcode_kernel, frontend=frontend, definitions=nested_kernel_mod) 
    driver = Subroutine.from_source(fcode_driver, frontend=frontend, definitions=kernel_mod)
    print(f"kernel.symbol_table: {dict(kernel_mod['kernel'].symbol_attrs)}")
    # kernel = Subroutine.from_source(fcode, frontend=frontend)
    LowerBlockIndexTransformation(blocking, recurse_to_kernels=recurse_to_kernels).apply(driver, role='driver', targets=('kernel',))
    LowerBlockIndexTransformation(blocking, recurse_to_kernels=recurse_to_kernels).apply(kernel_mod['kernel'], role='kernel', targets=('compute',))
    LowerBlockIndexTransformation(blocking, recurse_to_kernels=recurse_to_kernels).apply(nested_kernel_mod['compute'], role='kernel')

    kernel_call = FindNodes(ir.CallStatement).visit(driver.body)[0]
    if block_dim_arg:
        assert blocking.size in kernel_call.arguments
        assert blocking.index in kernel_call.arguments
    else:
        assert blocking.size in [kwarg[0] for kwarg in kernel_call.kwarguments]
        assert blocking.index in [kwarg[0] for kwarg in kernel_call.kwarguments]
    assert blocking.size in kernel_mod['kernel'].arguments
    assert blocking.index in kernel_mod['kernel'].arguments

    kernel_array_args = [arg for arg in kernel_mod['kernel'].arguments if isinstance(arg, sym.Array)]
    nested_kernel_array_args = [arg for arg in nested_kernel_mod['compute'].arguments if isinstance(arg, sym.Array)]
    for array in kernel_array_args:
        assert blocking.size in array.dimensions
        assert blocking.size in array.shape
    if recurse_to_kernels:
        for array in nested_kernel_array_args:
            assert blocking.size in array.dimensions
            assert blocking.size in array.shape
    else:
        for array in nested_kernel_array_args:
            assert blocking.size not in array.dimensions
            assert blocking.size not in array.shape

    arrays = [var for var in FindVariables().visit(kernel_mod['kernel'].body) if isinstance(var, sym.Array)]
    for array in arrays:
        if array.name.lower() in [arg.name.lower() for arg in kernel_mod['kernel'].arguments]:
            assert blocking.size in array.shape
            assert blocking.index not in array.dimensions

    InjectBlockIndexTransformation(blocking).apply(driver, role='driver', targets=('kernel',))
    InjectBlockIndexTransformation(blocking).apply(kernel_mod['kernel'], role='kernel', targets=('compute',))
    InjectBlockIndexTransformation(blocking).apply(nested_kernel_mod['compute'], role='kernel')

    arrays = [var for var in FindVariables().visit(kernel_mod['kernel'].body) if isinstance(var, sym.Array)]
    for array in arrays:
        if array.name.lower() in [arg.name.lower() for arg in kernel_mod['kernel'].arguments]:
            assert blocking.size in array.shape
            assert not array.dimensions or blocking.index in array.dimensions

    driver_loops = FindNodes(ir.Loop).visit(driver.body)
    kernel_loops = FindNodes(ir.Loop).visit(kernel_mod['kernel'].body)
    assert any(loop.variable == blocking.index for loop in driver_loops)
    assert not any(loop.variable == blocking.index for loop in kernel_loops)

    LowerBlockLoopTransformation(blocking).apply(driver, role='driver', targets=('kernel',))
    LowerBlockLoopTransformation(blocking).apply(kernel_mod['kernel'], role='kernel', targets=('compute',))
    LowerBlockLoopTransformation(blocking).apply(nested_kernel_mod['compute'], role='kernel')

    """
    driver_loops = FindNodes(ir.Loop).visit(driver.body)
    kernel_loops = FindNodes(ir.Loop).visit(kernel_mod['kernel'].body)
    assert not any(loop.variable == blocking.index for loop in driver_loops)
    assert any(loop.variable == blocking.index for loop in kernel_loops)
    if block_dim_arg:
        assert blocking.size in kernel_call.arguments
        assert blocking.index not in kernel_call.arguments
    else:
        assert blocking.size in [kwarg[0] for kwarg in kernel_call.kwarguments]
        assert blocking.index not in [kwarg[0] for kwarg in kernel_call.kwarguments]
    assert blocking.size in kernel_mod['kernel'].arguments
    assert blocking.index not in kernel_mod['kernel'].arguments
    """

    print(f"---------------\ndriver:\n{fgen(driver)}")
    print(f"---------------\nkernel:\n{fgen(kernel_mod['kernel'])}")
    print(f"---------------\nkernel:\n{fgen(nested_kernel_mod['compute'])}")
    print("\n\n")
    # print(f"kernel.symbol_table: {dict(kernel['kernel'].symbol_attrs)}")
    # assigns = FindNodes(Assignment).visit(kernel.body)
    # assert assigns[0].lhs == 'var(:,:,ibl)'
    # calls = FindNodes(CallStatement).visit(kernel.body)
    # assert 'var(:,:,ibl)' in calls[0].arguments

@pytest.mark.parametrize('frontend', available_frontends(xfail=[(OMNI,
                         'OMNI correctly complains about rank mismatch in assignment.')]))
def test_lower_loop(blocking, frontend):

    fcode_driver = """
subroutine driver(nlon,nlev,nb,var)
  implicit none
  use kernel_mod, only: kernel
  integer, intent(in) :: nlon,nlev,nb
  real, intent(inout) :: var(nlon,nlev,nb)
  real :: some_var(nlon,nlev,nb)
  integer :: jkglo, ibl
  do jkglo=1,nb,nlev
    ibl = (jkglo-1)/(nlev+1)
    call kernel(nlon,nlev,var(:,:,ibl), some_var(:,:,ibl))
  enddo
end subroutine driver
"""

    fcode_kernel = """
module kernel_mod
implicit none
contains
subroutine kernel(nlon,nlev,var,another_var)
  implicit none
  use compute_mod, only: compute
  integer, intent(in) :: nlon,nlev
  real, intent(inout) :: var(nlon,nlev)
  real, intent(inout) :: another_var(nlon, nlev)
  var(:,:) = 0.
  call compute(nlon,nlev,var)
  call compute(nlon,nlev,another_var)
end subroutine kernel
end module kernel_mod
"""

    fcode_nested_kernel = """
module compute_mod
implicit none
contains
subroutine compute(nlon,nlev,var)
  implicit none
  integer, intent(in) :: nlon,nlev
  real, intent(inout) :: var(nlon,nlev)
  var(:,:) = 0.
end subroutine compute
end module compute_mod
"""

    recurse_to_kernels = True
    # kernel = Subroutine.from_source(fcode, frontend=frontend)
    nested_kernel_mod = Module.from_source(fcode_nested_kernel, frontend=frontend)
    kernel_mod = Module.from_source(fcode_kernel, frontend=frontend, definitions=nested_kernel_mod)
    driver = Subroutine.from_source(fcode_driver, frontend=frontend, definitions=kernel_mod)
    print(f"kernel.symbol_table: {dict(kernel_mod['kernel'].symbol_attrs)}")
    # kernel = Subroutine.from_source(fcode, frontend=frontend)
    LowerBlockIndexTransformation(blocking, recurse_to_kernels=recurse_to_kernels).apply(driver, role='driver', targets=('kernel',))
    LowerBlockIndexTransformation(blocking, recurse_to_kernels=recurse_to_kernels).apply(kernel_mod['kernel'], role='kernel', targets=('compute',))
    LowerBlockIndexTransformation(blocking, recurse_to_kernels=recurse_to_kernels).apply(nested_kernel_mod['compute'], role='kernel')
    InjectBlockIndexTransformation(blocking).apply(driver, role='driver', targets=('kernel',))
    InjectBlockIndexTransformation(blocking).apply(kernel_mod['kernel'], role='kernel', targets=('compute',))
    InjectBlockIndexTransformation(blocking).apply(nested_kernel_mod['compute'], role='kernel')

    LowerBlockLoopTransformation(blocking).apply(driver, role='driver', targets=('kernel',))
    LowerBlockLoopTransformation(blocking).apply(kernel_mod['kernel'], role='kernel', targets=('compute',))
    LowerBlockLoopTransformation(blocking).apply(nested_kernel_mod['compute'], role='kernel')

    print(f"---------------\ndriver:\n{fgen(driver)}")
    print(f"---------------\nkernel:\n{fgen(kernel_mod['kernel'])}")
    print(f"---------------\nkernel:\n{fgen(nested_kernel_mod['compute'])}")
    print("\n\n")
    # print(f"kernel.symbol_table: {dict(kernel['kernel'].symbol_attrs)}")
    # assigns = FindNodes(Assignment).visit(kernel.body)
    # assert assigns[0].lhs == 'var(:,:,ibl)'
    # calls = FindNodes(CallStatement).visit(kernel.body)
    # assert 'var(:,:,ibl)' in calls[0].arguments
