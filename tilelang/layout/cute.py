"""CuTe layout IR objects and layout-algebra Python API.

This module is the single Python home for everything CuTe-layout related:

* the FFI-registered IR objects (:class:`Swizzle`, :class:`IntTuple` /
  :class:`IntScalar` / :class:`IntTupleArray`, :class:`CuteLayout`,
  :class:`ComposedLayout`);
* the recovery entry points :meth:`ComposedLayout.from_tilelang` /
  :meth:`CuteLayout.from_tilelang`;
* the flat layout-algebra ops (:func:`make_layout`, :func:`coalesce`,
  :func:`right_inverse`, :func:`composition`) and the TMA box derivation
  :func:`derive_tma_tile`.

All of these are registered C++-side under the ``tl.cute.*`` FFI namespace and
surfaced through :mod:`tilelang.layout._cute_ffi_api`. Import as
``from tilelang.layout import cute`` and call ``cute.coalesce(...)`` etc.
"""

from __future__ import annotations

import tvm
import tvm_ffi
from tvm.ir.base import Node
from tvm.runtime import Scriptable

from tilelang._typing import BufferLikeType
from . import _cute_ffi_api


# ---------------------------------------------------------------------------
# IR objects (FFI-registered; keys must match the C++ TVM_FFI type keys).
# ---------------------------------------------------------------------------
@tvm_ffi.register_object("tl.cute.Swizzle")
class Swizzle(Node, Scriptable):
    """A CuTe-style XOR swizzle functor <b_bits, m_base, s_shift>."""

    b_bits: int
    m_base: int
    s_shift: int

    @property
    def is_swizzled(self) -> bool:
        """Whether the layout applies an XOR swizzle (b_bits > 0)."""
        return self.b_bits > 0

    def recast(self, old_bits: int, new_bits: int) -> Swizzle:
        """Reinterpret the swizzle when the buffer is viewed as a different
        element type. Only ``m_base`` shifts (by ``log2(old_bits/new_bits)``);
        ``b_bits`` and ``s_shift`` are preserved. Sizes are element widths in
        bits and must be byte-aligned powers of two."""
        return _cute_ffi_api.swizzle_recast(self, int(old_bits), int(new_bits))


@tvm_ffi.register_object("tl.cute.IntTuple")
class IntTuple(Node, Scriptable):
    """Base of a CuTe hierarchical IntTuple (either an :class:`IntScalar` leaf or
    an :class:`IntTupleArray` branch)."""


@tvm_ffi.register_object("tl.cute.IntScalar")
class IntScalar(IntTuple):
    """Abstract base of the three scalar-leaf kinds (CuTe's integral leaf):
    :class:`IntConst`, :class:`IntExpr`, :class:`ScaledBasis`. Dispatch by
    ``isinstance`` on the concrete kind."""


@tvm_ffi.register_object("tl.cute.IntConst")
class IntConst(IntScalar):
    """A compile-time integer leaf."""

    value: int


@tvm_ffi.register_object("tl.cute.IntExpr")
class IntExpr(IntScalar):
    """A dynamic (runtime-valued) integer leaf, carrying a PrimExpr ``value``."""

    value: object


@tvm_ffi.register_object("tl.cute.ScaledBasis")
class ScaledBasis(IntScalar):
    """A CuTe ScaledBasis leaf ``value * E<basis...>``: ``value`` is itself an
    :class:`IntScalar` scale and ``basis`` is the mode-path of the unit basis."""

    value: IntScalar
    basis: list


@tvm_ffi.register_object("tl.cute.IntTupleArray")
class IntTupleArray(IntTuple):
    """A branch of an IntTuple: a tuple of :class:`IntTuple` children."""

    fields: list


@tvm_ffi.register_object("tl.cute.Layout")
class CuteLayout(Node, Scriptable):
    """A CuTe layout described by hierarchical ``shape`` and ``stride``
    :class:`IntTuple` trees. ``shape`` / ``stride`` expose the flattened per-leaf
    integer lists (the common view)."""

    @property
    def shape(self) -> list:
        """The flattened per-leaf extents as a list of ints."""
        return list(_cute_ffi_api.layout_flat_shape(self))

    @property
    def stride(self) -> list:
        """The flattened per-leaf strides as a list of ints. Errors if any leaf
        is dynamic or a ScaledBasis; use :attr:`stride_leaves` for those."""
        return list(_cute_ffi_api.layout_flat_stride(self))

    @property
    def stride_leaves(self) -> list:
        """The flattened stride leaves as :class:`IntScalar` objects (preserves
        dynamic :class:`IntExpr` and :class:`ScaledBasis` leaves)."""
        return list(_cute_ffi_api.layout_stride_leaves(self))

    @staticmethod
    def from_tilelang(layout):
        """The affine special case of :meth:`ComposedLayout.from_tilelang`:
        recover a TileLang layout as a flat :class:`CuteLayout` when it has no
        swizzle and zero offset, else ``None``."""
        return _cute_ffi_api.layout_from_tilelang(layout)


@tvm_ffi.register_object("tl.cute.ComposedLayout")
class ComposedLayout(Node, Scriptable):
    """A CuTe ComposedLayout ``Swizzle o offset o Layout``. Evaluating at a
    coordinate ``x`` gives ``swizzle.apply(offset + layout(x))``."""

    swizzle: Swizzle
    offset: int
    layout: CuteLayout

    def recast(self, old_bits: int, new_bits: int) -> ComposedLayout:
        """Recast the whole composed layout into a different element width: the
        swizzle's ``m_base`` shifts and the plain layout's strides/offset scale
        by ``log2(old_bits/new_bits)``. Sizes must be byte-aligned powers of
        two."""
        return _cute_ffi_api.composed_layout_recast(self, int(old_bits), int(new_bits))

    @staticmethod
    def from_tilelang(layout, buffer: BufferLikeType = None):
        """Recover an arbitrary CUTLASS/CuTe XOR swizzle over an affine layout
        from a TileLang layout, as a CuTe :class:`ComposedLayout`
        (``Swizzle o offset o Layout``).

        Returns ``.swizzle`` (with ``b_bits``, ``m_base``, ``s_shift``,
        ``is_swizzled``) and ``.layout`` (the recovered unswizzled affine
        layout). A non-swizzled (linear) layout is reported with
        ``swizzle.b_bits == 0``. ``None`` is returned when the layout cannot be
        analyzed.

        Detection is dtype-agnostic and reports addresses in element-offset
        positions. When ``buffer`` is given, the result is recast into
        byte-address space using its element size (the CuTe/CUTLASS convention
        where swizzles act on byte addresses)."""
        mode = _cute_ffi_api.composed_layout_from_tilelang(layout)
        if buffer is None or mode is None:
            return mode
        from tilelang.layout.swizzle import _get_buffer_info

        _, _, dtype = _get_buffer_info(buffer)
        return mode.recast(int(tvm.DataType(dtype).bits), 8)


def int_const(value) -> IntConst:
    """A compile-time integer leaf."""
    return _cute_ffi_api.make_int_const(int(value))


def int_expr(value) -> IntExpr:
    """A dynamic (runtime-valued) integer leaf, wrapping a PrimExpr."""
    return _cute_ffi_api.make_int_expr(value)


def scaled_basis(value, basis) -> ScaledBasis:
    """A CuTe ScaledBasis leaf ``value * E<basis...>``. ``value`` may be an int
    or an :class:`IntScalar`; ``basis`` is the integer mode-path."""
    if not isinstance(value, IntScalar):
        value = int_const(value)
    return _cute_ffi_api.make_scaled_basis(value, [int(b) for b in basis])


def _as_stride_leaf(s) -> IntScalar:
    """Coerce a stride entry to an IntScalar leaf: pass leaves through, wrap a
    Python int as IntConst, and a PrimExpr (or other) as IntExpr."""
    if isinstance(s, IntScalar):
        return s
    if isinstance(s, int):
        return int_const(s)
    return int_expr(s)


def make_layout(shape, stride):
    """Build a flat :class:`CuteLayout` from per-mode ``shape`` and ``stride``
    lists (fastest mode first). ``stride`` entries may be Python ints, PrimExprs,
    or :class:`IntScalar` leaves (incl. :class:`ScaledBasis`)."""
    if all(isinstance(s, int) for s in stride):
        return _cute_ffi_api.make_layout([int(s) for s in shape], [int(s) for s in stride])
    leaves = [_as_stride_leaf(s) for s in stride]
    return _cute_ffi_api.make_layout_leaves([int(s) for s in shape], leaves)


def make_identity_layout(shape):
    """CuTe ``make_identity_layout`` / ``make_basis_like``: a flat layout over
    ``shape`` whose per-mode strides are the unit ScaledBases ``E<k>``."""
    return _cute_ffi_api.make_identity_layout([int(s) for s in shape])


def coalesce(layout):
    """CuTe ``coalesce``: merge contiguous adjacent modes, drop size-1 modes."""
    return _cute_ffi_api.coalesce(layout)


def right_inverse(layout):
    """CuTe ``right_inverse``: the layout ``r`` with ``layout(r(i)) == i``."""
    return _cute_ffi_api.right_inverse(layout)


def composition(lhs, rhs):
    """CuTe ``composition`` lhs o rhs: ``result(c) == lhs(rhs(c))``."""
    return _cute_ffi_api.composition(lhs, rhs)


def derive_tma_tile_layouts(gmem, smem_plain, tile_shape):
    """Derive the faithful-CuTe TMA decomposition (CuTe ``construct_tma_gbasis``).
    Returns the three :class:`CuteLayout`s ``(box, rest_gmem, rest_smem)`` or
    ``None`` when not TMA-expressible. ``gmem`` is the global tensor as a
    :class:`CuteLayout` (extents as shape, element strides as stride; may be
    dynamic). See :func:`derive_tma_tile` for the decoded tuple form."""
    out = _cute_ffi_api.derive_tma_tile(gmem, smem_plain, [int(s) for s in tile_shape])
    if out is None:
        return None
    return tuple(out)


def derive_tma_tile(gmem, smem_plain, tile_shape):
    """Decode :func:`derive_tma_tile_layouts` into ``(box, rest)`` or ``None``:

    * ``box`` -- list of ``(extent, axis)`` per descriptor box mode (fastest
      first); the single TMA descriptor covers these contiguous (unit gmem step)
      modes.
    * ``rest`` -- list of ``(extent, scale, axis, smem_stride)`` iteration modes
      replayed as separate TMA instructions: for digit ``d`` in ``range(extent)``
      the gmem coord of ``axis`` shifts by ``d*scale`` and the SMEM pointer by
      ``d*smem_stride``. CuTe truncates the box at the first non-contiguous global
      mode; those (and any >256 box overflow) land here.
    """
    out = derive_tma_tile_layouts(gmem, smem_plain, tile_shape)
    if out is None:
        return None
    box_l, rest_gmem, rest_smem = out
    box = [(int(e), int(s.basis[0])) for e, s in zip(box_l.shape, box_l.stride_leaves)]
    rest = []
    for e, sg, ss in zip(rest_gmem.shape, rest_gmem.stride_leaves, rest_smem.stride_leaves):
        e = int(e)
        if e == 1:  # scalar (1):(0) -> no rest
            continue
        rest.append((e, int(sg.value.value), int(sg.basis[0]), int(ss.value)))
    return box, rest
