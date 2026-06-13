/*!
 * \file cute_layout.h
 * \brief CuTe-style layout IR types for TileLang.
 */

#ifndef TVM_TL_LAYOUT_CUTE_LAYOUT_H_
#define TVM_TL_LAYOUT_CUTE_LAYOUT_H_

#include "support/check.h"
#include <tvm/ffi/container/array.h>
#include <tvm/ffi/object.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/runtime/base.h>
#include <tvm/tirx/buffer.h>

#include <cstdint>
#include <optional>
#include <utility>
#include <vector>

namespace tvm {
namespace tl {

// Forward declaration: the TileLang layout (defined in layout.h).
class Layout;

namespace cute {

using namespace ffi;
using namespace tirx;

/* A generic CUTLASS/CuTe-style XOR swizzle, described by three bit-field
 * parameters (b_bits, m_base, s_shift):
 *
 * 0bxxxxxxxxxxxxxxxYYYxxxxxxxZZZxxxx
 *                               ^--^ m_base  (least-significant bits kept
 * fixed)
 *                  ^-^       ^-^     b_bits   (number of mask bits)
 *                    ^---------^     s_shift  (distance YYY is shifted onto
 * ZZZ)
 *
 * apply(x) = x ^ ((x & yyy_msk) >> s_shift), where the YYY bits live at
 * [m_base + s_shift, m_base + s_shift + b_bits) and the ZZZ (target) bits at
 * [m_base, m_base + b_bits). The map is GF(2)-linear and an involution when the
 * source and target regions are disjoint (s_shift >= b_bits).
 *
 * A plain linear (non-swizzled) layout is represented as b_bits == 0; use
 * IsSwizzled() to distinguish it from an actual XOR swizzle.
 */
class SwizzleNode : public Object {
public:
  int b_bits{0};
  int m_base{0};
  int s_shift{0};

  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("tl.cute.Swizzle", SwizzleNode, Object);

  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<SwizzleNode>()
        .def_ro("b_bits", &SwizzleNode::b_bits)
        .def_ro("m_base", &SwizzleNode::m_base)
        .def_ro("s_shift", &SwizzleNode::s_shift);
  }

  // Whether the layout actually applies an XOR swizzle. A linear layout has
  // b_bits == 0 and is not swizzled.
  bool IsSwizzled() const { return b_bits > 0; }

  // Apply the swizzle to a physical offset: ZZZ ^= YYY.
  int64_t Apply(int64_t offset) const;
};

class Swizzle : public ObjectRef {
public:
  TVM_DLL Swizzle(int b_bits, int m_base, int s_shift);

  // A linear (non-swizzled) swizzle.
  static Swizzle Identity() { return Swizzle(0, 0, 0); }

  // Reinterpret the swizzle when the underlying buffer is viewed as a different
  // element type. The swizzle is anchored to a fixed physical byte granularity,
  // so only m_base shifts: by log2(old_bits) - log2(new_bits). b_bits and
  // s_shift are unchanged. old_bits/new_bits are element sizes in bits and must
  // be byte-aligned powers of two.
  TVM_DLL Swizzle Recast(int old_bits, int new_bits) const;

  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NULLABLE(Swizzle, ObjectRef, SwizzleNode);
};

/* A hierarchical CuTe IntTuple: either a scalar leaf or a tuple of IntTuples,
 * mirroring CuTe's int_tuple. A leaf is an IntScalar — the generic "integral"
 * element CuTe allows at the leaves — and comes in three concrete kinds,
 * modeled as distinct subtypes of IntScalar so consuming code dispatches by
 * type
 * (`leaf.as<IntConstNode>()`) rather than on a mode flag:
 *
 *   IntScalar (abstract)
 *     IntConst     — a compile-time integer value.
 *     IntExpr      — a dynamic runtime value (a PrimExpr), e.g. a symbolic
 *                    global stride; CuTe treats these as integral leaves too.
 *     ScaledBasis  — a CuTe ScaledBasis (value * E<basis...>): `value` is
 * itself an IntScalar (the scale, possibly dynamic) and `basis` is the
 * hierarchical mode-path of the unit basis vector. CuTe treats a ScaledBasis as
 * an integral leaf (not a tuple), so it lives at a leaf here too.
 *
 * The leaf is deliberately the ONLY place value semantics live: all algebra
 * goes through the leaf-arithmetic helpers below (ScalarMul / AsConst /
 * IsBasis /
 * ...), so a future fourth leaf kind is a change to those helpers and nowhere
 * else.
 *
 * The node types follow this codebase's BaseExprNode -> IntImmNode /
 * SeqStmtNode pattern (TVM ObjectRef inheritance; Array<Self> for recursion).
 */
class IntTupleNode : public Object {
public:
  // Non-final base: reserve a child-index block spanning the leaf subtree
  // (IntScalar + its 3 concrete kinds) and the branch type. Must exceed
  // IntScalarNode::_type_child_slots.
  static constexpr uint32_t _type_child_slots = 6;
  TVM_FFI_DECLARE_OBJECT_INFO("tl.cute.IntTuple", IntTupleNode, Object);
};

class IntTuple : public ObjectRef {
public:
  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NULLABLE(IntTuple, ObjectRef, IntTupleNode);
};

// Abstract base of the three scalar-leaf kinds (CuTe's integral leaf). It holds
// no fields; the concrete kind carries the value. Dispatch on the subtype.
class IntScalarNode : public IntTupleNode {
public:
  // Reserve a child block for IntConst / IntExpr / ScaledBasis (and headroom).
  static constexpr uint32_t _type_child_slots = 4;
  TVM_FFI_DECLARE_OBJECT_INFO("tl.cute.IntScalar", IntScalarNode, IntTupleNode);
};

class IntScalar : public IntTuple {
public:
  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NULLABLE(IntScalar, IntTuple,
                                             IntScalarNode);
};

// A compile-time integer leaf.
class IntConstNode : public IntScalarNode {
public:
  int64_t value{0};

  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("tl.cute.IntConst", IntConstNode,
                                    IntScalarNode);

  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<IntConstNode>().def_ro("value", &IntConstNode::value);
  }
};

class IntConst : public IntScalar {
public:
  TVM_DLL explicit IntConst(int64_t value);
  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NULLABLE(IntConst, IntScalar, IntConstNode);
};

// A dynamic (runtime-valued) integer leaf, carrying a PrimExpr.
class IntExprNode : public IntScalarNode {
public:
  PrimExpr value;

  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("tl.cute.IntExpr", IntExprNode,
                                    IntScalarNode);

  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<IntExprNode>().def_ro("value", &IntExprNode::value);
  }
};

class IntExpr : public IntScalar {
public:
  TVM_DLL explicit IntExpr(PrimExpr value);
  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NULLABLE(IntExpr, IntScalar, IntExprNode);
};

// A CuTe ScaledBasis leaf: value * E<basis...>. `value` is a (possibly dynamic)
// IntScalar scale; `basis` is the mode-path of the unit basis vector.
class ScaledBasisNode : public IntScalarNode {
public:
  IntScalar value;
  Array<int64_t> basis;

  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("tl.cute.ScaledBasis", ScaledBasisNode,
                                    IntScalarNode);

  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<ScaledBasisNode>()
        .def_ro("value", &ScaledBasisNode::value)
        .def_ro("basis", &ScaledBasisNode::basis);
  }
};

class ScaledBasis : public IntScalar {
public:
  TVM_DLL ScaledBasis(IntScalar value, Array<int64_t> basis);
  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NULLABLE(ScaledBasis, IntScalar,
                                             ScaledBasisNode);
};

// A branch: a tuple of IntTuples.
class IntTupleArrayNode : public IntTupleNode {
public:
  Array<IntTuple> fields;

  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("tl.cute.IntTupleArray", IntTupleArrayNode,
                                    IntTupleNode);

  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<IntTupleArrayNode>().def_ro("fields",
                                                &IntTupleArrayNode::fields);
  }
};

class IntTupleArray : public IntTuple {
public:
  TVM_DLL explicit IntTupleArray(Array<IntTuple> fields);
  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NULLABLE(IntTupleArray, IntTuple,
                                             IntTupleArrayNode);
};

/* ---- IntScalar leaf arithmetic (CuTe's integral-leaf ops) --------------- *
 * These are the ONLY functions that dispatch on the concrete leaf kind. All of
 * the layout algebra (coalesce/composition/...) is expressed through them, so a
 * new leaf kind is a localized change here. They mirror CuTe's basis_value /
 * basis_get and the integral operator* / equality on ScaledBasis.            */

// Is `s` a ScaledBasis leaf?
TVM_DLL bool IsBasis(const IntScalar &s);

// CuTe basis_value: the scale of a ScaledBasis, or `s` itself if not a basis.
TVM_DLL IntScalar BasisValue(const IntScalar &s);

// The basis mode-path of a ScaledBasis, or empty for a non-basis leaf.
TVM_DLL Array<int64_t> BasisPath(const IntScalar &s);

// Whether the leaf is a plain compile-time integer (an IntConst).
TVM_DLL bool IsConst(const IntScalar &s);

// The leaf's compile-time integer value. ICHECKs unless IsConst(s) (i.e. not an
// IntExpr or ScaledBasis); guard with IsConst when the leaf may be
// non-constant.
TVM_DLL int64_t AsConst(const IntScalar &s);

// The leaf's value as a PrimExpr: an IntConst becomes an int32 IntImm, an
// IntExpr returns its PrimExpr. ICHECKs on a ScaledBasis (no scalar value).
// Used to read the (possibly dynamic) result of routing a global property
// through a box mode.
TVM_DLL PrimExpr AsPrimExpr(const IntScalar &s);

// CuTe integral operator*: multiply two leaves. At most one may be a
// ScaledBasis (the result keeps its path); const*const folds to IntConst,
// otherwise to an IntExpr, with 0/1 identities applied.
TVM_DLL IntScalar ScalarMul(const IntScalar &a, const IntScalar &b);

// Whether two leaves are provably equal (used by coalesce's merge test): a
// ScaledBasis matches only a ScaledBasis with the same path and provably-equal
// scale; constants compare by value; exprs by StructuralEqual; else false.
TVM_DLL bool ProvablyEqual(const IntScalar &a, const IntScalar &b);

// CuTe basis_get over a flat layout: index `flat` by the single-level basis
// `path` (path[0]). ICHECKs deeper paths (not produced in our use).
TVM_DLL IntScalar BasisGet(const Array<int64_t> &path,
                           const std::vector<IntScalar> &flat);

/* ---- IntTuple free functions (CuTe int_tuple.hpp analogues) ------------- */

// Is `t` a scalar leaf (CuTe: !is_tuple)?
inline bool IsScalar(const IntTuple &t) {
  return t.as<IntScalarNode>() != nullptr;
}

// The scalar leaf as an IntScalar (ICHECK if `t` is a tuple).
TVM_DLL IntScalar AsScalar(const IntTuple &t);

// Top-level rank: number of children (1 for a leaf), CuTe's rank().
TVM_DLL int Rank(const IntTuple &t);

TVM_DLL IntTuple Get(const IntTuple &t, int64_t index);

// Product of all leaf values, CuTe's product()/size().
TVM_DLL int64_t Product(const IntTuple &t);

// Flatten a tree into a depth-1 IntTupleArray of its leaves, CuTe's flatten().
TVM_DLL IntTuple Flatten(const IntTuple &t);

// The flattened scalar leaves, in order (CuTe: the leaves of flatten(t)).
TVM_DLL std::vector<IntScalar> Leaves(const IntTuple &t);

// Higher-order helpers over the scalar leaves, CuTe's transform_leaf / fold.
// TransformLeaf rebuilds the tree applying `f : IntScalar -> IntScalar` to each
// leaf; FoldLeaves accumulates `f : (Acc, IntScalar) -> Acc` left-to-right.
template <class F> IntTuple TransformLeaf(const IntTuple &t, F &&f) {
  if (const auto *s = t.as<IntScalarNode>())
    return f(GetRef<IntScalar>(s));
  const auto *a = t.as<IntTupleArrayNode>();
  ICHECK(a != nullptr) << "TransformLeaf on a null IntTuple";
  Array<IntTuple> out;
  out.reserve(a->fields.size());
  for (const auto &child : a->fields)
    out.push_back(TransformLeaf(child, f));
  return IntTupleArray(out);
}

template <class Acc, class F>
Acc FoldLeaves(const IntTuple &t, Acc init, F &&f) {
  if (const auto *s = t.as<IntScalarNode>())
    return f(std::move(init), GetRef<IntScalar>(s));
  const auto *a = t.as<IntTupleArrayNode>();
  ICHECK(a != nullptr) << "FoldLeaves on a null IntTuple";
  for (const auto &child : a->fields)
    init = FoldLeaves(child, std::move(init), f);
  return init;
}

/* A hierarchical CuTe layout: `shape` and `stride` are congruent IntTuples
 * (same tree structure), exactly as in CuTe. A flat (depth-1) layout with plain
 * integer strides is the common degenerate case and is what the
 * swizzle-recovery machinery produces; hierarchical shapes and ScaledBasis
 * strides arise only in the TMA box derivation. Evaluating maps a single linear
 * coordinate to a physical index: the coordinate is decomposed column-major
 * (CuTe idx2crd) over the flattened leaves, then dotted with the (flattened)
 * integer strides.
 */
class LayoutNode : public Object {
public:
  IntTuple shape;
  IntTuple stride;

  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("tl.cute.Layout", LayoutNode, Object);

  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<LayoutNode>()
        .def_ro("shape", &LayoutNode::shape)
        .def_ro("stride", &LayoutNode::stride);
  }

  // Map a single linear coordinate to a physical index by column-major
  // decomposition over the flattened leaves, dotted with the integer strides.
  // Requires plain (non-basis) strides.
  int64_t Eval(int64_t coord) const;

  // Symbolic Eval: like Eval but the coordinate (and result) are PrimExprs, so
  // a runtime loop variable can index the layout. Column-major idx2crd over the
  // (constant) leaf extents, dotted with the integer strides. Requires plain
  // (non-basis) strides; the shape leaves must be constant.
  PrimExpr EvalExpr(const PrimExpr &coord) const;
};

class Layout : public ObjectRef {
public:
  // Flat convenience ctor: depth-1 layout with plain integer strides.
  TVM_DLL Layout(Array<int64_t> shape, Array<int64_t> stride);
  // Hierarchical ctor.
  TVM_DLL Layout(IntTuple shape, IntTuple stride);

  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NULLABLE(Layout, ObjectRef, LayoutNode);
};

TVM_DLL int64_t Rank(const Layout &layout);
TVM_DLL Layout Get(const Layout &layout, int64_t index);
TVM_DLL int64_t Depth(const Layout &layout);
TVM_DLL Layout Flatten(const Layout &layout);
TVM_DLL int64_t Size(const Layout &layout);

// Merge adjacent modes that compose into one, following CuTe's coalesce: a
// faster mode k and the next-slower mode k+1 (modes are fastest-first here)
// collapse into a single mode (shape[k]*shape[k+1], stride[k]) when
// stride[k+1] == shape[k] * stride[k]; size-1 modes are dropped. The result
// maps every linear coordinate to the same index as the input. A fully
// collapsed (empty) layout is returned as the scalar mode (1, 0).
TVM_DLL Layout Coalesce(const Layout &layout);

// Right-inverse of a layout, following CuTe's right_inverse. Returns the layout
// `result` such that layout(result(i)) == i for all i < size(result) — i.e. it
// inverts the image of `layout` back to a contiguous coordinate. Only modes
// whose strides chain contiguously from 1 (1, then s0, then s0*s1, ...) are
// invertible and appear in the result; gaps stop the chain. All strides must be
// non-negative constants.
TVM_DLL Layout RightInverse(const Layout &layout);

// Composition lhs o rhs, following CuTe's composition: returns `result` with
// result(c) == lhs(rhs(c)) for all c in the domain of rhs (so size(result) ==
// size(rhs)). Each rhs mode is pushed through lhs under CuTe's stride/shape
// divisibility conditions, enforced with ICHECK. When an rhs stride is a
// ScaledBasis (value * E<path>), composition reroutes into the matching lhs
// axis (CuTe's is_scaled_basis branch), which is how box modes get tagged with
// the global axis they traverse.
TVM_DLL Layout Composition(const Layout &lhs, const Layout &rhs);

// CuTe coalesce_256: like Coalesce but only merges adjacent modes while the
// merged extent stays <= 256 (the TMA box per-mode limit). Used so a large
// contiguous run becomes one capped box mode rather than a single oversized
// one.
TVM_DLL Layout Coalesce256(const Layout &layout);

// Build a flat (depth-1) Layout from constant shapes and explicit stride leaves
// (IntConst / IntExpr / ScaledBasis). shape.size() must equal stride.size().
TVM_DLL Layout MakeLayoutFromLeaves(Array<int64_t> shape,
                                    Array<IntScalar> stride);

// Column major layout over `shape`.
TVM_DLL Layout MakeColumnMajorLayout(const std::vector<int64_t> &shape);

// Row major layout over `shape`.
TVM_DLL Layout MakeRowMajorLayout(const std::vector<int64_t> &shape);

// Identity layout over `shape` whose strides are the per-axis unit ScaledBases
// E<k> (CuTe make_identity_layout / make_basis_like). Used as the cta_v_map
// that tags each tile coordinate with the global axis it indexes.
TVM_DLL Layout MakeIdentityLayout(const std::vector<int64_t> &shape);

/* A CuTe ComposedLayout of the form Swizzle o offset o Layout. Evaluating at a
 * coordinate x yields swizzle.apply(offset + layout(x)), matching CuTe's
 * (A o O o B)(c) = layout_a(offset + layout_b(c)).
 */
class ComposedLayoutNode : public Object {
public:
  Swizzle swizzle;
  int64_t offset{0};
  Layout layout;

  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("tl.cute.ComposedLayout",
                                    ComposedLayoutNode, Object);

  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<ComposedLayoutNode>()
        .def_ro("swizzle", &ComposedLayoutNode::swizzle)
        .def_ro("offset", &ComposedLayoutNode::offset)
        .def_ro("layout", &ComposedLayoutNode::layout);
  }

  // Map a single linear coordinate through the layout, add the offset, then
  // apply the swizzle: Sw(offset + layout(coord)).
  int64_t Eval(int64_t coord) const {
    return swizzle->Apply(offset + layout->Eval(coord));
  }
};

class ComposedLayout : public ObjectRef {
public:
  TVM_DLL ComposedLayout(Swizzle swizzle, int64_t offset, Layout layout);

  // Recast the whole composed layout into a different element width: the
  // swizzle's m_base shifts (see Swizzle::Recast) and the plain layout's
  // strides and the composed offset are scaled by old_bits/new_bits. Element
  // sizes must be byte-aligned powers of two.
  TVM_DLL ComposedLayout Recast(int old_bits, int new_bits) const;

  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NULLABLE(ComposedLayout, ObjectRef,
                                             ComposedLayoutNode);
};

/* Evaluating, probing, and proving machinery is internal to cute_layout.cc;
 * the public surface is the three IR types above plus the conversions below.
 */

// Convert a TileLang layout into an affine cute::Layout. Returns nullopt if the
// layout's address is not purely affine in the logical coords (e.g. it
// contains a swizzle, a non-zero base offset, or a non-constant extent).
Optional<Layout> LayoutFromTileLang(const tvm::tl::Layout &layout);

// Recover a single CUTLASS/CuTe XOR swizzle over an affine layout from a
// TileLang layout, returning a CuTe ComposedLayout (Swizzle o offset o Layout).
// The result is verified symbolically: the layout's address and the composed
// layout are proven equal for every coordinate (no sampling). Addresses are
// reported in element-offset positions; to obtain the byte-address-space form
// (the CuTe/CUTLASS convention where swizzles act on byte addresses) recast the
// result with `.Recast(dtype.bits(), 8)`. Returns nullopt when the layout is
// not a single XOR swizzle over an affine layout, has non-constant extents, or
// the equivalence cannot be proven.
Optional<ComposedLayout>
ComposedLayoutFromTileLang(const tvm::tl::Layout &layout);

/* The faithful-CuTe decomposition of a bulk copy, mirroring CuTe's
 * construct_tma_gbasis (copy_traits_sm90_tma.hpp). All three members are
 * cute::Layouts; consumers recover concrete geometry by Composition (the
 * `scale @ axis` ScaledBasis strides route each mode into a global axis,
 * exactly CuTe's gtensor.compose(sidx2gmode) / basis_get) rather than by
 * parsing modes.
 *
 *  - box: the single TMA descriptor box (CuTe's tma_gbasis). shape = boxDim
 *    (IntConst, each <= 256, mode 0 unit-scale); stride = `1 @ axis` per mode
 * (in SMEM-vector order, 1:1 with distinct global axes). Composition(gmem, box)
 *    yields the per-mode global strides; Composition(gextent, box) the
 * globalDim.
 *  - rest_gmem: the iteration space CuTe truncates away (modes past smem_rank,
 *    plus any >256 box overflow), as a layout whose strides are `scale @ axis`,
 *    e.g. (8):(64@1). One TMA instruction is issued per coordinate of this
 *    layout; Composition(unit_axis, rest_gmem) gives that axis's gmem-coord
 * step.
 *  - rest_smem: congruent to rest_gmem (same shape) with plain SMEM element
 *    strides, e.g. (8):(4096); rest_smem.Eval(i) is the SMEM offset of the i-th
 *    instruction.
 *
 * With no rest, rest_gmem/rest_smem are the scalar layout (1):(0) -> one
 * instruction. This never permutes modes and never folds a non-contiguous
 * (scale != 1) global mode into the box, so per-box out-of-bounds is exact. */
struct TmaTile {
  Layout box;
  Layout rest_gmem;
  Layout rest_smem;
};

/* Derive the faithful-CuTe TMA decomposition, porting construct_tma_gbasis.
 * Pipeline:
 *   inv  = RightInverse(smem_plain)                     // SMEM idx -> SMEM
 * coord full = Coalesce(Composition(identity(tile_shape), inv))   //
 * sidx2gmode_full smem_rank = first mode with scale != 1; box =
 * take<0,smem_rank>(full) box dims sized via Coalesce256(Composition(gmem,
 * box))    // CuTe coalesce_256 The leading scale-1 run is the box; the
 * remainder (and any >256 box overflow) becomes rest_gmem/rest_smem. `gmem`
 * carries the global tensor's extents (shape) and element strides (stride);
 * strides/extents may be dynamic (IntExpr).
 *
 * `smem_plain` must be a bijection onto [0, size) (ICHECKed). Returns nullopt
 * when not TMA-expressible (innermost mode not unit-scale, a mode with no axis,
 * an unsplittable >256 mode, or box rank > 5). */
std::optional<TmaTile> DeriveTmaTile(const Layout &gmem,
                                     const Layout &smem_plain,
                                     const std::vector<int64_t> &tile_shape);

} // namespace cute
} // namespace tl
} // namespace tvm

#endif // TVM_TL_LAYOUT_CUTE_LAYOUT_H_
