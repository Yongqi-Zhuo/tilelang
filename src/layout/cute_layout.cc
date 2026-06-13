/*!
 * \file cute_layout.cc
 * \brief CuTe-style layout IR types and TileLang-to-CuTe layout recovery.
 *
 * The public surface (see cute_layout.h) is three IR types — Swizzle, Layout,
 * ComposedLayout — plus two conversions from a TileLang layout:
 *
 *   ComposedLayoutFromTileLang : recover Swizzle o offset o Layout, proven
 * exact. LayoutFromTileLang         : the affine special case (identity
 * swizzle, offset 0).
 *
 * Recovery works in three stages, all file-local:
 *   1. Probe  — evaluate the layout's linearized address A(x) at concrete
 *               coordinates (AddrProbe).
 *   2. Detect — find the XOR swizzle by GF(2) bit-incidence and the plain
 *               layout's per-mode strides by one-hot probing.
 *   3. Prove  — symbolically prove A(x) == Sw(offset + plain(x)) for every
 *               coordinate (ProveEquivalent). A result is returned only if the
 *               equivalence is proven; there is no sampling.
 */

#include "cute_layout.h"
#include "layout.h"

#include "support/check.h"
#include <algorithm>
#include <limits>
#include <map>
#include <optional>
#include <vector>

#include <tvm/arith/analyzer.h>
#include <tvm/ffi/extra/structural_equal.h>
#include <tvm/ffi/extra/structural_hash.h>
#include <tvm/runtime/logging.h>
#include <tvm/tirx/op.h>
#include <tvm/tirx/stmt_functor.h>

namespace tvm {
namespace tl {
namespace cute {

using namespace tirx;

namespace {

// Return log2(v) if v is a positive power of two (log2(1) == 0), else -1.
int Log2Exact(int64_t v) {
  if (v <= 0 || (v & (v - 1)) != 0)
    return -1;
  return __builtin_ctzll(static_cast<uint64_t>(v));
}

// Fully evaluate a PrimExpr to an integer constant. The expression must contain
// only integer constants and the operators a swizzled layout is built from:
// +, -, *, FloorDiv, FloorMod, and the bitwise/shift builtins. arith::Analyzer
// does not constant-fold bitwise ops (e.g. T.bitwise_xor) even when both
// operands are constant, so we evaluate them ourselves. Returns std::nullopt if
// any node is not a recognized constant-foldable form.
std::optional<int64_t> EvalConstExpr(const PrimExpr &e) {
  if (auto c = as_const_int(e))
    return *c;
  // Floored division/modulo (rounds toward -inf), matching FloorDiv/FloorMod.
  auto fdiv = [](int64_t a, int64_t b) {
    int64_t q = a / b, r = a % b;
    return q - ((r != 0) && ((r < 0) != (b < 0)) ? 1 : 0);
  };
  auto fmod = [&](int64_t a, int64_t b) { return a - fdiv(a, b) * b; };
  if (const auto *op = e.as<AddNode>()) {
    auto a = EvalConstExpr(op->a), b = EvalConstExpr(op->b);
    if (a && b)
      return *a + *b;
  } else if (const auto *op = e.as<SubNode>()) {
    auto a = EvalConstExpr(op->a), b = EvalConstExpr(op->b);
    if (a && b)
      return *a - *b;
  } else if (const auto *op = e.as<MulNode>()) {
    auto a = EvalConstExpr(op->a), b = EvalConstExpr(op->b);
    if (a && b)
      return *a * *b;
  } else if (const auto *op = e.as<FloorDivNode>()) {
    auto a = EvalConstExpr(op->a), b = EvalConstExpr(op->b);
    if (a && b && *b != 0)
      return fdiv(*a, *b);
  } else if (const auto *op = e.as<FloorModNode>()) {
    auto a = EvalConstExpr(op->a), b = EvalConstExpr(op->b);
    if (a && b && *b != 0)
      return fmod(*a, *b);
  } else if (const auto *op = e.as<CallNode>()) {
    if (op->args.size() == 2) {
      auto a = EvalConstExpr(op->args[0]), b = EvalConstExpr(op->args[1]);
      if (a && b) {
        if (op->op.same_as(builtin::bitwise_xor()))
          return *a ^ *b;
        if (op->op.same_as(builtin::bitwise_and()))
          return *a & *b;
        if (op->op.same_as(builtin::bitwise_or()))
          return *a | *b;
        if (op->op.same_as(builtin::shift_left())) {
          ICHECK(*b >= 0 && *b < 64);
          return *a << *b;
        }
        if (op->op.same_as(builtin::shift_right())) {
          ICHECK(*b >= 0 && *b < 64);
          return *a >> *b;
        }
      }
    } else if (op->args.size() == 1 && op->op.same_as(builtin::bitwise_not())) {
      if (auto a = EvalConstExpr(op->args[0]))
        return ~*a;
    }
  }
  return std::nullopt;
}

/* Probes a TileLang layout's row-major linearized physical address A(x).
 * The constant input shape, output strides, forward-index expressions, and
 * input placeholders are parsed once in the constructor. operator() folds
 * concrete integer coordinates to a constant address; Symbolic() substitutes
 * fresh variables for the equivalence proof. valid() is false when any extent
 * is non-constant.
 */
class AddrProbe {
public:
  explicit AddrProbe(const tvm::tl::Layout &layout) {
    if (!layout.defined())
      return;
    for (const auto &e : layout->InputShape()) {
      auto c = as_const_int(e);
      if (!c)
        return;
      shape_.push_back(*c);
    }
    std::vector<int64_t> out_sizes;
    for (const auto &e : layout->OutputShape()) {
      auto c = as_const_int(e);
      if (!c)
        return;
      out_sizes.push_back(*c);
    }
    out_strides_.assign(out_sizes.size(), 0);
    int64_t acc = 1;
    for (int d = static_cast<int>(out_sizes.size()) - 1; d >= 0; --d) {
      out_strides_[d] = acc;
      acc *= out_sizes[d];
    }
    forward_index_ = layout->GetForwardIndex();
    if (forward_index_.size() != out_strides_.size() ||
        layout->InputDim() != shape_.size())
      return;
    for (size_t k = 0; k < shape_.size(); ++k)
      placeholders_.push_back(InputPlaceholder(k));
    valid_ = true;
  }

  bool valid() const { return valid_; }
  const std::vector<int64_t> &shape() const { return shape_; }

  // Concrete address A(coords); nullopt if it does not fold to a constant.
  std::optional<int64_t> operator()(const std::vector<int64_t> &coords) const {
    Map<Var, PrimExpr> vmap;
    for (size_t k = 0; k < coords.size(); ++k)
      vmap.Set(placeholders_[k], IntImm(DataType::Int(32), coords[k]));
    int64_t addr = 0;
    for (size_t d = 0; d < forward_index_.size(); ++d) {
      PrimExpr e = Substitute(forward_index_[d], vmap);
      std::optional<int64_t> val = EvalConstExpr(e);
      if (!val)
        return std::nullopt;
      addr += *val * out_strides_[d];
    }
    return addr;
  }

  // Symbolic address A(vars). Built in int32 (the forward index's native
  // dtype): int64 casts in the spine are opaque to the simplifier's div/mod
  // rules and would defeat the equivalence proof. Returns nullopt if an output
  // stride exceeds int32.
  std::optional<PrimExpr> Symbolic(const std::vector<Var> &vars) const {
    Map<Var, PrimExpr> vmap;
    for (size_t k = 0; k < vars.size(); ++k)
      vmap.Set(placeholders_[k], vars[k]);
    PrimExpr a = IntImm(DataType::Int(32), 0);
    for (size_t d = 0; d < forward_index_.size(); ++d) {
      if (out_strides_[d] > std::numeric_limits<int32_t>::max())
        return std::nullopt;
      a = a + Substitute(forward_index_[d], vmap) *
                  IntImm(DataType::Int(32), out_strides_[d]);
    }
    return a;
  }

private:
  bool valid_{false};
  std::vector<int64_t> shape_;
  std::vector<int64_t> out_strides_;
  Array<PrimExpr> forward_index_;
  std::vector<Var> placeholders_;
};

/* ---- The equivalence prover -------------------------------------------- */

PrimExpr Bit(const PrimExpr &e, int k) {
  PrimExpr shifted =
      k > 0 ? FloorDiv(e, IntImm(DataType::Int(32), int64_t(1) << k)) : e;
  return FloorMod(shifted, IntImm(DataType::Int(32), 2));
}

// Expand every bitwise_xor call with bounded operands into its exact integer
// form sum_k ((bit_k(X) + bit_k(Y)) % 2) * 2^k. Single-bit parities cannot
// carry, so the expansion is an identity; it removes the one operator the
// arith machinery cannot reason about. Every other node recurses through
// ExprMutator's default child-rebuilding visitors.
class XorLowerer : public ExprMutator {
public:
  explicit XorLowerer(arith::Analyzer *ana) : ana_(ana) {}
  using ExprMutator::operator();

protected:
  PrimExpr VisitExpr_(const CallNode *op) final {
    if (op->args.size() == 2 && op->op.same_as(builtin::bitwise_xor())) {
      PrimExpr x = VisitExpr(op->args[0]), y = VisitExpr(op->args[1]);
      int64_t mx = std::max(ana_->const_int_bound(x)->max_value,
                            ana_->const_int_bound(y)->max_value);
      if (mx >= 0 && mx < (int64_t(1) << 20)) {
        int width = std::max<int>(
            1, 64 - __builtin_clzll(static_cast<uint64_t>(mx) | 1));
        PrimExpr out = IntImm(DataType::Int(32), 0);
        for (int k = 0; k < width; ++k)
          out = out +
                FloorMod(Bit(x, k) + Bit(y, k), IntImm(DataType::Int(32), 2)) *
                    IntImm(DataType::Int(32), int64_t(1) << k);
        return out;
      }
      return bitwise_xor(x, y);
    }
    return ExprMutator::VisitExpr_(op);
  }

private:
  arith::Analyzer *ana_;
};

PrimExpr LowerXor(const PrimExpr &e, arith::Analyzer *ana) {
  return XorLowerer(ana)(e);
}

// Canonicalize every parity node FloorMod(x, 2) in `e`. Working mod 2:
// +/- coincide, even-coefficient terms vanish, and a nested (y % 2) inside a
// parity sum equals y — so the parity's terms flatten into a bag. Each leaf
// term is keyed by Simplify(term % 2) (the simplifier's canonical single-bit
// form, e.g. (i // 2) % 2 == i % 4 // 2), keys appearing an even number of
// times cancel, and the survivors are rebuilt in sorted order. Two equal
// parities thus become structurally identical, which lets the surrounding
// Simplify cancel them — the step the rewrite simplifier cannot do by itself.
// Only FloorMod(_, 2) needs custom handling; every other node recurses through
// ExprMutator's default child-rebuilding visitors.
class ParityCanonicalizer : public ExprMutator {
public:
  explicit ParityCanonicalizer(arith::Analyzer *ana) : ana_(ana) {}
  using ExprMutator::operator();

protected:
  PrimExpr VisitExpr_(const FloorModNode *op) final {
    if (!is_two(op->b))
      return ExprMutator::VisitExpr_(op);
    std::vector<PrimExpr> terms;
    int parity = 0;
    flatten(op->a, &terms, &parity);
    std::stable_sort(terms.begin(), terms.end(),
                     [](const PrimExpr &a, const PrimExpr &b) {
                       return StructuralHash()(a) < StructuralHash()(b);
                     });
    PrimExpr sum = IntImm(DataType::Int(32), parity);
    for (size_t i = 0; i < terms.size();) {
      if (i + 1 < terms.size() && StructuralEqual()(terms[i], terms[i + 1])) {
        i += 2; // t + t == 0 (mod 2)
      } else {
        sum = sum + terms[i];
        ++i;
      }
    }
    return FloorMod(sum, IntImm(DataType::Int(32), 2));
  }

private:
  static bool is_two(const PrimExpr &x) {
    auto c = as_const_int(x);
    return c && *c == 2;
  }

  // Flatten `x` (interpreted mod 2) into leaf terms + a constant parity.
  void flatten(const PrimExpr &x, std::vector<PrimExpr> *terms, int *parity) {
    if (const auto *op = x.as<AddNode>()) {
      flatten(op->a, terms, parity);
      flatten(op->b, terms, parity);
      return;
    }
    if (const auto *op = x.as<SubNode>()) { // -t == t (mod 2)
      flatten(op->a, terms, parity);
      flatten(op->b, terms, parity);
      return;
    }
    if (const auto *imm = x.as<IntImmNode>()) {
      *parity ^= static_cast<int>(imm->value & 1);
      return;
    }
    if (const auto *op = x.as<MulNode>()) {
      const auto *ca = op->a.as<IntImmNode>();
      const auto *cb = op->b.as<IntImmNode>();
      if (ca || cb) {
        int64_t c = ca ? ca->value : cb->value;
        if (c % 2 == 0)
          return; // even coefficient: vanishes mod 2.
        flatten(ca ? op->b : op->a, terms, parity);
        return;
      }
    }
    if (const auto *op = x.as<FloorModNode>()) {
      if (is_two(op->b)) { // (y % 2) == y (mod 2)
        flatten(op->a, terms, parity);
        return;
      }
    }
    // Leaf: canonicalize via the simplifier's single-bit form (recursing into
    // any nested parities first). If that form is itself a parity of a sum,
    // keep flattening through it.
    PrimExpr key = ana_->Simplify(FloorMod(VisitExpr(x), 2));
    if (const auto *km = key.as<FloorModNode>()) {
      if (is_two(km->b) && (km->a.as<AddNode>() || km->a.as<SubNode>() ||
                            km->a.as<MulNode>() || km->a.as<IntImmNode>())) {
        flatten(km->a, terms, parity);
        return;
      }
    }
    if (const auto *imm = key.as<IntImmNode>()) {
      *parity ^= static_cast<int>(imm->value & 1);
      return;
    }
    terms->push_back(key);
  }

  arith::Analyzer *ana_;
};

PrimExpr ParityCanon(const PrimExpr &e, arith::Analyzer *ana) {
  return ParityCanonicalizer(ana)(e);
}

// Decide E == 0 for all assignments of the analyzer's range-bound variables,
// by iterating Simplify -> LowerXor -> ParityCanon to a fixpoint. Sound: every
// step is an exact identity, so a 0 result is a proof; a non-0 fixpoint means
// "not proven" and the caller must reject. Complete enough in practice that
// neither sampling nor an SMT solver is needed.
bool ProveZero(PrimExpr e, arith::Analyzer *ana, int max_rounds = 8) {
  PrimExpr d = ana->Simplify(e);
  size_t last_hash = 0;
  for (int round = 0; round < max_rounds; ++round) {
    if (is_zero(d))
      return true;
    size_t h = StructuralHash()(d);
    if (round > 0 && h == last_hash)
      return false; // fixpoint reached without proving 0.
    last_hash = h;
    d = ana->Simplify(ParityCanon(LowerXor(d, ana), ana));
  }
  return is_zero(d);
}

// Symbolically prove that the probed layout address equals the recovered
// composed layout, A(x) == Sw(offset + plain(linearize(x))), for ALL logical
// coordinates x. With mid(Q) = the swizzle's XOR of Q's target and source bit
// fields written as per-bit parities, and tgt(Q) = Q's target field, the
// equality is the single integer identity
//
//   A - Q - (mid(Q) - tgt(Q)) * 2^m_base == 0,
//
// which ProveZero decides with the parity-aware rewriting. All proof
// arithmetic is int32; layouts that could overflow it are rejected.
bool ProveEquivalent(const AddrProbe &probe, const Swizzle &swizzle,
                     int64_t offset, const Layout &plain) {
  const SwizzleNode *sw = swizzle.get();
  const LayoutNode *cl = plain.get();
  ICHECK(sw != nullptr && cl != nullptr);
  const std::vector<int64_t> &shape = probe.shape();
  // The recovered plain layout is flat with plain integer strides.
  std::vector<int64_t> cl_shape, cl_stride;
  for (const IntScalar &s : Leaves(cl->shape)) {
    if (!IsConst(s))
      return false; // a recovered affine shape is always a constant int.
    cl_shape.push_back(AsConst(s));
  }
  for (const IntScalar &s : Leaves(cl->stride)) {
    if (!IsConst(s))
      return false; // a basis/dynamic stride is never an affine recovery.
    cl_stride.push_back(AsConst(s));
  }

  // int32 overflow guards on every constant entering the proof.
  auto fits = [](int64_t v) {
    return v >= 0 && v <= std::numeric_limits<int32_t>::max();
  };
  int64_t domain = 1;
  for (int64_t s : shape) {
    domain *= std::max<int64_t>(s, 1);
    if (domain > (int64_t(1) << 30))
      return false;
  }
  if (!fits(offset))
    return false;
  for (size_t i = 0; i < cl_shape.size(); ++i)
    if (!fits(cl_shape[i]) || !fits(cl_stride[i]))
      return false;
  int b = sw->b_bits, m = sw->m_base, s = sw->s_shift;
  if (b > 0 && m + s + b >= 31)
    return false;

  // Analyzer with each coordinate range-bound: x_k in [0, shape[k]).
  arith::Analyzer ana;
  std::vector<Var> vars;
  for (size_t k = 0; k < shape.size(); ++k) {
    Var v("c" + std::to_string(k), DataType::Int(32));
    vars.push_back(v);
    if (shape[k] > 0)
      ana.Bind(v, Range(IntImm(DataType::Int(32), 0),
                        IntImm(DataType::Int(32), shape[k])));
  }
  auto i32 = [](int64_t v) { return IntImm(DataType::Int(32), v); };

  std::optional<PrimExpr> A = probe.Symbolic(vars);
  if (!A)
    return false;

  // Q(x) = offset + plain(linearize(x)): linearize x column-major over the
  // input shape, then evaluate the plain layout's CuTe idx2crd symbolically.
  PrimExpr coord = i32(0);
  int64_t place = 1;
  for (size_t k = 0; k < vars.size(); ++k) {
    coord = coord + vars[k] * i32(place);
    place *= std::max<int64_t>(shape[k], 1);
  }
  PrimExpr Q = i32(offset);
  {
    PrimExpr rem = coord;
    for (size_t i = 0; i < cl_shape.size(); ++i) {
      int64_t ext = cl_shape[i];
      PrimExpr crd = ext > 0 ? FloorMod(rem, i32(ext)) : PrimExpr(rem);
      Q = Q + crd * i32(cl_stride[i]);
      if (ext > 0)
        rem = FloorDiv(rem, i32(ext));
    }
  }

  if (b <= 0)
    return ProveZero(*A - Q, &ana);

  PrimExpr mid = i32(0);
  for (int p = 0; p < b; ++p)
    mid = mid + FloorMod(Bit(Q, m + p) + Bit(Q, m + s + p), i32(2)) *
                    i32(int64_t(1) << p);
  PrimExpr tgt =
      FloorMod(FloorDiv(Q, i32(int64_t(1) << m)), i32(int64_t(1) << b));
  return ProveZero(*A - Q - (mid - tgt) * i32(int64_t(1) << m), &ana);
}

} // namespace

int64_t SwizzleNode::Apply(int64_t offset) const {
  if (b_bits <= 0)
    return offset;
  uint64_t mask = (static_cast<uint64_t>(1) << b_bits) - 1;
  uint64_t yyy = mask << (m_base + s_shift);
  uint64_t x = static_cast<uint64_t>(offset);
  return static_cast<int64_t>(x ^ ((x & yyy) >> s_shift));
}

/* ---- IntTuple leaves --------------------------------------------------- */

IntConst::IntConst(int64_t value) {
  auto node = make_object<IntConstNode>();
  node->value = value;
  data_ = std::move(node);
}

IntExpr::IntExpr(PrimExpr value) {
  auto node = make_object<IntExprNode>();
  node->value = std::move(value);
  data_ = std::move(node);
}

ScaledBasis::ScaledBasis(IntScalar value, Array<int64_t> basis) {
  auto node = make_object<ScaledBasisNode>();
  node->value = std::move(value);
  node->basis = std::move(basis);
  data_ = std::move(node);
}

IntTupleArray::IntTupleArray(Array<IntTuple> fields) {
  auto node = make_object<IntTupleArrayNode>();
  node->fields = std::move(fields);
  data_ = std::move(node);
}

/* ---- IntScalar leaf arithmetic ----------------------------------------- */

bool IsBasis(const IntScalar &s) { return s.as<ScaledBasisNode>() != nullptr; }

IntScalar BasisValue(const IntScalar &s) {
  if (const auto *b = s.as<ScaledBasisNode>())
    return b->value;
  return s;
}

Array<int64_t> BasisPath(const IntScalar &s) {
  if (const auto *b = s.as<ScaledBasisNode>())
    return b->basis;
  return Array<int64_t>();
}

bool IsConst(const IntScalar &s) { return s.as<IntConstNode>() != nullptr; }

int64_t AsConst(const IntScalar &s) {
  const auto *c = s.as<IntConstNode>();
  ICHECK(c != nullptr)
      << "AsConst on a non-constant leaf (IntExpr/ScaledBasis)";
  return c->value;
}

PrimExpr AsPrimExpr(const IntScalar &s) {
  if (const auto *c = s.as<IntConstNode>())
    return IntImm(DataType::Int(32), c->value);
  if (const auto *e = s.as<IntExprNode>())
    return e->value;
  LOG(FATAL) << "AsPrimExpr on a ScaledBasis leaf";
  TVM_FFI_UNREACHABLE();
}

// Alias for building dynamic products inside the leaf arithmetic helpers.
static PrimExpr LeafExpr(const IntScalar &s) { return AsPrimExpr(s); }

// Multiply two plain (non-basis) leaves, folding constants and applying 0/1
// identities; falls back to an IntExpr product for dynamic operands.
static IntScalar MulPlain(const IntScalar &a, const IntScalar &b) {
  bool ca = IsConst(a), cb = IsConst(b);
  if (ca && cb)
    return IntConst(AsConst(a) * AsConst(b));
  if (ca) {
    if (AsConst(a) == 0)
      return IntConst(0);
    if (AsConst(a) == 1)
      return b;
  }
  if (cb) {
    if (AsConst(b) == 0)
      return IntConst(0);
    if (AsConst(b) == 1)
      return a;
  }
  return IntExpr(LeafExpr(a) * LeafExpr(b));
}

IntScalar ScalarMul(const IntScalar &a, const IntScalar &b) {
  // CuTe operator* on integral leaves: at most one operand is a ScaledBasis;
  // the result keeps that path and multiplies the scales.
  bool ba = IsBasis(a), bb = IsBasis(b);
  ICHECK(!(ba && bb)) << "ScalarMul of two ScaledBasis leaves";
  if (ba)
    return ScaledBasis(MulPlain(BasisValue(a), b), BasisPath(a));
  if (bb)
    return ScaledBasis(MulPlain(a, BasisValue(b)), BasisPath(b));
  return MulPlain(a, b);
}

bool ProvablyEqual(const IntScalar &a, const IntScalar &b) {
  bool ba = IsBasis(a), bb = IsBasis(b);
  if (ba != bb)
    return false;
  if (ba && !StructuralEqual()(BasisPath(a), BasisPath(b)))
    return false;
  IntScalar va = BasisValue(a), vb = BasisValue(b);
  bool ca = IsConst(va), cb = IsConst(vb);
  if (ca && cb)
    return AsConst(va) == AsConst(vb);
  if (ca || cb)
    return false; // one constant, one dynamic: not provably equal.
  return StructuralEqual()(LeafExpr(va), LeafExpr(vb));
}

IntScalar BasisGet(const Array<int64_t> &path,
                   const std::vector<IntScalar> &flat) {
  ICHECK_EQ(path.size(), 1u)
      << "BasisGet supports only single-level basis paths; got rank "
      << path.size();
  int64_t i = path[0];
  ICHECK(i >= 0 && i < static_cast<int64_t>(flat.size()))
      << "BasisGet index " << i << " out of range " << flat.size();
  return flat[i];
}

IntScalar AsScalar(const IntTuple &t) {
  const auto *s = t.as<IntScalarNode>();
  ICHECK(s != nullptr) << "AsScalar on a non-leaf IntTuple";
  return GetRef<IntScalar>(s);
}

int Rank(const IntTuple &t) {
  if (t.as<IntScalarNode>())
    return 1;
  const auto *a = t.as<IntTupleArrayNode>();
  ICHECK(a != nullptr) << "Rank on a null IntTuple";
  return static_cast<int>(a->fields.size());
}

IntTuple Get(const IntTuple &t, int64_t index) {
  if (t.as<IntScalarNode>()) {
    ICHECK(index == 0) << "Leaf index out of range: " << index << " vs 0";
    return t;
  }
  const auto *a = t.as<IntTupleArrayNode>();
  ICHECK(a != nullptr) << "Get on a null IntTuple";
  int64_t n = a->fields.size();
  ICHECK(index >= 0 && index < n) << "Index " << index << " out of range " << n;
  return a->fields[index];
}

int64_t Product(const IntTuple &t) {
  return FoldLeaves(t, int64_t(1), [](int64_t acc, const IntScalar &s) {
    return acc * AsConst(s);
  });
}

std::vector<IntScalar> Leaves(const IntTuple &t) {
  std::vector<IntScalar> out;
  FoldLeaves(t, 0, [&](int, const IntScalar &s) {
    out.push_back(s);
    return 0;
  });
  return out;
}

IntTuple Flatten(const IntTuple &t) {
  if (t.as<IntScalarNode>())
    return t;
  Array<IntTuple> out;
  for (const IntScalar &s : Leaves(t))
    out.push_back(s);
  return IntTupleArray(out);
}

int64_t Rank(const Layout &layout) { return Rank(layout->shape); }

Layout Get(const Layout &layout, int64_t index) {
  return Layout(Get(layout->shape, index), Get(layout->stride, index));
}

Layout Flatten(const Layout &layout) {
  return Layout(Flatten(layout->shape), Flatten(layout->stride));
}

int64_t Size(const Layout &layout) { return Product(layout->shape); }

namespace {

// CuTe operates on flat layouts as two congruent tuples (flat_shape,
// flat_stride). We mirror that with two parallel leaf vectors — no bespoke
// "Mode" struct — so a stride leaf is the full CuTe integral leaf (IntConst,
// IntExpr, or ScaledBasis) and the algebra goes entirely through the leaf
// helpers (ScalarMul / AsConst / IsBasis / ...).
struct FlatLayout {
  std::vector<IntScalar> shape;
  std::vector<IntScalar> stride;
  size_t size() const { return shape.size(); }
};

// A shape leaf as a constant int (shapes are always concrete in our use).
int64_t ShapeInt(const IntScalar &s) { return AsConst(s); }

// Flatten a layout into congruent leaf vectors.
FlatLayout ToFlat(const LayoutNode *n) {
  FlatLayout f;
  f.shape = Leaves(n->shape);
  f.stride = Leaves(n->stride);
  ICHECK_EQ(f.shape.size(), f.stride.size())
      << "Layout shape/stride are not congruent (different leaf counts)";
  return f;
}

// Build a flat (depth-1) Layout from leaf vectors.
Layout FromFlat(const FlatLayout &f) {
  ICHECK_EQ(f.shape.size(), f.stride.size());
  Array<IntTuple> shape_fields, stride_fields;
  for (size_t k = 0; k < f.shape.size(); ++k) {
    shape_fields.push_back(f.shape[k]);
    stride_fields.push_back(f.stride[k]);
  }
  return Layout(IntTuple(IntTupleArray(shape_fields)),
                IntTuple(IntTupleArray(stride_fields)));
}

// The scalar (1, 0) layout — CuTe's collapsed-layout identity.
FlatLayout ScalarFlat() {
  FlatLayout f;
  f.shape.push_back(IntConst(1));
  f.stride.push_back(IntConst(0));
  return f;
}

} // namespace

int64_t LayoutNode::Eval(int64_t coord) const {
  // CuTe idx2crd then dot with stride over the flattened leaves: decompose the
  // linear coordinate column-major (mode 0 fastest) into per-mode coordinates
  // and accumulate crd_k * stride_k. Requires plain (non-basis) strides.
  std::vector<IntScalar> sh = Leaves(shape), st = Leaves(stride);
  int64_t addr = 0;
  int64_t rem = coord;
  for (size_t k = 0; k < sh.size(); ++k) {
    int64_t ext = AsConst(sh[k]);
    int64_t crd = ext > 0 ? rem % ext : 0;
    ICHECK(IsConst(st[k])) << "Eval requires plain integer strides";
    addr += crd * AsConst(st[k]);
    if (ext > 0)
      rem /= ext;
  }
  return addr;
}

PrimExpr LayoutNode::EvalExpr(const PrimExpr &coord) const {
  // Symbolic idx2crd-then-dot: same as Eval but in PrimExpr arithmetic, so a
  // runtime loop variable can index the layout. Constant leaf extents/strides.
  std::vector<IntScalar> sh = Leaves(shape), st = Leaves(stride);
  PrimExpr addr = IntImm(coord->dtype, 0);
  PrimExpr rem = coord;
  for (size_t k = 0; k < sh.size(); ++k) {
    int64_t ext = AsConst(sh[k]);
    ICHECK(IsConst(st[k])) << "EvalExpr requires plain integer strides";
    int64_t s = AsConst(st[k]);
    if (ext > 0 && s != 0) {
      PrimExpr crd = FloorMod(rem, IntImm(rem->dtype, ext));
      addr = addr + crd * IntImm(addr->dtype, s);
    }
    if (ext > 0)
      rem = FloorDiv(rem, IntImm(rem->dtype, ext));
  }
  return addr;
}

Swizzle::Swizzle(int b_bits, int m_base, int s_shift) {
  auto node = make_object<SwizzleNode>();
  node->b_bits = b_bits;
  node->m_base = m_base;
  node->s_shift = s_shift;
  data_ = std::move(node);
}

Swizzle Swizzle::Recast(int old_bits, int new_bits) const {
  const SwizzleNode *n = get();
  ICHECK(n != nullptr) << "Recast on a null Swizzle";
  if (!n->IsSwizzled())
    return Swizzle::Identity();
  // Reinterpreting the element size scales every address by old_bits/new_bits,
  // so bit positions shift by log2(old_bits) - log2(new_bits). Element sizes
  // must be powers of two for this to be a clean bit shift.
  int old_log = Log2Exact(old_bits);
  int new_log = Log2Exact(new_bits);
  ICHECK(old_log >= 0 && new_log >= 0)
      << "Recast expects power-of-two element sizes, got " << old_bits << " -> "
      << new_bits;
  int m_base = n->m_base + (old_log - new_log);
  ICHECK(m_base >= 0) << "Recast produced negative m_base: " << m_base;
  return Swizzle(n->b_bits, m_base, n->s_shift);
}

Layout::Layout(Array<int64_t> shape, Array<int64_t> stride) {
  ICHECK_EQ(shape.size(), stride.size());
  Array<IntTuple> shape_fields, stride_fields;
  for (size_t k = 0; k < shape.size(); ++k) {
    shape_fields.push_back(IntConst(shape[k]));
    stride_fields.push_back(IntConst(stride[k]));
  }
  auto node = make_object<LayoutNode>();
  node->shape = IntTuple(IntTupleArray(shape_fields));
  node->stride = IntTuple(IntTupleArray(stride_fields));
  data_ = std::move(node);
}

Layout::Layout(IntTuple shape, IntTuple stride) {
  auto node = make_object<LayoutNode>();
  node->shape = std::move(shape);
  node->stride = std::move(stride);
  data_ = std::move(node);
}

namespace {
// CuTe bw_coalesce: fold flattened modes fastest-first, merging a mode into the
// running front (the back of `out`) when it continues it contiguously, i.e. the
// front's stride equals shape*stride of the mode (ProvablyEqual handles basis /
// dynamic leaves). Size-1 modes are dropped. `max_extent` caps the merged
// extent (CuTe coalesce_256 passes 256; plain Coalesce passes inf).
FlatLayout CoalesceFlat(const FlatLayout &in, int64_t max_extent) {
  FlatLayout out;
  for (size_t k = 0; k < in.size(); ++k) {
    int64_t s = AsConst(in.shape[k]);
    if (s == 1)
      continue; // rank-1 (size-1) mode: identity, drop it.
    if (!out.shape.empty()) {
      int64_t bs = AsConst(out.shape.back());
      // merged extent test: front.stride == back.shape * back.stride.
      IntScalar cont = ScalarMul(IntConst(bs), out.stride.back());
      if (ProvablyEqual(in.stride[k], cont) && bs * s <= max_extent) {
        out.shape.back() = IntConst(bs * s); // contiguous continuation.
        continue;
      }
    }
    out.shape.push_back(in.shape[k]);
    out.stride.push_back(in.stride[k]);
  }
  if (out.shape.empty())
    return ScalarFlat(); // fully collapsed: scalar (1, 0).
  return out;
}
} // namespace

Layout Coalesce(const Layout &layout) {
  ICHECK(layout.get() != nullptr) << "Coalesce on a null Layout";
  constexpr int64_t kInf = std::numeric_limits<int64_t>::max();
  return FromFlat(CoalesceFlat(ToFlat(layout.get()), kInf));
}

Layout Coalesce256(const Layout &layout) {
  // CuTe coalesce_256: like Coalesce but caps each merged run at 256 (the TMA
  // per-mode boxDim limit), so a contiguous run wider than 256 stays split.
  ICHECK(layout.get() != nullptr) << "Coalesce256 on a null Layout";
  return FromFlat(CoalesceFlat(ToFlat(layout.get()), 256));
}

Layout RightInverse(const Layout &layout) {
  ICHECK(layout.get() != nullptr) << "RightInverse on a null Layout";
  // Coalesce first, then greedily chain modes whose strides form 1, s0, s0*s1,
  // ...; the inverse maps each such mode back to its prefix product. Modes off
  // the chain (and any non-constant strides, which CuTe also filters) are
  // skipped. Inversion needs a concrete stride order, so this is const-only.
  FlatLayout m = ToFlat(Coalesce(layout).get());
  int r = static_cast<int>(m.size());

  std::vector<int64_t> sh(r), preprod(r);
  int64_t acc = 1;
  for (int k = 0; k < r; ++k) {
    sh[k] = AsConst(m.shape[k]);
    preprod[k] = acc;
    acc *= sh[k];
  }

  // Only modes with a constant stride participate (CuTe filters dynamic ones).
  std::vector<int> order;
  for (int k = 0; k < r; ++k)
    if (IsConst(m.stride[k]))
      order.push_back(k);
  std::stable_sort(order.begin(), order.end(), [&](int a, int b) {
    return AsConst(m.stride[a]) < AsConst(m.stride[b]);
  });

  FlatLayout out;
  int64_t current = 1;
  for (int k : order) {
    int64_t st = AsConst(m.stride[k]);
    ICHECK(st >= 0) << "RightInverse requires non-negative strides, got " << st;
    if (st == current) {
      out.shape.push_back(IntConst(sh[k]));
      out.stride.push_back(IntConst(preprod[k]));
      current = sh[k] * st;
    }
  }
  if (out.shape.empty())
    out = ScalarFlat();
  return Coalesce(FromFlat(out));
}

namespace {
// Compose a flat lhs with a single rhs mode (shape s, plain stride leaf d),
// appending result modes to `out`. Mirrors CuTe composition_impl's general
// case. Each emitted result stride is rest_stride * lhs[i].stride (CuTe scales
// the lhs stride leaf), so a ScaledBasis / dynamic LHS leaf flows onto the
// output via ScalarMul. `d` (the rhs-derived multiplier) must be a constant int
// — it divides against the lhs shapes.
void ComposeIntoLHS(const FlatLayout &lhs, int64_t s, int64_t d,
                    FlatLayout *out) {
  if (d == 0) {
    out->shape.push_back(IntConst(s));
    out->stride.push_back(IntConst(0));
    return;
  }
  int R = static_cast<int>(lhs.size());
  std::vector<IntScalar> res_shape, res_stride;
  int64_t rest_shape = s, rest_stride = d;
  for (int i = 0; i < R - 1; ++i) {
    int64_t cs = AsConst(lhs.shape[i]);
    ICHECK(rest_stride % cs == 0 || rest_stride < cs)
        << "Composition stride divisibility: rest_stride=" << rest_stride
        << ", curr_shape=" << cs;
    int64_t a = rest_stride < 0 ? -rest_stride : rest_stride;
    int sgn = (rest_stride > 0) - (rest_stride < 0);
    int64_t next_shape = (cs + a - 1) / a;
    int64_t next_stride = ((a + cs - 1) / cs) * sgn;
    if (next_shape == 1 || rest_shape == 1) {
      rest_stride = next_stride;
    } else {
      int64_t new_shape = std::min(next_shape, rest_shape);
      ICHECK(rest_shape % new_shape == 0)
          << "Composition shape divisibility: rest_shape=" << rest_shape
          << ", new_shape=" << new_shape;
      // result stride = rest_stride * lhs[i] stride leaf (basis/dynamic kept).
      res_shape.push_back(IntConst(new_shape));
      res_stride.push_back(ScalarMul(IntConst(rest_stride), lhs.stride[i]));
      rest_shape /= new_shape;
      rest_stride = next_stride;
    }
  }
  IntScalar last = lhs.stride[R - 1];
  if (res_shape.empty()) {
    out->shape.push_back(IntConst(rest_shape));
    out->stride.push_back(ScalarMul(IntConst(rest_stride), last));
  } else {
    for (size_t i = 0; i < res_shape.size(); ++i) {
      out->shape.push_back(res_shape[i]);
      out->stride.push_back(res_stride[i]);
    }
    if (rest_shape != 1) {
      out->shape.push_back(IntConst(rest_shape));
      out->stride.push_back(ScalarMul(IntConst(rest_stride), last));
    }
  }
}

// Compose a flat lhs with one rhs mode (shape s, stride leaf d), handling
// CuTe's is_scaled_basis(RHS) branch: a ScaledBasis rhs stride reroutes
// composition into the single lhs mode it selects (basis_get), with the basis
// scale as the new rhs multiplier. This is what tags / dynamic-strides flow
// through when the basis lives on the RHS (e.g. composition(A, identity)).
void ComposeMode(const FlatLayout &lhs, int64_t s, const IntScalar &d,
                 FlatLayout *out) {
  if (IsBasis(d)) {
    IntScalar sel = BasisGet(BasisPath(d), lhs.stride);
    ICHECK(IsConst(BasisValue(d)))
        << "Composition RHS basis scale must be a constant int";
    FlatLayout one;
    one.shape.push_back(
        IntConst(1)); // single-mode lhs (shape unused for d!=0).
    one.stride.push_back(sel);
    ComposeIntoLHS(one, s, AsConst(BasisValue(d)), out);
    return;
  }
  ICHECK(IsConst(d))
      << "Composition RHS stride must be a constant int or a ScaledBasis";
  ComposeIntoLHS(lhs, s, AsConst(d), out);
}
} // namespace

Layout Composition(const Layout &lhs, const Layout &rhs) {
  ICHECK(lhs.get() != nullptr && rhs.get() != nullptr)
      << "Composition on a null Layout";
  // CuTe coalesces the lhs via coalesce_x(lhs, coprofile(rhs)). When the rhs
  // has any ScaledBasis stride, its coprofile preserves the per-axis rank, so
  // the lhs is coalesced per-mode (i.e. NOT merged across modes) — basis_get
  // must index the original lhs modes. When the rhs is plain, coprofile is
  // scalar and the lhs is fully coalesced. Our lhs is always flat, so "coalesce
  // per-mode" is the identity: full-coalesce iff no rhs stride is a basis.
  FlatLayout rs = ToFlat(rhs.get());
  bool rhs_has_basis = false;
  for (const IntScalar &d : rs.stride)
    rhs_has_basis = rhs_has_basis || IsBasis(d);
  FlatLayout ls =
      rhs_has_basis ? ToFlat(lhs.get()) : ToFlat(Coalesce(lhs).get());

  FlatLayout out;
  for (size_t k = 0; k < rs.size(); ++k)
    ComposeMode(ls, AsConst(rs.shape[k]), rs.stride[k], &out);
  if (out.shape.empty())
    out = ScalarFlat();
  return FromFlat(out);
}

Layout MakeLayoutFromLeaves(Array<int64_t> shape, Array<IntScalar> stride) {
  ICHECK_EQ(shape.size(), stride.size());
  Array<IntTuple> shape_fields, stride_fields;
  for (size_t k = 0; k < shape.size(); ++k) {
    shape_fields.push_back(IntConst(shape[k]));
    stride_fields.push_back(stride[k]);
  }
  return Layout(IntTuple(IntTupleArray(shape_fields)),
                IntTuple(IntTupleArray(stride_fields)));
}

Layout MakeColumnMajorLayout(const std::vector<int64_t> &shape) {
  Array<IntTuple> shape_fields, stride_fields;
  int64_t stride = 1;
  for (size_t k = 0; k < shape.size(); ++k) {
    shape_fields.push_back(IntConst(shape[k]));
    stride_fields.push_back(IntConst(stride));
    stride *= shape[k];
  }
  return Layout(IntTuple(IntTupleArray(shape_fields)),
                IntTuple(IntTupleArray(stride_fields)));
}

Layout MakeRowMajorLayout(const std::vector<int64_t> &shape) {
  Array<IntTuple> shape_fields, stride_fields;
  int64_t stride = 1;
  for (int k = static_cast<int>(shape.size()) - 1; k >= 0; --k) {
    shape_fields.push_back(IntConst(shape[k]));
    stride_fields.push_back(IntConst(stride));
    stride *= shape[k];
  }
  return Layout(IntTuple(IntTupleArray(shape_fields)),
                IntTuple(IntTupleArray(stride_fields)));
}

Layout MakeIdentityLayout(const std::vector<int64_t> &shape) {
  // CuTe make_identity_layout / make_basis_like: per-mode unit ScaledBasis
  // E<k>.
  Array<IntTuple> shape_fields, stride_fields;
  for (size_t k = 0; k < shape.size(); ++k) {
    shape_fields.push_back(IntConst(shape[k]));
    stride_fields.push_back(ScaledBasis(IntConst(1), {int64_t(k)}));
  }
  return Layout(IntTuple(IntTupleArray(shape_fields)),
                IntTuple(IntTupleArray(stride_fields)));
}

std::optional<TmaTile> DeriveTmaTile(const Layout &gmem,
                                     const Layout &smem_plain,
                                     const std::vector<int64_t> &tile_shape) {
  // Port of CuTe construct_tma_gbasis (copy_traits_sm90_tma.hpp). All output is
  // cute::Layouts with `scale @ axis` ScaledBasis strides; consumers route them
  // through Composition rather than parsing modes.
  //
  // 1. cta_v_map: tile-coord -> per-axis basis E<axis> (make_identity_layout),
  //    column-major to match the SMEM layout's input linearization.
  Layout cta_v_map = MakeIdentityLayout(tile_shape);
  // 2. inv_smem = right_inverse(smem_plain): SMEM idx -> SMEM coord.
  Layout inv = RightInverse(smem_plain);
  // smem_plain must be a bijection onto [0, size): right_inverse keeps only the
  // stride chain from 1, so a gapped/non-injective layout undercovers the tile.
  ICHECK_EQ(Size(inv), Size(smem_plain))
      << "plain SMEM layout is not a bijection (gapped or non-injective): "
      << "RightInverse covers " << Size(inv) << " of " << Size(smem_plain)
      << " elements";
  // 3. sidx2gmode_full = coalesce(composition(cta_v_map, inv_smem)): SMEM idx
  // ->
  //    gmem mode, each mode carrying a `scale @ axis` basis, in SMEM-vector
  //    order. We do NOT permute it.
  Layout full = Coalesce(Composition(cta_v_map, inv));
  std::vector<IntScalar> full_shape = Leaves(full->shape);
  std::vector<IntScalar> full_stride = Leaves(full->stride);
  size_t n = full_shape.size();

  // Decode each full mode into (extent, basis-stride, SMEM stride). The SMEM
  // stride is the column-major prefix product over full's extents (full's
  // domain is the SMEM linear index).
  struct GMode {
    int64_t extent;   // mode extent.
    IntScalar stride; // `scale @ axis` ScaledBasis.
    int64_t smem;     // SMEM element stride (prefix product).
  };
  std::vector<GMode> modes;
  int64_t smem_acc = 1;
  for (size_t k = 0; k < n; ++k) {
    if (!IsBasis(full_stride[k]) || BasisPath(full_stride[k]).size() != 1u)
      return std::nullopt; // a mode not derived from a single global axis.
    int axis = static_cast<int>(BasisPath(full_stride[k])[0]);
    if (axis < 0 || axis >= static_cast<int>(tile_shape.size()))
      return std::nullopt;
    if (!IsConst(BasisValue(full_stride[k])) || !IsConst(full_shape[k]))
      return std::nullopt;
    int64_t ext = AsConst(full_shape[k]);
    modes.push_back(GMode{ext, full_stride[k], smem_acc});
    smem_acc *= ext;
  }

  // 4. smem_rank: truncate at the first mode whose basis scale != 1 (CuTe's
  //    find_if over basis_value != 1). The leading scale-1 run is the box (a
  //    contiguous SMEM vector at unit gmem step); the innermost mode must be
  //    unit-scale. Modes from smem_rank on become `rest` (replayed as separate
  //    TMA instructions). A box mode wider than 256 (the boxDim limit, which
  //    CuTe would static_assert) is capped at the largest divisor <= 256 and
  //    the quotient moves into `rest` on the same axis.
  if (modes.empty() || AsConst(BasisValue(modes[0].stride)) != 1)
    return std::nullopt;
  // box_* are the descriptor modes; rest_* the iteration modes.
  std::vector<IntScalar> box_shape, box_stride;
  std::vector<IntScalar> rg_shape, rg_stride; // rest gmem (scale @ axis)
  std::vector<IntScalar> rs_shape, rs_stride; // rest smem (plain strides)
  std::vector<bool> axis_in_box(tile_shape.size(), false);
  bool in_box = true;
  auto add_rest = [&](int64_t extent, const IntScalar &gstride, int64_t smem) {
    rg_shape.push_back(IntConst(extent));
    rg_stride.push_back(gstride);
    rs_shape.push_back(IntConst(extent));
    rs_stride.push_back(IntConst(smem));
  };
  for (const GMode &m : modes) {
    int axis = static_cast<int>(BasisPath(m.stride)[0]);
    int64_t scale = AsConst(BasisValue(m.stride));
    if (in_box && scale == 1) {
      int64_t S = m.extent;
      if (S > 256) {
        int64_t g = 256;
        while (g > 1 && S % g != 0)
          --g;
        if (g <= 1)
          return std::nullopt; // no divisor <= 256: not splittable.
        box_shape.push_back(IntConst(g));
        box_stride.push_back(m.stride); // 1 @ axis
        // Overflow -> same-axis rest mode: each digit steps the gmem coord by g
        // elements (g @ axis) and the SMEM pointer by g * smem.
        add_rest(S / g, ScaledBasis(IntConst(g), BasisPath(m.stride)),
                 m.smem * g);
      } else {
        box_shape.push_back(IntConst(S));
        box_stride.push_back(m.stride);
      }
      axis_in_box[axis] = true;
    } else {
      in_box = false; // first non-unit-scale mode truncates the box.
      add_rest(m.extent, m.stride, m.smem);
    }
  }

  // 5. Append a size-1 box mode for every axis not covered by the box (CuTe's
  //    "missing bases appended with size-1"): a rest-only axis needs a
  //    descriptor coordinate slot, and an unindexed axis (e.g. a batch dim)
  //    still selects a coordinate.
  for (size_t a = 0; a < tile_shape.size(); ++a) {
    if (axis_in_box[a])
      continue;
    box_shape.push_back(IntConst(1));
    box_stride.push_back(ScaledBasis(IntConst(1), {static_cast<int64_t>(a)}));
  }
  if (box_shape.size() > 5)
    return std::nullopt; // TMA descriptor rank limit.

  // gmem is accepted for signature fidelity with construct_tma_gbasis (and to
  // assert the box is contiguous against real strides); box sizing here is the
  // basis coalesce above, which matches CuTe's coalesce_256 result because
  // coalesce never merges across distinct gmem axes (different @axis).
  (void)gmem;

  TmaTile tile;
  tile.box = FromFlat(FlatLayout{box_shape, box_stride});
  tile.rest_gmem = rg_shape.empty() ? FromFlat(ScalarFlat())
                                    : FromFlat(FlatLayout{rg_shape, rg_stride});
  tile.rest_smem = rs_shape.empty() ? FromFlat(ScalarFlat())
                                    : FromFlat(FlatLayout{rs_shape, rs_stride});
  return tile;
}

ComposedLayout::ComposedLayout(Swizzle swizzle, int64_t offset, Layout layout) {
  auto node = make_object<ComposedLayoutNode>();
  node->swizzle = std::move(swizzle);
  node->offset = offset;
  node->layout = std::move(layout);
  data_ = std::move(node);
}

ComposedLayout ComposedLayout::Recast(int old_bits, int new_bits) const {
  const ComposedLayoutNode *n = get();
  ICHECK(n != nullptr) << "Recast on a null ComposedLayout";
  int old_log = Log2Exact(old_bits);
  int new_log = Log2Exact(new_bits);
  ICHECK(old_log >= 0 && new_log >= 0)
      << "Recast expects power-of-two element sizes, got " << old_bits << " -> "
      << new_bits;
  // Scaling the element width multiplies every address by old_bits/new_bits.
  // For a clean integer layout this must be an exact power-of-two scale; the
  // shift can be positive (smaller elements) or negative (larger elements).
  int shift = old_log - new_log;
  auto scale = [&](int64_t v) -> int64_t {
    if (shift >= 0)
      return v << shift;
    int64_t d = static_cast<int64_t>(1) << (-shift);
    ICHECK(v % d == 0) << "Recast cannot scale " << v << " by 2^" << shift
                       << " without remainder";
    return v / d;
  };
  // Scale every stride leaf, preserving the shape tree (TransformLeaf keeps the
  // hierarchy; the recovered layout's strides are plain integers).
  IntTuple new_stride = TransformLeaf(n->layout->stride, [&](IntScalar s) {
    ICHECK(IsConst(s)) << "Recast requires plain integer strides";
    return IntScalar(IntConst(scale(AsConst(s))));
  });
  Layout new_layout(n->layout->shape, new_stride);
  return ComposedLayout(n->swizzle.Recast(old_bits, new_bits), scale(n->offset),
                        new_layout);
}

// Recover a CUTLASS/CuTe XOR swizzle over an affine layout from a TileLang
// layout: A(x) = Sw(offset + sum over plain modes). Sizes and strides need not
// be powers of two.
//
// Stage 1 (detect the swizzle via GF(2) bit-incidence). For each bit-atom 2^p
// of a power-of-two-SIZE input dim, col = A(onehot) XOR A(0). A swizzle source
// bit yields a weight-2 column {lo, lo + s_shift} whose low bit lo is also some
// atom's weight-1 (identity) image — that cross-check excludes weight-2
// columns coming from non-power-of-2 plain strides. The swizzle is the lowest
// contiguous run of such targets sharing one s_shift.
//
// Stage 2 (recover the plain layout by probing). Sw is an involution, so
// offset = Sw(A(0)) and the plain address is P(x) = Sw(A(x)) - offset. Each
// logical dim contributes one mode per power-of-two bit (so a dim split across
// non-contiguous address regions is captured) plus a non-power-of-2 residual
// mode; each mode's stride is P at the coordinate that activates only it.
//
// Stage 3 (prove). The candidate is returned only if ProveEquivalent shows
// A(x) == Sw(offset + plain(x)) for every coordinate.
Optional<ComposedLayout>
ComposedLayoutFromTileLang(const tvm::tl::Layout &layout) {
  AddrProbe A(layout);
  if (!A.valid() || A.shape().empty())
    return std::nullopt;
  const std::vector<int64_t> &shape = A.shape();
  int n = static_cast<int>(shape.size());

  std::vector<int64_t> zero(n, 0);
  std::optional<int64_t> A0 = A(zero);
  if (!A0)
    return std::nullopt;

  // Stage 1: bit-incidence columns for every power-of-two bit-atom.
  std::vector<uint64_t> cols;
  uint64_t weight1 = 0; // bit positions that are some atom's identity image.
  for (int k = 0; k < n; ++k) {
    int nb = Log2Exact(shape[k]);
    for (int p = 0; p < nb; ++p) {
      std::vector<int64_t> x(n, 0);
      x[k] = int64_t(1) << p;
      std::optional<int64_t> a = A(x);
      if (!a)
        return std::nullopt;
      uint64_t col = static_cast<uint64_t>(*a ^ *A0);
      cols.push_back(col);
      if (__builtin_popcountll(col) == 1)
        weight1 |= col;
    }
  }
  std::map<int, int> s_at; // swizzle target bit -> s_shift.
  for (uint64_t col : cols) {
    if (__builtin_popcountll(col) != 2)
      continue;
    int lo = __builtin_ctzll(col);
    int hi = 63 - __builtin_clzll(col);
    if (!(weight1 & (uint64_t(1) << lo)))
      continue; // not a swizzle source: e.g. a non-pow2 plain stride.
    if (s_at.count(lo))
      return std::nullopt; // two sources at one target.
    s_at.emplace(lo, hi - lo);
  }
  Swizzle swizzle = Swizzle::Identity();
  if (!s_at.empty()) {
    // The lowest contiguous run of targets sharing one s_shift.
    int m_base = s_at.begin()->first;
    int s_shift = s_at.begin()->second;
    int b_bits = 0;
    for (auto it = s_at.find(m_base + b_bits);
         it != s_at.end() && it->second == s_shift;
         it = s_at.find(m_base + b_bits))
      ++b_bits;
    if (s_shift < b_bits)
      return std::nullopt; // CuTe requires disjoint source/target regions.
    swizzle = Swizzle(b_bits, m_base, s_shift);
  }

  // Stage 2: offset and per-mode plain strides via unswizzled probes
  // P(x) = Sw(A(x)) - offset, with offset = Sw(A(0)).
  int64_t offset = swizzle->Apply(*A0);
  auto P = [&](const std::vector<int64_t> &x) -> std::optional<int64_t> {
    std::optional<int64_t> a = A(x);
    if (!a)
      return std::nullopt;
    return swizzle->Apply(*a) - offset;
  };
  Array<int64_t> mode_shape, mode_stride;
  for (int k = 0; k < n; ++k) {
    int64_t rem = shape[k];
    if (rem <= 1) {
      mode_shape.push_back(1); // singleton dim: extent-1 mode, stride 0.
      mode_stride.push_back(0);
      continue;
    }
    // One mode per power-of-two bit (so a dim split across non-contiguous
    // address regions is captured), then a non-power-of-2 residual mode. The
    // modes are emitted dim-by-dim in increasing place order, so the layout's
    // single-coordinate domain is the column-major linearization over `shape`.
    for (int64_t place = 1; rem > 1;) {
      int64_t ext = rem % 2 == 0 ? 2 : rem;
      std::vector<int64_t> x(n, 0);
      x[k] = place;
      std::optional<int64_t> d = P(x);
      if (!d)
        return std::nullopt;
      if (*d == 0)
        return std::nullopt; // non-injective: the mode does not move the addr.
      mode_shape.push_back(ext);
      mode_stride.push_back(*d);
      rem /= ext;
      place *= ext;
    }
  }
  Layout plain = Coalesce(Layout(mode_shape, mode_stride));

  // Stage 3: only a proven-equivalent recovery is returned.
  if (!ProveEquivalent(A, swizzle, offset, plain))
    return std::nullopt;
  return ComposedLayout(swizzle, offset, plain);
}

Optional<Layout> LayoutFromTileLang(const tvm::tl::Layout &layout) {
  // The affine special case of the composed recovery: no swizzle, no offset.
  Optional<ComposedLayout> composed = ComposedLayoutFromTileLang(layout);
  if (!composed.defined())
    return std::nullopt;
  const ComposedLayoutNode *c = composed.value().get();
  if (c->swizzle->IsSwizzled() || c->offset != 0)
    return std::nullopt;
  return c->layout;
}

TVM_FFI_STATIC_INIT_BLOCK() {
  SwizzleNode::RegisterReflection();
  IntConstNode::RegisterReflection();
  IntExprNode::RegisterReflection();
  ScaledBasisNode::RegisterReflection();
  IntTupleArrayNode::RegisterReflection();
  LayoutNode::RegisterReflection();
  ComposedLayoutNode::RegisterReflection();
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  // All CuTe-layout FFI lives under the `tl.cute.*` namespace so the Python
  // side initializes a single module (tilelang.layout.cute).
  refl::GlobalDef()
      .def("tl.cute.composed_layout_from_tilelang",
           [](const tvm::tl::Layout &layout) {
             return ComposedLayoutFromTileLang(layout);
           })
      .def("tl.cute.swizzle_recast",
           [](const Swizzle &swizzle, int old_bits, int new_bits) {
             return swizzle.Recast(old_bits, new_bits);
           })
      .def("tl.cute.composed_layout_recast",
           [](const ComposedLayout &layout, int old_bits, int new_bits) {
             return layout.Recast(old_bits, new_bits);
           })
      .def("tl.cute.layout_from_tilelang",
           [](const tvm::tl::Layout &layout) {
             return LayoutFromTileLang(layout);
           })
      .def("tl.cute.make_layout",
           [](Array<int64_t> shape, Array<int64_t> stride) {
             return Layout(shape, stride);
           })
      .def("tl.cute.make_layout_leaves",
           [](Array<int64_t> shape, Array<IntScalar> stride) {
             return MakeLayoutFromLeaves(shape, stride);
           })
      .def("tl.cute.make_int_const",
           [](int64_t value) { return IntConst(value); })
      .def("tl.cute.make_int_expr",
           [](PrimExpr value) { return IntExpr(std::move(value)); })
      .def("tl.cute.make_scaled_basis",
           [](IntScalar value, Array<int64_t> basis) {
             return ScaledBasis(std::move(value), std::move(basis));
           })
      .def("tl.cute.make_identity_layout",
           [](Array<int64_t> shape) {
             return MakeIdentityLayout(
                 std::vector<int64_t>(shape.begin(), shape.end()));
           })
      .def("tl.cute.coalesce",
           [](const Layout &layout) { return Coalesce(layout); })
      .def("tl.cute.right_inverse",
           [](const Layout &layout) { return RightInverse(layout); })
      .def("tl.cute.composition",
           [](const Layout &lhs, const Layout &rhs) {
             return Composition(lhs, rhs);
           })
      .def("tl.cute.layout_flat_shape",
           [](const Layout &layout) {
             Array<int64_t> out;
             for (const IntScalar &s : Leaves(layout->shape)) {
               ICHECK(IsConst(s))
                   << "layout_flat_shape: non-constant shape leaf";
               out.push_back(AsConst(s));
             }
             return out;
           })
      .def("tl.cute.layout_stride_leaves",
           [](const Layout &layout) {
             // The flattened stride leaves as objects (preserves dynamic /
             // ScaledBasis leaves for inspection).
             Array<IntScalar> out;
             for (const IntScalar &s : Leaves(layout->stride))
               out.push_back(s);
             return out;
           })
      .def("tl.cute.derive_tma_tile",
           [](const Layout &gmem, const Layout &smem_plain,
              Array<int64_t> tile_shape) -> Optional<Array<Layout>> {
             // Returns {box, rest_gmem, rest_smem} cute::Layouts, or nullopt if
             // not TMA-expressible. See cute::DeriveTmaTile.
             auto t = DeriveTmaTile(
                 gmem, smem_plain,
                 std::vector<int64_t>(tile_shape.begin(), tile_shape.end()));
             if (!t.has_value())
               return std::nullopt;
             return Array<Layout>{t->box, t->rest_gmem, t->rest_smem};
           })
      .def("tl.cute.layout_flat_stride", [](const Layout &layout) {
        // Flattened strides as integers; errors on a dynamic / basis leaf
        // (which only appear in intermediate TMA derivations, never in a
        // recovered affine layout). Use layout_stride_leaves for the general
        // case.
        Array<int64_t> out;
        for (const IntScalar &s : Leaves(layout->stride)) {
          ICHECK(IsConst(s)) << "layout_flat_stride: non-constant stride leaf; "
                                "use layout_stride_leaves";
          out.push_back(AsConst(s));
        }
        return out;
      });
}

} // namespace cute
} // namespace tl
} // namespace tvm
