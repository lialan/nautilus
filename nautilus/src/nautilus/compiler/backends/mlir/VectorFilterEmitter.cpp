#include "nautilus/compiler/backends/mlir/VectorFilterEmitter.hpp"
#include "nautilus/compiler/ir/IRGraph.hpp"
#include "nautilus/compiler/ir/operations/CastOperation.hpp"
#include "nautilus/compiler/ir/operations/ConstIntOperation.hpp"
#include "nautilus/compiler/ir/operations/FunctionOperation.hpp"
#include "nautilus/compiler/ir/operations/LoadOperation.hpp"
#include "nautilus/compiler/ir/operations/LogicalOperations/CompareOperation.hpp"
#include <algorithm>
#include <climits>
#include <map>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/ControlFlow/IR/ControlFlowOps.h>
#include <mlir/Dialect/LLVMIR/LLVMDialect.h>
#include <mlir/Dialect/LLVMIR/LLVMTypes.h>
#include <mlir/Dialect/SCF/IR/SCF.h>
#include <mlir/IR/Builders.h>
#include <set>
#include <stdexcept>

namespace nautilus::compiler::mlir {

VectorFilterEmitter::VectorFilterEmitter(::mlir::MLIRContext& context, const engine::Options& options)
    : context(context), options(options), typeSize(options.getOptionOrDefault("vectorFilter.typeSize", 4)) {
}

VectorFilterEmitter::~VectorFilterEmitter() = default;

// ---------------------------------------------------------------------------
// Utility: collect all unique column indices from a predicate tree
// ---------------------------------------------------------------------------

void collectColumnIndices(const PredicateNode& node, std::vector<int>& indices) {
	std::visit(
	    [&](auto&& arg) {
		    using T = std::decay_t<decltype(arg)>;
		    if constexpr (std::is_same_v<T, CompareNode>) {
			    indices.push_back(arg.columnIndex);
		    } else if constexpr (std::is_same_v<T, std::unique_ptr<AndNode>>) {
			    collectColumnIndices(arg->left, indices);
			    collectColumnIndices(arg->right, indices);
		    } else if constexpr (std::is_same_v<T, std::unique_ptr<OrNode>>) {
			    collectColumnIndices(arg->left, indices);
			    collectColumnIndices(arg->right, indices);
		    } else if constexpr (std::is_same_v<T, std::unique_ptr<NotNode>>) {
			    collectColumnIndices(arg->child, indices);
		    }
	    },
	    node);
}

// ---------------------------------------------------------------------------
// IR Walking: Extract predicate from the traced IRGraph
// ---------------------------------------------------------------------------

ExtractedPredicate VectorFilterEmitter::extractPredicate(const ir::IRGraph& ir) {
	using OpType = ir::Operation::OperationType;

	const auto& funcOp = ir.getRootOperation();
	const auto& blocks = funcOp.getBasicBlocks();

	// Walk all operations in all basic blocks to find the CompareOperation.
	const ir::CompareOperation* compareOp = nullptr;
	const ir::ConstIntOperation* constOp = nullptr;
	// We don't need to track the LoadOperation for now -- just the constant and comparator.

	for (const auto& block : blocks) {
		for (const auto& op : block->getOperations()) {
			switch (op->getOperationType()) {
			case OpType::CompareOp: {
				if (compareOp != nullptr) {
					throw std::runtime_error("VectorFilterEmitter: multiple CompareOperations not yet supported");
				}
				compareOp = op->dynCast<ir::CompareOperation>();
				break;
			}
			case OpType::ConstIntOp: {
				// There may be multiple constants; we pick the one used by the compare.
				// For now, record the last one; we'll refine below.
				auto* candidate = op->dynCast<ir::ConstIntOperation>();
				if (candidate != nullptr) {
					constOp = candidate;
				}
				break;
			}
			default:
				break;
			}
		}
	}

	if (compareOp == nullptr) {
		throw std::runtime_error("VectorFilterEmitter: no CompareOperation found in IR");
	}

	// Now identify which side of the compare is the constant and which is the column load.
	// One side should trace back to a LoadOperation (the column value), the other to a ConstIntOperation.
	auto* left = compareOp->getLeftInput();
	auto* right = compareOp->getRightInput();

	// Helper: chase through CastOperations to find the underlying operation.
	auto chase = [](ir::Operation* op) -> ir::Operation* {
		while (op != nullptr && op->getOperationType() == OpType::CastOp) {
			op = const_cast<ir::Operation*>(static_cast<const ir::CastOperation*>(op)->getInput());
		}
		return op;
	};

	auto* leftResolved = chase(left);
	auto* rightResolved = chase(right);

	const ir::ConstIntOperation* foundConst = nullptr;
	// Determine which side is the constant.
	if (leftResolved != nullptr && leftResolved->getOperationType() == OpType::ConstIntOp) {
		foundConst = leftResolved->dynCast<ir::ConstIntOperation>();
	} else if (rightResolved != nullptr && rightResolved->getOperationType() == OpType::ConstIntOp) {
		foundConst = rightResolved->dynCast<ir::ConstIntOperation>();
	} else if (constOp != nullptr) {
		// Fallback: use the last ConstIntOp we found scanning the blocks.
		foundConst = constOp;
	} else {
		throw std::runtime_error("VectorFilterEmitter: could not find constant in predicate");
	}

	ExtractedPredicate result;
	result.comparator = compareOp->getComparator();
	result.constantValue = foundConst->getValue();
	result.columnIndex = 0; // single-column for now
	return result;
}

// ---------------------------------------------------------------------------
// MLIR Generation: Build a vectorized filter module from a predicate
// ---------------------------------------------------------------------------

static ::mlir::arith::CmpIPredicate comparatorToMLIR(ir::CompareOperation::Comparator comp) {
	switch (comp) {
	case ir::CompareOperation::EQ:
		return ::mlir::arith::CmpIPredicate::eq;
	case ir::CompareOperation::NE:
		return ::mlir::arith::CmpIPredicate::ne;
	case ir::CompareOperation::LT:
		return ::mlir::arith::CmpIPredicate::slt;
	case ir::CompareOperation::LE:
		return ::mlir::arith::CmpIPredicate::sle;
	case ir::CompareOperation::GT:
		return ::mlir::arith::CmpIPredicate::sgt;
	case ir::CompareOperation::GE:
		return ::mlir::arith::CmpIPredicate::sge;
	default:
		throw std::runtime_error("VectorFilterEmitter: unsupported comparator");
	}
}

::mlir::OwningOpRef<::mlir::ModuleOp>
VectorFilterEmitter::generateModuleFromPredicate(ir::CompareOperation::Comparator comparator, int64_t constantValue,
                                                 int columnIndex) {
	// Wrap as a single CompareNode and delegate to the predicate tree method.
	CompareNode node {comparator, constantValue, columnIndex};
	PredicateNode root = node;
	return generateModuleFromPredicateTree(root);
}

// ---------------------------------------------------------------------------
// Helper structures for predicate tree emission context
// ---------------------------------------------------------------------------

/// Context passed through recursive predicate emission, containing all the
/// state needed to emit vector and scalar predicate evaluations.
struct EmitContext {
	int typeSize;
	int vectorWidth;
	::mlir::Type elemTy;
	::mlir::VectorType vecElemTy;
	::mlir::Type ptrTy;
	::mlir::Type i64Ty;
	// Column pointers indexed by column index
	std::map<int, ::mlir::Value> colPtrs;
	// Whether null checks are enabled (only applied for typeSize >= 4)
	bool nullChecksEnabled;
	// Pointer to vars[] array (funcOp arg 3), used when isRuntimeVar == true
	::mlir::Value vars = {};
};

/// Emit a vector-mode predicate evaluation, returning a vector<Nxi1> mask.
static ::mlir::Value emitVectorPredicate(::mlir::OpBuilder& b, ::mlir::Location loc, const PredicateNode& node,
                                         const EmitContext& ctx, ::mlir::Value iv) {
	return std::visit(
	    [&](auto&& arg) -> ::mlir::Value {
		    using T = std::decay_t<decltype(arg)>;

		    if constexpr (std::is_same_v<T, CompareNode>) {
			    // Load vector from column[iv]
			    auto colPtr = ctx.colPtrs.at(arg.columnIndex);
			    auto colGep =
			        b.create<::mlir::LLVM::GEPOp>(loc, ctx.ptrTy, ctx.elemTy, colPtr, ::mlir::ValueRange {iv});
			    auto colVec = b.create<::mlir::LLVM::LoadOp>(loc, ctx.vecElemTy, colGep, /*alignment=*/ctx.typeSize);

			    // Build comparison vector: either from vars[] or compile-time constant
			    ::mlir::Value splatVec;
			    if (arg.isRuntimeVar) {
				    // Load vars[varSlotIndex] as i64, truncate to element type, splat
				    auto slotIdx = b.create<::mlir::arith::ConstantOp>(loc, b.getI64IntegerAttr(arg.varSlotIndex));
				    auto varGep = b.create<::mlir::LLVM::GEPOp>(loc, ctx.ptrTy, ctx.i64Ty, ctx.vars,
				                                                  ::mlir::ValueRange {slotIdx});
				    auto varI64 = b.create<::mlir::LLVM::LoadOp>(loc, ctx.i64Ty, varGep, /*alignment=*/8);

				    ::mlir::Value scalar;
				    if (ctx.typeSize == 8) {
					    scalar = varI64;
				    } else {
					    scalar = b.create<::mlir::arith::TruncIOp>(loc, ctx.elemTy, varI64);
				    }
				    splatVec = b.create<::mlir::LLVM::ShuffleVectorOp>(
				        loc,
				        b.create<::mlir::LLVM::InsertElementOp>(loc, b.create<::mlir::LLVM::UndefOp>(loc, ctx.vecElemTy),
				                                                scalar,
				                                                b.create<::mlir::arith::ConstantOp>(loc, b.getI32IntegerAttr(0))),
				        b.create<::mlir::LLVM::UndefOp>(loc, ctx.vecElemTy),
				        llvm::SmallVector<int32_t>(ctx.vectorWidth, 0));
			    } else {
				    // Existing path: splat compile-time constant
				    ::mlir::DenseElementsAttr splatAttr;
				    switch (ctx.typeSize) {
				    case 1:
					    splatAttr =
					        ::mlir::DenseElementsAttr::get(ctx.vecElemTy, static_cast<int8_t>(arg.constantValue));
					    break;
				    case 2:
					    splatAttr =
					        ::mlir::DenseElementsAttr::get(ctx.vecElemTy, static_cast<int16_t>(arg.constantValue));
					    break;
				    case 4:
					    splatAttr =
					        ::mlir::DenseElementsAttr::get(ctx.vecElemTy, static_cast<int32_t>(arg.constantValue));
					    break;
				    case 8:
					    splatAttr =
					        ::mlir::DenseElementsAttr::get(ctx.vecElemTy, static_cast<int64_t>(arg.constantValue));
					    break;
				    default:
					    throw std::runtime_error("VectorFilterEmitter: unsupported typeSize for splat");
				    }
				    splatVec = b.create<::mlir::arith::ConstantOp>(loc, splatAttr);
			    }

			    auto pred = comparatorToMLIR(arg.comparator);
			    auto rawMask = b.create<::mlir::arith::CmpIOp>(loc, pred, colVec, splatVec).getResult();

			    // Apply null correction when null checks are enabled and typeSize >= 4
			    if (ctx.nullChecksEnabled && ctx.typeSize >= 4) {
				    // Build null sentinel splat
				    int64_t nullSentinel = (ctx.typeSize == 4) ? static_cast<int64_t>(INT32_MIN) : INT64_MIN;
				    ::mlir::DenseElementsAttr nullSplatAttr;
				    if (ctx.typeSize == 4) {
					    nullSplatAttr =
					        ::mlir::DenseElementsAttr::get(ctx.vecElemTy, static_cast<int32_t>(nullSentinel));
				    } else {
					    nullSplatAttr =
					        ::mlir::DenseElementsAttr::get(ctx.vecElemTy, static_cast<int64_t>(nullSentinel));
				    }
				    auto nullSplat = b.create<::mlir::arith::ConstantOp>(loc, nullSplatAttr);

				    auto lhsNull =
				        b.create<::mlir::arith::CmpIOp>(loc, ::mlir::arith::CmpIPredicate::eq, colVec, nullSplat);
				    auto rhsNull =
				        b.create<::mlir::arith::CmpIOp>(loc, ::mlir::arith::CmpIPredicate::eq, splatVec, nullSplat);

				    auto maskType = ::mlir::cast<::mlir::VectorType>(rawMask.getType());
				    auto trueSplat =
				        b.create<::mlir::arith::ConstantOp>(loc, ::mlir::DenseElementsAttr::get(maskType, true));

				    using Comp = ir::CompareOperation::Comparator;
				    switch (arg.comparator) {
				    case Comp::GT:
				    case Comp::LT: {
					    auto eitherNull = b.create<::mlir::arith::OrIOp>(loc, lhsNull, rhsNull);
					    auto notNull = b.create<::mlir::arith::XOrIOp>(loc, eitherNull, trueSplat);
					    rawMask = b.create<::mlir::arith::AndIOp>(loc, rawMask, notNull);
					    break;
				    }
				    case Comp::GE:
				    case Comp::LE: {
					    auto oppositePred = (arg.comparator == Comp::GE) ? ::mlir::arith::CmpIPredicate::slt
					                                                     : ::mlir::arith::CmpIPredicate::sgt;
					    auto oppositeResult = b.create<::mlir::arith::CmpIOp>(loc, oppositePred, colVec, splatVec);
					    auto notOpposite = b.create<::mlir::arith::XOrIOp>(loc, oppositeResult, trueSplat);
					    auto nullXor = b.create<::mlir::arith::XOrIOp>(loc, lhsNull, rhsNull);
					    auto notNullXor = b.create<::mlir::arith::XOrIOp>(loc, nullXor, trueSplat);
					    rawMask = b.create<::mlir::arith::AndIOp>(loc, notOpposite, notNullXor);
					    break;
				    }
				    case Comp::EQ: {
					    auto bothNull = b.create<::mlir::arith::AndIOp>(loc, lhsNull, rhsNull);
					    rawMask = b.create<::mlir::arith::OrIOp>(loc, rawMask, bothNull);
					    break;
				    }
				    case Comp::NE: {
					    auto eqRaw =
					        b.create<::mlir::arith::CmpIOp>(loc, ::mlir::arith::CmpIPredicate::eq, colVec, splatVec);
					    auto bothNull = b.create<::mlir::arith::AndIOp>(loc, lhsNull, rhsNull);
					    auto eqWithNull = b.create<::mlir::arith::OrIOp>(loc, eqRaw, bothNull);
					    rawMask = b.create<::mlir::arith::XOrIOp>(loc, eqWithNull, trueSplat);
					    break;
				    }
				    default:
					    break;
				    }
			    }

			    return rawMask;

		    } else if constexpr (std::is_same_v<T, std::unique_ptr<AndNode>>) {
			    auto leftMask = emitVectorPredicate(b, loc, arg->left, ctx, iv);
			    auto rightMask = emitVectorPredicate(b, loc, arg->right, ctx, iv);
			    return b.create<::mlir::arith::AndIOp>(loc, leftMask, rightMask).getResult();

		    } else if constexpr (std::is_same_v<T, std::unique_ptr<OrNode>>) {
			    auto leftMask = emitVectorPredicate(b, loc, arg->left, ctx, iv);
			    auto rightMask = emitVectorPredicate(b, loc, arg->right, ctx, iv);
			    return b.create<::mlir::arith::OrIOp>(loc, leftMask, rightMask).getResult();

		    } else if constexpr (std::is_same_v<T, std::unique_ptr<NotNode>>) {
			    auto childMask = emitVectorPredicate(b, loc, arg->child, ctx, iv);
			    // XOR with all-true splat to negate
			    auto maskType = ::mlir::cast<::mlir::VectorType>(childMask.getType());
			    auto trueSplat =
			        b.create<::mlir::arith::ConstantOp>(loc, ::mlir::DenseElementsAttr::get(maskType, true));
			    return b.create<::mlir::arith::XOrIOp>(loc, childMask, trueSplat).getResult();

		    } else {
			    throw std::runtime_error("VectorFilterEmitter: unknown predicate node type");
		    }
	    },
	    node);
}

/// Emit a scalar-mode predicate evaluation, returning an i1 value.
static ::mlir::Value emitScalarPredicate(::mlir::OpBuilder& b, ::mlir::Location loc, const PredicateNode& node,
                                         const EmitContext& ctx, ::mlir::Value jv) {
	return std::visit(
	    [&](auto&& arg) -> ::mlir::Value {
		    using T = std::decay_t<decltype(arg)>;

		    if constexpr (std::is_same_v<T, CompareNode>) {
			    // Load single element from column[jv]
			    auto colPtr = ctx.colPtrs.at(arg.columnIndex);
			    auto gep = b.create<::mlir::LLVM::GEPOp>(loc, ctx.ptrTy, ctx.elemTy, colPtr, ::mlir::ValueRange {jv});
			    auto val = b.create<::mlir::LLVM::LoadOp>(loc, ctx.elemTy, gep, /*alignment=*/ctx.typeSize);

			    // Create scalar constant matching the element type
			    ::mlir::Value scalarCst;
			    switch (ctx.typeSize) {
			    case 1:
				    scalarCst =
				        b.create<::mlir::arith::ConstantOp>(loc, b.getIntegerAttr(b.getI8Type(), arg.constantValue));
				    break;
			    case 2:
				    scalarCst =
				        b.create<::mlir::arith::ConstantOp>(loc, b.getIntegerAttr(b.getI16Type(), arg.constantValue));
				    break;
			    case 4:
				    scalarCst =
				        b.create<::mlir::arith::ConstantOp>(loc, b.getIntegerAttr(b.getI32Type(), arg.constantValue));
				    break;
			    case 8:
				    scalarCst =
				        b.create<::mlir::arith::ConstantOp>(loc, b.getIntegerAttr(b.getI64Type(), arg.constantValue));
				    break;
			    default:
				    throw std::runtime_error("VectorFilterEmitter: unsupported typeSize for scalar constant");
			    }

			    auto pred = comparatorToMLIR(arg.comparator);
			    auto rawResult = b.create<::mlir::arith::CmpIOp>(loc, pred, val, scalarCst).getResult();

			    // Apply scalar null correction when null checks are enabled and typeSize >= 4
			    if (ctx.nullChecksEnabled && ctx.typeSize >= 4) {
				    // Build null sentinel constant
				    int64_t nullSentinel = (ctx.typeSize == 4) ? static_cast<int64_t>(INT32_MIN) : INT64_MIN;
				    ::mlir::Value nullCst;
				    if (ctx.typeSize == 4) {
					    nullCst =
					        b.create<::mlir::arith::ConstantOp>(loc, b.getIntegerAttr(b.getI32Type(), nullSentinel));
				    } else {
					    nullCst =
					        b.create<::mlir::arith::ConstantOp>(loc, b.getIntegerAttr(b.getI64Type(), nullSentinel));
				    }

				    auto lhsNull = b.create<::mlir::arith::CmpIOp>(loc, ::mlir::arith::CmpIPredicate::eq, val, nullCst);
				    auto rhsNull =
				        b.create<::mlir::arith::CmpIOp>(loc, ::mlir::arith::CmpIPredicate::eq, scalarCst, nullCst);

				    auto trueVal = b.create<::mlir::arith::ConstantOp>(loc, b.getBoolAttr(true));

				    using Comp = ir::CompareOperation::Comparator;
				    switch (arg.comparator) {
				    case Comp::GT:
				    case Comp::LT: {
					    // False if either null: result AND NOT(lhs_null OR rhs_null)
					    auto eitherNull = b.create<::mlir::arith::OrIOp>(loc, lhsNull, rhsNull);
					    auto notNull = b.create<::mlir::arith::XOrIOp>(loc, eitherNull, trueVal);
					    rawResult = b.create<::mlir::arith::AndIOp>(loc, rawResult, notNull);
					    break;
				    }
				    case Comp::GE:
				    case Comp::LE: {
					    // True if both null, false if only one null:
					    // NOT(opposite) AND NOT(lhs_null XOR rhs_null)
					    auto oppositePred = (arg.comparator == Comp::GE) ? ::mlir::arith::CmpIPredicate::slt
					                                                     : ::mlir::arith::CmpIPredicate::sgt;
					    auto oppositeResult = b.create<::mlir::arith::CmpIOp>(loc, oppositePred, val, scalarCst);
					    auto notOpposite = b.create<::mlir::arith::XOrIOp>(loc, oppositeResult, trueVal);
					    auto nullXor = b.create<::mlir::arith::XOrIOp>(loc, lhsNull, rhsNull);
					    auto notNullXor = b.create<::mlir::arith::XOrIOp>(loc, nullXor, trueVal);
					    rawResult = b.create<::mlir::arith::AndIOp>(loc, notOpposite, notNullXor);
					    break;
				    }
				    case Comp::EQ: {
					    // True if both null: (lhs == rhs) OR (lhs_null AND rhs_null)
					    auto bothNull = b.create<::mlir::arith::AndIOp>(loc, lhsNull, rhsNull);
					    rawResult = b.create<::mlir::arith::OrIOp>(loc, rawResult, bothNull);
					    break;
				    }
				    case Comp::NE: {
					    // NOT(eq with null handling)
					    auto eqRaw =
					        b.create<::mlir::arith::CmpIOp>(loc, ::mlir::arith::CmpIPredicate::eq, val, scalarCst);
					    auto bothNull = b.create<::mlir::arith::AndIOp>(loc, lhsNull, rhsNull);
					    auto eqWithNull = b.create<::mlir::arith::OrIOp>(loc, eqRaw, bothNull);
					    rawResult = b.create<::mlir::arith::XOrIOp>(loc, eqWithNull, trueVal);
					    break;
				    }
				    default:
					    break;
				    }
			    }

			    return rawResult;

		    } else if constexpr (std::is_same_v<T, std::unique_ptr<AndNode>>) {
			    auto leftVal = emitScalarPredicate(b, loc, arg->left, ctx, jv);
			    auto rightVal = emitScalarPredicate(b, loc, arg->right, ctx, jv);
			    return b.create<::mlir::arith::AndIOp>(loc, leftVal, rightVal).getResult();

		    } else if constexpr (std::is_same_v<T, std::unique_ptr<OrNode>>) {
			    auto leftVal = emitScalarPredicate(b, loc, arg->left, ctx, jv);
			    auto rightVal = emitScalarPredicate(b, loc, arg->right, ctx, jv);
			    return b.create<::mlir::arith::OrIOp>(loc, leftVal, rightVal).getResult();

		    } else if constexpr (std::is_same_v<T, std::unique_ptr<NotNode>>) {
			    auto childVal = emitScalarPredicate(b, loc, arg->child, ctx, jv);
			    // XOR with true to negate
			    auto trueVal = b.create<::mlir::arith::ConstantOp>(loc, b.getBoolAttr(true));
			    return b.create<::mlir::arith::XOrIOp>(loc, childVal, trueVal).getResult();

		    } else {
			    throw std::runtime_error("VectorFilterEmitter: unknown predicate node type");
		    }
	    },
	    node);
}

// ---------------------------------------------------------------------------
// MLIR Generation: Build a vectorized filter module from a predicate tree
// ---------------------------------------------------------------------------

::mlir::OwningOpRef<::mlir::ModuleOp> VectorFilterEmitter::generateModuleFromPredicateTree(const PredicateNode& root) {
	context.loadAllAvailableDialects();

	::mlir::OpBuilder builder(&context);
	auto loc = builder.getUnknownLoc();

	// --- Types ---
	auto i8Ty = builder.getI8Type();
	auto i32Ty = builder.getI32Type();
	auto i64Ty = builder.getI64Type();
	auto ptrTy = ::mlir::LLVM::LLVMPointerType::get(&context);

	// Compute vector width from typeSize: vectorWidth = 512 / (typeSize * 8) = 64 / typeSize
	int vectorWidth = 64 / typeSize;
	// Element MLIR type based on typeSize
	::mlir::Type elemTy;
	switch (typeSize) {
	case 1:
		elemTy = builder.getI8Type();
		break;
	case 2:
		elemTy = builder.getI16Type();
		break;
	case 4:
		elemTy = i32Ty;
		break;
	case 8:
		elemTy = i64Ty;
		break;
	default:
		throw std::runtime_error("VectorFilterEmitter: unsupported typeSize");
	}

	auto vecElemTy = ::mlir::VectorType::get({vectorWidth}, elemTy);
	auto vec8xi64Ty = ::mlir::VectorType::get({8}, i64Ty);

	// --- Module ---
	auto module = ::mlir::ModuleOp::create(loc);
	builder.setInsertionPointToEnd(module.getBody());

	// Function type: (ptr, i64, ptr, ptr, i64, ptr, i64, i64) -> i64
	auto funcTy = ::mlir::LLVM::LLVMFunctionType::get(i64Ty, {ptrTy, i64Ty, ptrTy, ptrTy, i64Ty, ptrTy, i64Ty, i64Ty});
	auto funcOp =
	    builder.create<::mlir::LLVM::LLVMFuncOp>(loc, "execute", funcTy, ::mlir::LLVM::Linkage::External, false);

	auto* entryBlock = funcOp.addEntryBlock(builder);
	builder.setInsertionPointToStart(entryBlock);

	// --- Extract function arguments ---
	::mlir::Value cols = funcOp.getArgument(0);
	::mlir::Value rows = funcOp.getArgument(5);
	::mlir::Value rowsCount = funcOp.getArgument(6);
	::mlir::Value rowsStartOffset = funcOp.getArgument(7);

	// --- Collect unique column indices and load all column pointers ---
	std::vector<int> allIndices;
	collectColumnIndices(root, allIndices);
	std::set<int> uniqueIndicesSet(allIndices.begin(), allIndices.end());
	std::map<int, ::mlir::Value> colPtrs;
	for (int colIdx : uniqueIndicesSet) {
		auto colIdxCst = builder.create<::mlir::arith::ConstantOp>(loc, builder.getI64IntegerAttr(colIdx));
		auto colGep = builder.create<::mlir::LLVM::GEPOp>(loc, ptrTy, i64Ty, cols, ::mlir::ValueRange {colIdxCst});
		auto colAddr = builder.create<::mlir::LLVM::LoadOp>(loc, i64Ty, colGep, /*alignment=*/8);
		auto colPtr =
		    builder.create<::mlir::LLVM::IntToPtrOp>(loc, ptrTy, colAddr, ::mlir::LLVM::DereferenceableAttr());
		colPtrs[colIdx] = colPtr;
	}

	// --- Build emission context ---
	EmitContext ctx;
	ctx.typeSize = typeSize;
	ctx.vectorWidth = vectorWidth;
	ctx.elemTy = elemTy;
	ctx.vecElemTy = vecElemTy;
	ctx.ptrTy = ptrTy;
	ctx.i64Ty = i64Ty;
	ctx.colPtrs = colPtrs;
	ctx.nullChecksEnabled = options.getOptionOrDefault("mlir.null_checks", false);

	// ==================== Runtime branch: row-index vs gather ====================
	// rowsStartOffset >= 0 -> vectorized row-index mode
	// rowsStartOffset < 0  -> scalar gather mode (gatherColIdx = -(rowsStartOffset + 1))
	auto cst0 = builder.create<::mlir::arith::ConstantOp>(loc, builder.getI64IntegerAttr(0));
	auto isRowMode =
	    builder.create<::mlir::arith::CmpIOp>(loc, ::mlir::arith::CmpIPredicate::sge, rowsStartOffset, cst0);

	// Create the two destination blocks inside the function's region
	auto& funcRegion = funcOp.getBody();
	auto* vectorizedBlock = new ::mlir::Block();
	auto* scalarGatherBlock = new ::mlir::Block();
	funcRegion.push_back(vectorizedBlock);
	funcRegion.push_back(scalarGatherBlock);

	builder.create<::mlir::cf::CondBranchOp>(loc, isRowMode, vectorizedBlock, /*trueArgs=*/::mlir::ValueRange {},
	                                         scalarGatherBlock, /*falseArgs=*/::mlir::ValueRange {});

	// ========================== Vectorized row-index path ==========================
	builder.setInsertionPointToStart(vectorizedBlock);

	auto cstVecWidth = builder.create<::mlir::arith::ConstantOp>(loc, builder.getI64IntegerAttr(vectorWidth));

	// vec_limit = rowsCount - (rowsCount % vectorWidth)
	auto remainder = builder.create<::mlir::arith::RemSIOp>(loc, rowsCount, cstVecWidth);
	auto vecLimit = builder.create<::mlir::arith::SubIOp>(loc, rowsCount, remainder);

	// Number of 8-element groups for compress-store processing.
	int numGroups = vectorWidth / 8;
	if (numGroups < 1) {
		numGroups = 1;
	}

	auto vecCst0 = builder.create<::mlir::arith::ConstantOp>(loc, builder.getI64IntegerAttr(0));
	auto vecCst1 = builder.create<::mlir::arith::ConstantOp>(loc, builder.getI64IntegerAttr(1));

	// ========================== Vectorized main loop ==========================
	auto vecLoop = builder.create<::mlir::scf::ForOp>(
	    loc, /*lb=*/vecCst0.getResult(), /*ub=*/vecLimit.getResult(), /*step=*/cstVecWidth.getResult(),
	    /*iterArgs=*/::mlir::ValueRange {vecCst0.getResult()},
	    [&](::mlir::OpBuilder& b, ::mlir::Location bodyLoc, ::mlir::Value iv, ::mlir::ValueRange iterArgs) {
		    ::mlir::Value outIdx = iterArgs[0];

		    // Recursively emit the predicate tree to get the full mask
		    auto maskFull = emitVectorPredicate(b, bodyLoc, root, ctx, iv);

		    // Row ID base = iv + rowsStartOffset
		    auto rowBase = b.create<::mlir::arith::AddIOp>(bodyLoc, iv, rowsStartOffset);

		    // Process each group of 8 lanes
		    ::mlir::Value currentIdx = outIdx;
		    for (int g = 0; g < numGroups; g++) {
			    // Extract 8-element mask slice via shufflevector
			    llvm::SmallVector<int32_t, 8> indices;
			    for (int k = 0; k < 8; k++) {
				    indices.push_back(g * 8 + k);
			    }
			    auto maskSlice = b.create<::mlir::LLVM::ShuffleVectorOp>(bodyLoc, maskFull, maskFull,
			                                                             llvm::ArrayRef<int32_t>(indices));

			    // Build row ID vector for this group: base + g*8 + 0..7
			    auto groupBase = b.create<::mlir::arith::ConstantOp>(bodyLoc, b.getI64IntegerAttr(g * 8));
			    auto groupStart = b.create<::mlir::arith::AddIOp>(bodyLoc, rowBase, groupBase);

			    ::mlir::Value ridVec = b.create<::mlir::LLVM::UndefOp>(bodyLoc, vec8xi64Ty);
			    for (int k = 0; k < 8; k++) {
				    auto offset = b.create<::mlir::arith::ConstantOp>(bodyLoc, b.getI64IntegerAttr(k));
				    auto elem = b.create<::mlir::arith::AddIOp>(bodyLoc, groupStart, offset);
				    auto idxI32 = b.create<::mlir::arith::ConstantOp>(bodyLoc, b.getI32IntegerAttr(k));
				    ridVec = b.create<::mlir::LLVM::InsertElementOp>(bodyLoc, ridVec, elem, idxI32);
			    }

			    // Compress-store: write matching row IDs to output
			    auto outGep =
			        b.create<::mlir::LLVM::GEPOp>(bodyLoc, ptrTy, i64Ty, rows, ::mlir::ValueRange {currentIdx});
			    b.create<::mlir::LLVM::masked_compressstore>(bodyLoc, ridVec, outGep, maskSlice);

			    // Popcount to advance output index
			    auto maskI8 = b.create<::mlir::LLVM::BitcastOp>(bodyLoc, i8Ty, maskSlice);
			    auto pop = b.create<::mlir::LLVM::CtPopOp>(bodyLoc, i8Ty, maskI8);
			    auto popI64 = b.create<::mlir::arith::ExtUIOp>(bodyLoc, i64Ty, pop);
			    currentIdx = b.create<::mlir::arith::AddIOp>(bodyLoc, currentIdx, popI64);
		    }

		    b.create<::mlir::scf::YieldOp>(bodyLoc, ::mlir::ValueRange {currentIdx});
	    });

	::mlir::Value vecOut = vecLoop.getResult(0);

	// ============================ Scalar tail loop ============================
	auto scalarLoop = builder.create<::mlir::scf::ForOp>(
	    loc, /*lb=*/vecLimit.getResult(), /*ub=*/rowsCount, /*step=*/vecCst1.getResult(),
	    /*iterArgs=*/::mlir::ValueRange {vecOut},
	    [&](::mlir::OpBuilder& b, ::mlir::Location bodyLoc, ::mlir::Value jv, ::mlir::ValueRange iterArgs) {
		    ::mlir::Value tidx = iterArgs[0];

		    // Recursively emit the scalar predicate
		    auto match = emitScalarPredicate(b, bodyLoc, root, ctx, jv);

		    // Compute row ID = jv + rowsStartOffset
		    auto rowId = b.create<::mlir::arith::AddIOp>(bodyLoc, jv, rowsStartOffset);

		    // Branchless: always store, advance by 0 or 1
		    auto outGep = b.create<::mlir::LLVM::GEPOp>(bodyLoc, ptrTy, i64Ty, rows, ::mlir::ValueRange {tidx});
		    b.create<::mlir::LLVM::StoreOp>(bodyLoc, rowId, outGep, /*alignment=*/8);

		    auto one = b.create<::mlir::arith::ConstantOp>(bodyLoc, b.getI64IntegerAttr(1));
		    auto zero = b.create<::mlir::arith::ConstantOp>(bodyLoc, b.getI64IntegerAttr(0));
		    auto advance = b.create<::mlir::arith::SelectOp>(bodyLoc, match, one, zero);
		    auto newTidx = b.create<::mlir::arith::AddIOp>(bodyLoc, tidx, advance);

		    b.create<::mlir::scf::YieldOp>(bodyLoc, ::mlir::ValueRange {newTidx});
	    });

	::mlir::Value totalMatches = scalarLoop.getResult(0);
	builder.create<::mlir::LLVM::ReturnOp>(loc, ::mlir::ValueRange {totalMatches});

	// ========================== Scalar gather path ==========================
	builder.setInsertionPointToStart(scalarGatherBlock);

	// Decode gatherColIdx = -(rowsStartOffset + 1)
	auto gCst1 = builder.create<::mlir::arith::ConstantOp>(loc, builder.getI64IntegerAttr(1));
	auto gCst0 = builder.create<::mlir::arith::ConstantOp>(loc, builder.getI64IntegerAttr(0));
	auto negOffsetPlusOne = builder.create<::mlir::arith::AddIOp>(loc, rowsStartOffset, gCst1);
	auto gatherColIdx = builder.create<::mlir::arith::SubIOp>(loc, gCst0, negOffsetPlusOne);

	// Load gather column pointer: cols[gatherColIdx] -> inttoptr
	auto gatherGep = builder.create<::mlir::LLVM::GEPOp>(loc, ptrTy, i64Ty, cols, ::mlir::ValueRange {gatherColIdx});
	auto gatherAddr = builder.create<::mlir::LLVM::LoadOp>(loc, i64Ty, gatherGep, /*alignment=*/8);
	auto gatherPtr =
	    builder.create<::mlir::LLVM::IntToPtrOp>(loc, ptrTy, gatherAddr, ::mlir::LLVM::DereferenceableAttr());

	// Scalar gather loop: for each row, evaluate predicate; if match, load from gather col, widen to i64, store
	auto gatherLoop = builder.create<::mlir::scf::ForOp>(
	    loc, /*lb=*/gCst0.getResult(), /*ub=*/rowsCount, /*step=*/gCst1.getResult(),
	    /*iterArgs=*/::mlir::ValueRange {gCst0.getResult()},
	    [&](::mlir::OpBuilder& b, ::mlir::Location bodyLoc, ::mlir::Value rowIdx, ::mlir::ValueRange iterArgs) {
		    ::mlir::Value outIdx = iterArgs[0];

		    // Evaluate scalar predicate for this row
		    auto match = emitScalarPredicate(b, bodyLoc, root, ctx, rowIdx);

		    // Load i32 value from gather column at rowIdx
		    auto gatherElemGep =
		        b.create<::mlir::LLVM::GEPOp>(bodyLoc, ptrTy, i32Ty, gatherPtr, ::mlir::ValueRange {rowIdx});
		    auto gatherVal = b.create<::mlir::LLVM::LoadOp>(bodyLoc, i32Ty, gatherElemGep, /*alignment=*/4);

		    // Widen i32 -> i64 via sign extension
		    auto widened = b.create<::mlir::arith::ExtSIOp>(bodyLoc, i64Ty, gatherVal);

		    // Branchless store: always store, advance by select(match, 1, 0)
		    auto outGep = b.create<::mlir::LLVM::GEPOp>(bodyLoc, ptrTy, i64Ty, rows, ::mlir::ValueRange {outIdx});
		    b.create<::mlir::LLVM::StoreOp>(bodyLoc, widened, outGep, /*alignment=*/8);

		    auto one = b.create<::mlir::arith::ConstantOp>(bodyLoc, b.getI64IntegerAttr(1));
		    auto zero = b.create<::mlir::arith::ConstantOp>(bodyLoc, b.getI64IntegerAttr(0));
		    auto advance = b.create<::mlir::arith::SelectOp>(bodyLoc, match, one, zero);
		    auto newOutIdx = b.create<::mlir::arith::AddIOp>(bodyLoc, outIdx, advance);

		    b.create<::mlir::scf::YieldOp>(bodyLoc, ::mlir::ValueRange {newOutIdx});
	    });

	::mlir::Value gatherCount = gatherLoop.getResult(0);
	builder.create<::mlir::LLVM::ReturnOp>(loc, ::mlir::ValueRange {gatherCount});

	return module;
}

// ---------------------------------------------------------------------------
// Main entry point: extract predicate from IR, then generate vectorized MLIR
// ---------------------------------------------------------------------------

::mlir::OwningOpRef<::mlir::ModuleOp>
VectorFilterEmitter::generateModuleFromIR(const std::shared_ptr<ir::IRGraph>& ir) {
	auto predicate = extractPredicate(*ir);
	return generateModuleFromPredicate(predicate.comparator, predicate.constantValue, predicate.columnIndex);
}

} // namespace nautilus::compiler::mlir
