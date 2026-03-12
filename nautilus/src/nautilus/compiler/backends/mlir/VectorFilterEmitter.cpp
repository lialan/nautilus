#include "nautilus/compiler/backends/mlir/VectorFilterEmitter.hpp"
#include "nautilus/compiler/ir/IRGraph.hpp"
#include "nautilus/compiler/ir/operations/CastOperation.hpp"
#include "nautilus/compiler/ir/operations/ConstIntOperation.hpp"
#include "nautilus/compiler/ir/operations/FunctionOperation.hpp"
#include "nautilus/compiler/ir/operations/LoadOperation.hpp"
#include "nautilus/compiler/ir/operations/LogicalOperations/CompareOperation.hpp"
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/LLVMIR/LLVMDialect.h>
#include <mlir/Dialect/LLVMIR/LLVMTypes.h>
#include <mlir/Dialect/SCF/IR/SCF.h>
#include <mlir/IR/Builders.h>
#include <stdexcept>

namespace nautilus::compiler::mlir {

VectorFilterEmitter::VectorFilterEmitter(::mlir::MLIRContext& context, const engine::Options& options)
    : context(context), options(options), typeSize(options.getOptionOrDefault("vectorFilter.typeSize", 4)) {
}

VectorFilterEmitter::~VectorFilterEmitter() = default;

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

	// --- Load column pointer: cols[columnIndex] → inttoptr ---
	auto colIdxCst = builder.create<::mlir::arith::ConstantOp>(loc, builder.getI64IntegerAttr(columnIndex));
	auto colGep = builder.create<::mlir::LLVM::GEPOp>(loc, ptrTy, i64Ty, cols, ::mlir::ValueRange {colIdxCst});
	auto colAddr = builder.create<::mlir::LLVM::LoadOp>(loc, i64Ty, colGep, /*alignment=*/8);
	auto colPtr = builder.create<::mlir::LLVM::IntToPtrOp>(loc, ptrTy, colAddr, ::mlir::LLVM::DereferenceableAttr());

	// --- Constants ---
	auto cst0 = builder.create<::mlir::arith::ConstantOp>(loc, builder.getI64IntegerAttr(0));
	auto cst1 = builder.create<::mlir::arith::ConstantOp>(loc, builder.getI64IntegerAttr(1));
	auto cstVecWidth = builder.create<::mlir::arith::ConstantOp>(loc, builder.getI64IntegerAttr(vectorWidth));

	// vec_limit = rowsCount - (rowsCount % vectorWidth)
	auto remainder = builder.create<::mlir::arith::RemSIOp>(loc, rowsCount, cstVecWidth);
	auto vecLimit = builder.create<::mlir::arith::SubIOp>(loc, rowsCount, remainder);

	// Splat constant for vector comparison
	::mlir::DenseElementsAttr splatAttr;
	switch (typeSize) {
	case 1:
		splatAttr = ::mlir::DenseElementsAttr::get(vecElemTy, static_cast<int8_t>(constantValue));
		break;
	case 2:
		splatAttr = ::mlir::DenseElementsAttr::get(vecElemTy, static_cast<int16_t>(constantValue));
		break;
	case 4:
		splatAttr = ::mlir::DenseElementsAttr::get(vecElemTy, static_cast<int32_t>(constantValue));
		break;
	case 8:
		splatAttr = ::mlir::DenseElementsAttr::get(vecElemTy, static_cast<int64_t>(constantValue));
		break;
	default:
		throw std::runtime_error("VectorFilterEmitter: unsupported typeSize for splat");
	}
	auto splatCst = builder.create<::mlir::arith::ConstantOp>(loc, splatAttr);

	auto mlirPredicate = comparatorToMLIR(comparator);

	// Number of 8-element groups for compress-store processing.
	// Each group processes 8 mask bits → 8 row IDs (i64).
	int numGroups = vectorWidth / 8;
	if (numGroups < 1) {
		numGroups = 1;
	}

	// ========================== Vectorized main loop ==========================
	auto vecLoop = builder.create<::mlir::scf::ForOp>(
	    loc, /*lb=*/cst0.getResult(), /*ub=*/vecLimit.getResult(), /*step=*/cstVecWidth.getResult(),
	    /*iterArgs=*/::mlir::ValueRange {cst0.getResult()},
	    [&](::mlir::OpBuilder& b, ::mlir::Location bodyLoc, ::mlir::Value iv, ::mlir::ValueRange iterArgs) {
		    ::mlir::Value outIdx = iterArgs[0];

		    // GEP to column[iv] and load vector
		    auto colGepInner = b.create<::mlir::LLVM::GEPOp>(bodyLoc, ptrTy, elemTy, colPtr, ::mlir::ValueRange {iv});
		    auto colVec = b.create<::mlir::LLVM::LoadOp>(bodyLoc, vecElemTy, colGepInner, /*alignment=*/typeSize);

		    // Compare: mask = (colVec <pred> splat<constant>)
		    auto maskFull = b.create<::mlir::arith::CmpIOp>(bodyLoc, mlirPredicate, colVec, splatCst);

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
	// Scalar constant for comparison (element-sized)
	auto scalarLoop = builder.create<::mlir::scf::ForOp>(
	    loc, /*lb=*/vecLimit.getResult(), /*ub=*/rowsCount, /*step=*/cst1.getResult(),
	    /*iterArgs=*/::mlir::ValueRange {vecOut},
	    [&](::mlir::OpBuilder& b, ::mlir::Location bodyLoc, ::mlir::Value jv, ::mlir::ValueRange iterArgs) {
		    ::mlir::Value tidx = iterArgs[0];

		    // Load single element from column[jv]
		    auto gep = b.create<::mlir::LLVM::GEPOp>(bodyLoc, ptrTy, elemTy, colPtr, ::mlir::ValueRange {jv});
		    auto val = b.create<::mlir::LLVM::LoadOp>(bodyLoc, elemTy, gep, /*alignment=*/typeSize);

		    // Create scalar constant matching the element type
		    ::mlir::Value scalarCst;
		    switch (typeSize) {
		    case 1:
			    scalarCst =
			        b.create<::mlir::arith::ConstantOp>(bodyLoc, b.getIntegerAttr(b.getI8Type(), constantValue));
			    break;
		    case 2:
			    scalarCst =
			        b.create<::mlir::arith::ConstantOp>(bodyLoc, b.getIntegerAttr(b.getI16Type(), constantValue));
			    break;
		    case 4:
			    scalarCst =
			        b.create<::mlir::arith::ConstantOp>(bodyLoc, b.getIntegerAttr(b.getI32Type(), constantValue));
			    break;
		    case 8:
			    scalarCst =
			        b.create<::mlir::arith::ConstantOp>(bodyLoc, b.getIntegerAttr(b.getI64Type(), constantValue));
			    break;
		    }

		    // Compare
		    auto match = b.create<::mlir::arith::CmpIOp>(bodyLoc, mlirPredicate, val, scalarCst);

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
