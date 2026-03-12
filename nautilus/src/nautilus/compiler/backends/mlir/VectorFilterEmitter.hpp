#pragma once
#include "nautilus/compiler/ir/IRGraph.hpp"
#include "nautilus/compiler/ir/operations/LogicalOperations/CompareOperation.hpp"
#include "nautilus/options.hpp"
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/MLIRContext.h>

namespace nautilus::compiler::mlir {

/// Extracted predicate from the IR graph: a single column comparison against a constant.
struct ExtractedPredicate {
	ir::CompareOperation::Comparator comparator;
	int64_t constantValue;
	int columnIndex; // which column (from cols[]) is being compared
};

class VectorFilterEmitter {
public:
	explicit VectorFilterEmitter(::mlir::MLIRContext& context, const engine::Options& options);
	~VectorFilterEmitter();

	::mlir::OwningOpRef<::mlir::ModuleOp> generateModuleFromIR(const std::shared_ptr<ir::IRGraph>& ir);

	/// Build a vectorized filter module directly from predicate parameters (for testing).
	::mlir::OwningOpRef<::mlir::ModuleOp> generateModuleFromPredicate(ir::CompareOperation::Comparator comparator,
	                                                                  int64_t constantValue, int columnIndex = 0);

	/// Extract the predicate from the IR graph.
	static ExtractedPredicate extractPredicate(const ir::IRGraph& ir);

private:
	::mlir::MLIRContext& context;
	const engine::Options& options;
	int typeSize; // from vectorFilter.typeSize option
};

} // namespace nautilus::compiler::mlir
