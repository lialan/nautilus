#pragma once
#include "nautilus/compiler/ir/IRGraph.hpp"
#include "nautilus/compiler/ir/operations/LogicalOperations/CompareOperation.hpp"
#include "nautilus/options.hpp"
#include <memory>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/MLIRContext.h>
#include <variant>

namespace nautilus::compiler::mlir {

/// Extracted predicate from the IR graph: a single column comparison against a constant.
struct ExtractedPredicate {
	ir::CompareOperation::Comparator comparator;
	int64_t constantValue;
	int columnIndex; // which column (from cols[]) is being compared
};

// --- Recursive predicate tree for compound predicates ---

struct CompareNode {
	ir::CompareOperation::Comparator comparator;
	int64_t constantValue;
	int columnIndex;
	int varSlotIndex = -1;
	bool isRuntimeVar = false;
};

struct AndNode;
struct OrNode;
struct NotNode;

using PredicateNode =
    std::variant<CompareNode, std::unique_ptr<AndNode>, std::unique_ptr<OrNode>, std::unique_ptr<NotNode>>;

struct AndNode {
	PredicateNode left;
	PredicateNode right;
};

struct OrNode {
	PredicateNode left;
	PredicateNode right;
};

struct NotNode {
	PredicateNode child;
};

class VectorFilterEmitter {
public:
	explicit VectorFilterEmitter(::mlir::MLIRContext& context, const engine::Options& options);
	~VectorFilterEmitter();

	::mlir::OwningOpRef<::mlir::ModuleOp> generateModuleFromIR(const std::shared_ptr<ir::IRGraph>& ir);

	/// Build a vectorized filter module directly from predicate parameters (for testing).
	::mlir::OwningOpRef<::mlir::ModuleOp> generateModuleFromPredicate(ir::CompareOperation::Comparator comparator,
	                                                                  int64_t constantValue, int columnIndex = 0);

	/// Build a vectorized filter module from a compound predicate tree.
	::mlir::OwningOpRef<::mlir::ModuleOp> generateModuleFromPredicateTree(const PredicateNode& root);

	/// Extract the predicate from the IR graph.
	static ExtractedPredicate extractPredicate(const ir::IRGraph& ir);

private:
	::mlir::MLIRContext& context;
	const engine::Options& options;
	int typeSize; // from vectorFilter.typeSize option
};

/// Collect all unique column indices referenced by a predicate tree.
void collectColumnIndices(const PredicateNode& node, std::vector<int>& indices);

} // namespace nautilus::compiler::mlir
