#pragma once
#include "nautilus/compiler/ir/IRGraph.hpp"
#include "nautilus/options.hpp"
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/MLIRContext.h>

namespace nautilus::compiler::mlir {

class VectorFilterEmitter {
public:
	explicit VectorFilterEmitter(::mlir::MLIRContext& context, const engine::Options& options);
	~VectorFilterEmitter();

	::mlir::OwningOpRef<::mlir::ModuleOp> generateModuleFromIR(const std::shared_ptr<ir::IRGraph>& ir);

private:
	::mlir::MLIRContext& context;
	const engine::Options& options;
	int typeSize; // from vectorFilter.typeSize option
};

} // namespace nautilus::compiler::mlir
