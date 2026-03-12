#include "nautilus/compiler/backends/mlir/VectorFilterEmitter.hpp"
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/LLVMIR/LLVMDialect.h>
#include <mlir/Dialect/LLVMIR/LLVMTypes.h>
#include <mlir/IR/Builders.h>

namespace nautilus::compiler::mlir {

VectorFilterEmitter::VectorFilterEmitter(::mlir::MLIRContext& context, const engine::Options& options)
    : context(context), options(options), typeSize(options.getOptionOrDefault("vectorFilter.typeSize", 4)) {
}

VectorFilterEmitter::~VectorFilterEmitter() = default;

::mlir::OwningOpRef<::mlir::ModuleOp> VectorFilterEmitter::generateModuleFromIR(
    const std::shared_ptr<ir::IRGraph>& /*ir*/) {

	context.loadAllAvailableDialects();

	::mlir::OpBuilder builder(&context);
	auto loc = builder.getUnknownLoc();

	auto i64Ty = builder.getI64Type();
	auto ptrTy = ::mlir::LLVM::LLVMPointerType::get(&context);

	auto module = ::mlir::ModuleOp::create(loc);
	builder.setInsertionPointToEnd(module.getBody());

	// Function type: (ptr, i64, ptr, ptr, i64, ptr, i64, i64) -> i64
	// Args: cols, colsCount, varsizeIndexes, vars, varsCount, rows, rowsCount, rowsStartOffset
	auto funcTy = ::mlir::LLVM::LLVMFunctionType::get(i64Ty, {ptrTy, i64Ty, ptrTy, ptrTy, i64Ty, ptrTy, i64Ty, i64Ty});
	auto funcOp = builder.create<::mlir::LLVM::LLVMFuncOp>(loc, "execute", funcTy, ::mlir::LLVM::Linkage::External,
	                                                        false);

	auto* entryBlock = funcOp.addEntryBlock(builder);
	builder.setInsertionPointToStart(entryBlock);

	// Return 0 (skeleton)
	auto zero = builder.create<::mlir::arith::ConstantOp>(loc, builder.getI64IntegerAttr(0));
	builder.create<::mlir::LLVM::ReturnOp>(loc, ::mlir::ValueRange{zero});

	return module;
}

} // namespace nautilus::compiler::mlir
