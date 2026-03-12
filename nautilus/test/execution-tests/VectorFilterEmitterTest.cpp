// VectorFilterEmitterTest.cpp — Tests for VectorFilterEmitter: vectorized equality filter (i32)
//
// Tests the generated MLIR module that implements column_value == constant with
// AVX-512 compress-store output, by calling VectorFilterEmitter::generateModuleFromPredicate()
// which bypasses IR walking and directly emits the vectorized MLIR.

#include "nautilus/compiler/DumpHandler.hpp"
#include "nautilus/compiler/backends/mlir/JITCompiler.hpp"
#include "nautilus/compiler/backends/mlir/LLVMIROptimizer.hpp"
#include "nautilus/compiler/backends/mlir/MLIRPassManager.hpp"
#include "nautilus/compiler/backends/mlir/VectorFilterEmitter.hpp"
#include "nautilus/compiler/ir/operations/LogicalOperations/CompareOperation.hpp"
#include "nautilus/options.hpp"
#include <catch2/catch_all.hpp>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <llvm/Support/TargetSelect.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/ControlFlow/IR/ControlFlow.h>
#include <mlir/Dialect/Func/Extensions/AllExtensions.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/LLVMIR/LLVMDialect.h>
#include <mlir/Dialect/LLVMIR/LLVMTypes.h>
#include <mlir/Dialect/LLVMIR/Transforms/InlinerInterfaceImpl.h>
#include <mlir/Dialect/Math/IR/Math.h>
#include <mlir/Dialect/SCF/IR/SCF.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/Verifier.h>
#include <mlir/Target/LLVMIR/Dialect/Builtin/BuiltinToLLVMIRTranslation.h>
#include <mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h>
#include <vector>

namespace nmlir = nautilus::compiler::mlir;
namespace ir = nautilus::compiler::ir;

// ---------------------------------------------------------------------------
// Function pointer type matching the NautilusFilterCallable signature.
// ---------------------------------------------------------------------------
using ExecuteFn = int64_t (*)(int64_t*, int64_t, int64_t*, int64_t*, int64_t, int64_t*, int64_t, int64_t);

struct CompiledFilter {
	std::unique_ptr<::mlir::ExecutionEngine> engine;
	ExecuteFn fn;
};

// ---------------------------------------------------------------------------
// Helper: compile the MLIR module through the full Nautilus pipeline.
// ---------------------------------------------------------------------------
static CompiledFilter compileModule(::mlir::OwningOpRef<::mlir::ModuleOp>& module) {
	if (::mlir::failed(::mlir::verify(*module))) {
		module->dump();
		FAIL("MLIR module verification failed");
	}

	if (nmlir::MLIRPassManager::lowerAndOptimizeMLIRModule(module, {})) {
		module->dump();
		FAIL("MLIRPassManager::lowerAndOptimizeMLIRModule failed");
	}

	nautilus::engine::Options options;
	options.setOption("engine.backend", std::string("mlir"));
	nautilus::compiler::CompilationUnitID unitId = "VectorFilterEmitterTest";
	nautilus::compiler::DumpHandler dumpHandler(options, unitId);
	auto optPipeline = nmlir::LLVMIROptimizer::getLLVMOptimizerPipeline(options, dumpHandler);

	std::vector<std::string> proxySymbols;
	std::vector<void*> proxyAddresses;
	auto engine = nmlir::JITCompiler::jitCompileModule(module, optPipeline, proxySymbols, proxyAddresses, options);
	REQUIRE(engine != nullptr);

	auto fnExpected = engine->lookup("execute");
	REQUIRE(!!fnExpected);
	auto* rawPtr = fnExpected.get();
	REQUIRE(rawPtr != nullptr);
	auto fn = reinterpret_cast<ExecuteFn>(rawPtr);

	return {std::move(engine), fn};
}

// ---------------------------------------------------------------------------
// Helper: build and compile a VectorFilterEmitter module for an i32 filter.
// ---------------------------------------------------------------------------
static CompiledFilter buildAndCompileFilter(ir::CompareOperation::Comparator comparator, int64_t constantValue,
                                            int typeSize = 4) {
	::mlir::DialectRegistry registry;
	registry.insert<::mlir::arith::ArithDialect, ::mlir::cf::ControlFlowDialect, ::mlir::math::MathDialect,
	                ::mlir::LLVM::LLVMDialect, ::mlir::func::FuncDialect>();
	registry.insert<::mlir::scf::SCFDialect>();
	::mlir::func::registerAllExtensions(registry);
	::mlir::registerBuiltinDialectTranslation(registry);
	::mlir::registerLLVMDialectTranslation(registry);
	::mlir::LLVM::registerInlinerInterface(registry);

	::mlir::MLIRContext context(registry);
	context.disableMultithreading();
	context.loadAllAvailableDialects();

	nautilus::engine::Options options;
	options.setOption("vectorFilter.enabled", true);
	options.setOption("vectorFilter.typeSize", typeSize);
	nmlir::VectorFilterEmitter emitter(context, options);

	auto module = emitter.generateModuleFromPredicate(comparator, constantValue, /*columnIndex=*/0);
	return compileModule(module);
}

static CompiledFilter buildAndCompileEqFilter(int64_t constantValue, int typeSize = 4) {
	return buildAndCompileFilter(ir::CompareOperation::EQ, constantValue, typeSize);
}

// ---------------------------------------------------------------------------
// Helper: invoke the compiled filter on a test column.
// ---------------------------------------------------------------------------
static int64_t runFilter(const CompiledFilter& compiled, int32_t* columnData, int64_t numRows, int64_t* rowsBuf,
                         int64_t rowsStartOffset = 0) {
	int64_t colAddr = reinterpret_cast<int64_t>(columnData);
	int64_t cols[1] = {colAddr};
	return compiled.fn(cols, 1, nullptr, nullptr, 0, rowsBuf, numRows, rowsStartOffset);
}

// ===========================================================================
// Test cases
// ===========================================================================

TEST_CASE("VectorFilterEmitter: basic eq filter (4096 rows, 3 matches)", "[vector-filter-emitter]") {
	llvm::InitializeNativeTarget();
	llvm::InitializeNativeTargetAsmPrinter();

	auto compiled = buildAndCompileEqFilter(42);

	constexpr int64_t NUM_ROWS = 4096;
	std::vector<int32_t> column(NUM_ROWS, 0);
	column[42] = 42;
	column[100] = 42;
	column[4000] = 42;

	std::vector<int64_t> rowsBuf(NUM_ROWS, -1);
	int64_t matchCount = runFilter(compiled, column.data(), NUM_ROWS, rowsBuf.data());

	CHECK(matchCount == 3);
	CHECK(rowsBuf[0] == 42);
	CHECK(rowsBuf[1] == 100);
	CHECK(rowsBuf[2] == 4000);
}

TEST_CASE("VectorFilterEmitter: eq filter with rowsStartOffset", "[vector-filter-emitter]") {
	llvm::InitializeNativeTarget();
	llvm::InitializeNativeTargetAsmPrinter();

	auto compiled = buildAndCompileEqFilter(42);

	constexpr int64_t NUM_ROWS = 64;
	std::vector<int32_t> column(NUM_ROWS, 0);
	column[10] = 42;
	column[42] = 42;

	constexpr int64_t OFFSET = 5000;
	std::vector<int64_t> rowsBuf(NUM_ROWS, -1);
	int64_t matchCount = runFilter(compiled, column.data(), NUM_ROWS, rowsBuf.data(), OFFSET);

	CHECK(matchCount == 2);
	CHECK(rowsBuf[0] == 10 + OFFSET);
	CHECK(rowsBuf[1] == 42 + OFFSET);
}

TEST_CASE("VectorFilterEmitter: edge cases", "[vector-filter-emitter]") {
	llvm::InitializeNativeTarget();
	llvm::InitializeNativeTargetAsmPrinter();

	SECTION("row_count = 0") {
		auto compiled = buildAndCompileEqFilter(42);

		std::vector<int32_t> column(1, 42); // at least 1 element so pointer is valid
		std::vector<int64_t> rowsBuf(1, -1);
		int64_t matchCount = runFilter(compiled, column.data(), 0, rowsBuf.data());

		CHECK(matchCount == 0);
	}

	SECTION("row_count < vector_width (tail only, 5 rows)") {
		auto compiled = buildAndCompileEqFilter(42);

		constexpr int64_t NUM_ROWS = 5;
		std::vector<int32_t> column(NUM_ROWS, 0);
		column[0] = 42;
		column[4] = 42;

		std::vector<int64_t> rowsBuf(NUM_ROWS, -1);
		int64_t matchCount = runFilter(compiled, column.data(), NUM_ROWS, rowsBuf.data());

		CHECK(matchCount == 2);
		CHECK(rowsBuf[0] == 0);
		CHECK(rowsBuf[1] == 4);
	}

	SECTION("row_count = exact multiple of vector width (no tail)") {
		auto compiled = buildAndCompileEqFilter(42);

		// For i32 with AVX-512: vectorWidth = 16
		constexpr int64_t NUM_ROWS = 32;
		std::vector<int32_t> column(NUM_ROWS, 0);
		column[15] = 42;
		column[16] = 42;
		column[31] = 42;

		std::vector<int64_t> rowsBuf(NUM_ROWS, -1);
		int64_t matchCount = runFilter(compiled, column.data(), NUM_ROWS, rowsBuf.data());

		CHECK(matchCount == 3);
		CHECK(rowsBuf[0] == 15);
		CHECK(rowsBuf[1] == 16);
		CHECK(rowsBuf[2] == 31);
	}

	SECTION("all match") {
		auto compiled = buildAndCompileEqFilter(42);

		constexpr int64_t NUM_ROWS = 50;
		std::vector<int32_t> column(NUM_ROWS, 42); // every element equals 42

		std::vector<int64_t> rowsBuf(NUM_ROWS, -1);
		int64_t matchCount = runFilter(compiled, column.data(), NUM_ROWS, rowsBuf.data());

		CHECK(matchCount == NUM_ROWS);
		for (int64_t i = 0; i < NUM_ROWS; i++) {
			CHECK(rowsBuf[i] == i);
		}
	}

	SECTION("no match") {
		auto compiled = buildAndCompileEqFilter(42);

		constexpr int64_t NUM_ROWS = 100;
		std::vector<int32_t> column(NUM_ROWS, 0);

		std::vector<int64_t> rowsBuf(NUM_ROWS, -1);
		int64_t matchCount = runFilter(compiled, column.data(), NUM_ROWS, rowsBuf.data());

		CHECK(matchCount == 0);
	}

	SECTION("misaligned pointer (offset by 4 bytes, no crash)") {
		auto compiled = buildAndCompileEqFilter(42);

		constexpr int64_t NUM_ROWS = 64;
		constexpr size_t PAYLOAD = NUM_ROWS * sizeof(int32_t);
		constexpr size_t ALLOC_SIZE = PAYLOAD + 128;

		auto* rawBuf = static_cast<uint8_t*>(std::aligned_alloc(64, ALLOC_SIZE));
		REQUIRE(rawBuf != nullptr);
		std::memset(rawBuf, 0, ALLOC_SIZE);

		// Offset by 4 bytes -> misaligned for 64-byte vector loads
		auto* colData = reinterpret_cast<int32_t*>(rawBuf + 4);
		colData[42] = 42;

		std::vector<int64_t> rowsBuf(NUM_ROWS, -1);
		int64_t matchCount = runFilter(compiled, colData, NUM_ROWS, rowsBuf.data());

		CHECK(matchCount == 1);
		CHECK(rowsBuf[0] == 42);

		std::free(rawBuf);
	}
}

TEST_CASE("VectorFilterEmitter: scalar tail correctness", "[vector-filter-emitter]") {
	llvm::InitializeNativeTarget();
	llvm::InitializeNativeTargetAsmPrinter();

	auto compiled = buildAndCompileEqFilter(42);

	// 50 rows: 48 handled by vector loop (3 iterations of 16), 2 by scalar tail
	constexpr int64_t NUM_ROWS = 50;
	std::vector<int32_t> column(NUM_ROWS, 0);
	column[49] = 42; // match is in the scalar tail

	std::vector<int64_t> rowsBuf(NUM_ROWS, -1);
	int64_t matchCount = runFilter(compiled, column.data(), NUM_ROWS, rowsBuf.data());

	CHECK(matchCount == 1);
	CHECK(rowsBuf[0] == 49);
}

TEST_CASE("VectorFilterEmitter: matches spanning vector and scalar regions", "[vector-filter-emitter]") {
	llvm::InitializeNativeTarget();
	llvm::InitializeNativeTargetAsmPrinter();

	auto compiled = buildAndCompileEqFilter(42);

	// 35 rows: 32 by vector (2 iters of 16), 3 by scalar tail
	constexpr int64_t NUM_ROWS = 35;
	std::vector<int32_t> column(NUM_ROWS, 0);
	column[0] = 42;  // vector region
	column[15] = 42; // vector region (end of first vector iteration)
	column[16] = 42; // vector region (start of second vector iteration)
	column[31] = 42; // vector region (end of second vector iteration)
	column[32] = 42; // scalar tail
	column[34] = 42; // scalar tail (last element)

	std::vector<int64_t> rowsBuf(NUM_ROWS, -1);
	int64_t matchCount = runFilter(compiled, column.data(), NUM_ROWS, rowsBuf.data());

	CHECK(matchCount == 6);
	CHECK(rowsBuf[0] == 0);
	CHECK(rowsBuf[1] == 15);
	CHECK(rowsBuf[2] == 16);
	CHECK(rowsBuf[3] == 31);
	CHECK(rowsBuf[4] == 32);
	CHECK(rowsBuf[5] == 34);
}

TEST_CASE("VectorFilterEmitter: comparison operators i32", "[vector-filter-emitter]") {
	llvm::InitializeNativeTarget();
	llvm::InitializeNativeTargetAsmPrinter();

	// Array: [0, 1, 2, ..., 255] (256 elements)
	constexpr int64_t NUM_ROWS = 256;
	std::vector<int32_t> column(NUM_ROWS);
	for (int32_t i = 0; i < NUM_ROWS; i++) {
		column[i] = i;
	}

	SECTION("gt: col > 200") {
		// expect 55 matches: 201..255
		auto compiled = buildAndCompileFilter(ir::CompareOperation::GT, 200);
		std::vector<int64_t> rowsBuf(NUM_ROWS, -1);
		int64_t matchCount = runFilter(compiled, column.data(), NUM_ROWS, rowsBuf.data());
		CHECK(matchCount == 55);
		for (int64_t i = 0; i < matchCount; i++) {
			CHECK(rowsBuf[i] == 201 + i);
		}
	}

	SECTION("lt: col < 10") {
		// expect 10 matches: 0..9
		auto compiled = buildAndCompileFilter(ir::CompareOperation::LT, 10);
		std::vector<int64_t> rowsBuf(NUM_ROWS, -1);
		int64_t matchCount = runFilter(compiled, column.data(), NUM_ROWS, rowsBuf.data());
		CHECK(matchCount == 10);
		for (int64_t i = 0; i < matchCount; i++) {
			CHECK(rowsBuf[i] == i);
		}
	}

	SECTION("ge: col >= 200") {
		// expect 56 matches: 200..255
		auto compiled = buildAndCompileFilter(ir::CompareOperation::GE, 200);
		std::vector<int64_t> rowsBuf(NUM_ROWS, -1);
		int64_t matchCount = runFilter(compiled, column.data(), NUM_ROWS, rowsBuf.data());
		CHECK(matchCount == 56);
		for (int64_t i = 0; i < matchCount; i++) {
			CHECK(rowsBuf[i] == 200 + i);
		}
	}

	SECTION("le: col <= 10") {
		// expect 11 matches: 0..10
		auto compiled = buildAndCompileFilter(ir::CompareOperation::LE, 10);
		std::vector<int64_t> rowsBuf(NUM_ROWS, -1);
		int64_t matchCount = runFilter(compiled, column.data(), NUM_ROWS, rowsBuf.data());
		CHECK(matchCount == 11);
		for (int64_t i = 0; i < matchCount; i++) {
			CHECK(rowsBuf[i] == i);
		}
	}

	SECTION("ne: col != 42") {
		// expect 255 matches: everything except index 42
		auto compiled = buildAndCompileFilter(ir::CompareOperation::NE, 42);
		std::vector<int64_t> rowsBuf(NUM_ROWS, -1);
		int64_t matchCount = runFilter(compiled, column.data(), NUM_ROWS, rowsBuf.data());
		CHECK(matchCount == 255);
		int64_t expected = 0;
		for (int64_t i = 0; i < matchCount; i++) {
			if (expected == 42)
				expected++;
			CHECK(rowsBuf[i] == expected);
			expected++;
		}
	}
}

// ---------------------------------------------------------------------------
// Helper: build and compile a VectorFilterEmitter module from a predicate tree.
// ---------------------------------------------------------------------------
static CompiledFilter buildAndCompilePredicateTree(const nmlir::PredicateNode& root, int typeSize = 4) {
	::mlir::DialectRegistry registry;
	registry.insert<::mlir::arith::ArithDialect, ::mlir::cf::ControlFlowDialect, ::mlir::math::MathDialect,
	                ::mlir::LLVM::LLVMDialect, ::mlir::func::FuncDialect>();
	registry.insert<::mlir::scf::SCFDialect>();
	::mlir::func::registerAllExtensions(registry);
	::mlir::registerBuiltinDialectTranslation(registry);
	::mlir::registerLLVMDialectTranslation(registry);
	::mlir::LLVM::registerInlinerInterface(registry);

	::mlir::MLIRContext context(registry);
	context.disableMultithreading();
	context.loadAllAvailableDialects();

	nautilus::engine::Options options;
	options.setOption("vectorFilter.enabled", true);
	options.setOption("vectorFilter.typeSize", typeSize);
	nmlir::VectorFilterEmitter emitter(context, options);

	auto module = emitter.generateModuleFromPredicateTree(root);
	return compileModule(module);
}

TEST_CASE("VectorFilterEmitter: compound predicates", "[vector-filter-emitter]") {
	llvm::InitializeNativeTarget();
	llvm::InitializeNativeTargetAsmPrinter();

	// Array: [0, 1, 2, ..., 255]
	constexpr int64_t NUM_ROWS = 256;
	std::vector<int32_t> column(NUM_ROWS);
	for (int32_t i = 0; i < NUM_ROWS; i++) {
		column[i] = i;
	}

	SECTION("col > 10 AND col < 20") {
		// expect 9 matches: 11..19
		auto andNode = std::make_unique<nmlir::AndNode>();
		andNode->left = nmlir::CompareNode {ir::CompareOperation::GT, 10, 0};
		andNode->right = nmlir::CompareNode {ir::CompareOperation::LT, 20, 0};
		nmlir::PredicateNode root = std::move(andNode);

		auto compiled = buildAndCompilePredicateTree(root);

		std::vector<int64_t> rowsBuf(NUM_ROWS, -1);
		int64_t colAddr = reinterpret_cast<int64_t>(column.data());
		int64_t cols[1] = {colAddr};
		int64_t matchCount = compiled.fn(cols, 1, nullptr, nullptr, 0, rowsBuf.data(), NUM_ROWS, 0);

		CHECK(matchCount == 9);
		for (int64_t i = 0; i < matchCount; i++) {
			CHECK(rowsBuf[i] == 11 + i);
		}
	}

	SECTION("col < 5 OR col > 250") {
		// expect 10 matches: 0..4, 251..255
		auto orNode = std::make_unique<nmlir::OrNode>();
		orNode->left = nmlir::CompareNode {ir::CompareOperation::LT, 5, 0};
		orNode->right = nmlir::CompareNode {ir::CompareOperation::GT, 250, 0};
		nmlir::PredicateNode root = std::move(orNode);

		auto compiled = buildAndCompilePredicateTree(root);

		std::vector<int64_t> rowsBuf(NUM_ROWS, -1);
		int64_t colAddr = reinterpret_cast<int64_t>(column.data());
		int64_t cols[1] = {colAddr};
		int64_t matchCount = compiled.fn(cols, 1, nullptr, nullptr, 0, rowsBuf.data(), NUM_ROWS, 0);

		CHECK(matchCount == 10);
		// First 5: indices 0..4
		for (int64_t i = 0; i < 5; i++) {
			CHECK(rowsBuf[i] == i);
		}
		// Last 5: indices 251..255
		for (int64_t i = 0; i < 5; i++) {
			CHECK(rowsBuf[5 + i] == 251 + i);
		}
	}

	SECTION("NOT (col == 42)") {
		// expect 255 matches
		auto notNode = std::make_unique<nmlir::NotNode>();
		notNode->child = nmlir::CompareNode {ir::CompareOperation::EQ, 42, 0};
		nmlir::PredicateNode root = std::move(notNode);

		auto compiled = buildAndCompilePredicateTree(root);

		std::vector<int64_t> rowsBuf(NUM_ROWS, -1);
		int64_t colAddr = reinterpret_cast<int64_t>(column.data());
		int64_t cols[1] = {colAddr};
		int64_t matchCount = compiled.fn(cols, 1, nullptr, nullptr, 0, rowsBuf.data(), NUM_ROWS, 0);

		CHECK(matchCount == 255);
		int64_t expected = 0;
		for (int64_t i = 0; i < matchCount; i++) {
			if (expected == 42)
				expected++;
			CHECK(rowsBuf[i] == expected);
			expected++;
		}
	}

	SECTION("col1 > 10 AND col2 < 100 (multi-column)") {
		// Setup two columns, same type i32
		// col1: [0, 1, 2, ..., 255]   (same as column above)
		// col2: [255, 254, ..., 0]     (reversed)
		// col1 > 10 means indices 11..255  (245 rows)
		// col2 < 100 means col2[i] < 100, i.e. 255-i < 100, i.e. i > 155, so indices 156..255 (100 rows)
		// AND: indices that satisfy both = 156..255 (100 rows)
		std::vector<int32_t> col2(NUM_ROWS);
		for (int32_t i = 0; i < NUM_ROWS; i++) {
			col2[i] = static_cast<int32_t>(NUM_ROWS - 1 - i);
		}

		auto andNode = std::make_unique<nmlir::AndNode>();
		andNode->left = nmlir::CompareNode {ir::CompareOperation::GT, 10, 0};
		andNode->right = nmlir::CompareNode {ir::CompareOperation::LT, 100, 1};
		nmlir::PredicateNode root = std::move(andNode);

		auto compiled = buildAndCompilePredicateTree(root);

		std::vector<int64_t> rowsBuf(NUM_ROWS, -1);
		int64_t cols[2] = {reinterpret_cast<int64_t>(column.data()), reinterpret_cast<int64_t>(col2.data())};
		int64_t matchCount = compiled.fn(cols, 2, nullptr, nullptr, 0, rowsBuf.data(), NUM_ROWS, 0);

		CHECK(matchCount == 100);
		for (int64_t i = 0; i < matchCount; i++) {
			CHECK(rowsBuf[i] == 156 + i);
		}
	}
}
