// VectorFilterSpikeTest.cpp — go/no-go spike for inttoptr + vector ops through Nautilus MLIR pipeline
//
// Hand-builds an MLIR module with:
//   llvm.inttoptr, vector<16xi32> load, arith.cmpi, shufflevector mask split,
//   llvm.intr.masked.compressstore, llvm.intr.ctpop
// and compiles it through the full MLIRPassManager → LLVMIROptimizer → JITCompiler path.

#include <catch2/catch_all.hpp>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <vector>

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

#include "nautilus/compiler/backends/mlir/JITCompiler.hpp"
#include "nautilus/compiler/backends/mlir/LLVMIROptimizer.hpp"
#include "nautilus/compiler/backends/mlir/MLIRPassManager.hpp"
#include "nautilus/compiler/DumpHandler.hpp"
#include "nautilus/options.hpp"

namespace nmlir = nautilus::compiler::mlir;

// ---------------------------------------------------------------------------
// Helper: build the hand-crafted MLIR module that implements a vectorized
// equality filter (column_value == 42) with compress-store output.
//
// Signature:
//   int64_t execute(int64_t* cols, int64_t colsCount,
//                   int64_t* varsizeIndexes, int64_t* vars, int64_t varsCount,
//                   int64_t* rows, int64_t rowsCount, int64_t rowsStartOffset)
// ---------------------------------------------------------------------------
static ::mlir::OwningOpRef<::mlir::ModuleOp> buildVectorFilterModule(::mlir::MLIRContext& ctx) {
	::mlir::OpBuilder builder(&ctx);
	auto loc = builder.getUnknownLoc();

	// --- Types ---
	auto i8Ty = builder.getI8Type();
	auto i32Ty = builder.getI32Type();
	auto i64Ty = builder.getI64Type();
	auto ptrTy = ::mlir::LLVM::LLVMPointerType::get(&ctx);
	auto vec16xi32Ty = ::mlir::VectorType::get({16}, i32Ty);
	auto vec8xi64Ty = ::mlir::VectorType::get({8}, i64Ty);

	// --- Module ---
	auto module = ::mlir::ModuleOp::create(loc);
	builder.setInsertionPointToEnd(module.getBody());

	// --- Function type: (ptr,i64,ptr,ptr,i64,ptr,i64,i64) -> i64 ---
	auto funcTy =
	    ::mlir::LLVM::LLVMFunctionType::get(i64Ty, {ptrTy, i64Ty, ptrTy, ptrTy, i64Ty, ptrTy, i64Ty, i64Ty});
	auto funcOp =
	    builder.create<::mlir::LLVM::LLVMFuncOp>(loc, "execute", funcTy, ::mlir::LLVM::Linkage::External, false);

	// Entry block with 8 arguments matching the function signature.
	auto* entryBlock = funcOp.addEntryBlock(builder);
	builder.setInsertionPointToStart(entryBlock);

	::mlir::Value cols = funcOp.getArgument(0);            // %cols        : !llvm.ptr
	/* colsCount = funcOp.getArgument(1) — unused */
	/* varsizeIndexes = funcOp.getArgument(2) — unused */
	/* vars = funcOp.getArgument(3) — unused */
	/* varsCount = funcOp.getArgument(4) — unused */
	::mlir::Value rows = funcOp.getArgument(5);            // %rows        : !llvm.ptr
	::mlir::Value rowsCount = funcOp.getArgument(6);       // %rowsCount   : i64
	::mlir::Value rowsStartOffset = funcOp.getArgument(7); // %rowsStartOffset : i64

	// ---- Load column 0 address from cols[0] ----
	auto colAddr = builder.create<::mlir::LLVM::LoadOp>(loc, i64Ty, cols, /*alignment=*/8);
	auto colPtr = builder.create<::mlir::LLVM::IntToPtrOp>(loc, ptrTy, colAddr,
	                                                        ::mlir::LLVM::DereferenceableAttr());

	// ---- Constants ----
	auto cst0 = builder.create<::mlir::arith::ConstantOp>(loc, builder.getI64IntegerAttr(0));
	auto cst1 = builder.create<::mlir::arith::ConstantOp>(loc, builder.getI64IntegerAttr(1));
	auto cst16 = builder.create<::mlir::arith::ConstantOp>(loc, builder.getI64IntegerAttr(16));

	// vec_limit = rowsCount - (rowsCount % 16)
	auto remainder = builder.create<::mlir::arith::RemSIOp>(loc, rowsCount, cst16);
	auto vecLimit = builder.create<::mlir::arith::SubIOp>(loc, rowsCount, remainder);

	// Splat constant 42 : vector<16xi32>
	auto splatAttr = ::mlir::DenseElementsAttr::get(vec16xi32Ty, static_cast<int32_t>(42));
	auto splat42 = builder.create<::mlir::arith::ConstantOp>(loc, splatAttr);

	// ========================== Vectorized main loop ==========================
	// scf.for %i = 0 to vecLimit step 16 iter_args(%idx = 0) -> i64
	auto vecLoop = builder.create<::mlir::scf::ForOp>(
	    loc, /*lb=*/cst0.getResult(), /*ub=*/vecLimit.getResult(), /*step=*/cst16.getResult(),
	    /*iterArgs=*/::mlir::ValueRange{cst0.getResult()},
	    /*bodyBuilder=*/
	    [&](::mlir::OpBuilder& b, ::mlir::Location bodyLoc, ::mlir::Value iv, ::mlir::ValueRange iterArgs) {
		    ::mlir::Value outIdx = iterArgs[0];

		    // GEP to column[i] and load vector<16xi32>
		    auto colGep =
		        b.create<::mlir::LLVM::GEPOp>(bodyLoc, ptrTy, i32Ty, colPtr, ::mlir::ValueRange{iv});
		    auto colVec = b.create<::mlir::LLVM::LoadOp>(bodyLoc, vec16xi32Ty, colGep, /*alignment=*/4);

		    // Compare: mask16 = (colVec == splat<42>)
		    auto mask16 =
		        b.create<::mlir::arith::CmpIOp>(bodyLoc, ::mlir::arith::CmpIPredicate::eq, colVec, splat42);

		    // Split mask into lo/hi halves of 8
		    auto maskLo = b.create<::mlir::LLVM::ShuffleVectorOp>(
		        bodyLoc, mask16, mask16, llvm::ArrayRef<int32_t>({0, 1, 2, 3, 4, 5, 6, 7}));
		    auto maskHi = b.create<::mlir::LLVM::ShuffleVectorOp>(
		        bodyLoc, mask16, mask16, llvm::ArrayRef<int32_t>({8, 9, 10, 11, 12, 13, 14, 15}));

		    // Build row-ID base = iv + rowsStartOffset
		    auto rowBase = b.create<::mlir::arith::AddIOp>(bodyLoc, iv, rowsStartOffset);

		    // Build row-ID vector: base, base+1, ..., base+7 via undef + 8 insertelements
		    auto buildRowIdVec = [&](::mlir::Value base) -> ::mlir::Value {
			    ::mlir::Value vec = b.create<::mlir::LLVM::UndefOp>(bodyLoc, vec8xi64Ty);
			    for (int k = 0; k < 8; k++) {
				    auto offset = b.create<::mlir::arith::ConstantOp>(bodyLoc, b.getI64IntegerAttr(k));
				    auto elem = b.create<::mlir::arith::AddIOp>(bodyLoc, base, offset);
				    auto idxI32 = b.create<::mlir::arith::ConstantOp>(bodyLoc, b.getI32IntegerAttr(k));
				    vec = b.create<::mlir::LLVM::InsertElementOp>(bodyLoc, vec, elem, idxI32);
			    }
			    return vec;
		    };

		    // Build lo/hi row-ID vectors
		    auto ridsLo = buildRowIdVec(rowBase);

		    // For hi: base + 8
		    auto cst8 = b.create<::mlir::arith::ConstantOp>(bodyLoc, b.getI64IntegerAttr(8));
		    auto hiBase = b.create<::mlir::arith::AddIOp>(bodyLoc, rowBase, cst8);
		    auto ridsHi = buildRowIdVec(hiBase);

		    // Compress-store lower half
		    auto outGep1 =
		        b.create<::mlir::LLVM::GEPOp>(bodyLoc, ptrTy, i64Ty, rows, ::mlir::ValueRange{outIdx});
		    b.create<::mlir::LLVM::masked_compressstore>(bodyLoc, ridsLo, outGep1, maskLo);

		    // Popcount lower mask
		    auto maskLoI8 = b.create<::mlir::LLVM::BitcastOp>(bodyLoc, i8Ty, maskLo);
		    auto popLo = b.create<::mlir::LLVM::CtPopOp>(bodyLoc, i8Ty, maskLoI8);
		    auto popLoI64 = b.create<::mlir::arith::ExtUIOp>(bodyLoc, i64Ty, popLo);
		    auto idx2 = b.create<::mlir::arith::AddIOp>(bodyLoc, outIdx, popLoI64);

		    // Compress-store upper half
		    auto outGep2 =
		        b.create<::mlir::LLVM::GEPOp>(bodyLoc, ptrTy, i64Ty, rows, ::mlir::ValueRange{idx2});
		    b.create<::mlir::LLVM::masked_compressstore>(bodyLoc, ridsHi, outGep2, maskHi);

		    // Popcount upper mask
		    auto maskHiI8 = b.create<::mlir::LLVM::BitcastOp>(bodyLoc, i8Ty, maskHi);
		    auto popHi = b.create<::mlir::LLVM::CtPopOp>(bodyLoc, i8Ty, maskHiI8);
		    auto popHiI64 = b.create<::mlir::arith::ExtUIOp>(bodyLoc, i64Ty, popHi);
		    auto idx3 = b.create<::mlir::arith::AddIOp>(bodyLoc, idx2, popHiI64);

		    b.create<::mlir::scf::YieldOp>(bodyLoc, ::mlir::ValueRange{idx3});
	    });

	// The vectorized loop result (total output index after main loop)
	::mlir::Value vecOut = vecLoop.getResult(0);

	// ============================ Scalar tail loop ============================
	// scf.for %j = vecLimit to rowsCount step 1 iter_args(%tidx = vecOut) -> i64
	auto scalarLoop = builder.create<::mlir::scf::ForOp>(
	    loc, /*lb=*/vecLimit.getResult(), /*ub=*/rowsCount, /*step=*/cst1.getResult(),
	    /*iterArgs=*/::mlir::ValueRange{vecOut},
	    /*bodyBuilder=*/
	    [&](::mlir::OpBuilder& b, ::mlir::Location bodyLoc, ::mlir::Value jv, ::mlir::ValueRange iterArgs) {
		    ::mlir::Value tidx = iterArgs[0];

		    // Load single i32 from column[j]
		    auto gep = b.create<::mlir::LLVM::GEPOp>(bodyLoc, ptrTy, i32Ty, colPtr, ::mlir::ValueRange{jv});
		    auto val = b.create<::mlir::LLVM::LoadOp>(bodyLoc, i32Ty, gep, /*alignment=*/4);

		    // Compare against 42
		    auto cst42 = b.create<::mlir::arith::ConstantOp>(bodyLoc, b.getI32IntegerAttr(42));
		    auto match = b.create<::mlir::arith::CmpIOp>(bodyLoc, ::mlir::arith::CmpIPredicate::eq, val, cst42);

		    // Compute row ID = j + rowsStartOffset
		    auto rowId = b.create<::mlir::arith::AddIOp>(bodyLoc, jv, rowsStartOffset);

		    // Always store (branchless), advance output index by 0 or 1
		    auto outGep =
		        b.create<::mlir::LLVM::GEPOp>(bodyLoc, ptrTy, i64Ty, rows, ::mlir::ValueRange{tidx});
		    b.create<::mlir::LLVM::StoreOp>(bodyLoc, rowId, outGep, /*alignment=*/8);

		    auto one = b.create<::mlir::arith::ConstantOp>(bodyLoc, b.getI64IntegerAttr(1));
		    auto zero = b.create<::mlir::arith::ConstantOp>(bodyLoc, b.getI64IntegerAttr(0));
		    auto advance = b.create<::mlir::arith::SelectOp>(bodyLoc, match, one, zero);
		    auto newTidx = b.create<::mlir::arith::AddIOp>(bodyLoc, tidx, advance);

		    b.create<::mlir::scf::YieldOp>(bodyLoc, ::mlir::ValueRange{newTidx});
	    });

	::mlir::Value totalMatches = scalarLoop.getResult(0);
	builder.create<::mlir::LLVM::ReturnOp>(loc, ::mlir::ValueRange{totalMatches});

	return module;
}

// ---------------------------------------------------------------------------
// Helper: compile the MLIR module through the full Nautilus pipeline and
// return a callable function pointer.
// ---------------------------------------------------------------------------
using ExecuteFn = int64_t (*)(int64_t*, int64_t, int64_t*, int64_t*, int64_t, int64_t*, int64_t, int64_t);

struct CompiledFilter {
	std::unique_ptr<::mlir::ExecutionEngine> engine;
	ExecuteFn fn;
};

static CompiledFilter compileModule(::mlir::OwningOpRef<::mlir::ModuleOp>& module) {
	// Verify the module
	if (::mlir::failed(::mlir::verify(*module))) {
		// Dump for diagnosis
		module->dump();
		FAIL("MLIR module verification failed");
	}

	// Lower through pass pipeline (SCF → CF, arith → LLVM, etc.)
	if (nmlir::MLIRPassManager::lowerAndOptimizeMLIRModule(module, {})) {
		module->dump();
		FAIL("MLIRPassManager::lowerAndOptimizeMLIRModule failed");
	}

	// Build the LLVM optimizer pipeline
	nautilus::engine::Options options;
	options.setOption("engine.backend", std::string("mlir"));
	nautilus::compiler::CompilationUnitID unitId = "VectorFilterSpikeTest";
	nautilus::compiler::DumpHandler dumpHandler(options, unitId);
	auto optPipeline = nmlir::LLVMIROptimizer::getLLVMOptimizerPipeline(options, dumpHandler);

	// JIT compile
	std::vector<std::string> proxySymbols;
	std::vector<void*> proxyAddresses;
	auto engine = nmlir::JITCompiler::jitCompileModule(module, optPipeline, proxySymbols, proxyAddresses, options);
	REQUIRE(engine != nullptr);

	// Look up the 'execute' function
	auto fnExpected = engine->lookup("execute");
	REQUIRE(!!fnExpected);
	auto* rawPtr = fnExpected.get();
	REQUIRE(rawPtr != nullptr);
	auto fn = reinterpret_cast<ExecuteFn>(rawPtr);

	return {std::move(engine), fn};
}

// ---------------------------------------------------------------------------
// Test cases
// ---------------------------------------------------------------------------

static CompiledFilter buildAndCompile() {
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

	auto module = buildVectorFilterModule(context);
	return compileModule(module);
}

TEST_CASE("VectorFilterSpike: inttoptr + vector load + compressstore",
          "[vector-filter-spike]") {
	// Initialize LLVM native target (idempotent)
	llvm::InitializeNativeTarget();
	llvm::InitializeNativeTargetAsmPrinter();

	SECTION("aligned 64-element column, one match at [42]") {
		auto compiled = buildAndCompile();

		// Prepare test data: 64 int32_t elements, all zero except [42] = 42
		constexpr int64_t NUM_ROWS = 64;
		std::vector<int32_t> column(NUM_ROWS, 0);
		column[42] = 42;

		// cols array: single entry holding the address of column data as int64_t
		int64_t colAddr = reinterpret_cast<int64_t>(column.data());
		int64_t cols[1] = {colAddr};

		// Output buffer for matched row IDs
		std::vector<int64_t> rowsBuf(NUM_ROWS, -1);

		// Call the JIT-compiled function
		int64_t matchCount = compiled.fn(
		    cols,             // cols
		    1,                // colsCount
		    nullptr,          // varsizeIndexes
		    nullptr,          // vars
		    0,                // varsCount
		    rowsBuf.data(),   // rows
		    NUM_ROWS,         // rowsCount
		    0                 // rowsStartOffset
		);

		CHECK(matchCount == 1);
		CHECK(rowsBuf[0] == 42);
	}

	SECTION("aligned 64-element column with rowsStartOffset") {
		auto compiled = buildAndCompile();

		constexpr int64_t NUM_ROWS = 64;
		std::vector<int32_t> column(NUM_ROWS, 0);
		column[42] = 42;

		int64_t colAddr = reinterpret_cast<int64_t>(column.data());
		int64_t cols[1] = {colAddr};
		std::vector<int64_t> rowsBuf(NUM_ROWS, -1);

		constexpr int64_t OFFSET = 1000;
		int64_t matchCount = compiled.fn(
		    cols, 1, nullptr, nullptr, 0,
		    rowsBuf.data(), NUM_ROWS, OFFSET
		);

		CHECK(matchCount == 1);
		CHECK(rowsBuf[0] == 42 + OFFSET);
	}

	SECTION("misaligned input — offset column pointer by 4 bytes from 64-byte boundary") {
		auto compiled = buildAndCompile();

		// Allocate with extra padding so we can misalign by 4 bytes off a 64-byte boundary.
		// We need at least 64 * sizeof(int32_t) = 256 bytes of payload,
		// plus 64 bytes of alignment padding, plus 4 bytes for misalignment.
		constexpr int64_t NUM_ROWS = 64;
		constexpr size_t PAYLOAD = NUM_ROWS * sizeof(int32_t);
		constexpr size_t ALLOC_SIZE = PAYLOAD + 128; // generous padding (multiple of 64)

		// Use aligned_alloc for 64-byte alignment, then offset by 4
		auto* rawBuf = static_cast<uint8_t*>(std::aligned_alloc(64, ALLOC_SIZE));
		REQUIRE(rawBuf != nullptr);
		std::memset(rawBuf, 0, ALLOC_SIZE);

		// Offset by 4 bytes → misaligned for 64-byte vector loads, but aligned for 4-byte i32 loads
		auto* colData = reinterpret_cast<int32_t*>(rawBuf + 4);
		colData[42] = 42;

		int64_t colAddr = reinterpret_cast<int64_t>(colData);
		int64_t cols[1] = {colAddr};
		std::vector<int64_t> rowsBuf(NUM_ROWS, -1);

		int64_t matchCount = compiled.fn(
		    cols, 1, nullptr, nullptr, 0,
		    rowsBuf.data(), NUM_ROWS, 0
		);

		CHECK(matchCount == 1);
		CHECK(rowsBuf[0] == 42);

		std::free(rawBuf);
	}

	SECTION("multiple matches") {
		auto compiled = buildAndCompile();

		constexpr int64_t NUM_ROWS = 64;
		std::vector<int32_t> column(NUM_ROWS, 0);
		// Set several elements to 42
		column[0] = 42;
		column[15] = 42;
		column[16] = 42;
		column[42] = 42;
		column[63] = 42;

		int64_t colAddr = reinterpret_cast<int64_t>(column.data());
		int64_t cols[1] = {colAddr};
		std::vector<int64_t> rowsBuf(NUM_ROWS, -1);

		int64_t matchCount = compiled.fn(
		    cols, 1, nullptr, nullptr, 0,
		    rowsBuf.data(), NUM_ROWS, 0
		);

		CHECK(matchCount == 5);
		// Row IDs should be in order
		CHECK(rowsBuf[0] == 0);
		CHECK(rowsBuf[1] == 15);
		CHECK(rowsBuf[2] == 16);
		CHECK(rowsBuf[3] == 42);
		CHECK(rowsBuf[4] == 63);
	}

	SECTION("no matches") {
		auto compiled = buildAndCompile();

		constexpr int64_t NUM_ROWS = 64;
		std::vector<int32_t> column(NUM_ROWS, 0); // all zeroes, nothing equals 42

		int64_t colAddr = reinterpret_cast<int64_t>(column.data());
		int64_t cols[1] = {colAddr};
		std::vector<int64_t> rowsBuf(NUM_ROWS, -1);

		int64_t matchCount = compiled.fn(
		    cols, 1, nullptr, nullptr, 0,
		    rowsBuf.data(), NUM_ROWS, 0
		);

		CHECK(matchCount == 0);
	}

	SECTION("non-multiple-of-16 row count exercises scalar tail") {
		auto compiled = buildAndCompile();

		// 50 rows: 48 handled by vector loop (3 iters), 2 by scalar tail
		constexpr int64_t NUM_ROWS = 50;
		std::vector<int32_t> column(NUM_ROWS, 0);
		column[49] = 42; // match in scalar tail

		int64_t colAddr = reinterpret_cast<int64_t>(column.data());
		int64_t cols[1] = {colAddr};
		std::vector<int64_t> rowsBuf(NUM_ROWS, -1);

		int64_t matchCount = compiled.fn(
		    cols, 1, nullptr, nullptr, 0,
		    rowsBuf.data(), NUM_ROWS, 0
		);

		CHECK(matchCount == 1);
		CHECK(rowsBuf[0] == 49);
	}
}
