
#pragma once

#include "nautilus/compiler/backends/CompilationBackend.hpp"

namespace nautilus::compiler::asmjit {

/**
 * @brief Compilation backend that generates x86-64 machine code via AsmJit.
 */
class AsmJitCompilationBackend : public CompilationBackend {
public:
	std::unique_ptr<Executable> compile(const std::shared_ptr<ir::IRGraph>& ir, const DumpHandler& dumpHandler,
	                                    const engine::Options& options) const override;
};

} // namespace nautilus::compiler::asmjit
