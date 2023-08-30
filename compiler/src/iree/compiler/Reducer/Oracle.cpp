// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Reducer/Oracle.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Program.h"
#include "llvm/Support/ToolOutputFile.h"

using namespace mlir;
using namespace mlir::iree_compiler;

bool Oracle::isInteresting(WorkItem &workItem) {
  // Print module to a temporary file.
  SmallString<128> filepath;
  int fd;
  std::error_code ec =
      llvm::sys::fs::createTemporaryFile("oracle", "mlir", fd, filepath);

  if (ec) {
    llvm::report_fatal_error(llvm::Twine("Failed to create temporary file: ") +
                             ec.message());
  }

  llvm::ToolOutputFile output(filepath, fd);
  workItem.getModule()->print(output.os());
  output.os().close();

  if (output.os().has_error()) {
    llvm::report_fatal_error(
        llvm::Twine("Failed to write to temporary file: ") +
        output.os().error().message());
  }

  // Run the oracle.
  SmallVector<StringRef> testerArgs;
  testerArgs.push_back(testScript);
  testerArgs.push_back(filepath);

  std::string errMsg;
  int exitCode = llvm::sys::ExecuteAndWait(testScript, testerArgs, std::nullopt,
                                           std::nullopt, 0, 0, &errMsg);

  if (exitCode < 0) {
    llvm::report_fatal_error(llvm::Twine("Failed to run oracle: ") + errMsg);
  }

  if (exitCode == 0) {
    return true;
  } else {
    return false;
  }
}
