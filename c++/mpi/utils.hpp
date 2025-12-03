// Copyright (c) 2024 Simons Foundation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0.txt
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
// Authors: Thomas Hahn, Alexander Hampel, Olivier Parcollet, Nils Wentzell

/**
 * @file
 * @brief Provides general utilities related to MPI.
 */

#pragma once

#include <mpi.h>

#include <stdexcept>
#include <string>

namespace mpi {

  /**
   * @addtogroup utilities
   * @{
   */

  /**
   * @brief Check the success of an MPI call.
   * 
   * @details It checks if the given error code returned by an MPI routine is equal to `MPI_SUCCESS`. If it isn't, it
   * throws an exception.
   *
   * @note MPI routines only return an error code if the error handler is set to `MPI_ERRORS_RETURN`. By default, it 
   * uses `MPI_ERRORS_ARE_FATAL`, which aborts the program in case of an error.
   *
   * @param errcode Error code returned by an MPI routine.
   * @param mpi_routine Name of the MPI routine used in the error message.
   */
  inline void check_mpi_call(int errcode, const std::string &mpi_routine) {
    if (errcode != MPI_SUCCESS) throw std::runtime_error("MPI error " + std::to_string(errcode) + " in MPI routine " + mpi_routine);
  }

  /** @} */

} // namespace mpi
