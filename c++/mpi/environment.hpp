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
 * @brief Provides an MPI environment for initializing and finalizing an MPI program.
 */

#pragma once

#include <mpi.h>

#include <cstdlib>

namespace mpi {

  /**
   * @addtogroup mpi_essentials
   * @{
   */

  /**
   * @brief Check if MPI has been initialized.
   * @return True if `MPI_Init` has been called, false otherwise.
   */
  [[nodiscard]] inline bool is_initialized() noexcept {
    int flag = 0;
    MPI_Initialized(&flag);
    return flag;
  }

  /**
   * @brief Boolean variable that is true, if one of the environment variables `OMPI_COMM_WORLD_RANK`,
   * `PMI_RANK`, `CRAY_MPICH_VERSION` or `FORCE_MPI_INIT` is set, false otherwise.
   *
   * @details The environment variables are set, when a program is executed with `mpirun` or `mpiexec`.
   */
  static const bool has_env = []() {
    if (std::getenv("OMPI_COMM_WORLD_RANK") != nullptr or std::getenv("PMI_RANK") != nullptr or std::getenv("CRAY_MPICH_VERSION") != nullptr
        or std::getenv("FORCE_MPI_INIT") != nullptr)
      return true;
    else
      return false;
  }();

  /**
   * @brief RAII class to initialize and finalize MPI.
   *
   * @details Calls `MPI_Init` upon construction and `MPI_Finalize` upon destruction i.e. when the environment object goes out of scope.
   * If mpi::has_env is false, this struct does nothing.
   */
  struct environment {
    /**
     * @brief Construct a new mpi environment object by calling `MPI_Init`.
     *
     * @details Checks first if the program is run with an MPI runtime environment and if it has not been
     * initialized before to avoid errors.
     *
     * @param argc Number of command line arguments.
     * @param argv Command line arguments.
     */
    environment(int argc, char *argv[]) { //  NOLINT (C-style array is wanted here)
      if (has_env && !is_initialized()) MPI_Init(&argc, &argv);
    }

    /**
     * @brief Destroy the mpi environment object by calling `MPI_Finalize`.
     *
     * @details Checks first if the program is run with an MPI runtime environment. Called automatically when the environment
     * object goes out of scope.
     */
    ~environment() {
      if (has_env) MPI_Finalize();
    }
  };

  /** @} */

} // namespace mpi
