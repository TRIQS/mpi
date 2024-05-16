// Copyright (c) 2019-2024 Simons Foundation
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
// Authors: Thomas Hahn, Olivier Parcollet, Nils Wentzell

/**
 * @file
 * @brief Provides an MPI broadcast for std::string.
 */

#pragma once

#include "./mpi.hpp"

#include <mpi.h>

#include <string>

namespace mpi {

  /**
   * @ingroup coll_comm
   * @brief Implementation of an MPI broadcast for a std::string.
   *
   * @details Simply calls `MPI_Bcast` for the underlying C-string.
   *
   * @param s std::string to broadcast.
   * @param c mpi::communicator.
   * @param root Rank of the root process.
   */
  inline void mpi_broadcast(std::string &s, communicator c, int root) {
    size_t len = s.size();
    broadcast(len, c, root);
    if (c.rank() != root) s.resize(len);
    if (len != 0) MPI_Bcast((void *)s.c_str(), static_cast<int>(s.size()), mpi_type<char>::get(), root, c.get());
  }

} // namespace mpi
