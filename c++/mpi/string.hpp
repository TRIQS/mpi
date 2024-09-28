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

#include "./communicator.hpp"
#include "./generic_communication.hpp"
#include "./ranges.hpp"

#include <string>

namespace mpi {

  /**
   * @addtogroup coll_comm
   * @{
   */

  /**
   * @brief Implementation of an MPI broadcast for a std::string.
   *
   * @details It first broadcasts the size of the string from the root process to all other processes, then resizes the
   * string on all non-root processes and calls mpi::broadcast_range with the (resized) input string.
   *
   * @param s std::string to broadcast.
   * @param c mpi::communicator.
   * @param root Rank of the root process.
   */
  inline void mpi_broadcast(std::string &s, communicator c, int root) {
    size_t len = s.size();
    broadcast(len, c, root);
    if (c.rank() != root) s.resize(len);
    broadcast_range(s, c, root);
  }

  /**
   * @brief Implementation of an MPI gather for a std::string.
   *
   * @details It first all-reduces the sizes of the input string from all processes and then calls mpi::gather_range.
   *
   * @param s std::string to gather.
   * @param c mpi::communicator.
   * @param root Rank of the root process.
   * @param all Should all processes receive the result.
   * @return std::string containing the result of the gather operation.
   */
  inline std::string mpi_gather(std::string const &s, communicator c = {}, int root = 0, bool all = false) {
    long len = static_cast<long>(all_reduce(s.size(), c));
    std::string res{};
    if (c.rank() == root || all) res.resize(len);
    gather_range(s, res, len, c, root, all);
    return res;
  }

  /** @} */

} // namespace mpi
