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
 * @brief Provides an MPI broadcast and gather for `std::string`.
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
   * @brief Implementation of an MPI broadcast for a `std::string`.
   *
   * @details It first broadcasts the size of the string from the root process to all other processes, then resizes the
   * string on all non-root processes and calls mpi::broadcast_range with the (resized) input string.
   *
   * @param s `std::string` to broadcast (into).
   * @param c mpi::communicator.
   * @param root Rank of the root process.
   */
  inline void mpi_broadcast(std::string &s, communicator c, int root) {
    auto count = s.size();
    broadcast(count, c, root);
    if (c.rank() != root) s.resize(count);
    broadcast_range(s, c, root);
  }

  /**
   * @brief Implementation of an MPI gather for a `std::string` that gathers directly into an existing output string.
   *
   * @details It first all-reduces the sizes of the input strings from all processes. On receiving ranks, the output
   * string is resized to the reduced size in case it has not the correct size. On non-receiving ranks, the output
   * string is always unmodified. Then mpi::gather_range with the input and (resized) output strings is called.
   *
   * @param s_in `std::string` to gather.
   * @param s_out `std::string` to gather into.
   * @param c mpi::communicator.
   * @param root Rank of the root process.
   * @param all Should all processes receive the result of the gather.
   */
  inline void mpi_gather_into(std::string const &s_in, std::string &s_out, communicator c = {}, int root = 0, bool all = false) {
    auto const gather_size = mpi::all_reduce(s_in.size(), c);
    if ((c.rank() == root || all) && s_out.size() != gather_size) s_out.resize(gather_size);
    gather_range(s_in, s_out, c, root, all);
  }

  /** @} */

} // namespace mpi
