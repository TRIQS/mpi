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
// Authors: Nils Wentzell

/**
 * @file
 * @brief Provides an MPI broadcast and reduce for `std::optional`.
 */

#pragma once

#include "./communicator.hpp"
#include "./generic_communication.hpp"
#include "./macros.hpp"

#include <mpi.h>

#include <optional>
#include <concepts>

namespace mpi {

  /**
   * @addtogroup coll_comm
   * @{
   */

  /**
   * @brief Implementation of an MPI broadcast for a `std::optional`.
   *
   * @details It first broadcasts a flag indicating whether the optional has a value. If the root's optional has a
   * value, the value is broadcast to all other processes. If the root's optional is empty, all other processes reset
   * their optional to empty.
   *
   * @tparam T Value type of the optional (must be default constructible).
   * @param opt `std::optional` to broadcast.
   * @param c mpi::communicator.
   * @param root Rank of the root process.
   */
  template <std::default_initializable T>
  void mpi_broadcast(std::optional<T> &opt, communicator c = {}, int root = 0) {
    bool has_val = opt.has_value();
    broadcast(has_val, c, root);

    if (has_val) {
      if (!opt.has_value()) opt.emplace();
      broadcast(*opt, c, root);
    } else {
      opt.reset();
    }
  }

  /**
   * @brief Implementation of an MPI reduce for a `std::optional` that reduces directly into a given output optional.
   *
   * @details All ranks must have consistent has_value state (all have values or all are empty).
   *
   * If all input optionals have values, the values are reduced into the output optional. If all input optionals are
   * empty, the output optional is reset to empty. On non-receiving ranks, the output optional is left untouched.
   *
   * @tparam T1 Value type of the optional to be reduced.
   * @tparam T2 Value type of the optional to be reduced into (must be default constructible).
   * @param opt_in `std::optional` to reduce.
   * @param opt_out `std::optional` to reduce into.
   * @param c mpi::communicator.
   * @param root Rank of the root process.
   * @param all Should all processes receive the result of the reduction.
   * @param op `MPI_Op` used in the reduction.
   */
  template <typename T1, std::default_initializable T2>
  void mpi_reduce_into(std::optional<T1> const &opt_in, std::optional<T2> &opt_out, communicator c = {}, int root = 0, bool all = false,
                       MPI_Op op = MPI_SUM) {
    // Verify consistency
    EXPECTS_WITH_MESSAGE(all_equal<int>(opt_in.has_value(), c),
                         "mpi::reduce_into for std::optional requires all ranks to have consistent has_value state");

    bool const receives = (c.rank() == root || all);

    if (opt_in.has_value()) {
      if (receives) {
        if (!opt_out.has_value()) opt_out.emplace();
        reduce_into(*opt_in, *opt_out, c, root, all, op);
      } else {
        T2 dummy;
        reduce_into(*opt_in, dummy, c, root, all, op);
      }
    } else if (receives) {
      opt_out.reset();
    }
  }

  /** @} */

} // namespace mpi
