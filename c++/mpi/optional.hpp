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

#include <mpi.h>

#include <optional>
#include <stdexcept>
#include <type_traits>
#include <utility>

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
   * @tparam T Value type of the optional.
   * @param opt `std::optional` to broadcast.
   * @param c mpi::communicator.
   * @param root Rank of the root process.
   */
  template <typename T> void mpi_broadcast(std::optional<T> &opt, communicator c = {}, int root = 0) {
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
   * @brief Implementation of an MPI reduce for a `std::optional`.
   *
   * @details All ranks must have consistent has_value state (all have values or all are empty). If this condition is
   * violated, a `std::runtime_error` is thrown.
   *
   * If all optionals have values, the values are reduced and returned in an optional. If all optionals are empty, an
   * empty optional is returned.
   *
   * @tparam T Value type of the optional.
   * @param opt `std::optional` to reduce.
   * @param c mpi::communicator.
   * @param root Rank of the root process.
   * @param all Should all processes receive the result of the reduction.
   * @param op `MPI_Op` used in the reduction.
   * @return `std::optional` containing the result of the reduction.
   */
  template <typename T>
  std::optional<T> mpi_reduce(std::optional<T> const &opt, communicator c = {}, int root = 0, bool all = false, MPI_Op op = MPI_SUM) {
    // Verify consistency: sum of has_value should be 0 or c.size()
    int has_val = opt.has_value() ? 1 : 0;
    int total   = mpi::all_reduce(has_val, c, MPI_SUM);
    if (total != 0 && total != c.size()) {
      throw std::runtime_error("mpi::reduce for std::optional requires all ranks to have consistent has_value state");
    }

    if (opt.has_value()) return reduce(*opt, c, root, all, op);
    else return {};
  }

  /**
   * @brief Implementation of an MPI reduce for a `std::optional` that reduces directly into a given output optional.
   *
   * @details All ranks must have consistent has_value state (all have values or all are empty). If this condition is
   * violated, a `std::runtime_error` is thrown.
   *
   * If all input optionals have values, the values are reduced into the output optional on all ranks, but only root
   * (or all ranks if `all` is true) receives the meaningful result. If all input optionals are empty, the output
   * optional is reset to empty on all ranks.
   *
   * @tparam T1 Value type of the optional to be reduced.
   * @tparam T2 Value type of the optional to be reduced into.
   * @param opt_in `std::optional` to reduce.
   * @param opt_out `std::optional` to reduce into.
   * @param c mpi::communicator.
   * @param root Rank of the root process.
   * @param all Should all processes receive the result of the reduction.
   * @param op `MPI_Op` used in the reduction.
   */
  template <typename T1, typename T2>
  void mpi_reduce_into(std::optional<T1> const &opt_in, std::optional<T2> &opt_out, communicator c = {}, int root = 0, bool all = false,
                       MPI_Op op = MPI_SUM) {
    // Verify consistency
    int has_val = opt_in.has_value() ? 1 : 0;
    int total   = mpi::all_reduce(has_val, c, MPI_SUM);
    if (total != 0 && total != c.size()) {
      throw std::runtime_error("mpi::reduce_into for std::optional requires all ranks to have consistent has_value state");
    }

    if (opt_in.has_value()) {
      if (!opt_out.has_value()) opt_out.emplace();
      reduce_into(*opt_in, *opt_out, c, root, all, op);
    } else {
      opt_out.reset();
    }
  }

  /** @} */

} // namespace mpi
