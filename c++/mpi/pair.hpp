// Copyright (c) 2021 Simons Foundation
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
 * @brief Provides an MPI broadcast and reduce for std::pair.
 */

#pragma once

#include "./mpi.hpp"

#include <mpi.h>

#include <utility>

namespace mpi {

  /**
   * @brief Implementation of an MPI broadcast for a std::pair.
   *
   * @details Simply calls the generic mpi::broadcast for the first and second element of the pair.
   *
   * @tparam T1 Type of the first element of the pair.
   * @tparam T2 Type of the second element of the pair.
   * @param p std::pair to broadcast.
   * @param c mpi::communicator.
   * @param root Rank of the root process.
   */
  template <typename T1, typename T2> void mpi_broadcast(std::pair<T1, T2> &p, communicator c = {}, int root = 0) {
    broadcast(p.first, c, root);
    broadcast(p.second, c, root);
  }

  /**
   * @brief Implementation of an MPI reduce for a std::pair.
   *
   * @details Simply calls the generic mpi::reduce for the first and second element of the pair.
   *
   * @tparam T1 Type of the first element of the pair.
   * @tparam T2 Type of the second element of the pair.
   * @param p std::pair to be reduced.
   * @param c mpi::communicator.
   * @param root Rank of the root process.
   * @param all Should all processes receive the result of the reduction.
   * @param op `MPI_Op` used in the reduction.
   * @return std::pair<T1, T2> containing the result of each individual reduction.
   */
  template <typename T1, typename T2>
  auto mpi_reduce(std::pair<T1, T2> const &p, communicator c = {}, int root = 0, bool all = false, MPI_Op op = MPI_SUM) {
    return std::make_pair(reduce(p.first, c, root, all, op), reduce(p.second, c, root, all, op));
  }

} // namespace mpi
