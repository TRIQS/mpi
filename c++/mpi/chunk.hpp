// Copyright (c) 2019-2022 Simons Foundation
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
// Authors: Alexander Hampel, Olivier Parcollet, Nils Wentzell

/**
 * @file
 * @brief Provides utilities to distribute a range across MPI processes.
 */

#pragma once

#include "./communicator.hpp"

#include <itertools/itertools.hpp>

#include <iterator>
#include <utility>

namespace mpi {

  /**
   * @ingroup utilities
   * @brief Get the length of the i<sup>th</sup> subrange after splitting the integer range `[0, end)` evenly across n subranges.
   *
   * @param end End of the integer range `[0, end)`.
   * @param n Number of subranges.
   * @param i Index of the subrange of interest.
   * @return Length of the i<sup>th</sup> subrange.
   */
  [[nodiscard]] inline long chunk_length(long end, int n, int i) {
    auto [node_begin, node_end] = itertools::chunk_range(0, end, n, i);
    return node_end - node_begin;
  }

  /**
   * @ingroup utilities
   * @brief Divide a given range as evenly as possible across the MPI processes in a communicator and get the subrange
   * assigned to the calling process.
   *
   * @tparam R Range type.
   * @param rg Range to divide.
   * @param c mpi::communicator.
   * @return An itertools::sliced range assigned to the calling process.
   */
  template <typename R> [[nodiscard]] auto chunk(R &&rg, communicator c = {}) {
    auto total_size           = itertools::distance(std::cbegin(rg), std::cend(rg));
    auto [start_idx, end_idx] = itertools::chunk_range(0, total_size, c.size(), c.rank());
    return itertools::slice(std::forward<R>(rg), start_idx, end_idx);
  }

} // namespace mpi
