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
 * @brief Provides utilities to distribute a range across MPI processes.
 */

#pragma once

#include "./communicator.hpp"

#include <itertools/itertools.hpp>

#include <iterator>
#include <stdexcept>
#include <utility>

namespace mpi {

  /**
   * @ingroup utilities
   * @brief Get the length of the i<sup>th</sup> subrange after splitting the integer range `[0, end)` as evenly as
   * possible across `n` subranges.
   *
   * @details The optional parameter `min_size` can be used to first divide the range into equal parts of size
   * `min_size` before distributing them as evenly as possible across the number of specified subranges.
   *
   * It throws an exception if `min_size < 1` or if it is not a divisor of `end`.
   *
   * @param end End of the integer range `[0, end)`.
   * @param nranges Number of subranges.
   * @param i Index of the subrange of interest.
   * @param min_size Minimum size of the subranges.
   * @return Length of the i<sup>th</sup> subrange.
   */
  [[nodiscard]] inline long chunk_length(long end, int nranges, int i, long min_size = 1) {
    if (min_size < 1 || end % min_size != 0) throw std::runtime_error("Error in mpi::chunk_length: min_size must be a divisor of end");
    auto [node_begin, node_end] = itertools::chunk_range(0, end / min_size, nranges, i);
    return (node_end - node_begin) * min_size;
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
