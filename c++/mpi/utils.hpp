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

  namespace detail {

    // Helper struct to get the regular type of a type.
    template <typename T, typename Enable = void> struct _regular {
      using type = T;
    };

    // Spezialization of _regular for types with a `regular_type` type alias.
    template <typename T> struct _regular<T, std::void_t<typename T::regular_type>> {
      using type = typename T::regular_type;
    };

  } // namespace detail

  /**
   * @ingroup utilities
   * @brief Type trait to get the regular type of a type.
   * @tparam T Type to check.
   */
  template <typename T> using regular_t = typename detail::_regular<std::decay_t<T>>::type;

  /**
   * @brief Check the success of an MPI call.
   * @details It checks if the given error code returned by an MPI routine is equal to `MPI_SUCCESS`. If it isn't, it
   * throws an exception.
   *
   * It is intended to simply wrap any calls to the MPI C library:
   * @code{.cpp}
   * int value = 5;
   * int result = 0;
   * check_mpi_call(MPI_Allreduce(&value, &result, 1, mpi::mpi_type<int>::get(), MPI_MAX, comm.get()), "MPI_Allreduce");
   * @endcode
   *
   * @param errcode Error code returned by an MPI routine.
   * @param mpi_routine Name of the MPI routine used in the error message.
   */
  inline void check_mpi_call(int errcode, const std::string &mpi_routine) {
    if (errcode != MPI_SUCCESS) throw std::runtime_error("MPI error " + std::to_string(errcode) + " in MPI routine " + mpi_routine);
  }

  /**
   * @brief A concept that checks if a range type is contiguous and sized.
   * @tparam R Range type.
   */
  template <typename R>
  concept contiguous_sized_range = std::ranges::contiguous_range<R> && std::ranges::sized_range<R>;

  /** @} */

} // namespace mpi
