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
 * @brief Provides a C++ wrapper class for an `MPI_Comm` object.
 */

#pragma once

#include "./environment.hpp"

#include <mpi.h>

#include <cstdlib>
#include <unistd.h>

namespace mpi {

  /**
   * @ingroup mpi_essentials
   * @brief C++ wrapper around `MPI_Comm` providing various convenience functions.
   *
   * @details It stores an `MPI_Comm` object as its only member which by default is set to `MPI_COMM_WORLD`.
   * Note that copying the communicator simply copies the `MPI_Comm` object, without calling `MPI_Comm_dup`.
   */
  class communicator {
    // Wrapped `MPI_Comm` object.
    MPI_Comm _com = MPI_COMM_WORLD;

    public:
    /// Construct a communicator with `MPI_COMM_WORLD`.
    communicator() = default;

    /**
     * @brief Construct a communicator with a given `MPI_Comm` object.
     * @details The `MPI_Comm` object is copied without calling `MPI_Comm_dup`.
     */
    communicator(MPI_Comm c) : _com(c) {}

    /// Get the wrapped `MPI_Comm` object.
    [[nodiscard]] MPI_Comm get() const noexcept { return _com; }

    /**
     * @brief Get the rank of the calling process in the communicator.
     * @return The result of `MPI_Comm_rank` if mpi::has_env is true, otherwise 0.
     */
    [[nodiscard]] int rank() const {
      if (has_env) {
        int num = 0;
        MPI_Comm_rank(_com, &num);
        return num;
      } else
        return 0;
    }

    /**
     * @brief Get the size of the communicator.
     * @return The result of `MPI_Comm_size` if mpi::has_env is true, otherwise 1.
     */
    [[nodiscard]] int size() const {
      if (has_env) {
        int num = 0;
        MPI_Comm_size(_com, &num);
        return num;
      } else
        return 1;
    }

    /**
     * @brief Split the communicator into disjoint subgroups.
     *
     * @details Calls `MPI_Comm_split` with the given color and key arguments. See the MPI documentation for more details,
     * e.g. <a href="https://docs.open-mpi.org/en/v5.0.x/man-openmpi/man3/MPI_Comm_split.3.html">open-mpi docs</a>.
     *
     * @return If mpi::has_env is true, return the split `MPI_Comm` object wrapped in a new mpi::communicator, otherwise
     * return a default constructed mpi::communicator.
     */
    [[nodiscard]] communicator split(int color, int key = 0) const {
      if (has_env) {
        communicator c;
        MPI_Comm_split(_com, color, key, &c._com);
        return c;
      } else
        return {};
    }

    /**
     * @brief Duplicate the communicator.
     *
     * @details Calls `MPI_Comm_dup` to duplicate the communicator. See the MPI documentation for more details, e.g.
     * <a href="https://docs.open-mpi.org/en/v5.0.x/man-openmpi/man3/MPI_Comm_dup.3.html">open-mpi docs</a>.
     *
     * @return If mpi::has_env is true, return the duplicated `MPI_Comm` object wrapped in a new mpi::communicator,
     * otherwise return a default constructed mpi::communicator.
     */
    [[nodiscard]] communicator duplicate() const {
      if (has_env) {
        communicator c;
        MPI_Comm_dup(_com, &c._com);
        return c;
      } else
        return {};
    }

    /**
     * @brief Free the communicator.
     *
     * @details Calls `MPI_Comm_free` to mark the communicator for deallocation. See the MPI documentation for more
     * details, e.g. <a href="https://docs.open-mpi.org/en/v5.0.x/man-openmpi/man3/MPI_Comm_free.3.html">open-mpi docs
     * </a>.
     *
     * Does nothing, if mpi::has_env is false.
     */
    void free() {
      if (has_env) { MPI_Comm_free(&_com); }
    }

    /**
     * @brief If mpi::has_env is true, `MPI_Abort` is called with the given error code, otherwise std::abort is called.
     * @param error_code The error code to pass to `MPI_Abort`.
     */
    void abort(int error_code) {
      if (has_env)
        MPI_Abort(_com, error_code);
      else
        std::abort();
    }

#ifdef BOOST_MPI_HPP
    // Conversion to and from boost communicator, Keep for backward compatibility
    inline operator boost::mpi::communicator() const { return boost::mpi::communicator(_com, boost::mpi::comm_duplicate); }
    inline communicator(boost::mpi::communicator c) : _com(c) {}
#endif // BOOST_MPI_HPP

    /**
     * @brief Barrier synchronization.
     *
     * @details Does nothing if mpi::has_env is false. Otherwise, it either uses a blocking `MPI_Barrier`
     * (if the given argument is 0) or a non-blocking `MPI_Ibarrier` call. The given parameter determines
     * in milliseconds how often each process calls `MPI_Test` to check if all processes have reached the barrier.
     * This can considerably reduce the CPU load:
     *     - 1 msec ~ 1% cpu load
     *     - 10 msec ~ 0.5% cpu load
     *     - 100 msec ~ 0.01% cpu load
     *
     * For a very unbalanced load that takes a long time to finish, 1000 msec is a good choice.
     *
     * @param poll_msec The polling interval in milliseconds. If set to 0, a simple `MPI_Barrier` call is used.
     */
    void barrier(long poll_msec = 1) {
      if (has_env) {
        if (poll_msec == 0) {
          MPI_Barrier(_com);
        } else {
          MPI_Request req{};
          int flag = 0;
          MPI_Ibarrier(_com, &req);
          while (!flag) {
            MPI_Test(&req, &flag, MPI_STATUS_IGNORE);
            usleep(poll_msec * 1000);
          }
        }
      }
    }
  };

} // namespace mpi
