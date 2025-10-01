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
#include "./utils.hpp"

#include <mpi.h>

#include <cstdlib>
#include <unistd.h>

namespace mpi {

  // Forward declaration.
  class shared_communicator;

  /**
   * @ingroup mpi_essentials
   * @brief C++ wrapper around `MPI_Comm` providing various convenience functions.
   *
   * @details It stores an `MPI_Comm` object as its only member which by default is set to `MPI_COMM_WORLD`. The
   * underlying `MPI_Comm` object is not freed when a communicator goes out of scope. It is the user's responsibility to
   * do so, in case it is needed. Note that copying the communicator simply copies the `MPI_Comm` object, without
   * calling `MPI_Comm_dup`.
   *
   * All functions that make direct calls to the MPI C library throw an exception in case the call fails.
   */
  class communicator {
    public:
    /// Construct a communicator with `MPI_COMM_WORLD`.
    communicator() = default;

    /**
     * @brief Construct a communicator with a given `MPI_Comm` object.
     * @details The `MPI_Comm` object is copied without calling `MPI_Comm_dup`.
     * @param c `MPI_Comm` object to wrap.
     */
    communicator(MPI_Comm c) : comm_(c) {}

    /// Get the wrapped `MPI_Comm` object.
    [[nodiscard]] MPI_Comm get() const noexcept { return comm_; }

    /// Check if the contained `MPI_Comm` is `MPI_COMM_NULL`.
    [[nodiscard]] bool is_null() const noexcept { return comm_ == MPI_COMM_NULL; }

    /**
     * @brief Get the rank of the calling process in the communicator.
     * @return The result of `MPI_Comm_rank` if mpi::has_env is true, otherwise 0.
     */
    [[nodiscard]] int rank() const {
      int r = 0;
      if (has_env) check_mpi_call(MPI_Comm_rank(comm_, &r), "MPI_Comm_rank");
      return r;
    }

    /**
     * @brief Get the size of the communicator.
     * @return The result of `MPI_Comm_size` if mpi::has_env is true, otherwise 1.
     */
    [[nodiscard]] int size() const {
      int s = 1;
      if (has_env) check_mpi_call(MPI_Comm_size(comm_, &s), "MPI_Comm_size");
      return s;
    }

    /**
     * @brief Split the communicator into disjoint subgroups.
     *
     * @details Calls `MPI_Comm_split` with the given color and key arguments. See the MPI documentation for more
     * details, e.g. <a href="https://docs.open-mpi.org/en/v5.0.x/man-openmpi/man3/MPI_Comm_split.3.html">open-mpi
     * docs</a>.
     *
     * @warning This allocates a new communicator object. Make sure to call free() on the returned communicator when it
     * is no longer needed.
     *
     * @param color Determines which processes are put into the same group.
     * @param key Determines the rank of the process in the new communicator.
     * @return If mpi::has_env is true, return the split `MPI_Comm` object wrapped in a new mpi::communicator, otherwise
     * return a default constructed mpi::communicator.
     */
    [[nodiscard]] communicator split(int color, int key = 0) const {
      communicator c{};
      if (has_env) check_mpi_call(MPI_Comm_split(comm_, color, key, &c.comm_), "MPI_Comm_split");
      return c;
    }

    /**
     * @brief Partition the communicator into subcommunicators according to their type.
     *
     * @details In the MPI3.0 standard the only supported split type is `MPI_COMM_TYPE_SHARED`. OpenMPI (and possibly
     * other implementations) provide more custom split types, however, they are not portable.
     *
     * @warning This allocates a new communicator object. Make sure to call free on the returned communicator when it
     * is no longer needed.
     *
     * @param split_type Type of processes to be grouped together.
     * @param key Determines the rank of the process in the new communicator.
     * @return If mpi::has_env is true, return the split `MPI_Comm` object wrapped in a new mpi::communicator, otherwise
     * return a default constructed mpi::communicator.
     */
    [[nodiscard]] shared_communicator split_shared(int split_type = MPI_COMM_TYPE_SHARED, int key = 0) const;

    /**
     * @brief Duplicate the communicator.
     *
     * @details Calls `MPI_Comm_dup` to duplicate the communicator. See the MPI documentation for more details, e.g.
     * <a href="https://docs.open-mpi.org/en/v5.0.x/man-openmpi/man3/MPI_Comm_dup.3.html">open-mpi docs</a>.
     *
     * @warning This allocates a new communicator object. Make sure to call free on the returned communicator when it
     * is no longer needed.
     *
     * @return If mpi::has_env is true, return the duplicated `MPI_Comm` object wrapped in a new mpi::communicator,
     * otherwise return a default constructed mpi::communicator.
     */
    [[nodiscard]] communicator duplicate() const {
      communicator c{};
      if (has_env) check_mpi_call(MPI_Comm_dup(comm_, &c.comm_), "MPI_Comm_dup");
      return c;
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
      if (has_env && !is_null()) check_mpi_call(MPI_Comm_free(&comm_), "MPI_Comm_free");
    }

    /**
     * @brief If mpi::has_env is true, `MPI_Abort` is called with the given error code, otherwise it calls `std::abort`.
     * @param error_code The error code to pass to `MPI_Abort`.
     */
    void abort(int error_code) const {
      if (has_env) {
        check_mpi_call(MPI_Abort(comm_, error_code), "MPI_Abort");
      } else {
        std::abort();
      }
    }

#ifdef BOOST_MPI_HPP
    // Conversion to and from boost communicator, Keep for backward compatibility
    inline operator boost::mpi::communicator() const { return boost::mpi::communicator(comm_, boost::mpi::comm_duplicate); }
    inline communicator(boost::mpi::communicator c) : comm_(c) {}
#endif // BOOST_MPI_HPP

    /**
     * @brief Barrier synchronization.
     *
     * @details Does nothing if mpi::has_env is false. Otherwise, it either uses a blocking `MPI_Barrier` (if the given
     * argument is 0) or a non-blocking `MPI_Ibarrier` call. The given parameter determines in milliseconds how often
     * each process calls `MPI_Test` to check if all processes have reached the barrier.
     *
     * This can considerably reduce the CPU load:
     * - 1 msec ~ 1% cpu load
     * - 10 msec ~ 0.5% cpu load
     * - 100 msec ~ 0.01% cpu load
     *
     * For a very unbalanced load that takes a long time to finish, 1000 msec is a good choice.
     *
     * @param poll_msec Polling interval in milliseconds. If set to 0, a simple `MPI_Barrier` call is used.
     */
    void barrier(long poll_msec = 1) const {
      if (has_env) {
        if (poll_msec == 0) {
          check_mpi_call(MPI_Barrier(comm_), "MPI_Barrier");
        } else {
          MPI_Request req{};
          int flag = 0;
          check_mpi_call(MPI_Ibarrier(comm_, &req), "MPI_Ibarrier");
          while (!flag) {
            check_mpi_call(MPI_Test(&req, &flag, MPI_STATUS_IGNORE), "MPI_Test");
            usleep(poll_msec * 1000);
          }
        }
      }
    }

    private:
    MPI_Comm comm_ = MPI_COMM_WORLD;
  };

  /**
   * @ingroup mpi_osc_shm
   * @brief C++ wrapper around `MPI_Comm` that is a result of the mpi::communicator::split_shared operation.
   *
   * @details In the plain MPI C API it is not distinguishable whether an `MPI_Comm` is local to a shared memory island
   * or not. Thus we introduce an extra type for that whose only purpose is to make that distinction on the type-level
   * to prevent wrong usage of the shared memory APIs.
   */
  class shared_communicator : public communicator {
    public:
    // Make the constructors of mpi::communicator accessible.
    using communicator::communicator;

    /// Construct a shared communicator with `MPI_COMM_NULL`.
    shared_communicator() : communicator(MPI_COMM_NULL) {}
  };

  [[nodiscard]] inline shared_communicator communicator::split_shared(int split_type, int key) const {
    shared_communicator c{};
    if (has_env) check_mpi_call(MPI_Comm_split_type(comm_, split_type, key, MPI_INFO_NULL, &c.comm_), "MPI_Comm_split_type");
    return c;
  }

} // namespace mpi
