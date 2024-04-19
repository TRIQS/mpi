// Copyright (c) 2020-2022 Simons Foundation
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
// Authors: Philipp Dumitrescu, Olivier Parcollet, Nils Wentzell

/**
 * @file
 * @brief Provides a class for monitoring and communicating exceptions and other errors of
 * individual processes.
 */

#pragma once

#include "./macros.hpp"
#include "./mpi.hpp"

#include <mpi.h>

#include <vector>
#include <unistd.h>

namespace mpi {

  /**
   * @ingroup err_handling
   * @brief Constructed on top of an MPI communicator, this class helps to monitor and communicate
   * exceptions and other errors of individual processes.
   *
   * @details The root process (process with rank 0) monitors all other processes. If a process encounters
   * an error, it sends an emergeny stop request to the root process which forwards it to all the other
   * processes.
   */
  class monitor {
    // Future struct for the non-blocking send/receive done on the root process.
    struct future {
      // MPI request for the non-blocking receive on the root process.
      MPI_Request request{};

      // 0 means that no error has occurred, 1 means that an error has occurred.
      int node_stop = 0;
    };

    // MPI communicator.
    mpi::communicator com;

    // Future objects stored on the root process for every non-root process.
    std::vector<future> root_futures;

    // MPI request for broadcasting the emergency stop to all non-root processes.
    MPI_Request req_ibcast{};

    // MPI request for sending the emergency stop request to the root process.
    MPI_Request req_isent{};

    // Set to 1, if the process has encountered a local error and requested an emergency stop.
    int local_stop = 0;

    // Set to 1, if the process has received an emergency stop broadcasted by the root process.
    int global_stop = 0;

    // Set to true, if finialize_communications() has been called.
    bool finalized = false;

    public:
    /**
     * @brief Construct a monitor on top of a given mpi::communicator.
     *
     * @details The root process performs a non-blocking receive for every non-root process and waits for
     * a non-root process to send an emergency stop request. Non-root processes make a non-blocking broadcast
     * call and wait for the root process to broadcast any emergency stop request it has received.
     *
     * @param c mpi::communicator.
     */
    monitor(mpi::communicator c) : com(c) {
      if (com.rank() == 0) {
        root_futures.resize(c.size() - 1);
        for (int rank = 1; rank < c.size(); ++rank) {
          MPI_Irecv(&(root_futures[rank - 1].node_stop), 1, MPI_INT, rank, 0, MPI_COMM_WORLD, &(root_futures[rank - 1].request));
        }
      } else {
        MPI_Ibcast(&global_stop, 1, MPI_INT, 0, MPI_COMM_WORLD, &req_ibcast);
      }
    }

    /// Deleted copy constructor.
    monitor(monitor const &) = delete;

    /// Deleted copy assignment operator.
    monitor &operator=(monitor const &) = delete;

    /// Destructor calls finalize_communications().
    ~monitor() { finalize_communications(); }

    /**
     * @brief Request an emergency stop.
     *
     * @details This function can be called on any process in case a local error has occurred. On the
     * root process, it sets its `local_stop` and `global_stop` members to 1 and broadcasts `global_stop`
     * to all non-root processes. On non-root processes, it sets `local_stop` to 1 and sends it to the
     * root process.
     */
    void request_emergency_stop() {
      EXPECTS(!finalized);
      // prevent sending multiple signals
      if (local_stop) { return; }

      // an error has occurred
      local_stop = 1;
      if (com.rank() == 0) {
        // root broadcasts the global_stop variable
        global_stop = 1;
        MPI_Ibcast(&global_stop, 1, MPI_INT, 0, MPI_COMM_WORLD, &req_ibcast);
      } else {
        // non-root sends the local_stop variable to root
        MPI_Isend(&local_stop, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, &req_isent);
      }
    }

    /**
     * @brief Check if an emergency stop has been requested.
     *
     * @details This function can be called on any process to check if an emergency has occurred somewhere.
     * It first checks, if its `local_stop` or `global_stop` members are set to 1 and returns `true` in case
     * one of them is. Otherwise, on the root process, it calls `root_check_nodes_and_bcast()` to check if
     * some other process has sent an emergency message and to possibly forward the received signal.
     * On non-root processes, it checks if the root process has broadcasted an emergency stop, which it has
     * received from some other process.
     *
     * @return True, if an emergency stop has been requested. Otherwise, it returns false.
     */
    [[nodiscard]] bool emergency_occured() {
      // if final_communications() has already been called, global_stop == 0 if no error has occurred, otherwise it is 1
      if (finalized) return global_stop;

      // either a local error has occurred or some other process has requested an emergency stop
      if (global_stop or local_stop) return true;

      if (com.rank() == 0) {
        // root checks if some other process has requested an emergency stop
        root_check_nodes_and_bcast();
      }
      return global_stop;
    }

    /**
     * @brief Finalize all pending communications.
     *
     * @details At the end of this function, all processes have completed their work or have had a local
     * emergency stop. The member `global_stop` is guaranteed to be the same on all processes when this
     * function returns.
     */
    void finalize_communications() {
      if (finalized) return;
      if (com.rank() == 0) {
        // root just listens to the other processes and bcasts the global_stop until everyone is done
        while (root_check_nodes_and_bcast()) { usleep(100); } // 100 us (micro seconds)
        // all others node have finished
        // if the root has never emitted the ibcast, we do it now
        if (not global_stop) { MPI_Ibcast(&global_stop, 1, MPI_INT, 0, MPI_COMM_WORLD, &req_ibcast); }
      } else {
        // on non-root node: either Isend was done when local_stop was set to 1 during request_emergency_stop,
        // or it has to happen now, i.e, work is done, and fine.
        if (not local_stop) { MPI_Isend(&local_stop, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, &req_isent); }
      }
      // all nodes wait for the ibcast of the global_stop to be complete
      MPI_Status status;
      MPI_Wait(&req_ibcast, &status);
      finalized = true;
    }

    private:
    /**
     * @brief Check if any non-root process has sent a stop request. If so, broadcast to all other processes
     * in case it has not been done yet.
     *
     * @return True, if at least one process has not finished the `MPI_Isend` of the `local_stop` variable to
     * the root process. Otherwise, it returns false.
     */
    bool root_check_nodes_and_bcast() {
      EXPECTS(!finalized);
      EXPECTS(com.rank() == 0);
      // loop over all non-root processes
      bool some_nodes_are_still_running = false;
      for (auto &[request, node_stop] : root_futures) {
        // check for an emergency stop request
        MPI_Status status;
        int comm_received = 0;
        MPI_Test(&request, &comm_received, &status);
        // for the first time an emergency stop has been requested -> root calls request_emergency_stop()
        // to broadcast to all other processes
        if (comm_received and (not global_stop) and node_stop) request_emergency_stop(); // the root requires the stop now. It also stops itself...
        some_nodes_are_still_running |= (not comm_received);
      }
      return some_nodes_are_still_running;
    }
  };

} // namespace mpi
