// Copyright (c) 2020-2024 Simons Foundation
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
// Authors: Philipp Dumitrescu, Thomas Hahn, Olivier Parcollet, Nils Wentzell

/**
 * @file
 * @brief Provides a class for monitoring and communicating events across multiple processes.
 */

#pragma once

#include "./macros.hpp"
#include "./mpi.hpp"

#include <mpi.h>

#include <vector>
#include <unistd.h>

namespace mpi {

  /**
   * @ingroup event_handling
   * @brief Constructed on top of an MPI communicator, this class helps to monitor and communicate events across
   * multiple processes.
   *
   * @details The root process (rank == 0) monitors all other processes. If a process encounters an event, it sends a
   * message to the root process by calling monitor::report_local_event. The root process then broadcasts this
   * information to all other processes.
   *
   * It can be used to check
   * - if an event has occurred on any process (monitor::event_on_any_rank) or
   * - if an event has occurred on all processes (monitor::event_on_all_ranks).
   *
   * It uses a duplicate communicator to not interfere with other MPI communications. The communicator is freed in the
   * `finalize_communications` function (which is called in the destructor if not called before).
   */
  class monitor {
    // Future struct for non-blocking MPI communication.
    struct future {
      // MPI request of the non-blocking MPI call.
      MPI_Request request{};

      // 0 means that no event has occurred, 1 means that an event has occurred.
      int event = 0;
    };

    // MPI communicator.
    mpi::communicator comm;

    // Future objects stored on the root process for local events on non-root processes.
    std::vector<future> root_futures;

    // MPI request for the broadcasting done on the root process in case an event has occurred on any rank.
    MPI_Request req_ibcast_any{};

    // MPI request for the broadcasting done on the root process in case an event has occurred on all ranks.
    MPI_Request req_ibcast_all{};

    // MPI request for the sending done on non-root processes.
    MPI_Request req_isent{};

    // Set to 1, if a local event has occurred on this process.
    int local_event = 0;

    // Set to 1, if an event has occurred on any process.
    int any_event = 0;

    // Set to 1, if an event has occurred on all processes.
    int all_events = 0;

    // Set to true, if finialize_communications() has been called.
    bool finalized = false;

    public:
    /**
     * @brief Construct a monitor on top of a given mpi::communicator.
     *
     * @details The communicator is duplicated to not interfere with other MPI communications.
     *
     * The root process (rank == 0) performs a non-blocking receive for every non-root process and waits for a
     * non-root process to send a message that an event has occurred.
     *
     * Non-root processes make two non-blocking broadcast calls and wait for the root process to broadcast a message in
     * case an event has occurred on any or on all processes.
     *
     * @param c mpi::communicator.
     */
    monitor(mpi::communicator c) : comm(c.duplicate()) {
      if (comm.rank() == 0) {
        root_futures.resize(c.size() - 1);
        for (int rank = 1; rank < c.size(); ++rank) {
          MPI_Irecv(&(root_futures[rank - 1].event), 1, MPI_INT, rank, rank, comm.get(), &(root_futures[rank - 1].request));
        }
      } else {
        MPI_Ibcast(&any_event, 1, MPI_INT, 0, comm.get(), &req_ibcast_any);
        MPI_Ibcast(&all_events, 1, MPI_INT, 0, comm.get(), &req_ibcast_all);
      }
    }

    /// Deleted copy constructor.
    monitor(monitor const &) = delete;

    /// Deleted copy assignment operator.
    monitor &operator=(monitor const &) = delete;

    /// Destructor calls finalize_communications().
    ~monitor() { finalize_communications(); }

    /**
     * @brief Report a local event to the root process (rank == 0).
     *
     * @details This function can be called on any process in case a local event has occurred.
     *
     * On the root process, it immediately broadcasts to all other processes that an event has occurred and further
     * checks if all other processes have reported an event as well. If so, it additionally broadcasts to all processes
     * that an event has occurred on all processes.
     *
     * On non-root processes, it sends a message to the root process that a local event has occurred.
     */
    void report_local_event() {
      // prevent sending multiple signals
      if (local_event or finalized) { return; }

      // a local event has occurred
      local_event = 1;
      if (comm.rank() == 0) {
        // on root process, check all other nodes and perform necessary broadcasts
        root_check_nodes_and_bcast();
      } else {
        // on non-root processes, let the root process know about the local event
        MPI_Isend(&local_event, 1, MPI_INT, 0, comm.rank(), comm.get(), &req_isent);
      }
    }

    /**
     * @brief Check if an event has occurred on any process.
     *
     * @details This function can be called on any process to check if an event has occurred somewhere.
     *
     * It returns true, if
     * - a local event has occurred or
     * - if an event has occurred on some other process which has already been reported to the root process and
     * broadcasted to all other processes.
     *
     * On the root process (rank == 0), it checks the status of all non-root processes and performs the necessary
     * broadcasts in case they have not been done yet.
     *
     * @return True, if an event has occurred on any process.
     */
    [[nodiscard]] bool event_on_any_rank() {
      // if final_communications() has already been called, any_event == 0 if no event has occurred, otherwise it is 1
      if (finalized) return any_event;

      // if a local event has occurred, we return true
      if (local_event) return true;

      // on the root process, we first check the status of all non-root processes, perform the necessary broadcasts and
      // return true if an event has occurred
      if (comm.rank() == 0) {
        root_check_nodes_and_bcast();
        return any_event;
      }

      // on non-root processes, we check the status of the corresponding broadcast and return true if an event has
      // occurred
      MPI_Status status;
      int has_received = 0;
      MPI_Test(&req_ibcast_any, &has_received, &status);
      return has_received and any_event;
    }

    /**
     * @brief Check if an event has occurred on all processes.
     *
     * @details This function can be called on any process to check if an event has occurred on all processes.
     *
     * It returns true, if an event has occurred on all processes which has already been reported to the root process
     * and broadcasted to all other processes.
     *
     * On the root process (rank == 0), it checks the status of all non-root processes and performs the necessary
     * broadcasts in case it has not been done yet.
     *
     * @return True, if an event has occurred on all processes.
     */
    [[nodiscard]] bool event_on_all_ranks() {
      // if final_communications() has already been called, all_events == 0 if an event has not occurred on every
      // process, otherwise it is 1
      if (finalized) return all_events;

      // on the root process, we first check the status of all non-root processes, perform the necessary broadcasts and
      // return true if an event has occurred on all of them
      if (comm.rank() == 0) {
        root_check_nodes_and_bcast();
        return all_events;
      }

      // on non-root processes, we check the status of the broadcast and return true if an event has occurred on all
      // processes
      MPI_Status status;
      int has_received = 0;
      MPI_Test(&req_ibcast_all, &has_received, &status);
      return has_received and all_events;
    }

    /**
     * @brief Finalize all pending communications.
     *
     * @details At the end of this function, all MPI communications have been completed and the values of the member
     * variables will not change anymore due to some member function calls.
     *
     * Furthermore, it frees the used communicator.
     */
    void finalize_communications() {
      // prevent multiple calls
      if (finalized) return;

      if (comm.rank() == 0) {
        // on root process, wait for all non-root processes to finish their MPI_Isend calls
        while (root_check_nodes_and_bcast()) {
          usleep(100); // 100 us (micro seconds)
        }
        // and perform broadcasts in case they have not been done yet
        if (not any_event) { MPI_Ibcast(&any_event, 1, MPI_INT, 0, comm.get(), &req_ibcast_any); }
        if (not all_events) { MPI_Ibcast(&all_events, 1, MPI_INT, 0, comm.get(), &req_ibcast_all); }
      } else {
        // on non-root processes, perform MPI_Isend call in case it has not been done yet
        if (not local_event) { MPI_Isend(&local_event, 1, MPI_INT, 0, comm.rank(), comm.get(), &req_isent); }
      }

      // all nodes wait for the broadcasts to be completed
      MPI_Status status_any, status_all;
      MPI_Wait(&req_ibcast_any, &status_any);
      MPI_Wait(&req_ibcast_all, &status_all);

      // free the communicator
      comm.free();
      finalized = true;
    }

    private:
    // Root process checks the status of all non-root processes, performs necessary broadcasts and returns a boolean
    // that is true if at least one non-root process has not performed its MPI_Isend call yet.
    bool root_check_nodes_and_bcast() {
      EXPECTS(!finalized);
      EXPECTS(comm.rank() == root);
      bool any      = false;
      bool all      = true;
      bool finished = true;
      for (auto &[request, rank_event] : root_futures) {
        MPI_Status status;
        int rank_received = 0;
        MPI_Test(&request, &rank_received, &status);
        any |= (rank_received and rank_event);
        all &= (rank_received and rank_event);
        finished &= rank_received;
      }
      if (not any_event and (any or local_event)) {
        any_event = 1;
        MPI_Ibcast(&any_event, 1, MPI_INT, 0, comm.get(), &req_ibcast_any);
      }
      if (not all_events and all and local_event) {
        all_events = 1;
        MPI_Ibcast(&all_events, 1, MPI_INT, 0, comm.get(), &req_ibcast_all);
      }
      return not finished;
    }
  };

} // namespace mpi
