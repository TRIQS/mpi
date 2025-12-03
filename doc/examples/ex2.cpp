#include <mpi/mpi.hpp>
#include <mpi/monitor.hpp>
#include <iostream>

int main(int argc, char *argv[]) {
  // initialize MPI environment
  mpi::environment env(argc, argv);
  mpi::communicator world;

  // initialize monitor
  mpi::monitor monitor(world);

  // in case an event has occurred, print some info and return true
  auto stop = [&monitor, world](int i) {
    bool res = false;
    if (monitor.event_on_any_rank()) {
      std::cerr << "Processor " << world.rank() << ": After " << i << " steps an event has been communicated.\n";
      res = true;
    }
    return res;
  };

  // loop as long as no event has occurred
  int event_rank = 3;
  for (int i = 0; i < 1000000; ++i) {
    // report a local event on the event_rank
    if (world.rank() == event_rank) {
      std::cerr << "Processor " << event_rank << ": Local event reported.\n";
      monitor.report_local_event();
    }

    // should we stop the loop?
    if (stop(i)) break;
  }

  // check if all processes finished the loop
  if (world.rank() == 0) {
    if (monitor.event_on_any_rank()) {
      std::cout << "Oh no! An event occurred somewhere and the loop has not been finished on all processes.\n";
    } else {
      std::cout << "No worries, all processes have finished the loop.\n";
    }
  }
}