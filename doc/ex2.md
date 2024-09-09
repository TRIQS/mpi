@page ex2 Example 2: Use monitor to communicate errors

[TOC]

In this example, we show how to use the mpi::monitor class to communicate and process errors across a communicator.

```cpp
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
    if (monitor.some_event_occurred()) {
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
    if (monitor.some_event_occurred()) {
      std::cout << "Oh no! An event occurred somewhere and loop has not been finished on all processes.\n";
    } else {
      std::cout << "No worries, all processes have finished the loop.\n";
    }
  }
}
```

Output (running with `-n 12`):

```
Processor 3: Local event reported.
Processor 3: After 0 steps an event has been communicated.
Processor 4: After 8428 steps an event has been communicated.
Processor 0: After 0 steps an event has been communicated.
Processor 8: After 10723 steps an event has been communicated.
Processor 5: After 10426 steps an event has been communicated.
Processor 6: After 12172 steps an event has been communicated.
Processor 7: After 9014 steps an event has been communicated.
Processor 1: After 400 steps an event has been communicated.
Processor 2: After 1646 steps an event has been communicated.
Processor 11: After 12637 steps an event has been communicated.
Processor 10: After 9120 steps an event has been communicated.
Processor 9: After 1 steps an event has been communicated.
Oh no! An event occurred somewhere and the loop has not been finished on all processes.
```

Output (running with `-n 3`):

```
No worries, all processes have finished the loop.
```