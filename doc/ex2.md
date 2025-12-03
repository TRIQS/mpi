@page ex2 Example 2: Use monitor to communicate errors

[TOC]

In this example, we show how to use the mpi::monitor class to communicate and process errors across a communicator.

@include ex2.cpp

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