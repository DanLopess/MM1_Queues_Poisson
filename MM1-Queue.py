import numpy as np
from typing import List
from queue import Queue
import enum
import math
import random
import matplotlib.pyplot as plt


class EventType(enum.Enum):
    arriving = 1
    processing_end = 2


class ServerState(enum.Enum):
    free = 1
    busy = 2


class Event:
    def __init__(self, t: float, ev_type: EventType):
        self.ev_type = ev_type
        self.t = t
        self.end_t = None
        self.q_in_t = None
        self.q_out_t = None

    def __str__(self):
        return f'Event({self.ev_type}, {self.t})'


def print_busy_time(s_time: float, end_time: float, process_times: List[float]):
    total_time = end_time - s_time
    time_busy = sum(process_times)
    busy_time = time_busy / total_time
    print(f'- Server was busy {busy_time * 100:.3f}% of the time.')


def get_random_variable(_lambda):
    u = random.random()
    return -(math.log(1 - u)) / _lambda


def theoreticalAvg(lamb, mu):
    ro = lamb / mu
    if lamb == mu:
        avg_num_packets_queue = "NA"
        avg_time_in_system = "NA"
        avg_time_in_queue = "NA"
    else:
        avg_num_packets_queue = ro / (1 - ro)
        avg_time_in_system = 1 / (mu - lamb)
        avg_time_in_queue = lamb / (mu ** 2 - lamb * mu)
    print("- Theoretical average number of packets in M/M/1 queue for is: ", str(avg_num_packets_queue),
          "\n- Theoretical average time in M/M/1 system for is: ", str(avg_time_in_system) +
          "\n- Theoretical average time in M/M/1 queue for is: ", str(avg_time_in_queue))


# For avg_num_packets_queue = 1 => 1 = ro / (1-r0) <=> 1 - ro = ro <=> ro = 1/2 => lamb = 1 and mu = 2
# For avg_num_packets_queue = 10 => 10 = ro / (1-r0) <=> 10 - 10ro = ro <=> ro = 10/11 => lamb = 100 and mu = 110
# For avg_num_packets_queue = 100 => 100 = ro / (1-r0) <=> 100 - 100ro = ro <=> ro = 100/101 => lamb = 100 and mu = 101
# For avg_num_packets_queue = 1000 => 1000 = ro / (1-r0) <=> 1000 - 1000ro = ro <=> ro = 1000/1001 => lamb = 1000 and mu = 1001

def MM1(event_limit, _lambda, _mu):
    server_state: ServerState = ServerState.free
    start_time = 0
    time = 0

    server_queue: Queue[Event] = Queue()
    server_queue_sizes: List[int] = []
    time_in_queue: List[float] = []
    time_in_system: List[float] = []
    processing_times: List[float] = []

    # generate single event to bootstrap process
    event_list: List[Event] = [Event(t=0, ev_type=EventType.arriving)]
    events_completed = 0

    while event_limit > events_completed >= 0:
        event_list.sort(key=lambda x: x.t)
        event_to_process: Event = event_list.pop(0)
        current_time = event_to_process.t

        if event_to_process.ev_type == EventType.arriving:
            server_queue.put(event_to_process)
            server_queue_sizes.append(server_queue.qsize())

            t1 = current_time + get_random_variable(_lambda)
            new_event: Event = Event(t=t1, ev_type=EventType.arriving)
            event_list.append(new_event)

        elif event_to_process.ev_type == EventType.processing_end:
            server_state = ServerState.free

        if server_state == ServerState.free and server_queue.qsize() > 0:
            server_state = ServerState.busy
            event_from_queue: Event = server_queue.get()
            server_queue_sizes.append(server_queue.qsize())
            time_in_queue.append(current_time - event_from_queue.t)

            t1 = current_time + get_random_variable(_mu)
            new_event: Event = Event(t=t1, ev_type=EventType.processing_end)
            processing_times.append(t1 - current_time)
            time_in_system.append(t1 - event_from_queue.t)
            event_list.append(new_event)

            events_completed += 1
    time = current_time
    plt.plot(server_queue_sizes)
    plt.title(f'Queue size (λ = {_lambda} and μ = {_mu})')
    plt.savefig(f'./images/3/queue_size-lamb-{_lambda}-mu-{_mu}-events-{event_limit}.png')
    plt.show()
    plt.title(f'Time in queue (λ = {_lambda} and μ = {_mu})')
    plt.plot(time_in_queue)
    plt.savefig(f'./images/3/time_queue-lamb-{_lambda}-mu-{_mu}-events-{event_limit}.png')
    plt.show()
    plt.title(f'Time in system (λ = {_lambda} and μ = {_mu})')
    plt.plot(time_in_system)
    plt.savefig(f'./images/3/time_system-lamb-{_lambda}-mu-{_mu}-events-{event_limit}.png')
    plt.show()
    print("\nFor 𝜆 =", _lambda, ", 𝜇 =", _mu, "and", event_limit, "events:")
    theoreticalAvg(_lambda, _mu)
    print("- Experimental average number of packets in M/M/1 queue is: ", np.mean(server_queue_sizes),
          "\n- Experimental average time in M/M/1 system is: ", np.mean(time_in_system),
          "\n- Experimental average time in M/M/1 queue is: ", np.mean(time_in_queue))
    print_busy_time(start_time, time, processing_times)


if __name__ == '__main__':
    # 𝜇 < 𝜆
    MM1(10000, 100, 50)
    # 𝜇 = 𝜆
    MM1(10000, 50, 50)
    # 𝜇 > 𝜆; For avg_num_packets_queue = 1 => 1 = ro / (1-r0) <=> 1 - ro = ro <=> ro = 1/2 => lamb = 50 and mu = 100
    MM1(10000, 50, 100)
    # For avg_num_packets_queue = 10 => 10 = ro / (1-r0) <=> 10 - 10ro = ro <=> ro = 10/11 => lamb = 100 and mu = 110
    MM1(10000, 100, 110)
    # For avg_num_packets_queue = 100 and 100 generated events
    MM1(100, 100, 101)
    # For avg_num_packets_queue = 1000 and 1 000 generated events
    MM1(1000, 100, 101)
    # For avg_num_packets_queue = 100 and 10 000 generated events => 100 = ro / (1-r0) <=> 100 - 100ro = ro <=> ro = 100/101 => lamb = 100 and mu = 101
    MM1(10000, 100, 101)
    # For avg_num_packets_queue = 1000 => 1000 = ro / (1-r0) <=> 1000 - 1000ro = ro <=> ro = 1000/1001 => lamb = 1000 and mu = 1001
    MM1(10000, 1000, 1001)
