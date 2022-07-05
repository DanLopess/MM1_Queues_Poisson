import math as m
import numpy as np
import matplotlib.pyplot as plt


# -------------------------------- Ex. 1 --------------------------------

# 1. Generate a random number u with a uniform distribution on [0,1).
def gen_uniform():
    return np.random.uniform(0, 1)


# 2. Apply the transformation
# 𝛿𝑡 = − log(1 − 𝑢)/𝜆
def exp_distribution(u, l):
    return -m.log(1 - u) / l


def run_exercise_2_1(lamb1=2, lamb2=5, lamb3=9):
    u = gen_uniform()
    print("Generated number: ", u)
    print("For 𝜆 =", lamb1, ": ", exp_distribution(u, lamb1))
    print("For 𝜆 =", lamb2, ": ", exp_distribution(u, lamb2))
    print("For 𝜆 =", lamb3, ": ", exp_distribution(u, lamb3))


# -------------------------------- Ex. 2 --------------------------------
# 1. Generate a sequence of N=70 events with time interval dt between successive events.
#   Generate dt according to an exponential distribution of parameter l. Make l=5 for initial tests

def gen_events(l, n_events):
    events = []
    for x in range(n_events):
        u = gen_uniform()
        interval = -m.log(1 - u) / l
        if x == 0:
            events.append(interval)
        else:
            events.append(events[-1] + interval)
    return events


# 2. Use the above sequence to make a histogram (bar chart) of the number of events occurring in a unitary time
# interval. The use of the Matlab / Octave hist() function is forbidden, develop your own version! x coordinate is
# range(70) = k values

# For a given time range (e.g. 0 - 12), this will count how many events
# happened in each unit of time
def get_event_count(unitary_times, events):
    counts = []
    big_count = 0
    for time in unitary_times:
        count = 0
        for event in events:
            if round(event) == time:
                count += 1
        counts.append(count)
        big_count += count
    return counts


# To obtain a histogram similar to poisson probabilities, we need to look at single unit of times
# and how many events occurred in that time. Then we can plot the distribution of how many events
# occur in a given time unit. (e.g. for k = 4 we usually get 5 events, for k = 5 we usually get 7 events)
def calculate_occurrences(events):
    unitary_times = np.arange(round(min(events)), round(max(events)) + 1, 1)
    event_counts = get_event_count(unitary_times, events)
    occurrences_dict = {}
    for count in event_counts:
        if not occurrences_dict.get(count):
            occurrences_dict[count] = 1
        else:
            occurrences_dict[count] += 1
    return occurrences_dict


# Since we have now the number of occurrences for each unit of time, we have now to calculate what is the probability
# of having an event in that unit. For that, we sum all the occurrences and divide each occurrence for that value in
# each unit of time.
def calculate_probabilities(occurrences):
    probs = {}
    total_occurrences = m.fsum(occurrences.values())
    for key, value in occurrences.items():
        probs[key] = value / total_occurrences
    return probs


def plot_experimental_poisson(probs, _lambda, exercise="2.2"):
    plt.bar(probs.keys(), probs.values())
    plt.xlabel('K value')
    plt.ylabel('Number of Occurrences')
    plt.title(f'Poisson Process (λ = {_lambda})')
    plt.savefig(f'./images/{exercise}/exp-lamb-{_lambda}-k-{len(probs.keys())}.png')
    plt.show()


# This function plots the poisson probabilities chart for a given k range and lambda

# 3. Use any plotting software (matlab, gnuplot, excel, ...) to display the above histogram against the theoretical
# Poisson distribution. The theoretical distribution and the experimental one (histogram) must be displayed in the
# same plot.

# generate Poisson distribution with sample size 70 and lambda=5
def theoretical_formula(lamb, k):
    return ((lamb ** k) / m.factorial(k)) * m.e ** (-lamb)


def get_events(lamb, k):
    x = []
    for j in range(k):
        x.append(theoretical_formula(lamb, j))
    return x


def calculate_theoretical_probs(_lambda, p_threshold=0.01, k_threshold=12):
    theoretical_probs = []
    k = 0
    while True:
        p = theoretical_formula(_lambda, k)
        theoretical_probs.append(p)
        k += 1
        if k > k_threshold and p <= p_threshold:
            break
    return theoretical_probs


def plot_theoretical_poisson(probs, _lambda):
    k = range(len(probs))
    plt.bar(k, probs)
    plt.xlabel('K value')
    plt.ylabel('Probability')
    plt.title(f'Theoretical Poisson Process (λ = {_lambda})')
    plt.savefig(f'./images/2.2/theo-lamb-{_lambda}-k-{len(k)}.png')
    plt.show()


def plot_theoretical_and_experimental_poisson(probs, theoretical_probs, _lambda):
    k = range(len(theoretical_probs))
    plt.bar(probs.keys(), probs.values(), label='Experimental')
    plt.scatter(k, theoretical_probs, color='orange')
    plt.plot(k, theoretical_probs, label='Theoretical', color='orange')
    plt.xlabel('K value')
    plt.ylabel('Probability')
    plt.title(f'Poisson Process (λ = {_lambda})')
    plt.legend()
    plt.savefig(f'./images/2.2/exp-theo-lamb-{_lambda}-k-{len(k)}.png')
    plt.show()


def run_exercise_2_2(_lambda=5, n_events=1000):
    generated_events = gen_events(_lambda, n_events)
    occur = calculate_occurrences(generated_events)
    probs = calculate_probabilities(occur)
    theoretical_probs = calculate_theoretical_probs(_lambda)
    plot_experimental_poisson(probs, _lambda)
    plot_theoretical_poisson(theoretical_probs, _lambda)
    plot_theoretical_and_experimental_poisson(probs, theoretical_probs, _lambda)


# -------------------------------- Ex. 3 --------------------------------

# 1. Modify the program above to generate 3 independent sequences with different lambda values;

# This function generates events not based on count but based on achieved time
# e.g. if the max time is 10seconds, it will generate N events until 10 seconds (no matter the rate of events)
def generate_timed_events(max_time, _lambda):
    events = []
    running_event_time = 0

    while running_event_time <= max_time:
        uniform = gen_uniform()
        time_interval = exp_distribution(uniform, _lambda)
        running_event_time = running_event_time + time_interval
        events.append(running_event_time)
    return events


# to be able to statistically obtain the superposition of all 3 sequences
# it is necessary to generate events for a long time (e.g. 1000secs) so that the
# statistic will approximate a poisson process
def run_exercise_2_3(max_time=1000, _lambda_1=2, _lambda_2=5, _lambda_3=10):
    generated_events_1 = generate_timed_events(max_time, _lambda_1)
    generated_events_2 = generate_timed_events(max_time, _lambda_2)
    generated_events_3 = generate_timed_events(max_time, _lambda_3)

    single_generated_events = np.concatenate((generated_events_1, generated_events_2, generated_events_3))
    occur = calculate_occurrences(single_generated_events)
    probs = calculate_probabilities(occur)
    _lambda = _lambda_1 + _lambda_2 + _lambda_3
    plot_experimental_poisson(probs, _lambda, exercise="2.3")


if __name__ == '__main__':
    # run_exercise_2_1()
    # run_exercise_2_2()
    run_exercise_2_3()

# When we thought that the theoretical distribution could be made with this already created function.

# x = poisson.rvs(mu=5, size=1000)

# create plot of Poisson distribution
# plt.hist(x,  density=True, edgecolor='black')
# plt.show()
