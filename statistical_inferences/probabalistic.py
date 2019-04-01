# random number generators an hacker statistics 

# BERNOULLI FUNCTION 
def perform_bernoulli_trials(n, p):
    """Perform n Bernoulli trials with success probability p
    and return number of successes."""
    # Initialize number of successes: n_success
    n_success = 0

    # Perform trials
    for i in range(n):
        # Choose random number between zero and one: random_number
        random_number = np.random.random()

        # If less than p, it's a success so add one to n_success
        if random_number < p:
            n_success += 1

    return n_success

# How many defaults might we expect?
# Let's say a bank made 100 mortgage loans. It is possible that anywhere between 0 and 100 of the loans will be defaulted upon. 
# You would like to know the probability of getting a given number of defaults, 
# given that the probability of a default is p = 0.05. To investigate this, you will do a simulation. 
# You will perform 100 Bernoulli trials using the perform_bernoulli_trials() function you wrote in the previous exercise 
# and record how many defaults we get. Here, a success is a default. 
# (Remember that the word "success" just means that the Bernoulli trial evaluates to True, i.e., did the loan recipient default?) 
# You will do this for another 100 Bernoulli trials. And again and again until we have tried it 1000 times. 
# Then, you will plot a histogram describing the probability of the number of defaults.

# Seed random number generator
np.random.seed(42)

# Initialize the number of defaults: n_defaults
n_defaults = np.empty(1000)

# Compute the number of defaults
for i in range(1000):
    n_defaults[i] = perform_bernoulli_trials(100, 0.05)


# Plot the histogram with default number of bins; label your axes
_ = plt.hist(n_defaults, normed=True)
_ = plt.xlabel('number of defaults out of 100 loans')
_ = plt.ylabel('probability')

# Show the plot
plt.show()

# BINOMIAL DISTRIBUTION STORY

# Sampling out of the Binomial distribution
# Compute the probability mass function for the number of defaults we would expect for 100 loans as in the last section, 
# but instead of simulating all of the Bernoulli trials, perform the sampling using np.random.binomial(). 
# This is identical to the calculation you did in the last set of exercises using your custom-written perform_bernoulli_trials() 
# function, but far more computationally efficient. Given this extra efficiency, we will take 10, 000 samples instead of 1000. 
# After taking the samples, plot the CDF as last time. This CDF that you are plotting is that of the Binomial distribution.

# Note: For this exercise and all going forward, the random number generator is 
# pre-seeded for you(with np.random.seed(42)) to save you typing that each time.

# Take 10,000 samples out of the binomial distribution: n_defaults
n_defaults = np.random.binomial(100, 0.05, size=10000)

# Compute CDF: x, y
x, y = ecdf(n_defaults)

# Plot the CDF with axis labels
_ = plt.plot(x, y, marker='.', linestyle='none')
_ = plt.xlabel('Number of defaults out of 100 loans')
_ = plt.ylabel('CDF')

# Show the plot
plt.show()

# creating a histogram out of that
# Compute bin edges: bins
bins = np.arange(0, max(n_defaults) + 1.5) - 0.5

# Generate histogram
_ = plt.hist(n_defaults, normed=True, bins=bins)

# Label axes
_ = plt.xlabel('Number of deafults out of 100')
_ = plt.ylabel('something')

# Show the plot
plt.show()


# POISSON DISTRIBUTION AND STORY 
