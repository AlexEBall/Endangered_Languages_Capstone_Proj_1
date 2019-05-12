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

# Relationship between Binomial and Poisson distributions
# You just heard that the Poisson distribution is a limit of the Binomial distribution for rare events. 
# This makes sense if you think about the stories. Say we do a Bernoulli trial every minute for an hour, 
# each with a success probability of 0.1. We would do 60 trials, and the number of successes is Binomially distributed, 
# and we would expect to get about 6 successes. This is just like the Poisson story we discussed in the video, 
# where we get on average 6 hits on a website per hour. So, the Poisson distribution with arrival rate equal 
# to np approximates a Binomial distribution for n Bernoulli trials with probability p of success(with n large and p small). 
# Importantly, the Poisson distribution is often simpler to work with because it has only one parameter instead of two 
# for the Binomial distribution.

# Let's explore these two distributions computationally. You will compute the mean and standard deviation of samples 
# from a Poisson distribution with an arrival rate of 10. Then, you will compute the mean and standard deviation of 
# samples from a Binomial distribution with parameters n and p such that np = 10.

# Draw 10,000 samples out of Poisson distribution: samples_poisson
samples_poisson = np.random.poisson(10, size=10000)

# Print the mean and standard deviation
print('Poisson:     ', np.mean(samples_poisson),
      np.std(samples_poisson))

# Specify values of n and p to consider for Binomial: n, p
n = [20, 100, 1000]
p = [0.5, 0.1, 0.01]

# Draw 10,000 samples for each n,p pair: samples_binomial
for i in range(3):
    samples_binomial = np.random.binomial(n[i], p[i], size=10000)

    # Print results
    print('n =', n[i], 'Binom:', np.mean(samples_binomial),
          np.std(samples_binomial))

# Was 2015 anomalous?
# 1990 and 2015 featured the most no-hitters of any season of baseball(there were seven).
# Given that there are on average 251/115 no-hitters per season, what is the probability of 
# having seven or more in a season?

# Draw 10,000 samples out of Poisson distribution: n_nohitters
n_nohitters = np.random.poisson(251/115, size=10000)

# Compute number of samples that are seven or greater: n_large
n_large = np.sum(n_nohitters >= 7)

# Compute probability of getting seven or more: p_large
p_large = n_large/10000

# Print the result
print('Probability of seven or more no-hitters:', p_large)


# PROBABLITY DENSITY FUNCTIONS 

# The Normal PDF
# In this exercise, you will explore the Normal PDF and also learn a way to plot a PDF of a known 
# distribution using hacker statistics. Specifically, you will plot a Normal PDF for various values of the variance.

# Draw 100000 samples from Normal distribution with stds of interest: samples_std1, samples_std3, samples_std10
samples_std1 = np.random.normal(20, 1, size=100000)
samples_std3 = np.random.normal(20, 3, size=100000)
samples_std10 = np.random.normal(20, 10, size=100000)

# Make histograms
_ = plt.hist(samples_std1, bins=100, normed=True, histtype='step')
_ = plt.hist(samples_std3, bins=100, normed=True, histtype='step')
_ = plt.hist(samples_std10, bins=100, normed=True, histtype='step')

# Make a legend, set limits and show plot
_ = plt.legend(('std = 1', 'std = 3', 'std = 10'))
plt.ylim(-0.01, 0.42)
plt.show()

# Generate CDFs
x_std1, y_std1 = ecdf(samples_std1)
x_std3, y_std3 = ecdf(samples_std3)
x_std10, y_std10 = ecdf(samples_std10)

# Plot CDFs
_ = plt.plot(x_std1, y_std1, marker='.', linestyle='none')
_ = plt.plot(x_std3, y_std3, marker='.', linestyle='none')
_ = plt.plot(x_std10, y_std10, marker='.', linestyle='none')

# Make a legend and show the plot
_ = plt.legend(('std = 1', 'std = 3', 'std = 10'), loc='lower right')
plt.show()

# Are the Belmont Stakes results Normally distributed?
# Since 1926, the Belmont Stakes is a 1.5 mile-long race of 3-year old thoroughbred horses. 
# Secretariat ran the fastest Belmont Stakes in history in 1973. While that was the fastest year, 
# 1970 was the slowest because of unusually wet and sloppy conditions. With these two outliers removed 
# from the data set, compute the mean and standard deviation of the Belmont winners' times. 
# Sample out of a Normal distribution with this mean and standard deviation using the np.random.normal() 
# function and plot a CDF. Overlay the ECDF from the winning Belmont times. Are these close to Normally distributed?

# Note: Justin scraped the data concerning the Belmont Stakes from the Belmont Wikipedia page.

# Compute mean and standard deviation: mu, sigma
mu = np.mean(belmont_no_outliers)
sigma = np.std(belmont_no_outliers)

# Sample out of a normal distribution with this mu and sigma: samples
samples = np.random.normal(mu, sigma, size=10000)

# Get the CDF of the samples and of the data
x_theor, y_theor = ecdf(samples)
x, y = ecdf(belmont_no_outliers)

# Plot the CDFs and show the plot
_ = plt.plot(x_theor, y_theor)
_ = plt.plot(x, y, marker='.', linestyle='none')
_ = plt.xlabel('Belmont winning time (sec.)')
_ = plt.ylabel('CDF')
plt.show()

# What are the chances of a horse matching or beating Secretariat's record?
# Assume that the Belmont winners' times are Normally distributed(with the 1970 and 1973 years removed), 
# what is the probability that the winner of a given Belmont Stakes will run it as fast or faster than Secretariat?

# Take a million samples out of the Normal distribution: samples
samples = np.random.normal(mu, sigma, size=1000000)

# Compute the fraction that are faster than 144 seconds: prob
prob = np.sum(samples <= 144) / len(samples)

# Print the result
print('Probability of besting Secretariat:', prob)


# EXPONENTIAL DISTRIBUTION 

# If you have a story, you can simulate it!
# Sometimes, the story describing our probability distribution does not have a named distribution to go along with it. 
# In these cases, fear not! You can always simulate it. We'll do that in this and the next exercise.

# In earlier exercises, we looked at the rare event of no-hitters in Major League Baseball. 
# Hitting the cycle is another rare baseball event. When a batter hits the cycle, he gets all four kinds of hits, 
# a single, double, triple, and home run, in a single game. Like no-hitters, this can be modeled as a Poisson process, 
# so the time between hits of the cycle are also Exponentially distributed.

# How long must we wait to see both a no-hitter and then a batter hit the cycle? The idea is that we have to 
# wait some time for the no-hitter, and then after the no-hitter, we have to wait for hitting the cycle. 
# Stated another way, what is the total waiting time for the arrival of two different Poisson processes? 
# The total waiting time is the time waited for the no-hitter, plus the time waited for the hitting the cycle.

# Now, you will write a function to sample out of the distribution described by this story.

def successive_poisson(tau1, tau2, size=1):
    """Compute time for arrival of 2 successive Poisson processes."""
    # Draw samples out of first exponential distribution: t1
    t1 = np.random.exponential(tau1, size=size)

    # Draw samples out of second exponential distribution: t2
    t2 = np.random.exponential(tau2, size=size)

    return t1 + t2


# Distribution of no-hitters and cycles
# Now, you'll use your sampling function to compute the waiting time to observe a no-hitter and 
# hitting of the cycle. The mean waiting time for a no-hitter is 764 games, and the mean waiting time 
# for hitting the cycle is 715 games.

# Draw samples of waiting times: waiting_times
waiting_times = successive_poisson(764, 715, size=100000)

# Make the histogram
_ = plt.hist(waiting_times, normed=True, bins=100, histtype='step')


# Label axes
_ = plt.xlabel('waiting time')
_ = plt.ylabel('cdf')

# Show the plot
plt.show()
