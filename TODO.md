# FIX
* `improve`

* algorithms appear more flat when they log more near neighbors into the population, so you can make the population look `better` by the metric simply by capturing more nearby swaps into the population distribution

# Genetic algorithm improvement ideas
* Consider multi-child generations, not just 1 child per loop.
* annealing for the most promising children some times
* Add some diversity preservation (duplicate filtering or crowding)


# Fixes
* annealing as default, clean up code
* annealing, is it possible to make better use of the temperature setup?
* is there a better way to do genetic?
* fix the failing test_optim
* improve a design (vs generate), fix _optimize


# Features
* jalo analyze to show high freq ngram for each metric
* jalo `show` to just visualize the design
* jalo KEPL memory management for many layouts


* keygen metrics
* carpalx metrics
* corpus
* shift and delete

* README
* LICENSE
