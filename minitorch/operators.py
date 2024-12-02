"""Collection of the core mathematical operators used throughout the code base."""

import math
from typing import Callable, Iterable

# ## Task 0.1
#
# Implementation of a prelude of elementary functions.


# Mathematical functions:
# - mul
# - id
# - add
# - neg
# - lt
# - eq
# - max
# - is_close
# - sigmoid
# - relu
# - log
# - exp
# - log_back
# - inv
# - inv_back
# - relu_back
#
# For sigmoid calculate as:
# $f(x) =  \frac{1.0}{(1.0 + e^{-x})}$ if x >=0 else $\frac{e^x}{(1.0 + e^{x})}$
# For is_close:
# $f(x) = |x - y| < 1e-2$


# TODO: Implement for Task 0.1.


# - mul
def mul(x: float, y: float) -> float:
    """Multiplies two numbers
    $f(x, y) = x * y$
    """
    return x * y


# - id
def id(x: float) -> float:
    """Returns the input unchanged
    $f(x) = x$
    """
    return x


# - add
def add(x: float, y: float) -> float:
    """Adds two numbers
    $f(x, y) = x + y$
    """
    return x + y


# - neg
def neg(x: float) -> float:
    """Negates a number
    $f(x) = -x$
    """
    return -x


# - lt
def lt(x: float, y: float) -> float:
    """Checks if one number is less than another
    $f(x) =$ 1.0 if x is less than y else 0.0
    """
    return 1.0 if x < y else 0.0


# - eq
def eq(x: float, y: float) -> float:
    """Checks if two numbers are equal
    $f(x) =$ 1.0 if x is equal to y else 0.0
    """
    return 1.0 if x == y else 0.0


# - max
def max(x: float, y: float) -> float:
    """Returns the larger of two numbers
    $f(x) =$ x if x is greater than y else y
    """
    return x if x > y else y


# - is_close
def is_close(x: float, y: float) -> bool:
    """Checks if two numbers are close

    For is_close:
    $f(x) = |x - y| < 1e-2$
    """
    return ((x - y) < 1e-2) and ((y - x) < 1e-2)


# - sigmoid
def sigmoid(x: float) -> float:
    r"""Calculates the sigmoid function

    For sigmoid calculate as:
    $f(x) =  \frac{1.0}{(1.0 + e^{-x})}$ if x >=0 else $\frac{e^x}{(1.0 + e^{x})}$
    (See https://en.wikipedia.org/wiki/Sigmoid_function )
    for stability
    """
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    else:
        return math.exp(x) / (1.0 + math.exp(x))


# - relu
def relu(x: float) -> float:
    """Applies the ReLU activation function

    $f(x) =$ x if x is greater than 0, else 0
    (See https://en.wikipedia.org/wiki/Rectifier_(neural_networks) )
    """
    return x if x > 0 else 0.0


# - relu_back
def relu_back(x: float, d: float) -> float:
    r"""Computes the derivative of ReLU times a second arg

    If $f = relu$ compute $d \times f'(x)$
    """
    return d if x > 0 else 0.0


EPS = 1e-6


# - log
def log(x: float) -> float:
    """Calculates the natural logarithm
    $f(x) = log(x)$
    """
    return math.log(x + EPS)


# - log_back
def log_back(x: float, d: float) -> float:
    r"""Computes the derivative of log times a second arg

    If $f = log$ as above, compute $d \times f'(x)$
    """
    return d / (x + EPS)


# - exp
def exp(x: float) -> float:
    """Calculates the exponential function
    $f(x) = e^{x}$
    """
    return math.exp(x)


# - inv
def inv(x: float) -> float:
    """Calculates the reciprocal
    $f(x) = 1/x$
    """
    return 1.0 / x


# - inv_back
def inv_back(x: float, d: float) -> float:
    r"""Computes the derivative of reciprocal times a second arg

    If $f(x) = 1/x$ compute $d \times f'(x)$
    """
    return -(1.0 / x**2) * d


# ## Task 0.3

# Small practice library of elementary higher-order functions.

# Implement the following core functions
# - map: Higher-order function that applies a given function to each element of an iterable
# - zipWith: Higher-order function that combines elements from two iterables using a given function
# - reduce: Higher-order function that reduces an iterable to a single value using a given function
#
# Use these to implement
# - negList : negate a list
# - addLists : add two lists together
# - sum: sum lists
# - prod: take the product of lists


# TODO: Implement for Task 0.3.


def map(fn: Callable[[float], float]) -> Callable[[Iterable[float]], Iterable[float]]:
    """Higher-order map.

    See https://en.wikipedia.org/wiki/Map_(higher-order_function)

    Args:
    ----
        fn: Function from one value to one value.

    Returns:
    -------
        A function that takes a list, applies `fn` to each element, and returns a
         new list

    """

    def _map(ls: Iterable[float]) -> Iterable[float]:
        ret = []
        for x in ls:
            ret.append(fn(x))
        return ret

    return _map


def zipWith(
    fn: Callable[[float, float], float],
) -> Callable[[Iterable[float], Iterable[float]], Iterable[float]]:
    """Higher-order zipwith (or map2).

    See https://en.wikipedia.org/wiki/Map_(higher-order_function)

    Args:
    ----
        fn: combine two values

    Returns:
    -------
        Function that takes two equally sized lists `ls1` and `ls2`, produce a new list by
        applying fn(x, y) on each pair of elements.

    """

    def _zipWith(ls1: Iterable[float], ls2: Iterable[float]) -> Iterable[float]:
        ret = []
        for x, y in zip(ls1, ls2):
            ret.append(fn(x, y))
        return ret

    return _zipWith


def reduce(
    fn: Callable[[float, float], float], start: float
) -> Callable[[Iterable[float]], float]:
    """Higher-order reduce.

    Args:
    ----
        fn: combine two values
        start: start value $x_0$

    Returns:
    -------
        Function that takes a list `ls` of elements
        $x_1 ... x_n$ and computes the reduction :math:`fn(x_3, fn(x_2,
        fn(x_1, x_0)))`

    """

    def _reduce(ls: Iterable[float]) -> float:
        val = start
        for l in ls:
            val = fn(val, l)
        return val

    return _reduce


# Higher-order functions: Using nested functions for mathematical operations that work on lists
def negList(ls: Iterable[float]) -> Iterable[float]:
    """Use `map` and `neg` to negate each element in `ls`"""
    return map(neg)(ls)


def addLists(ls1: Iterable[float], ls2: Iterable[float]) -> Iterable[float]:
    """Add the elements of `ls1` and `ls2` using `zipWith` and `add`"""
    return zipWith(add)(ls1, ls2)


def sum(ls: Iterable[float]) -> float:
    """Sum up a list using `reduce` and `add`."""
    return reduce(add, 0.0)(ls)


def prod(ls: Iterable[float]) -> float:
    """Product of a list using `reduce` and `mul`."""
    return reduce(mul, 1.0)(ls)
