from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, List, Tuple, Protocol


# ## Task 1.1
# Central Difference calculation


def central_difference(f: Any, *vals: Any, arg: int = 0, epsilon: float = 1e-6) -> Any:
    r"""Computes an approximation to the derivative of `f` with respect to one arg.

    See :doc:`derivative` or https://en.wikipedia.org/wiki/Finite_difference for more details.

    Args:
    ----
        f : arbitrary function from n-scalar args to one value
        *vals : n-float values $x_0 \ldots x_{n-1}$. The data type of *vals is a tuple, where each element of the tuple is expected to be a floating-point value
        arg : the number $i$ of the arg to compute the derivative
        epsilon : a small constant

    Returns:
    -------
        An approximation of $f'_i(x_0, \ldots, x_{n-1})$

    """
    # TODO: Implement for Task 1.1.
    # Create a new list of values for the perturbed argument
    vals1 = [v for v in vals]
    vals2 = [v for v in vals]
    vals1[arg] = vals[arg] + epsilon
    vals2[arg] = vals[arg] - epsilon
    delta = f(*vals1) - f(*vals2)
    return delta / (2 * epsilon)


variable_count = 1


class Variable(Protocol):
    def accumulate_derivative(self, x: Any) -> None:
        """Accumulates the derivative of the variable."""
        pass

    @property
    def unique_id(self) -> int:
        """Returns a unique identifier for the variable."""
        ...

    def is_leaf(self) -> bool:
        """Checks if the variable is a leaf node in the computation graph."""
        ...

    def is_constant(self) -> bool:
        """Checks if the variable is a constant."""
        ...

    @property
    def parents(self) -> Iterable["Variable"]:
        """Returns the parent variables of the current variable."""
        ...

    def chain_rule(self, d_output: Any) -> Iterable[Tuple[Variable, Any]]:
        """Applies the chain rule to compute the gradients of the parent variables."""
        ...


def topological_sort(variable: Variable) -> Iterable[Variable]:
    """Computes the topological order of the computation graph.
    In neural networks and many other computational models that use gradients for optimization,
    the gradient calculation typically starts from the output and moves backward to the inputs.
    This process is known as backpropagation.
    Topological ordering is not unique.

    Args:
    ----
        variable: The right-most variable

    Returns:
    -------
        Iterable[Variable]: Non-constant Variables in topological order.

    """
    # TODO: Implement for Task 1.4.
    order: List[Variable] = []
    seen = set()

    def visit(var: Variable) -> None:
        """DFS Recursive helper function to visit each variable in the computation graph
        and add it to the topological order list.

        Args:
        ----
            var (Variable): The current variable to process.

        """
        # base case
        if var.unique_id in seen or var.is_constant():
            return
        if not var.is_leaf():
            for m in var.parents:
                if not m.is_constant():
                    visit(m)
        seen.add(var.unique_id)
        order.insert(0, var)

    visit(variable)
    return order


def backpropagate(variable: Variable, deriv: Any) -> None:
    """Runs backpropagation on the computation graph
    in order to compute derivatives for the leaf nodes.

    Args:
    ----
        variable: The right-most variable in the computation graph.
        deriv: The derivative of the output with respect to the variable.

    Returns:
    -------
        None. The function updates the derivative values of each leaf node through `accumulate_derivative`.

    """
    # TODO: Implement for Task 1.4.
    queue = topological_sort(variable)
    derivatives = {}
    derivatives[variable.unique_id] = deriv
    for var in queue:
        deriv = derivatives[var.unique_id]
        if var.is_leaf():
            var.accumulate_derivative(deriv)
        else:
            for v, d in var.chain_rule(deriv):
                if v.is_constant():
                    continue
                derivatives.setdefault(v.unique_id, 0.0)
                derivatives[v.unique_id] = derivatives[v.unique_id] + d


@dataclass
class Context:
    """Context class is used by `Function` to store information during the forward pass."""

    no_grad: bool = False
    saved_values: Tuple[Any, ...] = ()

    def save_for_backward(self, *values: Any) -> None:
        """Store the given `values` if they need to be used during backpropagation."""
        if self.no_grad:
            return
        self.saved_values = values

    @property
    def saved_tensors(self) -> Tuple[Any, ...]:
        """Returns the saved values from the forward pass."""
        return self.saved_values
