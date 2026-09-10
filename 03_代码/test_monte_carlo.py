"""Regression tests for the five production Monte Carlo p-value expressions."""
import ast
from pathlib import Path
import unittest
import numpy as np


class MonteCarloTests(unittest.TestCase):
    def test_all_production_expressions(self):
        expressions = []
        for p in Path(__file__).parent.rglob('*.py'):
            if p == Path(__file__):
                continue
            for node in ast.walk(ast.parse(p.read_text(encoding='utf-8-sig'))):
                if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'p_perm' for t in node.targets):
                    expressions.append((p, compile(ast.Expression(node.value), str(p), 'eval')))
        self.assertEqual(len(expressions), 5)
        cases = [
            ([0., .1, -.1], 1., .25),  # no extreme samples: strictly positive
            ([1., -1., 0.], 1., .75),  # both tails and equality included
            ([0., 0., 0.], 0., 1.),    # identical methods: p=1
            ([2., -2., 1.], 1., 1.),   # all samples extreme
        ]
        for path, expression in expressions:
            for values, observed, expected in cases:
                with self.subTest(file=path.name, values=values):
                    actual = eval(expression, {'np': np, 'perm_means': np.array(values), 'obs_mean': observed})
                    self.assertAlmostEqual(actual, expected)


if __name__ == '__main__':
    unittest.main()
