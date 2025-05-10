import time
import random
import psutil
from typing import List, Tuple, Dict, Optional
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
from pysat.solvers import Glucose3


class CNF:
    def __init__(self, clauses: List[List[int]], num_vars: int):
        self.clauses = clauses
        self.num_vars = num_vars


class SATSolver:
    def __init__(self):
        self.stats = {
            'decisions': 0,
            'backtracks': 0,
            'time': 0.0,
            'memory': 0.0,
            'timeout': False,
            'resolvents': 0,
            'units': 0,
            'pures': 0
        }
        self.timeout = 30
        self.start_time = 0
        self.start_memory = 0

    def check_timeout(self) -> bool:
        if time.perf_counter() - self.start_time > self.timeout:
            self.stats['timeout'] = True
            return True
        return False

    def get_memory_usage(self):
        try:
            process = psutil.Process()
            return process.memory_info().rss / (1024 * 1024)  # Convert to MB
        except:
            return 0

    def update_memory_stats(self):
        current_mem = self.get_memory_usage()
        self.stats['memory'] = max(self.stats['memory'], current_mem - self.start_memory)

    def solve(self, cnf: CNF) -> Optional[Dict[int, bool]]:
        raise NotImplementedError

    @staticmethod
    def calculate_confidence(data):
        arr = np.array(data)
        return stats.t.interval(0.95, len(arr) - 1, loc=np.mean(arr), scale=stats.sem(arr))


class ResolutionSolver(SATSolver):
    def solve(self, cnf: CNF) -> Optional[Dict[int, bool]]:
        self.start_time = time.perf_counter()
        self.start_memory = self.get_memory_usage()
        clauses = set(tuple(sorted(clause)) for clause in cnf.clauses)
        new_clauses = set()
        seen_pairs = set()
        self.stats['resolvents'] = 0
        self.stats['timeout'] = False

        while True:
            if self.check_timeout():
                self.stats['time'] = time.perf_counter() - self.start_time
                self.update_memory_stats()
                return None

            if any(len(clause) == 0 for clause in clauses):
                self.stats['time'] = time.perf_counter() - self.start_time
                self.update_memory_stats()
                return {}

            clause_list = list(clauses)
            progress = False

            for i in range(len(clause_list)):
                for j in range(i + 1, len(clause_list)):
                    if self.check_timeout():
                        self.stats['time'] = time.perf_counter() - self.start_time
                        self.update_memory_stats()
                        return None

                    c1, c2 = clause_list[i], clause_list[j]

                    if (c1, c2) in seen_pairs or (c2, c1) in seen_pairs:
                        continue
                    seen_pairs.add((c1, c2))

                    for lit in c1:
                        if -lit in c2:
                            resolvent = tuple(sorted(
                                set(l for l in c1 if l != lit) |
                                set(l for l in c2 if l != -lit)
                            ))
                            self.stats['resolvents'] += 1

                            if len(resolvent) == 0:
                                self.stats['time'] = time.perf_counter() - self.start_time
                                self.update_memory_stats()
                                return {}

                            if resolvent not in clauses and resolvent not in new_clauses:
                                new_clauses.add(resolvent)
                                progress = True
                            break

                    self.update_memory_stats()

            if not progress:
                self.stats['time'] = time.perf_counter() - self.start_time
                self.update_memory_stats()
                return {1: True}

            clauses.update(new_clauses)
            new_clauses.clear()
            self.update_memory_stats()


class DPSolver(SATSolver):
    def solve(self, cnf: CNF) -> Optional[Dict[int, bool]]:
        self.start_time = time.perf_counter()
        self.start_memory = self.get_memory_usage()
        self.stats['decisions'] = 0
        self.stats['backtracks'] = 0
        self.stats['units'] = 0
        self.stats['pures'] = 0
        self.stats['timeout'] = False

        clauses = [frozenset(clause) for clause in cnf.clauses]
        assignment = {}

        result = self._dp(clauses, assignment)
        self.stats['time'] = time.perf_counter() - self.start_time
        self.update_memory_stats()
        return result

    def _dp(self, clauses: List[frozenset], assignment: Dict[int, bool]) -> Optional[Dict[int, bool]]:
        if self.check_timeout():
            return None

        self.update_memory_stats()

        unit_clauses = [c for c in clauses if len(c) == 1]
        while unit_clauses:
            self.stats['units'] += 1
            unit = next(iter(unit_clauses[0]))
            var = abs(unit)
            val = unit > 0

            if var in assignment and assignment[var] != val:
                self.stats['backtracks'] += 1
                return {}

            assignment[var] = val
            new_clauses = []
            for clause in clauses:
                if unit in clause:
                    continue
                new_clause = clause.difference({-unit})
                if not new_clause:
                    self.stats['backtracks'] += 1
                    return {}
                new_clauses.append(new_clause)

            clauses = new_clauses
            unit_clauses = [c for c in clauses if len(c) == 1]
            self.update_memory_stats()

        if not clauses:
            return assignment

        all_literals = {lit for clause in clauses for lit in clause}
        pure_literals = {lit for lit in all_literals if -lit not in all_literals}
        if pure_literals:
            self.stats['pures'] += len(pure_literals)
            new_clauses = []
            for clause in clauses:
                if not any(lit in pure_literals for lit in clause):
                    new_clauses.append(clause)

            for lit in pure_literals:
                var = abs(lit)
                assignment[var] = lit > 0

            return self._dp(new_clauses, assignment)

        var_counts = {}
        for clause in clauses:
            for lit in clause:
                var = abs(lit)
                var_counts[var] = var_counts.get(var, 0) + 1

        if not var_counts:
            return assignment

        var = max(var_counts.items(), key=lambda x: x[1])[0]
        self.stats['decisions'] += 1

        true_result = self._dp([c.difference({-var}) for c in clauses if var not in c],
                               {**assignment, var: True})
        if true_result is not None and true_result != {}:
            return true_result

        self.stats['backtracks'] += 1

        false_result = self._dp([c.difference({var}) for c in clauses if -var not in c],
                                {**assignment, var: False})
        if false_result is not None and false_result != {}:
            return false_result

        self.stats['backtracks'] += 1
        return {}


class DPLLSolver(SATSolver):
    def __init__(self, heuristic='random'):
        super().__init__()
        self.heuristic = heuristic
        self.timeout = 30

    def solve(self, cnf: CNF) -> Optional[Dict[int, bool]]:
        self.start_time = time.perf_counter()
        self.start_memory = self.get_memory_usage()
        self.stats = {
            'decisions': 0,
            'backtracks': 0,
            'time': 0.0,
            'memory': 0.0,
            'timeout': False,
            'units': 0,
            'pures': 0
        }
        result = self._dpll(cnf.clauses, {})
        self.stats['time'] = time.perf_counter() - self.start_time
        self.update_memory_stats()
        return result

    def _unit_propagate(self, clauses: List[List[int]], assignment: Dict[int, bool]) -> Tuple[
        List[List[int]], Dict[int, bool]]:
        changed = True
        new_clauses = [clause.copy() for clause in clauses]

        while changed:
            changed = False
            unit_clauses = [c for c in new_clauses if len(c) == 1]

            if not unit_clauses:
                break

            lit = unit_clauses[0][0]
            var = abs(lit)
            val = lit > 0

            if var in assignment and assignment[var] != val:
                return [[]], assignment

            if var not in assignment:
                assignment[var] = val
                changed = True
                self.stats['units'] += 1

            simplified_clauses = []
            for clause in new_clauses:
                if any((lit > 0 and assignment.get(abs(lit), False)) or
                       (lit < 0 and not assignment.get(abs(lit), True)) for lit in clause):
                    continue

                new_clause = [l for l in clause if abs(l) != var or (l > 0) == val]
                if not new_clause:
                    return [[]], assignment
                simplified_clauses.append(new_clause)

            new_clauses = simplified_clauses
            self.update_memory_stats()

        return new_clauses, assignment

    def _choose_literal(self, clauses: List[List[int]]) -> int:
        if not clauses:
            return 0

        all_vars = set(abs(lit) for clause in clauses for lit in clause)
        if not all_vars:
            return 0

        if self.heuristic == 'random':
            return clauses[0][0] if clauses[0] else 0
        elif self.heuristic == 'moms':
            min_len = min((len(c) for c in clauses if c), default=0)
            if min_len == 0:
                return 0

            min_clauses = [c for c in clauses if len(c) == min_len]
            lit_counts = {}
            for clause in min_clauses:
                for lit in clause:
                    lit_counts[lit] = lit_counts.get(lit, 0) + 1

            return max(lit_counts.items(), key=lambda x: x[1])[0] if lit_counts else (
                clauses[0][0] if clauses[0] else 0)
        elif self.heuristic == 'jw':
            j_scores = {}
            for clause in clauses:
                weight = 2 ** -len(clause)
                for lit in clause:
                    j_scores[lit] = j_scores.get(lit, 0) + weight

            return max(j_scores.items(), key=lambda x: x[1])[0] if j_scores else (clauses[0][0] if clauses[0] else 0)
        else:
            raise ValueError(f"Unknown heuristic: {self.heuristic}")

    def _dpll(self, clauses: List[List[int]], assignment: Dict[int, bool]) -> Optional[Dict[int, bool]]:
        if self.check_timeout():
            return None

        self.update_memory_stats()

        clauses, assignment = self._unit_propagate(clauses, assignment)

        if any(len(c) == 0 for c in clauses):
            self.stats['backtracks'] += 1
            return {}

        if not clauses:
            return assignment

        lit = self._choose_literal(clauses)
        if lit == 0:
            self.stats['backtracks'] += 1
            return {}

        self.stats['decisions'] += 1
        var = abs(lit)

        try_first_value = lit > 0

        first_assignment_clauses = []
        for clause in clauses:
            if (lit if try_first_value else -lit) in clause:
                continue
            new_clause = [l for l in clause if l != (-lit if try_first_value else lit)]
            if not new_clause:
                break
            first_assignment_clauses.append(new_clause)
        else:
            first_result = self._dpll(first_assignment_clauses, {**assignment, var: try_first_value})
            if first_result is not None and first_result != {}:
                return first_result

        self.stats['backtracks'] += 1

        second_assignment_clauses = []
        for clause in clauses:
            if (-lit if try_first_value else lit) in clause:
                continue
            new_clause = [l for l in clause if l != (lit if try_first_value else -lit)]
            if not new_clause:
                self.stats['backtracks'] += 1
                return {}
            second_assignment_clauses.append(new_clause)

        second_result = self._dpll(second_assignment_clauses, {**assignment, var: not try_first_value})
        if second_result is not None and second_result != {}:
            return second_result

        self.stats['backtracks'] += 1
        return {}  # UNSAT


class PySATWrapper(SATSolver):

    def solve(self, cnf: CNF) -> Optional[Dict[int, bool]]:
        self.start_time = time.perf_counter()
        self.start_memory = self.get_memory_usage()
        self.stats['timeout'] = False

        try:
            with Glucose3(bootstrap_with=cnf.clauses) as solver:
                model = solver.solve()
                self.stats['time'] = time.perf_counter() - self.start_time
                self.update_memory_stats()

                if model:  # SAT
                    # We're just indicating satisfiability here
                    return {1: True}
                else:  # UNSAT
                    return {}
        except Exception as e:
            print(f"PySAT error: {e}")
            self.stats['time'] = time.perf_counter() - self.start_time
            self.update_memory_stats()
            return None


def generate_assembly_problem(num_tasks: int, num_workstations: int, seed=None) -> CNF:
    if seed is not None:
        random.seed(seed)

    clauses = []
    var_map = {}
    var_idx = 1

    for task in range(1, num_tasks + 1):
        for ws in range(1, num_workstations + 1):
            var_map[(task, ws)] = var_idx
            var_idx += 1

    for task in range(1, num_tasks + 1):
        clauses.append([var_map[(task, ws)] for ws in range(1, num_workstations + 1)])
        for ws1 in range(1, num_workstations + 1):
            for ws2 in range(ws1 + 1, num_workstations + 1):
                clauses.append([-var_map[(task, ws1)], -var_map[(task, ws2)]])

    conflict_density = 0.3
    for t1 in range(1, num_tasks + 1):
        for t2 in range(t1 + 1, num_tasks + 1):
            if random.random() < conflict_density:
                ws = random.randint(1, num_workstations)
                clauses.append([-var_map[(t1, ws)], -var_map[(t2, ws)]])

    for ws in range(1, num_workstations + 1):
        max_tasks_per_ws = num_tasks // 2
        if max_tasks_per_ws >= 3:
            all_tasks = list(range(1, num_tasks + 1))
            for _ in range(min(5, num_tasks)):
                task_subset = random.sample(all_tasks, max_tasks_per_ws + 1)
                clauses.append([-var_map[(task, ws)] for task in task_subset])

    return CNF(clauses, var_idx - 1)


def generate_random_3sat(num_vars: int, num_clauses: int, seed=None) -> CNF:
    if seed is not None:
        random.seed(seed)

    clauses = []
    for _ in range(num_clauses):
        clause = []
        vars_in_clause = random.sample(range(1, num_vars + 1), 3)
        for var in vars_in_clause:
            clause.append(var if random.random() > 0.5 else -var)
        clauses.append(clause)
    return CNF(clauses, num_vars)


def test_backtracking():
    clauses = [
        [1, 2],
        [1, -2],
        [-1, 2],
        [-1, -2]
    ]
    cnf = CNF(clauses, 2)

    print("\nTesting backtracking with unsatisfiable problem:")
    for name, solver in [('DPLL', DPLLSolver()), ('DP', DPSolver())]:
        result = solver.solve(cnf)
        print(f"{name}: Result={'UNSAT' if result == {} else 'SAT' if result else 'TIMEOUT'}, "
              f"Backtracks={solver.stats['backtracks']}, "
              f"Time={solver.stats['time']:.6f}s, "
              f"Memory={solver.stats['memory']:.1f}MB")

    clauses = [
        [1, 2],
        [-1, 2],
        [-2, 3],
        [-3]
    ]
    cnf = CNF(clauses, 3)

    print("\nTesting backtracking with a problem requiring backtracking:")
    for name, solver in [('DPLL', DPLLSolver()), ('DP', DPSolver())]:
        result = solver.solve(cnf)
        print(f"{name}: Result={'UNSAT' if result == {} else 'SAT' if result else 'TIMEOUT'}, "
              f"Backtracks={solver.stats['backtracks']}, "
              f"Time={solver.stats['time']:.6f}s, "
              f"Memory={solver.stats['memory']:.1f}MB")


def run_benchmarks():
    random.seed(42)

    # Define number of runs
    NUM_RUNS = 30
    PROBLEM_SIZES = [20, 40, 60]

    scaling_benchmarks = {}
    for size in PROBLEM_SIZES:
        scaling_benchmarks[f'3sat_{size}'] = generate_random_3sat(size, size * 4, seed=42)
        scaling_benchmarks[f'assembly_{size}'] = generate_assembly_problem(size, size // 4, seed=42)

    benchmarks = [
        ('Basic Task Assignment Constraints (Porsche-inspired)', CNF([[1, 2, 3], [4, 5, 6], [7, 8, 9],
                                                                      [-1, -2], [-1, -3], [-2, -3],
                                                                      [-4, -5], [-4, -6], [-5, -6],
                                                                      [-7, -8], [-7, -9], [-8, -9]], 9)),
        ('Precedence Constraints (Audi-inspired)', CNF([[-1, 4], [-1, 5], [-1, 6], [-2, 7], [-2, 8], [-3, 9]], 9)),
        ('Workstation Capacity Limits (Volvo-inspired)', CNF([[-1, -4, -7], [-2, -5, -8], [-3, -6, -9]], 9))
    ]

    for name, cnf in scaling_benchmarks.items():
        benchmarks.append((name, cnf))

    solvers = {
        'Resolution': ResolutionSolver(),
        'DP': DPSolver(),
        'DPLL (Random)': DPLLSolver('random'),
        'DPLL (MOMS)': DPLLSolver('moms'),
        'DPLL-JW': DPLLSolver('jw'),
        'Glucose': PySATWrapper()
    }

    results = []
    scaling_data = {solver_name: {size: [] for size in PROBLEM_SIZES} for solver_name in ['DPLL-JW', 'Glucose']}

    print("\n=== Running Standard Benchmarks ===")
    for name, cnf in benchmarks:
        print(f"\nRunning benchmark: {name} ({len(cnf.clauses)} clauses, {cnf.num_vars} vars)")

        benchmark_cnfs = {solver_name: CNF([clause.copy() for clause in cnf.clauses], cnf.num_vars)
                          for solver_name in solvers}

        for solver_name, solver in solvers.items():
            if solver_name in ['Resolution', 'DP'] and ('3sat_60' in name or 'assembly_60' in name):
                print(f"  Skipping {solver_name} for large problem {name}")
                continue

            print(f"  Testing {solver_name}...", end=' ', flush=True)
            times = []
            memories = []
            status = None

            for run in range(NUM_RUNS):
                solver.stats['timeout'] = False

                result = solver.solve(benchmark_cnfs[solver_name])

                status = "SAT" if result and result != {} else "UNSAT" if result == {} else "TIMEOUT"
                times.append(solver.stats['time'])
                memories.append(solver.stats['memory'])

                if status == "TIMEOUT" or solver_name in ['Resolution', 'DP']:
                    break

            mean_time = sum(times) / len(times)
            mean_memory = sum(memories) / len(memories)

            if len(times) > 1:
                ci = SATSolver.calculate_confidence(times)
                ci_str = f" ± {ci[1] - mean_time:.6f}s (95% CI)"
            else:
                ci_str = ""

            print(f"{status} in {mean_time:.6f}s{ci_str}, {mean_memory:.1f}MB")

            results.append({
                'Benchmark': name,
                'Solver': solver_name,
                'Time': mean_time,
                'Memory': mean_memory,
                'Decisions': solver.stats.get('decisions', 0),
                'Backtracks': solver.stats.get('backtracks', 0),
                'Resolvents': solver.stats.get('resolvents', 0),
                'Units': solver.stats.get('units', 0),
                'Pures': solver.stats.get('pures', 0),
                'Result': status,
                'CI': ci_str if len(times) > 1 else ""
            })

            if solver_name in ['DPLL-JW', 'Glucose']:
                for size in PROBLEM_SIZES:
                    if f'3sat_{size}' in name:
                        scaling_data[solver_name][size].append(mean_time)

    plt.figure(figsize=(10, 6))
    for solver_name, size_data in scaling_data.items():
        sizes = []
        times = []
        for size, time_list in sorted(size_data.items()):
            if time_list:
                sizes.append(size)
                times.append(time_list[0])

        if sizes and times:
            plt.plot(sizes, times, marker='o', label=solver_name)

    plt.xlabel('Problem Size (Variables)')
    plt.ylabel('Execution Time (s)')
    plt.title('SAT Solver Scaling Performance')
    plt.grid(True)
    plt.legend()
    plt.savefig('scaling_plot.png')
    plt.close()

    print("\nBenchmark Results Summary:")
    print(
        "+--------------------------------+----------------+-----------+------------+-----------+-----------+--------+")
    print(
        "|           Benchmark            |     Solver     | Time (s)  | Memory (MB)| Decisions | Backtracks| Result |")
    print(
        "+--------------------------------+----------------+-----------+------------+-----------+-----------+--------+")

    current_benchmark = None
    for res in results:
        benchmark_name = res['Benchmark']
        if len(benchmark_name) > 30:
            benchmark_name = benchmark_name[:27] + "..."

        if benchmark_name != current_benchmark:
            print(f"| {benchmark_name.ljust(30)} |", end="")
            current_benchmark = benchmark_name
        else:
            print("|" + " ".ljust(32) + "|", end="")

        time_str = f"{res['Time']:9.6f}{res['CI']}"
        print(f" {res['Solver'].ljust(14)} | {time_str[:9]} | {res['Memory']:10.1f} | "
              f"{res['Decisions']:9} | {res['Backtracks']:9} | {res['Result']:6} |")

    print(
        "+--------------------------------+----------------+-----------+------------+-----------+-----------+--------+")

    consistency_check = {}
    for res in results:
        key = res['Benchmark']
        if key not in consistency_check:
            consistency_check[key] = {}
        consistency_check[key][res['Solver']] = res['Result']

    import csv
    with open('sat_solver_benchmarks.csv', 'w', newline='') as csvfile:
        fieldnames = ['Benchmark', 'Solver', 'Time', 'Memory', 'Decisions', 'Backtracks', 'Resolvents', 'Units',
                      'Pures', 'Result', 'CI']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    print("\nResults saved to sat_solver_benchmarks.csv")
    print("Scaling plot saved to scaling_plot.png")


if __name__ == "__main__":
    run_benchmarks()
    # test_backtracking()