from tabulate import tabulate
from collections import defaultdict
from typet5.type_check import parse_type_str

class Problem:
    repo_name: str
    src_path: str
    file_path: str
    target: str
    correct_num_list: list
    params: list
    preds: list
    expects: list
    before_solutions: list
    after_solutions: list
    cat: str
    generic: bool
    total_preds: list

    def __init__(self, repo_name, src_path, file_path, proj_result_path, target, correct_num_list, params, preds, expects, cat=None, generic=None, total_preds=None):
        self.repo_name = repo_name
        self.src_path = src_path
        self.file_path = file_path
        self.proj_result_path = proj_result_path
        self.target = target

        # if target == "Topic.add_occurrence":
        #     for pred in preds:
        #         print(pred)

        new_correct_num_list = []
        for i, pred in enumerate(preds):
            try:
                normalized_pred = [str(parse_type_str(p).normalized()) for p in pred]
            except SyntaxError:
                continue
            normalized_expects = [str(parse_type_str(e).normalized()) for e in expects]
            if normalized_pred == normalized_expects:
                new_correct_num_list.append(i)

        # if set(new_correct_num_list) != set(correct_num_list):
        #     print(f"Correct num list is changed: {correct_num_list} -> {new_correct_num_list}")

        self.correct_num_list = new_correct_num_list
        self.params = params
        self.preds = preds
        self.expects = expects

        self.before_solutions = []
        self.after_solutions = []

        self.cat = cat
        self.generic = generic
        self.total_preds = total_preds

    def set_before_solutions(self, before_solutions):
        self.before_solutions = before_solutions

    def set_after_solutions(self, after_solutions):
        self.after_solutions = after_solutions

    def add_correct_num(self, correct_num):
        self.correct_num_list.append(correct_num)

    def add_pred(self, num, pred):
        if len(self.preds) <= num:
            self.preds.extend([None] * (num - len(self.preds) + 1))

        self.preds[num] = pred

    def remove_duplicate(self):
        dup_dict = {}

        for i, pred in enumerate(self.preds):
            dup_idx = dup_dict.get(str(pred), [])
            dup_idx.append(i)
            dup_dict[str(pred)] = dup_idx

        # print(dup_dict)
        # input()

        new_before_solutions = []
        new_after_solutions = []

        skip_idx = set([])
        for k in self.before_solutions:
            if k in skip_idx:
                continue

            for dup_idx in dup_dict.values():
                if k in dup_idx:
                    skip_idx.update(dup_idx)
                    new_before_solutions.append(k)
                    break

        skip_idx = set([])
        for k in self.after_solutions:
            if k in skip_idx:
                continue

            for dup_idx in dup_dict.values():
                if k in dup_idx:
                    skip_idx.update(dup_idx)
                    new_after_solutions.append(k)
                    break

        # if len(new_before_solutions) == 1:
        #     print(new_before_solutions)
        #     print(new_after_solutions)

        #     for pred in self.preds:
        #         print(pred)

        # if len(new_before_solutions) != len(self.before_solutions):
        #     print(f"Before: {(self.before_solutions)} -> {(new_before_solutions)}")
        # if len(new_after_solutions) != len(self.after_solutions):
        #     print(f"After: {(self.after_solutions)} -> {(new_after_solutions)}")
        # input()

        for i in new_before_solutions:
            for dup_idx in dup_dict.values():
                if i in dup_idx and i in self.correct_num_list:
                    try:
                        assert set(dup_idx) <= set(self.correct_num_list), f"{dup_idx} {self.correct_num_list}"
                    except AssertionError as e:
                        print(dup_dict)
                        print("---")
                        for pred in self.preds:
                            print(pred)
                        print(new_before_solutions)
                        print("---")
                        print(self.expects)
                        print(self.correct_num_list)

                        print("---")

                        new_correct_num_list = []
                        for i, pred in enumerate(self.preds):
                            if pred == self.expects:
                                new_correct_num_list.append(i)
                        print(new_correct_num_list)
                        print(self.target)
                        raise e
        self.before_solutions = new_before_solutions
        self.after_solutions = new_after_solutions

class ProblemList:
    problem_list: list[Problem]

    def __init__(self):
        self.problem_list = []

    def get_problem_list(self):
        return self.problem_list

    def add_problem(self, problem):
        self.problem_list.append(problem)

    def remove_duplicate(self):
        pass
        # for problem in self.problem_list:
        #     problem.remove_duplicate()

    def var_to_type_dict(self):
        var_to_type = defaultdict(dict)



        for problem in self.problem_list:
            if not problem.after_solutions:
                continue

            before_pred = problem.preds[0]

            for var, type_str in zip(problem.params, before_pred):
                if var == "__RET__":
                    continue 
                name = problem.file_path + "@" + problem.target
                var_to_type[var][name] = {
                    "before_type": type_str,
                    "expected_type": problem.expects[0],
                    "before_correct": 0 in problem.correct_num_list,
                    "after_correct": problem.after_solutions[0] in problem.correct_num_list
                }

                type_set = var_to_type[var].get("type_set", [])
                type_set.append(type_str)
                var_to_type[var]["type_set"] = list(set(type_set))

        return var_to_type

    def check_base_correct(self, preds, expects):
        for pred, expect in zip(preds, expects):
            try:
                if isinstance(pred, str):
                    pred_type = parse_type_str(pred)
                else:
                    pred_type = pred

                if isinstance(expect, str):
                    expect_type = parse_type_str(expect)
                else:
                    expect_type = expect

                pred_type = pred_type.normalized()
                expect_type = expect_type.normalized()

                if pred_type.head != expect_type.head:
                    return False
            except SyntaxError: # Failed Parsing
                return False

        return True

    def calc_before_top_k(self):
        top_k = [0] * 15

        for problem in self.problem_list:
            for i, before in enumerate(problem.before_solutions):
                
                if before in problem.correct_num_list:
                    # add 1 from i-th index to the end
                    top_k[i:] = [x + 1 for x in top_k[i:]]
                    break

        return top_k

    def calc_before_base_top_k(self):
        top_k = [0] * 15

        for problem in self.problem_list:
            for i, preds in enumerate(problem.preds):
                if preds is None:
                    continue

                if self.check_base_correct(preds, problem.expects):
                    top_k[i:] = [x + 1 for x in top_k[i:]]
                    break

        return top_k

    def calc_after_top_k(self):
        top_k = [0] * 15

        for problem in self.problem_list:
            for i, after in enumerate(problem.after_solutions):
                if after in problem.correct_num_list:
                    # add 1 from i-th index to the end
                    top_k[i:] = [x + 1 for x in top_k[i:]]
                    break

        return top_k

    def calc_after_base_top_k(self):
        top_k = [0] * 15

        for problem in self.problem_list:
            after_preds = [problem.preds[x] for x in problem.after_solutions]
            for i, preds in enumerate(after_preds):
                if preds is None:
                    continue

                if self.check_base_correct(preds, problem.expects):
                    top_k[i:] = [x + 1 for x in top_k[i:]]
                    break

        return top_k

    def return_top_k(self):
        before_top_k = self.calc_before_top_k()
        before_top_k_percent = [x/len(self.problem_list)*100 for x in before_top_k]
        before_top_k_percent = [round(x, 1) for x in before_top_k_percent]

        after_top_k = self.calc_after_top_k()
        after_top_k_percent = [x/len(self.problem_list)*100 for x in after_top_k]
        after_top_k_percent = [round(x, 1) for x in after_top_k_percent]

        diff_k = [after_top_k[i] - before_top_k[i] for i in range(15)]
        diff_k_percent = [round((after_top_k_percent[i] - before_top_k_percent[i]) / before_top_k_percent[i] * 100, 1) for i in range(15)]


        return before_top_k, before_top_k_percent, after_top_k, after_top_k_percent, diff_k, diff_k_percent

    def print_top_k(self):
        # self.remove_duplicate()
        before_top_k = self.calc_before_top_k()
        before_top_k_percent = [x/len(self.problem_list)*100 for x in before_top_k]
        before_top_k_percent = [round(x, 1) for x in before_top_k_percent]

        after_top_k = self.calc_after_top_k()
        after_top_k_percent = [x/len(self.problem_list)*100 for x in after_top_k]
        after_top_k_percent = [round(x, 1) for x in after_top_k_percent]

        diff_k = [after_top_k[i] - before_top_k[i] for i in range(15)]
        diff_k_percent = [round((after_top_k_percent[i] - before_top_k_percent[i]) / before_top_k_percent[i] * 100, 1) for i in range(15)]

        print(
            tabulate([["Top 1", before_top_k[0], f"{before_top_k_percent[0]}%", after_top_k[0], f"{after_top_k_percent[0]}%", f"{diff_k_percent[0]}%"],
                        ["Top 3", before_top_k[2], f"{before_top_k_percent[2]}%", after_top_k[2], f"{after_top_k_percent[2]}%", f"{diff_k_percent[2]}%"],
                        ["Top 5", before_top_k[4], f"{before_top_k_percent[4]}%", after_top_k[4], f"{after_top_k_percent[4]}%", f"{diff_k_percent[4]}%"]],
                       headers=["Rank", "Before", "Before (%)", "After", "After (%)", "Diff"], tablefmt="pretty"))

    def return_base_top_k(self):
        before_top_k = self.calc_before_base_top_k()
        before_top_k_percent = [x/len(self.problem_list)*100 for x in before_top_k]
        before_top_k_percent = [round(x, 1) for x in before_top_k_percent]

        after_top_k = self.calc_after_base_top_k()
        after_top_k_percent = [x/len(self.problem_list)*100 for x in after_top_k]
        after_top_k_percent = [round(x, 1) for x in after_top_k_percent]

        diff_k = [after_top_k[i] - before_top_k[i] for i in range(15)]
        diff_k_percent = [round((after_top_k_percent[i] - before_top_k_percent[i]) / before_top_k_percent[i] * 100, 1) for i in range(15)]

        return before_top_k, before_top_k_percent, after_top_k, after_top_k_percent, diff_k, diff_k_percent

    def print_base_top_k(self):
        # self.remove_duplicate()
        before_top_k = self.calc_before_base_top_k()
        before_top_k_percent = [x/len(self.problem_list)*100 for x in before_top_k]
        before_top_k_percent = [round(x, 1) for x in before_top_k_percent]

        after_top_k = self.calc_after_base_top_k()
        after_top_k_percent = [x/len(self.problem_list)*100 for x in after_top_k]
        after_top_k_percent = [round(x, 1) for x in after_top_k_percent]

        diff_k = [after_top_k[i] - before_top_k[i] for i in range(15)]
        diff_k_percent = [round((after_top_k_percent[i] - before_top_k_percent[i]) / before_top_k_percent[i] * 100, 1) for i in range(15)]

        print(
            tabulate([["Top 1", before_top_k[0], f"{before_top_k_percent[0]}%", after_top_k[0], f"{after_top_k_percent[0]}%", f"{diff_k_percent[0]}%"],
                        ["Top 3", before_top_k[2], f"{before_top_k_percent[2]}%", after_top_k[2], f"{after_top_k_percent[2]}%", f"{diff_k_percent[2]}%"],
                        ["Top 5", before_top_k[4], f"{before_top_k_percent[4]}%", after_top_k[4], f"{after_top_k_percent[4]}%", f"{diff_k_percent[4]}%"]],
                       headers=["Rank", "Before", "Before (%)", "After", "After (%)", "Diff"], tablefmt="pretty"))

    def return_categorized_top_k(self):
        # self.remove_duplicate()
        element_problems = ProblemList()
        parameterized_problems = ProblemList()
        user_problems = ProblemList()

        for problem in self.problem_list:
            if problem.cat == "builtins":
                if problem.generic is False:
                    element_problems.add_problem(problem)
                else:
                    parameterized_problems.add_problem(problem)
            else:
                user_problems.add_problem(problem)

        print("Element Problems: ", len(element_problems.problem_list))
        print("Parameterized Problems: ", len(parameterized_problems.problem_list))
        print("User Problems: ", len(user_problems.problem_list))

        data = {
            "Elem": element_problems.return_top_k(),
            "Param": parameterized_problems.return_top_k(),
            "User": user_problems.return_top_k()
        }

        return data

    def print_categorized_top_k(self):
        # self.remove_duplicate()
        element_problems = ProblemList()
        parameterized_problems = ProblemList()
        user_problems = ProblemList()


        for problem in self.problem_list:
            if problem.cat == "builtins":
                if problem.generic is False:
                    element_problems.add_problem(problem)
                else:
                    parameterized_problems.add_problem(problem)

            else:
                user_problems.add_problem(problem)

        print("Element Problems")
        element_problems.print_top_k()
        print()
        print("Parameterized Problems")
        parameterized_problems.print_top_k()
        print()
        print("User Problems")
        user_problems.print_top_k()

    def return_arg_ret_top_k(self):
        # self.remove_duplicate()
        arg_problem_num = 0
        ret_problem_num = 0

        before_arg_top_k = [0] * 10
        before_ret_top_k = [0] * 10
    
        before_arg_base_top_k = [0] * 10
        before_ret_base_top_k = [0] * 10

        after_arg_top_k = [0] * 10
        after_ret_top_k = [0] * 10

        after_arg_base_top_k = [0] * 10
        after_ret_base_top_k = [0] * 10

        for problem in self.problem_list:
            assert len(problem.params) >= len(problem.expects) 
            before_preds = [
                problem.preds[i] for i in problem.before_solutions
            ]
            after_preds = [
                problem.preds[i] for i in problem.after_solutions
            ]

            # Check is correct
            for i, expect in enumerate(problem.expects):
                param = problem.params[i]
                if param == "__RET__":
                    
                    ret_problem_num += 1

                    for j, before in enumerate(before_preds):
                        if before is None:
                            continue

                        try:
                            if str(parse_type_str(before[i]).normalized()) == str(parse_type_str(expect).normalized()):
                                before_ret_top_k[j:] = [x + 1 for x in before_ret_top_k[j:]]
                                break
                        except SyntaxError:
                            continue

                    for j, after in enumerate(after_preds):
                        if after is None:
                            continue

                        try:
                            if str(parse_type_str(after[i]).normalized()) == str(parse_type_str(expect).normalized()):
                                after_ret_top_k[j:] = [x + 1 for x in after_ret_top_k[j:]]
                                break
                        except SyntaxError:
                            continue

                else:
                    arg_problem_num += 1

                    for j, before in enumerate(before_preds):
                        if before is None:
                            continue

                        try:
                            if str(parse_type_str(before[i]).normalized()) == str(parse_type_str(expect).normalized()):
                                before_arg_top_k[j:] = [x + 1 for x in before_arg_top_k[j:]]
                                break
                        except SyntaxError:
                            continue

                    for j, after in enumerate(after_preds):
                        if after is None:
                            continue

                        try:
                            if str(parse_type_str(after[i]).normalized()) == str(parse_type_str(expect).normalized()):
                                after_arg_top_k[j:] = [x + 1 for x in after_arg_top_k[j:]]
                                break
                        except SyntaxError:
                            continue
            # Check is correct by base
            for i, expect in enumerate(problem.expects):
                param = problem.params[i]
                if param == "__RET__":
                    for j, before in enumerate(before_preds):
                        if before is None:
                            continue

                        if self.check_base_correct([before[i]], [expect]):
                            before_ret_base_top_k[j:] = [x + 1 for x in before_ret_base_top_k[j:]]
                            break

                    for j, after in enumerate(after_preds):
                        if after is None:
                            continue

                        if self.check_base_correct([after[i]], [expect]):
                            after_ret_base_top_k[j:] = [x + 1 for x in after_ret_base_top_k[j:]]
                            break

                else:
                    for j, before in enumerate(before_preds):
                        if before is None:
                            continue

                        if self.check_base_correct([before[i]], [expect]):
                            before_arg_base_top_k[j:] = [x + 1 for x in before_arg_base_top_k[j:]]
                            break

                    for j, after in enumerate(after_preds):
                        if after is None:
                            continue

                        if self.check_base_correct([after[i]], [expect]):
                            after_arg_base_top_k[j:] = [x + 1 for x in after_arg_base_top_k[j:]]
                            break

        total_problem_num = arg_problem_num + ret_problem_num
        # Calculate total top k
        total_before_top_k = [0] * 10
        total_after_top_k = [0] * 10
        for i in range(10):
            total_before_top_k[i] = before_arg_top_k[i] + before_ret_top_k[i]
            total_after_top_k[i] = after_arg_top_k[i] + after_ret_top_k[i]

        before_top_k_percent = [x/total_problem_num*100 for x in total_before_top_k]
        before_top_k_percent = [round(x, 1) for x in before_top_k_percent]
        after_top_k_percent = [x/total_problem_num*100 for x in total_after_top_k]
        after_top_k_percent = [round(x, 1) for x in after_top_k_percent]


        before_arg_top_k_percent = [x/arg_problem_num*100 for x in before_arg_top_k]
        before_arg_top_k_percent = [round(x, 1) for x in before_arg_top_k_percent]
        after_arg_top_k_percent = [x/arg_problem_num*100 for x in after_arg_top_k]
        after_arg_top_k_percent = [round(x, 1) for x in after_arg_top_k_percent]
        before_ret_top_k_percent = [x/ret_problem_num*100 for x in before_ret_top_k]
        before_ret_top_k_percent = [round(x, 1) for x in before_ret_top_k_percent]
        after_ret_top_k_percent = [x/ret_problem_num*100 for x in after_ret_top_k]
        after_ret_top_k_percent = [round(x, 1) for x in after_ret_top_k_percent]

        diff_total_k = [total_after_top_k[i] - total_before_top_k[i] for i in range(10)]
        diff_total_k_percent = [round((after_top_k_percent[i] - before_top_k_percent[i]) / before_top_k_percent[i] * 100, 1) for i in range(10)]
        diff_arg_k = [after_arg_top_k[i] - before_arg_top_k[i] for i in range(10)]
        diff_arg_k_percent = [round((after_arg_top_k_percent[i] - before_arg_top_k_percent[i]) / before_arg_top_k_percent[i] * 100, 1) for i in range(10)]
        diff_ret_k = [after_ret_top_k[i] - before_ret_top_k[i] for i in range(10)]
        diff_ret_k_percent = [round((after_ret_top_k_percent[i] - before_ret_top_k_percent[i]) / before_ret_top_k_percent[i] * 100, 1) for i in range(10)]

        # Calculate base top k
        total_before_base_top_k = [0] * 10
        total_after_base_top_k = [0] * 10

        for i in range(10):
            total_before_base_top_k[i] = before_arg_base_top_k[i] + before_ret_base_top_k[i]
            total_after_base_top_k[i] = after_arg_base_top_k[i] + after_ret_base_top_k[i]
        before_base_top_k_percent = [x/total_problem_num*100 for x in total_before_base_top_k]
        before_base_top_k_percent = [round(x, 1) for x in before_base_top_k_percent]
        after_base_top_k_percent = [x/total_problem_num*100 for x in total_after_base_top_k]
        after_base_top_k_percent = [round(x, 1) for x in after_base_top_k_percent]
        before_arg_base_top_k_percent = [x/arg_problem_num*100 for x in before_arg_base_top_k]
        before_arg_base_top_k_percent = [round(x, 1) for x in before_arg_base_top_k_percent]
        after_arg_base_top_k_percent = [x/arg_problem_num*100 for x in after_arg_base_top_k]
        after_arg_base_top_k_percent = [round(x, 1) for x in after_arg_base_top_k_percent]
        before_ret_base_top_k_percent = [x/ret_problem_num*100 for x in before_ret_base_top_k]
        before_ret_base_top_k_percent = [round(x, 1) for x in before_ret_base_top_k_percent]
        after_ret_base_top_k_percent = [x/ret_problem_num*100 for x in after_ret_base_top_k]
        after_ret_base_top_k_percent = [round(x, 1) for x in after_ret_base_top_k_percent]

        diff_total_base_k = [total_after_base_top_k[i] - total_before_base_top_k[i] for i in range(10)]
        diff_total_base_k_percent = [round((after_base_top_k_percent[i] - before_base_top_k_percent[i]) / before_base_top_k_percent[i] * 100, 1) for i in range(10)]
        diff_arg_base_k = [after_arg_base_top_k[i] - before_arg_base_top_k[i] for i in range(10)]
        diff_arg_base_k_percent = [round((after_arg_base_top_k_percent[i] - before_arg_base_top_k_percent[i]) / before_arg_base_top_k_percent[i] * 100, 1) for i in range(10)]
        diff_ret_base_k = [after_ret_base_top_k[i] - before_ret_base_top_k[i] for i in range(10)]
        diff_ret_base_k_percent = [round((after_ret_base_top_k_percent[i] - before_ret_base_top_k_percent[i]) / before_ret_base_top_k_percent[i] * 100, 1) for i in range(10)]

        data = {
            "Exact": {
                "Args": (before_arg_top_k, before_arg_top_k_percent, after_arg_top_k, after_arg_top_k_percent, diff_arg_k, diff_arg_k_percent),
                "Returns": (before_ret_top_k, before_ret_top_k_percent, after_ret_top_k, after_ret_top_k_percent, diff_ret_k, diff_ret_k_percent),
                "Total": (total_before_top_k, before_top_k_percent, total_after_top_k, after_top_k_percent, diff_total_k, diff_total_k_percent)
            },
            "Base": {
                "Args": (before_arg_base_top_k, before_arg_top_k_percent, after_arg_base_top_k, after_arg_top_k_percent, diff_arg_k, diff_arg_k_percent),
                "Returns": (before_ret_base_top_k, before_ret_top_k_percent, after_ret_base_top_k, after_ret_top_k_percent, diff_ret_k, diff_ret_k_percent),
                "Total": (total_before_base_top_k, before_base_top_k_percent, total_after_base_top_k, after_base_top_k_percent, diff_total_base_k, diff_total_base_k_percent)
            }
        }

        return data

    def print_arg_ret_top_k(self):
        # self.remove_duplicate()
        arg_problem_num = 0
        ret_problem_num = 0

        before_arg_top_k = [0] * 10
        before_ret_top_k = [0] * 10
    
        before_arg_base_top_k = [0] * 10
        before_ret_base_top_k = [0] * 10

        after_arg_top_k = [0] * 10
        after_ret_top_k = [0] * 10

        after_arg_base_top_k = [0] * 10
        after_ret_base_top_k = [0] * 10

        for problem in self.problem_list:
            assert len(problem.params) >= len(problem.expects) 
            before_preds = [
                problem.preds[i] for i in problem.before_solutions
            ]
            after_preds = [
                problem.preds[i] for i in problem.after_solutions
            ]

            # print(before_preds)
            # print(after_preds)

            # print(problem.expects)
            # input()

            # Check is correct
            for i, expect in enumerate(problem.expects):
                param = problem.params[i]
                if param == "__RET__":
                    
                    ret_problem_num += 1

                    for j, before in enumerate(before_preds):
                        if before is None:
                            continue

                        try:
                            if str(parse_type_str(before[i]).normalized()) == str(parse_type_str(expect).normalized()):
                                before_ret_top_k[j:] = [x + 1 for x in before_ret_top_k[j:]]
                                break
                        except SyntaxError:
                            continue

                    for j, after in enumerate(after_preds):
                        if after is None:
                            continue

                        try:
                            if str(parse_type_str(after[i]).normalized()) == str(parse_type_str(expect).normalized()):
                                # print(j)
                                after_ret_top_k[j:] = [x + 1 for x in after_ret_top_k[j:]]
                                break
                        except SyntaxError:
                            continue

                    # assert str(after_preds[0][i]) == expect, f"{after_preds[0][i]} {expect}"

                else:
                    arg_problem_num += 1

                    for j, before in enumerate(before_preds):
                        if before is None:
                            continue

                        try:
                            if str(parse_type_str(before[i]).normalized()) == str(parse_type_str(expect).normalized()):
                                before_arg_top_k[j:] = [x + 1 for x in before_arg_top_k[j:]]
                                break
                        except SyntaxError:
                            continue

                    for j, after in enumerate(after_preds):
                        if after is None:
                            continue

                        try:
                            if str(parse_type_str(after[i]).normalized()) == str(parse_type_str(expect).normalized()):
                                after_arg_top_k[j:] = [x + 1 for x in after_arg_top_k[j:]]
                                break
                        except SyntaxError:
                            continue

            # Check is correct by base
            for i, expect in enumerate(problem.expects):
                param = problem.params[i]
                if param == "__RET__":
                    for j, before in enumerate(before_preds):
                        if before is None:
                            continue

                        if self.check_base_correct([before[i]], [expect]):
                            before_ret_base_top_k[j:] = [x + 1 for x in before_ret_base_top_k[j:]]
                            break

                    for j, after in enumerate(after_preds):
                        if after is None:
                            continue

                        if self.check_base_correct([after[i]], [expect]):
                            after_ret_base_top_k[j:] = [x + 1 for x in after_ret_base_top_k[j:]]
                            break

                else:
                    for j, before in enumerate(before_preds):
                        if before is None:
                            continue

                        if self.check_base_correct([before[i]], [expect]):
                            before_arg_base_top_k[j:] = [x + 1 for x in before_arg_base_top_k[j:]]
                            break

                    for j, after in enumerate(after_preds):
                        if after is None:
                            continue

                        if self.check_base_correct([after[i]], [expect]):
                            after_arg_base_top_k[j:] = [x + 1 for x in after_arg_base_top_k[j:]]
                            break

        total_problem_num = arg_problem_num + ret_problem_num

        # Calculate total top k
        total_before_top_k = [0] * 10
        total_after_top_k = [0] * 10

        for i in range(10):
            total_before_top_k[i] = before_arg_top_k[i] + before_ret_top_k[i]
            total_after_top_k[i] = after_arg_top_k[i] + after_ret_top_k[i]

        before_top_k_percent = [x/total_problem_num*100 for x in total_before_top_k]
        before_top_k_percent = [round(x, 1) for x in before_top_k_percent]
        after_top_k_percent = [x/total_problem_num*100 for x in total_after_top_k]
        after_top_k_percent = [round(x, 1) for x in after_top_k_percent]


        before_arg_top_k_percent = [x/arg_problem_num*100 for x in before_arg_top_k]
        before_arg_top_k_percent = [round(x, 1) for x in before_arg_top_k_percent]
        after_arg_top_k_percent = [x/arg_problem_num*100 for x in after_arg_top_k]
        after_arg_top_k_percent = [round(x, 1) for x in after_arg_top_k_percent]
        before_ret_top_k_percent = [x/ret_problem_num*100 for x in before_ret_top_k]
        before_ret_top_k_percent = [round(x, 1) for x in before_ret_top_k_percent]
        after_ret_top_k_percent = [x/ret_problem_num*100 for x in after_ret_top_k]
        after_ret_top_k_percent = [round(x, 1) for x in after_ret_top_k_percent]

        diff_total_k = [total_after_top_k[i] - total_before_top_k[i] for i in range(10)]
        diff_total_k_percent = [round((after_top_k_percent[i] - before_top_k_percent[i]) / before_top_k_percent[i] * 100, 1) for i in range(10)]
        
        diff_arg_k = [after_arg_top_k[i] - before_arg_top_k[i] for i in range(10)]
        diff_arg_k_percent = [round((after_arg_top_k_percent[i] - before_arg_top_k_percent[i]) / before_arg_top_k_percent[i] * 100, 1) for i in range(10)]

        diff_ret_k = [after_ret_top_k[i] - before_ret_top_k[i] for i in range(10)]
        diff_ret_k_percent = [round((after_ret_top_k_percent[i] - before_ret_top_k_percent[i]) / before_ret_top_k_percent[i] * 100, 1) for i in range(10)]

        print("=== Argument and Return Type Top-K ===")
        print()
        print("Argument Type")
        print(
            tabulate([["Top 1", before_arg_top_k[0], f"{before_arg_top_k_percent[0]}%", after_arg_top_k[0], f"{after_arg_top_k_percent[0]}%", f"{diff_arg_k_percent[0]}%"],
                        ["Top 3", before_arg_top_k[2], f"{before_arg_top_k_percent[2]}%", after_arg_top_k[2], f"{after_arg_top_k_percent[2]}%", f"{diff_arg_k_percent[2]}%"],
                        ["Top 5", before_arg_top_k[4], f"{before_arg_top_k_percent[4]}%", after_arg_top_k[4], f"{after_arg_top_k_percent[4]}%", f"{diff_arg_k_percent[4]}%"]],
                       headers=["Rank", "Before", "Before (%)", "After", "After (%)", "Diff"], tablefmt="pretty"))
        print()
        print("Return Type")
        print(
            tabulate([["Top 1", before_ret_top_k[0], f"{before_ret_top_k_percent[0]}%", after_ret_top_k[0], f"{after_ret_top_k_percent[0]}%", f"{diff_ret_k_percent[0]}%"],
                        ["Top 3", before_ret_top_k[2], f"{before_ret_top_k_percent[2]}%", after_ret_top_k[2], f"{after_ret_top_k_percent[2]}%", f"{diff_ret_k_percent[2]}%"],
                        ["Top 5", before_ret_top_k[4], f"{before_ret_top_k_percent[4]}%", after_ret_top_k[4], f"{after_ret_top_k_percent[4]}%", f"{diff_ret_k_percent[4]}%"]],
                       headers=["Rank", "Before", "Before (%)", "After", "After (%)", "Diff"], tablefmt="pretty"))
        print()
        print("Total Type")
        print(
            tabulate([["Top 1", total_before_top_k[0], f"{before_top_k_percent[0]}%", total_after_top_k[0], f"{after_top_k_percent[0]}%", f"{diff_total_k_percent[0]}%"],
                        ["Top 3", total_before_top_k[2], f"{before_top_k_percent[2]}%", total_after_top_k[2], f"{after_top_k_percent[2]}%", f"{diff_total_k_percent[2]}%"],
                        ["Top 5", total_before_top_k[4], f"{before_top_k_percent[4]}%", total_after_top_k[4], f"{after_top_k_percent[4]}%", f"{diff_total_k_percent[4]}%"]],
                       headers=["Rank", "Before", "Before (%)", "After", "After (%)", "Diff"], tablefmt="pretty"))
       
        # Calculate base top k
        total_before_base_top_k = [0] * 10
        total_after_base_top_k = [0] * 10

        for i in range(10):
            total_before_base_top_k[i] = before_arg_base_top_k[i] + before_ret_base_top_k[i]
            total_after_base_top_k[i] = after_arg_base_top_k[i] + after_ret_base_top_k[i]
        before_base_top_k_percent = [x/total_problem_num*100 for x in total_before_base_top_k]
        before_base_top_k_percent = [round(x, 1) for x in before_base_top_k_percent]
        after_base_top_k_percent = [x/total_problem_num*100 for x in total_after_base_top_k]
        after_base_top_k_percent = [round(x, 1) for x in after_base_top_k_percent]

        before_arg_base_top_k_percent = [x/arg_problem_num*100 for x in before_arg_base_top_k]
        before_arg_base_top_k_percent = [round(x, 1) for x in before_arg_base_top_k_percent]
        after_arg_base_top_k_percent = [x/arg_problem_num*100 for x in after_arg_base_top_k]
        after_arg_base_top_k_percent = [round(x, 1) for x in after_arg_base_top_k_percent]
        before_ret_base_top_k_percent = [x/ret_problem_num*100 for x in before_ret_base_top_k]
        before_ret_base_top_k_percent = [round(x, 1) for x in before_ret_base_top_k_percent]
        after_ret_base_top_k_percent = [x/ret_problem_num*100 for x in after_ret_base_top_k]
        after_ret_base_top_k_percent = [round(x, 1) for x in after_ret_base_top_k_percent] 

        diff_total_base_k = [total_after_base_top_k[i] - total_before_base_top_k[i] for i in range(10)]
        diff_total_base_k_percent = [round((after_base_top_k_percent[i] - before_base_top_k_percent[i]) / before_base_top_k_percent[i] * 100, 1) for i in range(10)]
        diff_arg_base_k = [after_arg_base_top_k[i] - before_arg_base_top_k[i] for i in range(10)]
        diff_arg_base_k_percent = [round((after_arg_base_top_k_percent[i] - before_arg_base_top_k_percent[i]) / before_arg_base_top_k_percent[i] * 100, 1) for i in range(10)]
        diff_ret_base_k = [after_ret_base_top_k[i] - before_ret_base_top_k[i] for i in range(10)]
        diff_ret_base_k_percent = [round((after_ret_base_top_k_percent[i] - before_ret_base_top_k_percent[i]) / before_ret_base_top_k_percent[i] * 100, 1) for i in range(10)]
        
        print("=== Argument and Return Type Base Top-K ===")
        print()
        print("Argument Type")
        print(
            tabulate([["Top 1", before_arg_base_top_k[0], f"{before_arg_base_top_k_percent[0]}%", after_arg_base_top_k[0], f"{after_arg_base_top_k_percent[0]}%", f"{diff_arg_base_k_percent[0]}%"],
                        ["Top 3", before_arg_base_top_k[2], f"{before_arg_base_top_k_percent[2]}%", after_arg_base_top_k[2], f"{after_arg_base_top_k_percent[2]}%", f"{diff_arg_base_k_percent[2]}%"],
                        ["Top 5", before_arg_base_top_k[4], f"{before_arg_base_top_k_percent[4]}%", after_arg_base_top_k[4], f"{after_arg_base_top_k_percent[4]}%", f"{diff_arg_base_k_percent[4]}%"]],
                       headers=["Rank", "Before", "Before (%)", "After", "After (%)", "Diff"], tablefmt="pretty"))
        
        print()
        print("Return Type")
        print(
            tabulate([["Top 1", before_ret_base_top_k[0], f"{before_ret_base_top_k_percent[0]}%", after_ret_base_top_k[0], f"{after_ret_base_top_k_percent[0]}%", f"{diff_ret_base_k_percent[0]}%"],
                        ["Top 3", before_ret_base_top_k[2], f"{before_ret_base_top_k_percent[2]}%", after_ret_base_top_k[2], f"{after_ret_base_top_k_percent[2]}%", f"{diff_ret_base_k_percent[2]}%"],
                        ["Top 5", before_ret_base_top_k[4], f"{before_ret_base_top_k_percent[4]}%", after_ret_base_top_k[4], f"{after_ret_base_top_k_percent[4]}%", f"{diff_ret_base_k_percent[4]}%"]],
                       headers=["Rank", "Before", "Before (%)", "After", "After (%)", "Diff"], tablefmt="pretty"))
        print()
        print("Total Type")
        print(
            tabulate([["Top 1", total_before_base_top_k[0], f"{before_base_top_k_percent[0]}%", total_after_base_top_k[0], f"{after_base_top_k_percent[0]}%", f"{diff_total_base_k_percent[0]}%"],
                        ["Top 3", total_before_base_top_k[2], f"{before_base_top_k_percent[2]}%", total_after_base_top_k[2], f"{after_base_top_k_percent[2]}%", f"{diff_total_base_k_percent[2]}%"],
                        ["Top 5", total_before_base_top_k[4], f"{before_base_top_k_percent[4]}%", total_after_base_top_k[4], f"{after_base_top_k_percent[4]}%", f"{diff_total_base_k_percent[4]}%"]],
                       headers=["Rank", "Before", "Before (%)", "After", "After (%)", "Diff"], tablefmt="pretty"))

       
        print()
        print(f"Total Problem Number: {total_problem_num}")
        print(f"Argument Problem Number: {arg_problem_num}")
        print(f"Return Problem Number: {ret_problem_num}")

