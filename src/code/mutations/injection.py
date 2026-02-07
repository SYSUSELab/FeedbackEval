import ast
import astor
import random
from core.utils import DataLoader, write_jsonl


class ErrorInjector(ast.NodeTransformer):
    def __init__(self, mut_type):
        self.mut_type = mut_type

    def visit_BinOp(self, node):
        operators = [ast.Add, ast.Sub, ast.Mult, ast.Div]
        if self.mut_type == "AOR":
            new_op = random.choice(operators)()
            while isinstance(new_op, type(node.op)):
                new_op = random.choice(operators)()
            node.op = new_op
        return self.generic_visit(node)

    def visit_Compare(self, node):
        operators = [ast.Eq, ast.NotEq, ast.Lt, ast.LtE, ast.Gt, ast.GtE]
        if self.mut_type == "ROR":
            new_op = random.choice(operators)()
            while isinstance(new_op, type(node.ops[0])):
                new_op = random.choice(operators)()
            node.ops = [new_op]
        return self.generic_visit(node)

    def visit_BoolOp(self, node):
        operators = [ast.And, ast.Or]
        if self.mut_type == "COR":
            new_op = random.choice(operators)()
            while isinstance(new_op, type(node.op)):
                new_op = random.choice(operators)()
            node.op = new_op
        return self.generic_visit(node)

    def visit_Constant(self, node):
        if self.mut_type == "LVR":
            if isinstance(node.value, int):
                node.value = node.value + 1
        elif self.mut_type == "CTR":
            if isinstance(node.value, int):
                node.value = float(node.value)
            elif isinstance(node.value, float):
                node.value = int(node.value)
        return self.generic_visit(node)

    def visit_For(self, node):
        if self.mut_type == "LOR":
            node = ast.While(
                test=ast.Name(id="True", ctx=ast.Load()),
                body=node.body,
                orelse=node.orelse,
            )
        return self.generic_visit(node)

    def visit_While(self, node):
        if self.mut_type == "LOR":
            node = ast.For(
                target=ast.Name(id="i", ctx=ast.Store()),
                iter=ast.Call(
                    func=ast.Name(id="range", ctx=ast.Load()),
                    args=[ast.Constant(value=5)],
                    keywords=[],
                ),
                body=node.body,
                orelse=node.orelse,
            )
        return self.generic_visit(node)

    def visit_Call(self, node):
        if self.mut_type == "MCR":
            if random.choice([True, False]):
                if node.args:
                    index_to_remove = random.randint(0, len(node.args) - 1)
                    node.args.pop(index_to_remove)
            else:
                new_arg = ast.Constant(value=random.randint(0, 100))
                node.args.append(new_arg)

        return self.generic_visit(node)


def is_contained(tree, node_types):
    check_constant = ast.Constant in node_types
    for node in ast.walk(tree):
        if any(isinstance(node, node_type) for node_type in node_types):
            if check_constant and isinstance(node, ast.Constant):
                if isinstance(node.value, int):
                    return True
            else:
                return True
    return False


def adjust_indentation(code: str, spaces: int = 4) -> str:
    """
    Adjust the indentation of the code by moving it left by the specified number of spaces.
    :param code: The input code as a multiline string.
    :param spaces: Number of spaces to remove from the beginning of each line.
    :return: Adjusted code with reduced indentation.
    """
    if not code[0].isspace():
        return code
    else:
        lines = code.split("\n")
        adjusted_lines = []
        for line in lines:
            if line.startswith(" " * spaces) or line.startswith("\t"):
                # Remove the specified number of spaces
                line = line.replace("\t", " " * 4)
                adjusted_lines.append(line[spaces:])
            else:
                # Strip only the existing leading spaces if less than the specified amount
                adjusted_lines.append(line.lstrip())
        return "\n".join(adjusted_lines)


class GenMut:
    def __init__(self, input_path, mutation_type):
        self.data = DataLoader(input_path).data
        self.input_path = input_path
        self.mut_type = mutation_type

    def _process_mutations(self, code):
        """Process mutations for a given code snippet."""
        false_codes = []
        for mut in self.mut_type:
            tree = ast.parse(code)
            if is_contained(tree, mut_dict[mut]):
                mutation = ErrorInjector(mut).visit(tree)
                mutated_code = astor.to_source(mutation)
                false_codes.append(
                    {"source": "rule_based", "generate_code": mutated_code}
                )
        return false_codes

    def generate_eval(self):
        """Generate evaluation mutants for different datasets."""
        mutated_data = []
        output_path = ""
        if "HumanEval" in self.input_path:
            for data in self.data:
                output_path = f"../../output/human_eval/mutants.jsonl"
                code = f"{data['prompt']}\n{data['canonical_solution']}"
                false_codes = self._process_mutations(code)
                mutated_data.append(
                    {
                        "task_id": data["task_id"],
                        "false_results": false_codes,
                        "test": f"{data['test']}\ncheck({data['entry_point']})",
                    }
                )
        elif "CoderEval" in self.input_path:
            for data in self.data:
                output_path = f"../../output/coder_eval/mutants.jsonl"
                code = adjust_indentation(data["code"])
                false_codes = self._process_mutations(code)
                mutated_data.append({"_id": data["_id"], "false_results": false_codes})
        else:
            raise ValueError(f"Invalid file path.")

        write_jsonl(output_path, mutated_data)


if __name__ == "__main__":
    """
    AOR: Arithmetic Operator Replacement
    ROR: Relational Operator Replacement
    COR: Logical Operator Replacement
    LVR: Literal Value Replacement
    CTR: Constant Type Replacement
    LOR: Loop Operator Replacement
    MCR: Method Call Replacement
    """
    mut_dict = {
        "AOR": [ast.BinOp],
        "ROR": [ast.Compare],
        "COR": [ast.BoolOp],
        "LVR": [ast.Constant],
        "CTR": [ast.Constant],
        "LOR": [ast.For, ast.While],
        "MCR": [ast.Call],
    }
    mut_type = ["AOR", "ROR", "COR", "LVR", "CTR", "LOR", "MCR"]
    generator = GenMut(f"../input/HumanEval.jsonl", mut_type)
    generator.generate_eval()
