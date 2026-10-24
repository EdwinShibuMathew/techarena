"""Extract the supplied feature calculations without changing their formulas."""
import ast
from pathlib import Path

source = Path(r"C:\Users\Edwin\Downloads\techarena_prediction_sources.txt").read_text(encoding="utf-8")
source = source.split("========== build_combiner_features.py ==========", 1)[1]
source = source.split("========== test_exposure_adjustment.py ==========", 1)[0]
original = next(n for n in ast.parse(source).body if isinstance(n, ast.FunctionDef) and n.name == "build_fold")
start = next(i for i, n in enumerate(original.body) if isinstance(n, ast.FunctionDef) and n.name == "array")


class AdaptInputs(ast.NodeTransformer):
    def visit_Subscript(self, node):
        if (isinstance(node.value, ast.Subscript)
            and isinstance(node.value.value, ast.Name)
            and node.value.value.id == "raw"
            and isinstance(node.slice, ast.Name)
            and node.slice.id == "fit_idx"):
            return ast.Subscript(value=ast.Name(id="training", ctx=ast.Load()), slice=node.value.slice, ctx=node.ctx)
        return self.generic_visit(node)

    def visit_Name(self, node):
        if node.id == "raw":
            return ast.Name(id="target", ctx=node.ctx)
        return node


body = ast.parse('target_idx = np.arange(len(target["u"]))').body
for node in original.body[start:]:
    if isinstance(node, ast.With):
        continue
    if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id in {"n_users", "n_items", "y"} for t in node.targets):
        continue
    if isinstance(node, ast.Return):
        node = ast.parse("return X, feature_names").body[0]
    body.append(AdaptInputs().visit(node))

signature = ast.parse("def build_features(training, target, state, coefficients, intercept, n_users, n_items, probabilities, raw_probabilities):\n    pass\n").body[0]
signature.body = body
module = ast.Module(body=[ast.Import(names=[ast.alias(name="numpy", asname="np")]), signature], type_ignores=[])
ast.fix_missing_locations(module)
destination = Path(__file__).parent / "submission_features.py"
destination.write_text('"""The same 37 feature formulas used by the validated combiners."""\n\n' + ast.unparse(module) + "\n", encoding="utf-8")
print(f"Generated: {destination}")
