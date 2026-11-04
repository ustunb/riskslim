"""Check the installed CPLEX Python library beyond the Community Edition limit."""

import hashlib
import json
from pathlib import Path
import socket
import sys

import cplex


internal = Path(cplex.__file__).parent / "_internal"
library = internal / "libcplex2220.dylib"
result = {
    "host": socket.gethostname(),
    "python": sys.executable,
    "cplex_package": cplex.__file__,
    "internal": str(internal),
    "library_sha256": hashlib.sha256(library.read_bytes()).hexdigest(),
    "variables": 1500,
}
try:
    with cplex.Cplex() as model:
        result["version"] = model.get_version()
        model.set_log_stream(None)
        model.set_results_stream(None)
        model.set_warning_stream(None)
        model.set_error_stream(None)
        model.variables.add(types="B" * 1500, obj=[1.0] * 1500)
        model.solve()
        result["status"] = model.solution.get_status_string()
        result["objective"] = model.solution.get_objective_value()
        result["success"] = model.solution.is_primal_feasible()
except cplex.exceptions.CplexError as error:
    result["success"] = False
    result["error"] = str(error)
print(json.dumps(result, indent=2))
sys.exit(0 if result["success"] else 1)
