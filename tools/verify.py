"""Run the suite and retain a machine-readable local validation report."""
import importlib.metadata
import json
import platform
import sys
import unittest
import tempfile
from pathlib import Path

root=Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root))
report_dir=root/"local"
report_dir.mkdir(exist_ok=True)
test_temp=report_dir/"test_tmp"
test_temp.mkdir(exist_ok=True)
tempfile.tempdir=str(test_temp)
suite=unittest.defaultTestLoader.discover(str(root/"tests"))
with (report_dir/"test_output.txt").open("w",encoding="utf-8") as stream:
    result=unittest.TextTestRunner(stream=stream,verbosity=2).run(suite)
versions={}
for name in ["torch","transformers","accelerate","tokenizers","safetensors","numpy","osiris-research"]:
    try: versions[name]=importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError: versions[name]=None
report={"successful":result.wasSuccessful(),"tests":result.testsRun,"failures":len(result.failures),
        "errors":len(result.errors),"skipped":len(result.skipped),"python":platform.python_version(),
        "platform":platform.platform(),"versions":versions,"scope":"CPU software validation using random tiny weights; no 27B behavior rerun"}
(report_dir/"validation.json").write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding="utf-8")
print(json.dumps(report,ensure_ascii=False))
sys.exit(0 if result.wasSuccessful() else 1)
