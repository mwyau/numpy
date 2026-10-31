"""Offline validation for native DUCC ND workload coverage and grouping."""
import csv
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

SCRIPTS=Path(__file__).resolve().parent
def import_script(name):
    spec=importlib.util.spec_from_file_location(name,SCRIPTS/f"{name}.py")
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module
bench=import_script("fft_native_nd_sweep")
group=import_script("fft_native_nd_group")


class NativeNdSweepTests(unittest.TestCase):
    def test_case_coverage_and_memory_bound(self):
        all_cases=bench.cases()
        self.assertEqual(len(all_cases),len(set(all_cases)))
        seen=set()
        for family,shape_text,batch,dtype,direction,output,axes_order in all_cases:
            dims=tuple(map(int,shape_text.split("x")))
            self.assertEqual(len(dims),2 if family=="nd2" else 3)
            self.assertTrue(all(d>0 and (d&(d-1))==0 for d in dims))
            size=16
            for d in dims:
                size*=d
            self.assertLessEqual(size*batch,bench.MAX_INPUT_BYTES)
            self.assertIn(dtype,("complex64","complex128"))
            seen.add((family,shape_text,batch,dtype,direction,output,axes_order))
        for family,shapes in (("nd2",bench.SHAPES_2D),("nd3",bench.SHAPES_3D)):
            for shape in shapes:
                text="x".join(map(str,shape))
                upper=min(64,bench.MAX_INPUT_BYTES//(
                    16 * __import__("math").prod(shape)))
                for batch in range(1,upper+1):
                    for dtype in ("complex64","complex128"):
                        for direction in ("forward","inverse"):
                            for output in ("new","fixed"):
                                self.assertIn((family,text,batch,dtype,direction,
                                               output,"last"),seen)
        self.assertTrue(any(r[6]=="reverse" for r in all_cases))
        self.assertTrue(any(r[2]==64 for r in all_cases))
        self.assertTrue(any(r[2] in (20,33,40,63) for r in all_cases))

    def test_profiles_pinned_and_reuse_build(self):
        self.assertEqual(tuple(x[0] for x in bench.PROFILES),group.PROFILES)
        self.assertEqual(len(set(x[2] for x in bench.PROFILES)),3)
        self.assertEqual([x[3] for x in bench.PROFILES],
                         ["V2","V3","V4","V4","V4"])

    def test_cpu_aggregation_two_models(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)/"input"
            for run,slot,model,scale in (
                ("run-1",1,"Intel Demo",1.0),
                ("run-2",4,"Intel Demo",1.2),
                ("run-3",2,"AMD Demo",1.4)
            ):
                dest=root/run/f"fft-native-nd-experiments-{slot}"
                dest.mkdir(parents=True)
                (dest/"machine.json").write_text(json.dumps({
                    "cpu_model":model,"vendor":"GenuineIntel" if "Intel" in model else "AuthenticAMD",
                    "family":"6","model":"1","stepping":"1","cpu_id":0,
                    "microcode":"0x12","compiler":"g++ test"}))
                (dest/"commits.json").write_text(json.dumps([
                    dict(zip(("variant","branch","sha","target"),x))
                    for x in bench.PROFILES]))
                (dest/"REPORT.md").write_text("placeholder")
                fields=["group",*group.CASE_FIELDS]+[f"{v}_us" for v in group.PROFILES]
                with (dest/"all-cases.csv").open("w",newline="") as stream:
                    w=csv.DictWriter(stream,fieldnames=fields)
                    w.writeheader()
                    for fam,shape in (("nd2","16x1024"),("nd3","4x8x128")):
                        for batch in (1,20,64):
                            row={"group":fam,"shape":shape,"batch":batch,
                                 "dtype":"complex64","direction":"forward",
                                 "output":"fixed","axis_order":"last"}
                            for v,t in zip(group.PROFILES,(100,90,75,64,60)):
                                row[f"{v}_us"]=t*scale
                            w.writerow(row)
            output=Path(tmp)/"output"
            group.summarize(root,output,planned=64)
            report=(output/"CPU_GROUPED_REPORT.md").read_text()
            self.assertIn("3/64",report)
            self.assertIn("Intel Demo (2 successful runners)",report)
            self.assertIn("AMD Demo (1 successful runners)",report)
            with (output/"CPU_GROUPED_CASES.csv").open(newline="") as fh:
                cases=list(csv.DictReader(fh))
            self.assertEqual(len(cases),12)
            row=next(x for x in cases if x["cpu_model"]=="Intel Demo"
                     and x["shape"]=="16x1024" and x["batch"]=="20")
            self.assertAlmostEqual(float(row["median_D_over_E"]),60/64)


if __name__=="__main__":
    unittest.main()
