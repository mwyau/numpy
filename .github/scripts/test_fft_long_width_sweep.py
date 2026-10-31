"""Offline checks for the long FFT 8-vs-16 SIMD experiment."""
import csv
import importlib.util
import json
import math
from pathlib import Path
import tempfile
import unittest

HOME=Path(__file__).resolve().parent
def load(name):
    spec=importlib.util.spec_from_file_location(name,HOME/f"{name}.py")
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module
bench=load("fft_long_width_sweep")
group=load("fft_long_width_group")


class LongWidthTests(unittest.TestCase):
    def test_new_lengths_and_large_batches(self):
        cases=bench.cases()
        self.assertEqual(len(cases),len(set(cases)))
        self.assertEqual(bench.LENGTHS,tuple(2**p for p in range(13,20)))
        self.assertGreater(max(bench.BATCHES),512)
        for family,cfg in cases:
            kind,n,batch,dtype,layout,pad,offset,output=cfg
            self.assertEqual(kind,"fft")
            self.assertEqual(family,"long_power2" if n<=65536 else "huge_power2")
            self.assertLessEqual(n*batch*16,bench.MAX_INPUT_BYTES)
            self.assertFalse(n==8192 and batch<=64)
            self.assertIn(dtype,("complex64","complex128"))
            self.assertIn(output,("new","fixed"))
        self.assertTrue(any(cfg[2]>512 for _,cfg in cases))
        self.assertTrue(any(cfg[1]==524288 for _,cfg in cases))
        for n in bench.LENGTHS:
            selected=[cfg for _,cfg in cases if cfg[1]==n and cfg[4]=="C"]
            self.assertTrue(selected)
            self.assertIn("complex64",{cfg[3] for cfg in selected})
            self.assertIn("complex128",{cfg[3] for cfg in selected})

    def test_pinned_policy_controls(self):
        self.assertEqual(bench.VARIANTS,group.PROFILES)
        self.assertEqual(len(bench.PROFILES),7)
        self.assertEqual([x[3] for x in bench.PROFILES],
                         ["V2","V3","V4","V4","V4","V4","V4"])
        self.assertEqual(len({x[2] for x in bench.PROFILES}),5)

    def test_grouped_synthetic(self):
        with tempfile.TemporaryDirectory() as base:
            root=Path(base)/"raw"
            for run_id,slot,cpu,scale in [
                ("run-1",1,"AMD example",1.0),
                ("run-2",3,"AMD example",1.1),
                ("run-3",5,"Intel example",1.3)]:
                d=root/run_id/f"fft-long-width-experiments-{slot}"
                d.mkdir(parents=True)
                (d/"machine.json").write_text(json.dumps({
                    "cpu_model":cpu,"vendor":"AMD" if "AMD" in cpu else "Intel",
                    "family":"25","model":"3","stepping":"1","microcode":"0x1",
                    "cpu_id":0,"compiler":"g++ test"}))
                (d/"commits.json").write_text(json.dumps([
                    dict(zip(("variant","branch","sha","target"),p))
                    for p in bench.PROFILES]))
                (d/"REPORT.md").write_text("report")
                fields=["group",*group.FIELDS]+[f"{p}_us" for p in bench.VARIANTS]
                with (d/"all-cases.csv").open("w",newline="") as f:
                    w=csv.DictWriter(f,fieldnames=fields)
                    w.writeheader()
                    for fam,n in [("long_power2",16384),("huge_power2",131072)]:
                        for batch in [1,20]:
                            data={"group":fam,"kind":"fft","n":n,"batch":batch,
                                  "dtype":"complex64","layout":"C","pad":0,
                                  "offset":0,"output":"fixed"}
                            for profile,t in zip(bench.VARIANTS,
                                (100,90,80,75,74,68,64)):
                                data[f"{profile}_us"]=t*scale
                            w.writerow(data)
            out=Path(base)/"summaries"
            group.summarize(root,out,planned=64)
            report=(out/"CPU_GROUPED_REPORT.md").read_text()
            self.assertIn("3 / 64",report)
            with (out/"CPU_GROUPED_CASES.csv").open(newline="") as f:
                rows=list(csv.DictReader(f))
            self.assertEqual(len(rows),8)
            target=next(x for x in rows if x["cpu_model"]=="AMD example" and
                        x["n"]=="16384" and x["batch"]=="20")
            self.assertAlmostEqual(float(target["median_F16_over_F8"]),68/64)


if __name__=="__main__":
    unittest.main()
