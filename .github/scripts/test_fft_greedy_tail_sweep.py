"""Sanity checks for two-family greedy-tail FFT benchmark and CPU aggregator."""
import csv
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

ROOT=Path(__file__).resolve().parent

def load(name):
    spec=importlib.util.spec_from_file_location(name,ROOT/f"{name}.py")
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

benchmark=load("fft_greedy_tail_sweep")
aggregate=load("fft_greedy_group")


class SweepTests(unittest.TestCase):
    def test_full_batch_coverage_and_partition(self):
        cases=benchmark.cases()
        self.assertEqual(len(cases),len(set(cases)))
        seen=set()
        for family,cfg in cases:
            self.assertEqual(cfg[0],"fft")
            seen.add((family,cfg[1],cfg[2],cfg[3],cfg[7]))
            self.assertIn(family,("power2","mixed_radix"))
        for family,lengths in (("power2",benchmark.POWER2),
                               ("mixed_radix",benchmark.MIXED)):
            for n in lengths:
                for batch in range(1,65):
                    for dtype in ("complex64","complex128"):
                        for output in ("new","fixed"):
                            self.assertIn((family,n,batch,dtype,output),seen)
        self.assertTrue(all(s in {c[0] for c in cases} for s in
                            ("power2","mixed_radix")))

    def test_profiles_are_five_and_share_a_baseline(self):
        profiles=benchmark.PROFILES
        self.assertEqual(tuple(x[0] for x in profiles),aggregate.PROFILES)
        self.assertEqual(len(set(x[2] for x in profiles)),3)
        self.assertEqual([x[3] for x in profiles],["V2","V3","V4","V4","V4"])

    def test_grouped_aggregator_with_two_cpu_generations(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)/"results"
            for cpu,run,slot,scale in [
                ("Example AMD Zen","run-1","1",1.0),
                ("Example AMD Zen","run-2","3",1.2),
                ("Example Intel","run-3","1",1.4),
            ]:
                dst=root/run/f"fft-greedy-experiments-{slot}"
                dst.mkdir(parents=True)
                (dst/"REPORT.md").write_text("# report\n")
                (dst/"machine.json").write_text(json.dumps({
                    "cpu_model":cpu,"vendor":"AuthenticAMD" if "AMD" in cpu else "GenuineIntel",
                    "family":"25","model":"3","cpu_id":0,"stepping":"1",
                    "microcode":"0xab","compiler":"g++"}))
                (dst/"commits.json").write_text(json.dumps([
                    {"variant":v,"sha":sha,"target":target}
                    for v,_,sha,target in benchmark.PROFILES]))
                columns=["group",*aggregate.FIELDS]+[
                    f"{p}_us" for p in aggregate.PROFILES]
                with (dst/"all-cases.csv").open("w",newline="") as f:
                    writer=csv.DictWriter(f,fieldnames=columns)
                    writer.writeheader()
                    for group,n in (("power2",1024),("mixed_radix",2047)):
                        for batch in (1,16):
                            row=dict(group=group,kind="fft",n=n,batch=batch,
                                     dtype="complex64",layout="C",pad=0,offset=0,
                                     output="fixed")
                            for p,t in zip(aggregate.PROFILES,(100,88,75,60,65)):
                                row[f"{p}_us"]=scale*t
                            writer.writerow(row)
            out=Path(temp)/"summary"
            aggregate.summarize(root,out,planned=64)
            report=(out/"CPU_GROUPED_REPORT.md").read_text()
            self.assertIn("3 / 64",report)
            self.assertIn("Example AMD Zen (2 successful runners)",report)
            self.assertIn("Example Intel (1 successful runners)",report)
            with (out/"CPU_GROUPED_CASES.csv").open(newline="") as f:
                cases=list(csv.DictReader(f))
            self.assertEqual(len(cases),8)
            amd=next(x for x in cases if x["cpu_model"]=="Example AMD Zen"
                     and x["suite"]=="power2" and x["batch"]=="16")
            self.assertAlmostEqual(float(amd["median_D_over_E"]),65/60)


if __name__=="__main__":
    unittest.main()
