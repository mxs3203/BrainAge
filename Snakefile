import os
from glob import glob

# Get all subfolders in "data/" (e.g., BA1, BA2, ...)
SAMPLES = [os.path.basename(d) for d in glob("data/*") if os.path.isdir(d)]

rule all:
    input:
        expand("results/{sample}/total_data.csv",sample=SAMPLES),
        expand("results/{sample}/mean_sd.csv",sample=SAMPLES)

rule process_csv:
    input:
        data="data/{sample}/data.csv",
        markers="data/{sample}/marker.csv"
    output:
        data="results/{sample}/total_data.csv",
        stat="results/{sample}/mean_sd.csv"
    log:
        "logs/{sample}.log"
    shell:
        """
        python isolate_stimuli.py {input.data} {input.markers} {output.data} {output.stat} > {log} 2>&1
        """
