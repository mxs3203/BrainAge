import os
from glob import glob

# Get all subfolders in "data/" (e.g., BA1, BA2, ...)
SAMPLES = [os.path.basename(d) for d in glob("data/*") if os.path.isdir(d)]

rule all:
    input:
        expand("results/{sample}/processed.csv", sample=SAMPLES)

rule process_csv:
    input:
        data="data/{sample}/data.csv",
        markers="data/{sample}/markers.csv"
    output:
        "results/{sample}/processed.csv"
    log:
        "logs/{sample}.log"
    shell:
        """
        python scripts/process.py {input.data} {input.markers} {output} > {log} 2>&1
        """
