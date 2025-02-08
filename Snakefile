import os
from glob import glob

# Get all subfolders in "data/" (e.g., BA1, BA2, ...)
SAMPLES = [os.path.basename(d) for d in glob("data/*") if os.path.isdir(d)]

rule all:
    input:
        expand("results/{sample}/1_normalized_data.csv",sample=SAMPLES),
        expand("results/{sample}/0_mean_sd.csv",sample=SAMPLES)

rule process_csv:
    input:
        data="data/{sample}/data.csv",
        markers="data/{sample}/marker.csv"
    output:
        data="results/{sample}/0_total_data.csv",
        stat="results/{sample}/0_mean_sd.csv"
    log:
        "logs/{sample}.log"
    shell:
        """
        python isolate_stimuli.py {input.data} {input.markers} {output.data} {output.stat} > {log} 2>&1
        """

rule normalize_stimuli:
    input:
        data=rules.process_csv.output.data
    output:
        data="results/{sample}/1_normalized_data.csv"
    log:
        "logs/{sample}.log"
    shell:
        """
        python normalize_stimuli.py {input.data} {output.data} > {log} 2>&1
        """
