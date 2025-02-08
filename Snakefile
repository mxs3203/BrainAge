import os
from glob import glob

# Get all subfolders in "data/" (e.g., BA1, BA2, ...)
SAMPLES = [os.path.basename(d) for d in glob("data/*") if os.path.isdir(d)]

rule all:
    input:
        expand("results/{sample}/0_mean_sd.csv",sample=SAMPLES),
        expand("results/{sample}/1_quality_filter.csv", sample=SAMPLES),
        expand("results/{sample}/2_normalized_data.csv",sample=SAMPLES),
        expand("results/{sample}/3_fft_data.pck", sample=SAMPLES),
        expand("results/{sample}/4_brain_score.csv", sample=SAMPLES)

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
        python 0_isolate_stimuli.py {input.data} {input.markers} {output.data} {output.stat} > {log} 2>&1
        """
rule filter_samples:
    input:
        stat=rules.process_csv.output.stat
    output:
        report="results/{sample}/1_quality_filter.csv"
    log:
        "logs/{sample}.log"
    shell:
        """
        python 1_inspect_quality.py {input.stat} {output.report} > {log} 2>&1
        """

rule normalize_stimuli:
    input:
        data=rules.process_csv.output.data
    output:
        data="results/{sample}/2_normalized_data.csv"
    log:
        "logs/{sample}.log"
    shell:
        """
        python 2_normalize_stimuli.py {input.data} {output.data} > {log} 2>&1
        """

rule perform_fft:
    input:
        data=rules.normalize_stimuli.output.data
    output:
        data="results/{sample}/3_fft_data.pck"
    log:
        "logs/{sample}.log"
    shell:
        """
        python 3_perform_fft.py {input.data} {output.data} > {log} 2>&1
        """

rule compute_brian_score:
    input:
        data=rules.normalize_stimuli.output.data
    output:
        data="results/{sample}/4_brain_score.csv"
    log:
        "logs/{sample}.log"
    shell:
        """
        python 4_compute_brain_score.py {input.data} {output.data} > {log} 2>&1
        """