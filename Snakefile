import os
from glob import glob

from PIL.features import features

# Get all subfolders in "data/" (e.g., BA1, BA2, ...)
SAMPLES = [os.path.basename(d) for d in glob("data/*") if os.path.isdir(d)]

rule all:
    input:
        expand("results/{sample}/0_mean_sd.csv",sample=SAMPLES),
        expand("results/{sample}/1_quality_filter.csv", sample=SAMPLES),
        expand("results/{sample}/2_emotiv_fft_values.csv", sample=SAMPLES),
        expand("results/{sample}/2_non_normalized_dictionary_of_stimuli.pck", sample=SAMPLES),
        expand("results/{sample}/2_consecutive_rows_report.csv", sample=SAMPLES),
        expand("results/{sample}/3_normalized_dictionary_of_stimuli.pck",sample=SAMPLES),
        expand("results/{sample}/4_fft_data.pck", sample=SAMPLES),
        expand("results/{sample}/4_EEG_features.csv", sample=SAMPLES),
        expand("results/{sample}/5_lstm_rdy_data.txt", sample=SAMPLES)


rule isolate_stimuli_based_on_markers:
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

rule extract_emotiv_fft:
    input:
        data=rules.isolate_stimuli_based_on_markers.output.data
    output:
        emotiv_fft="results/{sample}/2_emotiv_fft_values.csv"
    log:
        "logs/{sample}.log"
    shell:
        """
           python 2_isolate_emotiv_fft.py {input.data} {output.emotiv_fft} > {log} 2>&1
        """

rule inspect_quality:
    input:
        stat=rules.isolate_stimuli_based_on_markers.output.stat
    output:
        report="results/{sample}/1_quality_filter.csv"
    log:
        "logs/{sample}.log"
    shell:
        """
        python 1_inspect_quality.py {input.stat} {output.report} > {log} 2>&1
        """

rule filter_samples:
    input:
        data = rules.isolate_stimuli_based_on_markers.output.data
    output:
        dict="results/{sample}/2_non_normalized_dictionary_of_stimuli.pck",
        report="results/{sample}/2_consecutive_rows_report.csv"
    log:
        "logs/{sample}.log"
    shell:
        """
        python 2_consecutive_rows.py {input.data} {output.dict} {output.report} > {log} 2>&1
        """

rule normalize_stimuli:
    input:
        data=rules.filter_samples.output.dict
    output:
        data="results/{sample}/3_normalized_dictionary_of_stimuli.pck"
    log:
        "logs/{sample}.log"
    shell:
        """
        python 3_normalize_stimuli.py {input.data} {output.data} > {log} 2>&1
        """

rule perform_feature_extract:
    input:
        non_normalized_data=rules.filter_samples.output.dict,
        normalized_data=rules.normalize_stimuli.output.data
    output:
        data="results/{sample}/4_fft_data.pck",
        features="results/{sample}/4_EEG_features.csv"
    log:
        "logs/{sample}.log"
    shell:
        """
        python 4_perform_feature_extract.py {input.non_normalized_data} {input.normalized_data} {output.data} {output.features} > {log} 2>&1
        """


rule prepare_data_for_lstm:
    input:
        data=rules.normalize_stimuli.output.data
    output:
        flag="results/{sample}/5_lstm_rdy_data.txt"
    log:
        "logs/{sample}.log"
    shell:
        """
        python 5_make_data_for_LSTM.py {input.data} {output.flag} > {log} 2>&1
        """