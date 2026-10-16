"""Semantic grouping of stimulus labels, so "which conditions carry age" can be"""
import re

# "closed_eyes" added for the in-house gyrus_left/right_closed_eyes stimuli, which
# are recorded with eyes closed and so belong with the eyes-closed rest condition
EC = ("eyesclose", "eyesclosed", "resteyesc", "eyes_closed", "closed_eyes")
EO = ("eyesopen", "eyes_open", "_open")


def group(dataset, stimulus):
    s = str(stimulus).lower()
    # phase_Stimuli_phase_15 is an eyes-CLOSED segment, so it belongs with the
    # eyes-closed rest condition rather than with the undifferentiated phase_*
    if dataset == "ours" and s == "phase_stimuli_phase_15":   # s is lower-cased above
        return "rest_ec"
    if dataset == "ours" and s.startswith("phase_"):
        return "inter"
    if any(p in s for p in EC):
        return "rest_ec"
    if any(p in s for p in EO):
        return "rest_eo"
    if re.match(r"^rest(_|\d|$)", s):
        return "rest_un"
    return "task"


SETS = {
    "all":      None,                                  # everything
    "rest_any": {"rest_ec", "rest_eo", "rest_un"},
    "rest_ec":  {"rest_ec"},
    "rest_eo":  {"rest_eo"},
    "task":     {"task"},
    "no_inter": {"rest_ec", "rest_eo", "rest_un", "task"},   # drop in-house gaps
}


def keep_mask(dataset, stimuli, set_name):
    want = SETS[set_name]
    if want is None:
        return [True] * len(stimuli)
    return [group(dataset, s) in want for s in stimuli]
