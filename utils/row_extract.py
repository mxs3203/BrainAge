import pandas as pd

def extract_quality_segments(df, cq_overall_col='CQ.Overall', cq_prefix='CQ.',
                             eeg_prefix='EEG.', task_col='task_name',
                             overall_thrs=70, channel_thrs=2, min_len=10):
    # ch quality columns
    cq_cols = [c for c in df.columns if c.startswith(cq_prefix) and c != cq_overall_col]

    # overal Q must be larger than X and each channel quality must be larger than Y
    good_mask = (df[cq_overall_col] > overall_thrs) & (df[cq_cols] > channel_thrs).all(axis=1)

    segments_by_task = {}
    report_rows = []

    # Iterate by task
    for task, task_df in df.groupby(task_col):
        task_mask = good_mask.loc[task_df.index]
        segment_ids = (task_mask != task_mask.shift()).cumsum()

        task_segments = []
        for seg_id, seg_df in task_df.groupby(segment_ids):
            if task_mask.loc[seg_df.index].all() and len(seg_df) >= min_len:
                task_segments.append(seg_df.reset_index(drop=True))

        # Count kept vs discarded
        kept_rows = sum(len(seg) for seg in task_segments)
        total_rows = len(task_df)
        discarded_rows = total_rows - kept_rows
        discarded_pct = 100 * discarded_rows / total_rows if total_rows > 0 else 0

        report_rows.append({
            "task_name": task,
            "total_rows": total_rows,
            "kept_rows": kept_rows,
            "discarded_rows": discarded_rows,
            "discarded_percent": round(discarded_pct, 2)
        })

        segments_by_task[task] = task_segments

    report_df = pd.DataFrame(report_rows)
    return segments_by_task, report_df
