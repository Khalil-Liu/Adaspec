# AdaSpec Revision Experiment Backup

This directory contains only the Python scripts and corresponding JSON outputs for the additional revision experiments. Logs, ZIP archives, model files, datasets, figures, and working notes are excluded.

The public copy removes machine-specific absolute paths and temporary publisher access query strings. This sanitization does not alter predictions, scores, statistical results, or experiment configurations. Set the `MODEL` environment variable and project asset paths for your local environment before rerunning the historical replay scripts.

## Directory Structure

- `scripts/table3_paired_truthfulqa.py`: Recomputes the paired confidence intervals and statistical tests in Table 3 from the saved SEA and AdaSpec question-level records.
- `scripts/controlled_audit/`: Scripts for the prompt-only probe, calibration, bypass, and routing audit.
- `scripts/historical_replay/`: Scripts for the matched TruthfulQA and BBQ replays under the candidate-continuation scoring protocol.
- `outputs/table3_paired_truthfulqa/`: Question-level SEA and AdaSpec records and the resulting `paired_statistics.json` for Table 3.
- `outputs/controlled_audit/`: Question-level and summary JSON outputs for the controlled TruthfulQA and BBQ audit, including the probe results used in Table 6.
- `outputs/matched_truthfulqa/`: Per-policy question-level results and `matched_policy_report.json` used in Table 7.
- `outputs/matched_bbq/`: Ambiguous and disambiguated BBQ results and `matched_bbq_audit.json` used in Table 4.

## Main Result Files

- Table 3: `outputs/table3_paired_truthfulqa/paired_statistics.json`
- Table 4: `outputs/matched_bbq/matched_bbq_audit.json`
- Table 6: `outputs/controlled_audit/router_test_report.json`
- Table 7: `outputs/matched_truthfulqa/matched_policy_report.json`

The controlled prompt-only audit and the candidate-continuation replay use different evaluation protocols. Their results are stored separately and should not be treated as numerical reproductions of one another. Except for the standalone Table 3 statistics script, the supplied scripts require the original project environment, including its models, datasets, and saved projectors.
