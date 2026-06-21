"""Manifest schemas (documented as pure dicts, no pydantic).

Each stage reads upstream manifest and writes its own. Field contracts:

ingest        -> {audio_id, audio_path, speaker_id, language, domain,
                  recording_type, transcript_path, audio_hash}
audio_qc      -> + {snr_db, lufs, peak_db, clipping_ratio, effective_bw_hz,
                    duration_sec, qc_status, reject_reasons[]}
resample      -> + {audio_train_path, audio_align_path, train_sr, align_sr}
vad_coarse    -> seg-level: {seg_id, source_audio_id, audio_align_path,
                  audio_train_path, speaker_id, speaker_label, start, end,
                  duration, diar_confidence}
asr           -> + {asr_text, asr_text_normalized?, asr_confidence}
text_normalize-> + {text_normalized, manual_text?, cer_vs_manual?}
align         -> + {alignment_json_path, align_conf_mean, align_conf_p10,
                    high_conf_char_ratio, text_audio_duration_ratio}
fine_segment  -> utt-level: {utt_id, speaker_id, speaker_label, source_audio_id,
                  wav, text, prev_text, next_text, start, end, duration,
                  align_conf_mean}
spk_consistency -> + {spk_consistency: {passed, n_windows, max_dist_to_center,
                       max_neighbor_delta, cluster_silhouette,
                       overlap_ratio, ref_outlier_ratio}}
extract_f0    -> + {f0_mean_hz, f0_median_hz, f0_std_st, f0_range_st,
                    f0_delta_p95_st, voiced_ratio, f0_confidence,
                    f0_npy_path}
rate_metrics  -> + {global_rate_cps, local_rate_mean, local_rate_std,
                    local_rate_cv, local_rate_p5_p95_range, pause_ratio,
                    long_pause_count, n_chars}
filter_score  -> + {audio_quality_score, align_quality_score, pitch_score,
                    rate_score, text_quality_score, quality_score, grade,
                    prosody_bucket, status, reject_reasons[]}
split_dataset -> writes train/valid/test/rejected jsonl
"""
