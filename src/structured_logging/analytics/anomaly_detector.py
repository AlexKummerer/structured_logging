"""
Anomaly detection for structured logs

This module provides anomaly detection capabilities to identify unusual
patterns, outliers, and deviations in log data.
"""

import statistics
from collections import Counter, deque
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from ..logger import get_logger


@dataclass
class AnomalyDetectorConfig:
    """Configuration for anomaly detection"""

    # Detection methods
    enable_statistical_detection: bool = True  # Statistical anomaly detection
    enable_pattern_deviation: bool = True  # Pattern-based anomaly detection
    enable_frequency_anomalies: bool = True  # Frequency/rate anomalies
    enable_value_anomalies: bool = True  # Field value anomalies

    # Statistical settings
    zscore_threshold: float = 3.0  # Z-score threshold for outliers
    iqr_multiplier: float = 1.5  # IQR multiplier for outlier detection
    min_baseline_samples: int = 100  # Minimum samples for baseline

    # Time series settings
    time_bucket_minutes: int = 5  # Time bucket size for rate analysis
    seasonality_period_hours: int = 24  # Period for seasonality detection
    trend_window_hours: int = 1  # Window for trend analysis

    # Pattern deviation settings
    pattern_similarity_threshold: float = 0.7  # Threshold for pattern matching
    rare_pattern_threshold: float = 0.01  # Threshold for rare patterns

    # Fields to analyze
    numeric_fields: List[str] = field(
        default_factory=lambda: [
            "response_time",
            "duration",
            "size",
            "count",
            "cpu_usage",
            "memory_usage",
            "error_rate",
        ]
    )
    categorical_fields: List[str] = field(
        default_factory=lambda: [
            "status_code",
            "error_type",
            "user_id",
            "endpoint",
            "operation",
            "level",
        ]
    )

    # Anomaly scoring
    min_anomaly_score: float = 0.7  # Minimum score to flag as anomaly
    combine_scores: bool = True  # Combine multiple detection methods

    # Context preservation
    include_context_window: int = 5  # Include N logs before/after anomaly
    track_anomaly_chains: bool = True  # Track related anomalies


@dataclass
class AnomalyScore:
    """Represents an anomaly score for a log entry"""

    score: float  # 0.0 (normal) to 1.0 (highly anomalous)
    detection_method: str  # statistical, pattern, frequency, value
    reason: str  # Human-readable reason
    details: Dict[str, Any] = field(default_factory=dict)

    @property
    def is_anomaly(self) -> bool:
        """Check if score indicates an anomaly"""
        return self.score >= 0.7


@dataclass
class LogAnomaly:
    """Represents a detected log anomaly"""

    anomaly_id: str
    log_entry: Dict[str, Any]
    timestamp: datetime
    scores: List[AnomalyScore]
    total_score: float
    anomaly_type: str  # primary anomaly type
    context_logs: List[Dict[str, Any]] = field(default_factory=list)
    related_anomalies: List[str] = field(default_factory=list)

    @property
    def primary_reason(self) -> str:
        """Get primary reason for anomaly"""
        if not self.scores:
            return "Unknown"
        return max(self.scores, key=lambda s: s.score).reason


class AnomalyDetector:
    """
    Detects anomalies in structured log data

    Uses multiple detection methods:
    - Statistical anomaly detection (z-score, IQR)
    - Pattern deviation detection
    - Frequency/rate anomalies
    - Categorical value anomalies
    """

    def __init__(self, config: Optional[AnomalyDetectorConfig] = None):
        self.config = config or AnomalyDetectorConfig()
        self.logger = get_logger("analytics.anomaly_detector")

        # Baseline statistics
        self._baseline_stats: Dict[str, Dict[str, float]] = {}
        self._value_frequencies: Dict[str, Counter] = {}
        self._pattern_history = deque(maxlen=1000)

        # Time series data
        self._rate_history: Dict[str, List[Tuple[datetime, int]]] = {}
        self._seasonal_baselines: Dict[int, Dict[str, float]] = {}

    def detect_anomalies(
        self,
        logs: List[Dict[str, Any]],
        baseline_logs: Optional[List[Dict[str, Any]]] = None,
    ) -> List[LogAnomaly]:
        """
        Detect anomalies in log data

        Args:
            logs: Logs to analyze for anomalies
            baseline_logs: Optional baseline logs for comparison

        Returns:
            List of detected anomalies
        """
        if not logs:
            return []

        # Build baseline if provided
        if baseline_logs:
            self._build_baseline(baseline_logs)
        elif not self._baseline_stats and len(logs) >= self.config.min_baseline_samples:
            # Use first portion of logs as baseline
            split_point = len(logs) // 3
            self._build_baseline(logs[:split_point])
            logs = logs[split_point:]

        anomalies = []

        # Analyze each log
        for i, log in enumerate(logs):
            scores = []

            # Run different detection methods
            if self.config.enable_statistical_detection:
                scores.extend(self._detect_statistical_anomalies(log))

            if self.config.enable_pattern_deviation:
                scores.extend(self._detect_pattern_anomalies(log))

            if self.config.enable_frequency_anomalies:
                scores.extend(self._detect_frequency_anomalies(log, logs))

            if self.config.enable_value_anomalies:
                scores.extend(self._detect_value_anomalies(log))

            # Calculate total score
            if scores:
                if self.config.combine_scores:
                    # Combine scores using maximum
                    total_score = max(s.score for s in scores)
                else:
                    # Average scores
                    total_score = sum(s.score for s in scores) / len(scores)

                # Check if anomaly
                if total_score >= self.config.min_anomaly_score:
                    anomaly = self._create_anomaly(log, scores, total_score, i, logs)
                    anomalies.append(anomaly)

        # Track anomaly chains
        if self.config.track_anomaly_chains:
            self._identify_anomaly_chains(anomalies)

        self.logger.info(
            f"Detected {len(anomalies)} anomalies in {len(logs)} logs",
            extra={
                "anomaly_count": len(anomalies),
                "log_count": len(logs),
                "anomaly_rate": len(anomalies) / len(logs) if logs else 0,
            },
        )

        return anomalies

    def _build_baseline(self, baseline_logs: List[Dict[str, Any]]) -> None:
        """Build baseline statistics from logs"""
        # Reset baseline
        self._baseline_stats.clear()
        self._value_frequencies.clear()

        # Collect numeric values
        numeric_values = {field: [] for field in self.config.numeric_fields}

        for log in baseline_logs:
            # Numeric fields
            for field in self.config.numeric_fields:
                value = log.get(field)
                if isinstance(value, (int, float)):
                    numeric_values[field].append(value)

            # Categorical fields
            for field in self.config.categorical_fields:
                value = log.get(field)
                if value is not None:
                    if field not in self._value_frequencies:
                        self._value_frequencies[field] = Counter()
                    self._value_frequencies[field][str(value)] += 1

        # Calculate statistics
        for field, values in numeric_values.items():
            if len(values) >= 2:
                self._baseline_stats[field] = {
                    "mean": statistics.mean(values),
                    "stdev": statistics.stdev(values),
                    "median": statistics.median(values),
                    "q1": self._percentile(values, 25),
                    "q3": self._percentile(values, 75),
                    "min": min(values),
                    "max": max(values),
                }

    def _detect_statistical_anomalies(self, log: Dict[str, Any]) -> List[AnomalyScore]:
        """Detect statistical anomalies in numeric fields"""
        scores = []

        for field in self.config.numeric_fields:
            value = log.get(field)
            if not isinstance(value, (int, float)):
                continue

            if field not in self._baseline_stats:
                continue

            stats = self._baseline_stats[field]

            # Z-score test
            if stats["stdev"] > 0:
                zscore = abs((value - stats["mean"]) / stats["stdev"])
                if zscore > self.config.zscore_threshold:
                    scores.append(
                        AnomalyScore(
                            score=min(zscore / (self.config.zscore_threshold * 2), 1.0),
                            detection_method="statistical",
                            reason=f"{field} value {value} is {zscore:.1f} standard deviations from mean",
                            details={
                                "field": field_name,
                                "value": value,
                                "zscore": zscore,
                                "mean": stats["mean"],
                                "stdev": stats["stdev"],
                            },
                        )
                    )

            # IQR test
            iqr = stats["q3"] - stats["q1"]
            if iqr > 0:
                lower_bound = stats["q1"] - self.config.iqr_multiplier * iqr
                upper_bound = stats["q3"] + self.config.iqr_multiplier * iqr

                if value < lower_bound or value > upper_bound:
                    distance = min(abs(value - lower_bound), abs(value - upper_bound))
                    scores.append(
                        AnomalyScore(
                            score=min(distance / (iqr * 2), 1.0),
                            detection_method="statistical",
                            reason=f"{field} value {value} is outside IQR bounds [{lower_bound:.2f}, {upper_bound:.2f}]",
                            details={
                                "field": field_name,
                                "value": value,
                                "iqr": iqr,
                                "bounds": [lower_bound, upper_bound],
                            },
                        )
                    )

        return scores

    def _detect_pattern_anomalies(self, log: Dict[str, Any]) -> List[AnomalyScore]:
        """Detect anomalies based on pattern deviation"""
        scores = []

        # Create pattern signature
        pattern_sig = self._create_pattern_signature(log)

        # Check against historical patterns
        if self._pattern_history:
            # Find most similar historical pattern
            max_similarity = 0
            for hist_pattern in self._pattern_history:
                similarity = self._calculate_pattern_similarity(
                    pattern_sig, hist_pattern
                )
                max_similarity = max(max_similarity, similarity)

            # Low similarity indicates anomaly
            if max_similarity < self.config.pattern_similarity_threshold:
                scores.append(
                    AnomalyScore(
                        score=1.0 - max_similarity,
                        detection_method="pattern",
                        reason="Log pattern deviates significantly from historical patterns",
                        details={
                            "max_similarity": max_similarity,
                            "pattern_signature": pattern_sig,
                        },
                    )
                )

        # Add to history
        self._pattern_history.append(pattern_sig)

        return scores

    def _detect_frequency_anomalies(
        self, log: Dict[str, Any], all_logs: List[Dict[str, Any]]
    ) -> List[AnomalyScore]:
        """Detect frequency/rate anomalies"""
        scores = []

        timestamp = log.get("timestamp", datetime.now())

        # Group logs by time buckets
        time_buckets = self._group_by_time_buckets(all_logs)

        # Find current bucket
        bucket_key = self._get_time_bucket_key(timestamp)
        bucket_size = len(time_buckets.get(bucket_key, []))

        # Calculate historical average
        if len(time_buckets) > 1:
            sizes = [len(logs) for logs in time_buckets.values()]
            avg_size = statistics.mean(sizes)
            stdev_size = statistics.stdev(sizes) if len(sizes) > 1 else 0

            if stdev_size > 0:
                zscore = abs((bucket_size - avg_size) / stdev_size)
                if zscore > 2:  # Lower threshold for rate anomalies
                    scores.append(
                        AnomalyScore(
                            score=min(zscore / 4, 1.0),
                            detection_method="frequency",
                            reason=f"Unusual log rate: {bucket_size} logs in {self.config.time_bucket_minutes} minutes (avg: {avg_size:.1f})",
                            details={
                                "bucket_size": bucket_size,
                                "average_size": avg_size,
                                "zscore": zscore,
                            },
                        )
                    )

        return scores

    def _detect_value_anomalies(self, log: Dict[str, Any]) -> List[AnomalyScore]:
        """Detect anomalies in categorical values"""
        scores = []

        for field in self.config.categorical_fields:
            value = log.get(field)
            if value is None:
                continue

            value_str = str(value)

            if field in self._value_frequencies:
                freq_counter = self._value_frequencies[field]
                total_count = sum(freq_counter.values())

                if total_count > 0:
                    # Check if value is rare
                    value_count = freq_counter.get(value_str, 0)
                    value_freq = value_count / total_count

                    if value_freq < self.config.rare_pattern_threshold:
                        # Rare or new value
                        rarity_score = 1.0 - value_freq
                        scores.append(
                            AnomalyScore(
                                score=rarity_score * 0.8,  # Scale down rarity scores
                                detection_method="value",
                                reason=f"Rare value for {field}: '{value_str}' (frequency: {value_freq:.2%})",
                                details={
                                    "field": field_name,
                                    "value": value_str,
                                    "frequency": value_freq,
                                    "count": value_count,
                                },
                            )
                        )

        return scores

    def _create_anomaly(
        self,
        log: Dict[str, Any],
        scores: List[AnomalyScore],
        total_score: float,
        index: int,
        all_logs: List[Dict[str, Any]],
    ) -> LogAnomaly:
        """Create an anomaly object"""
        timestamp = log.get("timestamp", datetime.now())

        # Determine primary anomaly type
        primary_score = max(scores, key=lambda s: s.score)
        anomaly_type = primary_score.detection_method

        anomaly = LogAnomaly(
            anomaly_id=f"anomaly_{timestamp.timestamp()}_{hash(str(log))}",
            log_entry=log,
            timestamp=timestamp,
            scores=scores,
            total_score=total_score,
            anomaly_type=anomaly_type,
        )

        # Add context logs if configured
        if self.config.include_context_window > 0:
            context_start = max(0, index - self.config.include_context_window)
            context_end = min(
                len(all_logs), index + self.config.include_context_window + 1
            )
            anomaly.context_logs = all_logs[context_start:context_end]

        return anomaly

    def _identify_anomaly_chains(self, anomalies: List[LogAnomaly]) -> None:
        """Identify related anomalies that form chains"""
        if len(anomalies) < 2:
            return

        # Sort by timestamp
        anomalies.sort(key=lambda a: a.timestamp)

        # Look for chains
        for i, anomaly1 in enumerate(anomalies):
            for j, anomaly2 in enumerate(anomalies[i + 1 :], i + 1):
                # Check time proximity
                time_diff = (anomaly2.timestamp - anomaly1.timestamp).total_seconds()
                if time_diff > 300:  # 5 minutes
                    break

                # Check for relationship
                if self._are_anomalies_related(anomaly1, anomaly2):
                    anomaly1.related_anomalies.append(anomaly2.anomaly_id)
                    anomaly2.related_anomalies.append(anomaly1.anomaly_id)

    def _are_anomalies_related(
        self, anomaly1: LogAnomaly, anomaly2: LogAnomaly
    ) -> bool:
        """Check if two anomalies are related"""
        # Same user/session/request
        for field in ["user_id", "session_id", "request_id", "trace_id"]:
            val1 = anomaly1.log_entry.get(field)
            val2 = anomaly2.log_entry.get(field)
            if val1 and val2 and val1 == val2:
                return True

        # Similar anomaly types
        if anomaly1.anomaly_type == anomaly2.anomaly_type:
            return True

        # Check for cause-effect patterns
        if anomaly1.anomaly_type == "value" and anomaly2.anomaly_type == "statistical":
            # Value anomaly might cause statistical anomaly
            return True

        return False

    def _create_pattern_signature(self, log: Dict[str, Any]) -> Dict[str, Any]:
        """Create a pattern signature for a log"""
        signature = {}

        # Include important fields
        for field in ["level", "status_code", "error_type", "operation"]:
            if field in log:
                signature[field] = log[field]

        # Include field presence
        signature["fields"] = set(log.keys())

        # Include value ranges for numeric fields
        for field in self.config.numeric_fields:
            value = log.get(field)
            if isinstance(value, (int, float)):
                # Bucket the value
                if field in self._baseline_stats:
                    stats = self._baseline_stats[field]
                    if value < stats["q1"]:
                        signature[f"{field}_range"] = "low"
                    elif value > stats["q3"]:
                        signature[f"{field}_range"] = "high"
                    else:
                        signature[f"{field}_range"] = "normal"

        return signature

    def _calculate_pattern_similarity(
        self, pattern1: Dict[str, Any], pattern2: Dict[str, Any]
    ) -> float:
        """Calculate similarity between two patterns"""
        if not pattern1 or not pattern2:
            return 0.0

        # Compare fields
        fields1 = pattern1.get("fields", set())
        fields2 = pattern2.get("fields", set())
        field_similarity = (
            len(fields1 & fields2) / len(fields1 | fields2) if fields1 | fields2 else 0
        )

        # Compare values
        common_keys = set(pattern1.keys()) & set(pattern2.keys()) - {"fields"}
        if not common_keys:
            return field_similarity

        value_matches = sum(1 for k in common_keys if pattern1[k] == pattern2[k])
        value_similarity = value_matches / len(common_keys)

        # Weighted average
        return 0.3 * field_similarity + 0.7 * value_similarity

    def _group_by_time_buckets(
        self, logs: List[Dict[str, Any]]
    ) -> Dict[str, List[Dict[str, Any]]]:
        """Group logs by time buckets"""
        buckets = {}

        for log in logs:
            timestamp = log.get("timestamp", datetime.now())
            bucket_key = self._get_time_bucket_key(timestamp)

            if bucket_key not in buckets:
                buckets[bucket_key] = []
            buckets[bucket_key].append(log)

        return buckets

    def _get_time_bucket_key(self, timestamp: datetime) -> str:
        """Get time bucket key for timestamp"""
        bucket_time = timestamp.replace(
            minute=(timestamp.minute // self.config.time_bucket_minutes)
            * self.config.time_bucket_minutes,
            second=0,
            microsecond=0,
        )
        return bucket_time.strftime("%Y-%m-%d %H:%M")

    def _percentile(self, values: List[float], percentile: float) -> float:
        """Calculate percentile of values"""
        if not values:
            return 0.0

        sorted_values = sorted(values)
        index = (len(sorted_values) - 1) * percentile / 100
        lower = sorted_values[int(index)]
        upper = sorted_values[min(int(index) + 1, len(sorted_values) - 1)]
        return lower + (upper - lower) * (index - int(index))


def detect_anomalies(
    logs: List[Dict[str, Any]],
    config: Optional[AnomalyDetectorConfig] = None,
    baseline_logs: Optional[List[Dict[str, Any]]] = None,
) -> List[LogAnomaly]:
    """
    Convenience function to detect anomalies in logs

    Args:
        logs: Logs to analyze
        config: Optional configuration
        baseline_logs: Optional baseline for comparison

    Returns:
        List of detected anomalies
    """
    detector = AnomalyDetector(config)
    return detector.detect_anomalies(logs, baseline_logs)


def create_anomaly_detector(
    config: Optional[AnomalyDetectorConfig] = None,
) -> AnomalyDetector:
    """
    Create an anomaly detector instance

    Args:
        config: Optional configuration

    Returns:
        AnomalyDetector instance
    """
    return AnomalyDetector(config)
