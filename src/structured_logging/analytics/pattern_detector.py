"""
Pattern detection for structured logs

This module provides pattern detection capabilities to identify common
patterns, recurring sequences, and structured patterns in log data.
"""

import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Pattern, Set, Tuple

from ..logger import get_logger


@dataclass
class PatternDetectorConfig:
    """Configuration for pattern detection"""

    # Pattern detection settings
    min_pattern_frequency: int = 3  # Minimum occurrences to be considered a pattern
    pattern_similarity_threshold: float = 0.8  # Similarity threshold for fuzzy matching
    max_pattern_length: int = 100  # Maximum length of pattern to detect

    # Time window settings
    time_window_minutes: int = 60  # Time window for pattern analysis
    sliding_window: bool = True  # Use sliding window analysis

    # Pattern types to detect
    detect_error_patterns: bool = True  # Detect error message patterns
    detect_sequence_patterns: bool = True  # Detect sequential patterns
    detect_field_patterns: bool = True  # Detect field value patterns
    detect_frequency_patterns: bool = True  # Detect frequency-based patterns

    # Field analysis
    fields_to_analyze: List[str] = field(
        default_factory=lambda: [
            "level",
            "message",
            "error_type",
            "status_code",
            "user_id",
            "endpoint",
            "operation",
        ]
    )
    exclude_fields: Set[str] = field(
        default_factory=lambda: {"timestamp", "request_id", "trace_id"}
    )

    # Pattern templates
    error_patterns: List[Pattern] = field(
        default_factory=lambda: [
            re.compile(r"(?i)error:?\s*(.+)"),
            re.compile(r"(?i)exception:?\s*(.+)"),
            re.compile(r"(?i)failed:?\s*(.+)"),
            re.compile(r"(?i)timeout:?\s*(.+)"),
        ]
    )

    # Output settings
    include_samples: bool = True  # Include sample logs for each pattern
    max_samples: int = 5  # Maximum samples per pattern
    aggregate_similar: bool = True  # Aggregate similar patterns


@dataclass
class LogPattern:
    """Represents a detected log pattern"""

    pattern_id: str
    pattern_type: str  # error, sequence, field, frequency
    pattern_value: str
    frequency: int
    first_seen: datetime
    last_seen: datetime
    samples: List[Dict[str, Any]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def duration(self) -> timedelta:
        """Duration between first and last occurrence"""
        return self.last_seen - self.first_seen

    @property
    def rate_per_minute(self) -> float:
        """Average rate per minute"""
        duration_minutes = max(self.duration.total_seconds() / 60, 1)
        return self.frequency / duration_minutes


@dataclass
class PatternMatch:
    """Represents a pattern match in a log entry"""

    pattern: LogPattern
    log_entry: Dict[str, Any]
    match_score: float
    matched_fields: List[str]


class PatternDetector:
    """
    Detects patterns in structured log data

    Identifies common patterns including:
    - Error message patterns
    - Sequential operation patterns
    - Field value patterns
    - Frequency-based patterns
    """

    def __init__(self, config: Optional[PatternDetectorConfig] = None):
        self.config = config or PatternDetectorConfig()
        self.logger = get_logger("analytics.pattern_detector")
        self._pattern_cache: Dict[str, LogPattern] = {}
        self._sequence_buffer: List[Dict[str, Any]] = []
        self._field_counters: Dict[str, Counter] = defaultdict(Counter)

    def detect_patterns(
        self, logs: List[Dict[str, Any]], time_window: Optional[timedelta] = None
    ) -> List[LogPattern]:
        """
        Detect patterns in a collection of logs

        Args:
            logs: List of structured log entries
            time_window: Optional time window to analyze

        Returns:
            List of detected patterns
        """
        if not logs:
            return []

        # Filter logs by time window if specified
        if time_window:
            logs = self._filter_by_time_window(logs, time_window)

        patterns = []

        # Detect different types of patterns
        if self.config.detect_error_patterns:
            patterns.extend(self._detect_error_patterns(logs))

        if self.config.detect_sequence_patterns:
            patterns.extend(self._detect_sequence_patterns(logs))

        if self.config.detect_field_patterns:
            patterns.extend(self._detect_field_patterns(logs))

        if self.config.detect_frequency_patterns:
            patterns.extend(self._detect_frequency_patterns(logs))

        # Aggregate similar patterns if configured
        if self.config.aggregate_similar:
            patterns = self._aggregate_similar_patterns(patterns)

        # Sort by frequency
        patterns.sort(key=lambda p: p.frequency, reverse=True)

        self.logger.info(
            f"Detected {len(patterns)} patterns in {len(logs)} logs",
            extra={
                "pattern_count": len(patterns),
                "log_count": len(logs),
                "pattern_types": Counter(p.pattern_type for p in patterns),
            },
        )

        return patterns

    def _detect_error_patterns(self, logs: List[Dict[str, Any]]) -> List[LogPattern]:
        """Detect error message patterns"""
        error_patterns = []
        error_groups = defaultdict(list)

        for log in logs:
            # Check log level
            if log.get("level", "").lower() in ["error", "critical", "fatal"]:
                message = log.get("message", "")

                # Try to match against known error patterns
                for pattern in self.config.error_patterns:
                    match = pattern.search(message)
                    if match:
                        error_key = self._normalize_error_message(match.group(1))
                        error_groups[error_key].append(log)
                        break
                else:
                    # No pattern matched, use full message
                    error_key = self._normalize_error_message(message)
                    if error_key:
                        error_groups[error_key].append(log)

        # Create patterns from groups
        for error_key, error_logs in error_groups.items():
            if len(error_logs) >= self.config.min_pattern_frequency:
                pattern = self._create_pattern(
                    pattern_type="error", pattern_value=error_key, logs=error_logs
                )
                error_patterns.append(pattern)

        return error_patterns

    def _detect_sequence_patterns(self, logs: List[Dict[str, Any]]) -> List[LogPattern]:
        """Detect sequential operation patterns"""
        sequence_patterns = []

        # Group logs by user/session/request
        grouped_logs = self._group_logs_by_context(logs)

        # Analyze sequences within each group
        for context_key, context_logs in grouped_logs.items():
            if len(context_logs) < 2:
                continue

            # Extract operation sequences
            sequences = self._extract_sequences(context_logs)

            # Count sequence occurrences
            sequence_counts = Counter(sequences)

            # Create patterns for frequent sequences
            for sequence, count in sequence_counts.items():
                if count >= self.config.min_pattern_frequency:
                    pattern = LogPattern(
                        pattern_id=f"seq_{hash(sequence)}",
                        pattern_type="sequence",
                        pattern_value=str(sequence),
                        frequency=count,
                        first_seen=min(
                            log.get("timestamp", datetime.now()) for log in context_logs
                        ),
                        last_seen=max(
                            log.get("timestamp", datetime.now()) for log in context_logs
                        ),
                        metadata={"sequence_length": len(sequence)},
                    )
                    sequence_patterns.append(pattern)

        return sequence_patterns

    def _detect_field_patterns(self, logs: List[Dict[str, Any]]) -> List[LogPattern]:
        """Detect patterns in field values"""
        field_patterns = []

        # Analyze each configured field
        for field_name in self.config.fields_to_analyze:
            if field_name in self.config.exclude_fields:
                continue

            # Count field values
            value_counts = Counter()
            field_logs = defaultdict(list)

            for log in logs:
                value = log.get(field_name)
                if value is not None:
                    value_str = str(value)
                    value_counts[value_str] += 1
                    field_logs[value_str].append(log)

            # Create patterns for frequent values
            for value, count in value_counts.most_common():
                if count >= self.config.min_pattern_frequency:
                    pattern = self._create_pattern(
                        pattern_type="field",
                        pattern_value=f"{field_name}={value}",
                        logs=field_logs[value],
                        metadata={"field": field_name, "value": value},
                    )
                    field_patterns.append(pattern)

        return field_patterns

    def _detect_frequency_patterns(
        self, logs: List[Dict[str, Any]]
    ) -> List[LogPattern]:
        """Detect frequency-based patterns (bursts, periodic patterns)"""
        frequency_patterns = []

        # Group logs by time buckets
        time_buckets = self._group_by_time_buckets(logs, minutes=5)

        # Analyze frequency patterns
        for bucket_key, bucket_logs in time_buckets.items():
            if len(bucket_logs) >= self.config.min_pattern_frequency * 5:
                # High frequency burst detected
                pattern = LogPattern(
                    pattern_id=f"freq_burst_{bucket_key}",
                    pattern_type="frequency",
                    pattern_value=f"High frequency burst at {bucket_key}",
                    frequency=len(bucket_logs),
                    first_seen=min(
                        log.get("timestamp", datetime.now()) for log in bucket_logs
                    ),
                    last_seen=max(
                        log.get("timestamp", datetime.now()) for log in bucket_logs
                    ),
                    metadata={
                        "burst_type": "high_frequency",
                        "logs_per_minute": len(bucket_logs) / 5,
                    },
                )

                if self.config.include_samples:
                    pattern.samples = bucket_logs[: self.config.max_samples]

                frequency_patterns.append(pattern)

        return frequency_patterns

    def _normalize_error_message(self, message: str) -> str:
        """Normalize error message for grouping"""
        # Remove numbers, IDs, timestamps
        normalized = re.sub(r"\b\d+\b", "N", message)
        normalized = re.sub(
            r"[a-f0-9]{8}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{12}",
            "UUID",
            normalized,
        )
        normalized = re.sub(
            r"\d{4}-\d{2}-\d{2}[T\s]\d{2}:\d{2}:\d{2}", "TIMESTAMP", normalized
        )

        # Remove extra whitespace
        normalized = " ".join(normalized.split())

        # Truncate if too long
        if len(normalized) > self.config.max_pattern_length:
            normalized = normalized[: self.config.max_pattern_length] + "..."

        return normalized

    def _group_logs_by_context(
        self, logs: List[Dict[str, Any]]
    ) -> Dict[str, List[Dict[str, Any]]]:
        """Group logs by context (user, session, request)"""
        grouped = defaultdict(list)

        for log in logs:
            # Try different context keys
            context_key = (
                log.get("request_id")
                or log.get("session_id")
                or log.get("user_id")
                or log.get("trace_id")
                or "global"
            )
            grouped[context_key].append(log)

        return grouped

    def _extract_sequences(self, logs: List[Dict[str, Any]]) -> List[Tuple[str, ...]]:
        """Extract operation sequences from logs"""
        sequences = []

        # Sort logs by timestamp
        sorted_logs = sorted(
            logs, key=lambda log: log.get("timestamp", datetime.now())
        )

        # Extract sequences of operations
        for i in range(len(sorted_logs) - 1):
            sequence = []
            for j in range(i, min(i + 5, len(sorted_logs))):  # Max sequence length of 5
                operation = (
                    sorted_logs[j].get("operation")
                    or sorted_logs[j].get("action")
                    or sorted_logs[j].get("event")
                    or sorted_logs[j].get("message", "")[:50]
                )
                sequence.append(operation)

                if len(sequence) >= 2:
                    sequences.append(tuple(sequence))

        return sequences

    def _group_by_time_buckets(
        self, logs: List[Dict[str, Any]], minutes: int = 5
    ) -> Dict[str, List[Dict[str, Any]]]:
        """Group logs into time buckets"""
        buckets = defaultdict(list)

        for log in logs:
            timestamp = log.get("timestamp", datetime.now())
            # Round down to nearest bucket
            bucket_time = timestamp.replace(
                minute=(timestamp.minute // minutes) * minutes, second=0, microsecond=0
            )
            bucket_key = bucket_time.strftime("%Y-%m-%d %H:%M")
            buckets[bucket_key].append(log)

        return buckets

    def _create_pattern(
        self,
        pattern_type: str,
        pattern_value: str,
        logs: List[Dict[str, Any]],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> LogPattern:
        """Create a pattern from logs"""
        pattern = LogPattern(
            pattern_id=f"{pattern_type}_{hash(pattern_value)}",
            pattern_type=pattern_type,
            pattern_value=pattern_value,
            frequency=len(logs),
            first_seen=min(log.get("timestamp", datetime.now()) for log in logs),
            last_seen=max(log.get("timestamp", datetime.now()) for log in logs),
            metadata=metadata or {},
        )

        if self.config.include_samples:
            pattern.samples = logs[: self.config.max_samples]

        return pattern

    def _aggregate_similar_patterns(
        self, patterns: List[LogPattern]
    ) -> List[LogPattern]:
        """Aggregate similar patterns"""
        if not patterns:
            return patterns

        # Group similar patterns
        aggregated = []
        used = set()

        for i, pattern1 in enumerate(patterns):
            if i in used:
                continue

            similar_group = [pattern1]

            for j, pattern2 in enumerate(patterns[i + 1 :], i + 1):
                if j in used:
                    continue

                if self._are_patterns_similar(pattern1, pattern2):
                    similar_group.append(pattern2)
                    used.add(j)

            # Create aggregated pattern
            if len(similar_group) > 1:
                aggregated_pattern = self._merge_patterns(similar_group)
                aggregated.append(aggregated_pattern)
            else:
                aggregated.append(pattern1)

        return aggregated

    def _are_patterns_similar(self, pattern1: LogPattern, pattern2: LogPattern) -> bool:
        """Check if two patterns are similar"""
        if pattern1.pattern_type != pattern2.pattern_type:
            return False

        # Calculate similarity based on pattern type
        if pattern1.pattern_type == "error":
            return (
                self._calculate_string_similarity(
                    pattern1.pattern_value, pattern2.pattern_value
                )
                >= self.config.pattern_similarity_threshold
            )

        elif pattern1.pattern_type == "field":
            # Similar if same field
            return pattern1.metadata.get("field") == pattern2.metadata.get("field")

        return False

    def _calculate_string_similarity(self, str1: str, str2: str) -> float:
        """Calculate similarity between two strings"""
        # Simple character-based similarity
        common_chars = sum(1 for c1, c2 in zip(str1, str2) if c1 == c2)
        max_len = max(len(str1), len(str2))
        return common_chars / max_len if max_len > 0 else 0

    def _merge_patterns(self, patterns: List[LogPattern]) -> LogPattern:
        """Merge similar patterns into one"""
        # Use the most frequent pattern as base
        base_pattern = max(patterns, key=lambda p: p.frequency)

        # Merge data
        merged = LogPattern(
            pattern_id=base_pattern.pattern_id,
            pattern_type=base_pattern.pattern_type,
            pattern_value=base_pattern.pattern_value,
            frequency=sum(p.frequency for p in patterns),
            first_seen=min(p.first_seen for p in patterns),
            last_seen=max(p.last_seen for p in patterns),
            metadata={
                **base_pattern.metadata,
                "merged_count": len(patterns),
                "variations": [p.pattern_value for p in patterns],
            },
        )

        # Merge samples
        if self.config.include_samples:
            all_samples = []
            for p in patterns:
                all_samples.extend(p.samples)
            merged.samples = all_samples[: self.config.max_samples]

        return merged

    def _filter_by_time_window(
        self, logs: List[Dict[str, Any]], time_window: timedelta
    ) -> List[Dict[str, Any]]:
        """Filter logs by time window"""
        if not logs:
            return logs

        # Find the latest timestamp
        latest_time = max(log.get("timestamp", datetime.now()) for log in logs)
        cutoff_time = latest_time - time_window

        # Filter logs
        filtered = [
            log for log in logs if log.get("timestamp", datetime.now()) >= cutoff_time
        ]

        return filtered

    def find_pattern_matches(
        self, log: Dict[str, Any], patterns: List[LogPattern]
    ) -> List[PatternMatch]:
        """Find which patterns match a given log entry"""
        matches = []

        for pattern in patterns:
            match_score = 0
            matched_fields = []

            # Check pattern type
            if pattern.pattern_type == "error":
                message = log.get("message", "")
                normalized = self._normalize_error_message(message)
                if normalized == pattern.pattern_value:
                    match_score = 1.0
                    matched_fields.append("message")

            elif pattern.pattern_type == "field":
                field = pattern.metadata.get("field")
                value = pattern.metadata.get("value")
                if field and str(log.get(field)) == str(value):
                    match_score = 1.0
                    matched_fields.append(field)

            # Add match if found
            if match_score > 0:
                matches.append(
                    PatternMatch(
                        pattern=pattern,
                        log_entry=log,
                        match_score=match_score,
                        matched_fields=matched_fields,
                    )
                )

        return matches


def detect_patterns(
    logs: List[Dict[str, Any]], config: Optional[PatternDetectorConfig] = None
) -> List[LogPattern]:
    """
    Convenience function to detect patterns in logs

    Args:
        logs: List of structured log entries
        config: Optional configuration

    Returns:
        List of detected patterns
    """
    detector = PatternDetector(config)
    return detector.detect_patterns(logs)


def create_pattern_detector(
    config: Optional[PatternDetectorConfig] = None,
) -> PatternDetector:
    """
    Create a pattern detector instance

    Args:
        config: Optional configuration

    Returns:
        PatternDetector instance
    """
    return PatternDetector(config)
