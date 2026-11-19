"""Network security constants, column definitions, and enumerations for NetSentry."""

from enum import Enum
from typing import Dict, List


class DatasetType(str, Enum):
    """Supported network intrusion dataset benchmarks."""
    UNSW_NB15 = "unsw_nb15"
    CIC_IDS2017 = "cic_ids2017"
    GENERIC = "generic"


class MLAlgorithm(str, Enum):
    """Supported machine learning classification algorithms."""
    LOGISTIC_REGRESSION = "logistic_regression"
    DECISION_TREE = "decision_tree"
    RANDOM_FOREST = "random_forest"
    KNN = "knn"


class FeatureSelectionStrategy(str, Enum):
    """Supported feature selection strategies for the research component."""
    NONE = "none"
    VARIANCE_THRESHOLD = "variance_threshold"
    CORRELATION_FILTER = "correlation_filter"
    RF_IMPORTANCE = "rf_importance"
    MUTUAL_INFO = "mutual_info"


class ThreatSeverity(str, Enum):
    """Intrusion alert threat severity levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class TrafficClass(int, Enum):
    """Binary traffic classification values."""
    NORMAL = 0
    MALICIOUS = 1


# Attack category mappings
ATTACK_CATEGORIES: List[str] = [
    "Normal",
    "Fuzzers",
    "Analysis",
    "Backdoors",
    "DoS",
    "Exploits",
    "Generic",
    "Reconnaissance",
    "Shellcode",
    "Worms",
]

# Attack severity mapping
ATTACK_SEVERITY_MAP: Dict[str, ThreatSeverity] = {
    "Normal": ThreatSeverity.LOW,
    "Fuzzers": ThreatSeverity.MEDIUM,
    "Analysis": ThreatSeverity.MEDIUM,
    "Backdoors": ThreatSeverity.CRITICAL,
    "DoS": ThreatSeverity.HIGH,
    "Exploits": ThreatSeverity.CRITICAL,
    "Generic": ThreatSeverity.MEDIUM,
    "Reconnaissance": ThreatSeverity.MEDIUM,
    "Shellcode": ThreatSeverity.CRITICAL,
    "Worms": ThreatSeverity.CRITICAL,
}

# UNSW-NB15 Standard Schema
UNSW_CATEGORICAL_COLUMNS: List[str] = ["proto", "service", "state"]

UNSW_NUMERICAL_COLUMNS: List[str] = [
    "dur", "sbytes", "dbytes", "sttl", "dttl", "sloss", "dloss",
    "Sload", "Dload", "Spkts", "Dpkts", "swin", "dwin", "stcpb", "dtcpb",
    "smeansz", "dmeansz", "trans_depth", "res_bdy_len", "sjit", "djit",
    "sintpkt", "dintpkt", "tcprtt", "synack", "ackdat",
    "is_sm_ips_ports", "ct_state_ttl", "ct_flw_http_mthd", "is_ftp_login",
    "ct_ftp_cmd", "ct_srv_src", "ct_srv_dst", "ct_dst_ltm", "ct_src_ltm",
    "ct_src_dport_ltm", "ct_dst_sport_ltm", "ct_dst_src_ltm"
]

UNSW_TARGET_LABEL: str = "label"
UNSW_ATTACK_CAT: str = "attack_cat"

# Common Network Protocols
COMMON_PROTOCOLS: List[str] = ["tcp", "udp", "icmp", "arp", "ospf", "sctp"]
COMMON_SERVICES: List[str] = ["-", "http", "ftp", "smtp", "ssh", "dns", "ftp-data", "pop3", "ssl", "snmp"]
COMMON_STATES: List[str] = ["FIN", "CON", "INT", "REQ", "RST", "ACC", "CLO"]
