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

# MITRE ATT&CK Technique ID mappings for UNSW attack classifications
MITRE_ATTACK_MAPPING: Dict[str, Dict[str, str]] = {
    "Normal": {"technique_id": "T0000", "technique_name": "Benign Traffic", "tactic": "None"},
    "Fuzzers": {"technique_id": "T1499.004", "technique_name": "Application or System Exploitation (Fuzzing)", "tactic": "Impact"},
    "Analysis": {"technique_id": "T1040", "technique_name": "Network Sniffing / Port Scanning", "tactic": "Credential Access / Discovery"},
    "Backdoors": {"technique_id": "T1059", "technique_name": "Command and Scripting Interpreter", "tactic": "Execution / Persistence"},
    "DoS": {"technique_id": "T1498", "technique_name": "Network Denial of Service", "tactic": "Impact"},
    "Exploits": {"technique_id": "T1203", "technique_name": "Exploitation for Client Execution", "tactic": "Execution"},
    "Generic": {"technique_id": "T1071", "technique_name": "Standard Cryptographic / Generic Protocol Abuse", "tactic": "Command and Control"},
    "Reconnaissance": {"technique_id": "T1595", "technique_name": "Active Scanning", "tactic": "Reconnaissance"},
    "Shellcode": {"technique_id": "T1055", "technique_name": "Process Injection / Shellcode Execution", "tactic": "Defense Evasion / Privilege Escalation"},
    "Worms": {"technique_id": "T1570", "technique_name": "Lateral Tool Transfer / Self-Replication", "tactic": "Lateral Movement"},
}

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

# Default Port Mappings for Well-Known Network Services
DEFAULT_SERVICE_PORTS: Dict[str, int] = {
    "http": 80,
    "ssl": 443,
    "ftp": 21,
    "ftp-data": 20,
    "ssh": 22,
    "dns": 53,
    "smtp": 25,
    "pop3": 110,
    "snmp": 161,
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
