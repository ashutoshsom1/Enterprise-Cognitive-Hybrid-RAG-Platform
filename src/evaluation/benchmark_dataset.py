"""Enterprise Golden Evaluation Dataset for RAG Benchmarking."""

from typing import Any, Dict, List

SAMPLE_ENTERPRISE_DOCUMENTS: List[Dict[str, Any]] = [
    {
        "doc_id": "doc_contract_sla_2024",
        "source": "legal/contracts/master_cloud_sla_v4.pdf",
        "title": "Master Cloud Enterprise Service Level Agreement (SLA)",
        "content": (
            "Section 8.1 - High Availability and Uptime Commitments: The Cloud Provider guarantees a Monthly Uptime "
            "Percentage of at least 99.99% for Tier-1 mission-critical enterprise workloads. If the uptime drops below "
            "99.99% but remains above 99.90%, the Customer is entitled to a 15% service credit. If uptime drops below 99.90%, "
            "the Customer is entitled to a 30% service credit. Liquidated damages are governed by Clause 14.2, capped at "
            "$2,500,000 per annual billing cycle. Either party may terminate this agreement with 60 days written notice "
            "under Clause 19.4 in the event of an uncurable material breach."
        ),
        "metadata": {"department": "legal", "classification": "confidential"},
    },
    {
        "doc_id": "doc_hardware_spec_h100",
        "source": "hardware/specs/datacenter_gpu_cluster_h100.pdf",
        "title": "Hyperscale AI Accelerator Cluster Technical Specifications",
        "content": (
            "Hardware Component Specifications: The high-throughput compute rack utilizes NVIDIA H100-SXM5-80GB accelerators. "
            "Each compute module has a Thermal Design Power (TDP) rating of 700W. Operating temperature must be maintained "
            "between 18°C and 24°C using chilled water liquid cooling loops. If junction temperature exceeds 90°C, the safety "
            "subsystem raises hardware fault error code ERR_THERMAL_THROTTLE_90C, automatically throttling clock frequencies "
            "to 405 MHz. Power distribution units must adhere to spec PDU-400A-3P with N+2 redundancy."
        ),
        "metadata": {"department": "infrastructure", "classification": "internal"},
    },
    {
        "doc_id": "doc_sre_k8s_runbook",
        "source": "devops/runbooks/kubernetes_p1_incident_remediation.md",
        "title": "Kubernetes Cluster P1 Incident Remediation Guide",
        "content": (
            "P1 Critical Incident Triage: When pod eviction triggers exit code OOMKilled_137 across worker nodes, "
            "the on-call SRE must execute failover procedures within 15 minutes. Step 1: Run 'kubectl drain --ignore-daemonsets "
            "--delete-emptydir-data <node_name>'. Step 2: Trigger automated cluster autoscaler rebalance via script "
            "'/opt/scripts/rebalance_nodepools.sh --force'. Step 3: Notify the Incident Commander on channel #sre-p1-war-room. "
            "Disaster recovery RTO is 30 minutes and RPO is 0 seconds."
        ),
        "metadata": {"department": "devops", "classification": "internal"},
    },
    {
        "doc_id": "doc_finance_procurement",
        "source": "finance/policies/procurement_governance_2024.json",
        "title": "Corporate Procurement and Expenditure Governance Policy",
        "content": (
            "Procurement Governance Article IV: Capital expenditures exceeding $250,000 require Level-4 executive sign-off "
            "from both the Chief Financial Officer (CFO) and Chief Information Officer (CIO). Any purchase order with foreign "
            "entities must undergo third-party vendor compliance check SEC-COMPLIANCE-99. Audit retention logs and invoices "
            "must be preserved immutably in tamper-evident storage for a mandatory statutory duration of 7 years."
        ),
        "metadata": {"department": "finance", "classification": "restricted"},
    },
]

GOLDEN_BENCHMARK_DATASET: List[Dict[str, Any]] = [
    {
        "id": "bench_001",
        "query": "What is the penalty credit if Monthly Uptime drops to 99.85% under the Master Cloud SLA?",
        "expected_answer": (
            "If uptime drops below 99.90%, the customer is entitled to a 30% service credit. "
            "Liquidated damages are governed by Clause 14.2 and capped at $2,500,000 per annual cycle."
        ),
        "ground_truth_doc_id": "doc_contract_sla_2024",
        "key_phrases": ["30% service credit", "Clause 14.2", "99.90%"],
    },
    {
        "id": "bench_002",
        "query": "What hardware error code is raised when H100 accelerator junction temperature exceeds 90°C?",
        "expected_answer": (
            "Hardware fault error code ERR_THERMAL_THROTTLE_90C is raised when junction temperature exceeds 90°C, "
            "throttling clock frequencies to 405 MHz."
        ),
        "ground_truth_doc_id": "doc_hardware_spec_h100",
        "key_phrases": ["ERR_THERMAL_THROTTLE_90C", "405 MHz", "700W"],
    },
    {
        "id": "bench_003",
        "query": "What is the immediate Step 1 action when pods are terminated with exit code OOMKilled_137?",
        "expected_answer": (
            "Step 1: Run 'kubectl drain --ignore-daemonsets --delete-emptydir-data <node_name>' within 15 minutes."
        ),
        "ground_truth_doc_id": "doc_sre_k8s_runbook",
        "key_phrases": ["kubectl drain", "OOMKilled_137", "15 minutes"],
    },
    {
        "id": "bench_004",
        "query": "Who must approve capital expenditures exceeding $250,000 and how long are audit logs kept?",
        "expected_answer": (
            "Capital expenditures exceeding $250,000 require Level-4 executive sign-off from both the CFO and CIO. "
            "Audit logs and invoices must be retained for 7 years."
        ),
        "ground_truth_doc_id": "doc_finance_procurement",
        "key_phrases": ["Level-4", "CFO", "CIO", "7 years"],
    },
]
