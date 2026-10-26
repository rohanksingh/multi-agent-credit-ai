
from orchestrator import orchestrator

tests = [
    (
        "Microsoft financial analysis",
        "Analyze Microsoft's financial performance in 2025.",
        "MICROSOFT",
        {"financial_rag"}
    ),
    (
        "ABC credit analysis",
        "Analyze ABC Corporation's credit utilization and payment history.",
        "ABC_CORP",
        {"credit_rag"}
    ),
    (
        "ABC combined analysis",
        "Evaluate ABC Corporation's financial statements and credit report.",
        "ABC_CORP",
        {"financial_rag", "credit_rag"}
    )
]

for name, question, expected_company, expected_agents in tests:
    decision = orchestrator(question)

    actual_agents = set(decision.selected_agents)

    passed = (
        decision.company_id == expected_company
        and actual_agents == expected_agents
    )

    print(f"\n{name}")
    print("Company:", decision.company_id)
    print("Agents:", decision.selected_agents)
    print("Result:", "PASS" if passed else "FAIL")

    if not passed:
        print("Expected company:", expected_company)
        print("Expected agents:", sorted(expected_agents))
