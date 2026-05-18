# 13 - references

## foundational frameworks & kill chain models

1. Hutchins, E., Cloppert, M., & Amin, R. (2011). *Intelligence-driven computer network defense informed by analysis of adversary campaigns and intrusion kill chains*. Lockheed Martin Corporation. [cyber kill chain - foundational framing for recon as phase 1]

2. MITRE ATT&CK Framework - Reconnaissance Tactic (TA0043). https://attack.mitre.org/tactics/TA0043/. [detailed sub-technique taxonomy for initial recon]

3. MITRE ATT&CK - Active Scanning (T1595), Gather Victim Host Information (T1592), Search Open Websites/Domains (T1593). [specific techniques this project automates]

---

## osint & recon tools (primary)

4. SpiderFoot Documentation. https://www.spiderfoot.net/documentation/. [osint automation framework - compared against in related work]

5. Amass Project. https://github.com/owasp/amass. [industry-standard subdomain enumeration - used as a passive source]

6. Shodan API Documentation. https://developer.shodan.io/api. [primary network intelligence source]

7. Censys Search API v2. https://search.censys.io/api. [secondary network + certificate intelligence source]

8. Certificate Transparency Logs - crt.sh. https://crt.sh. [passive subdomain discovery via certificate transparency]

---

## agentic ai frameworks

9. Chase, H. (2022). *LangChain: Building Applications with LLMs through Composability*. https://github.com/langchain-ai/langchain. [tool-use agent foundations]

10. Moura, J. (2023). *CrewAI: Role-based AI agents*. https://github.com/joaomdmoura/crewai. [primary agent framework for this project]

11. LangGraph Documentation. https://langchain-ai.github.io/langgraph/. [stateful multi-agent graph framework - considered for v2]

12. Microsoft AutoGen. https://github.com/microsoft/autogen. [alternative multi-agent framework - compared in related work]

---

## llm & agentic research papers

13. Yao, S., Zhao, J., Yu, D., Du, N., Shafran, I., Narasimhan, K., & Cao, Y. (2022). *ReAct: Synergizing Reasoning and Acting in Language Models*. arXiv:2210.03629. [react loop - core reasoning pattern for the planner agent]

14. Wei, J., Wang, X., Schuurmans, D., et al. (2022). *Chain-of-Thought Prompting Elicits Reasoning in Large Language Models*. arXiv:2201.11903. [cot prompting for planner decision quality]

15. Significant Gravitas. (2023). *AutoGPT: An Autonomous GPT-4 Experiment*. https://github.com/Significant-Gravitas/AutoGPT. [early autonomous agent - demonstrates multi-step tool use]

---

## ai for cybersecurity papers

16. Happe, A., & Cito, J. (2023). *Getting pwn'd by AI: Penetration testing with large language models*. arXiv:2308.00121. [empirical study of llm-guided pentesting - directly relevant]

17. Xu, Z., Liu, Z., Zheng, L., et al. (2024). *AutoAttacker: A Large Language Model Guided System to Implement Automatic Cyber-attacks*. arXiv:2403.01038. [llm-driven attack automation - related work context]

18. Deng, G., Liu, Y., Mayoral-Vilches, V., et al. (2023). *PentestGPT: An LLM-empowered Automatic Penetration Testing Tool*. arXiv:2308.06782. [closest related work - key differentiator analysis needed]

19. Zhang, H., et al. (2024). *LLMs as Agents for Cybersecurity: A Survey*. [broad survey of the space - confirms gap this project fills]

---

## legal & ethical frameworks

20. Computer Fraud and Abuse Act (CFAA), 18 U.S.C. § 1030. [us legal framework for unauthorized computer access]

21. Criminal Code of Canada, R.S.C. 1985, c. C-46, s. 342.1. [canadian legal framework for unauthorized computer access]

22. OWASP Testing Guide v4.2 - Information Gathering. https://owasp.org/www-project-web-security-testing-guide/. [authorized osint methodology reference]

---

## data sources (api documentation)

23. VirusTotal API v3. https://developers.virustotal.com/reference/overview. [reputation + passive dns queries]

24. Hunter.io API. https://hunter.io/api-documentation/v2. [email harvesting]

25. GitHub REST API - Search Code. https://docs.github.com/en/rest/search. [github dorking for leaked secrets]

26. IPWhois Python Library. https://pypi.org/project/ipwhois/. [asn/ip ownership lookups]

27. DNSPython. https://www.dnspython.org/. [dns record resolution library]
