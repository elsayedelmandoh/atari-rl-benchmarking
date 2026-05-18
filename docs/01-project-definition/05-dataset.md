# 05 - dataset & data sources

## data type

this project does not use a traditional ml training dataset. instead, it consumes **live osint data from public apis and data sources** at runtime, and uses **curated ground-truth target profiles** for evaluation.

---

## runtime data sources (osint inputs)

| source | data type | access method | rate limits |
|--------|-----------|---------------|-------------|
| **shodan** | exposed ports, services, banners, vulns | api (free/paid tier) | 1 req/sec (free), higher on paid |
| **censys** | tls certs, port scans, host data | api (free researcher tier) | 0.2 req/sec (free) |
| **crt.sh** | certificate transparency logs, subdomains | api (public, no key) | no hard limit, be polite |
| **virustotal** | domain/ip reputation, historical dns, passive dns | api (free: 4 req/min) | 4 req/min free |
| **whois / arin / ripe** | ip ownership, asn, org info | whois protocol / api | minimal |
| **dns resolvers** | a, mx, ns, txt, aaaa records | system dns / dnspython | no limit |
| **github api** | code search (dorking for leaked secrets/configs) | api (auth required) | 30 req/min authenticated |
| **hunter.io** | email address enumeration for a domain | api (free: 25/month) | 25 req/month free |
| **google dorks (serp api)** | indexed pages, exposed files, admin panels | serp api or google custom search | paid or limited free |

---

## evaluation dataset (ground truth)

for benchmarking the system's recall and accuracy, a set of **manually-built target profiles** will be created against:

| target type | examples | justification |
|-------------|----------|---------------|
| ctf/intentionally vulnerable domains | hackthebox machines, vulnhub, tryhackme | legal, safe, reproducible |
| researcher-owned infrastructure | personal vps, lab domains | fully authorized |
| bug bounty public scopes | hackerone / bugcrowd public programs | explicitly authorized for recon |
| public osint challenges | osint dojo, osint framework exercises | designed for this use case |

**ground truth construction process:**
1. manually execute full recon against target using standard tools (amass, shodan manual search, dnsx, etc.)
2. document all discovered assets in a structured json schema
3. run the agent system against the same target
4. compare agent output vs. ground truth using precision / recall / f1

---

## evaluation schema (ground truth json format)

```json
{
  "target": "example.com",
  "timestamp": "2025-01-01T00:00:00Z",
  "subdomains": ["api.example.com", "mail.example.com"],
  "ips": ["93.184.216.34"],
  "open_ports": [{"host": "93.184.216.34", "port": 443, "service": "https"}],
  "emails": ["admin@example.com"],
  "technologies": ["nginx", "cloudflare", "wordpress"],
  "certificates": [...],
  "risk_indicators": ["exposed admin panel", "leaked api key in github"]
}
```

---

## data handling notes

- no pii is stored beyond what's publicly available
- all runtime api keys scoped to read-only osint queries
- evaluation targets confirmed as legally authorized before any queries
- rate limit compliance enforced at the agent level (token bucket per source)
