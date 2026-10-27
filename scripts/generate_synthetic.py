"""Seeded synthetic document generator.

Produces fictional CVs, invoices, contracts and technical spec sheets with
character-level entity spans and token-level BIO labels. Everything is random
and fictional: Faker (fixed seed) names plus invented company names, and
reserved example.* e-mail domains.

    python scripts/generate_synthetic.py --n-per-type 200 --out data/generated
    python scripts/generate_synthetic.py --n-per-type 5 --out data/sample --seed 7
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from faker import Faker

from ftsdoc.tokens import tokenize

DOC_TYPES = ["cv", "invoice", "contract", "tech_spec"]
SKILLS = ["Python", "PyTorch", "TensorFlow", "SQL", "Docker", "Kubernetes", "React", "TypeScript",
          "Go", "Rust", "Spark", "Airflow", "FastAPI", "PostgreSQL", "Terraform", "AWS", "scikit-learn",
          "LangChain", "Java", "Linux", "Git", "Pandas", "Kafka", "GraphQL", "Redis"]
ROLES = ["Software Engineer", "Data Scientist", "ML Engineer", "Backend Developer", "DevOps Engineer",
         "Data Analyst", "Product Engineer", "Research Engineer"]
ORG_ROOTS = ["Veltrix", "Orbana", "Quillon", "Zephyra", "Marlow", "Kestrel", "Altavue", "Brindle",
             "Cobalto", "Dunmore", "Elmstead", "Fenwick", "Garnet", "Halcyon", "Ironvale", "Juniper"]
ORG_SUFFIX = ["Labs", "Systems", "Dynamics", "Holdings", "Industries", "Solutions", "Works", "GmbH", "Ltd", "Inc"]
EMAIL_DOMAINS = ["example.com", "example.org", "example.net"]
CURRENCIES = ["$", "EUR ", "GBP "]
PART_PREFIX = ["KX", "RT", "ZN", "QM", "HV", "TL"]


class Builder:
    """Accumulates text and entity spans."""

    def __init__(self) -> None:
        self.parts: list[str] = []
        self.spans: list[tuple[int, int, str]] = []
        self.pos = 0

    def t(self, s: str) -> "Builder":
        self.parts.append(s)
        self.pos += len(s)
        return self

    def e(self, s: str, label: str) -> "Builder":
        self.spans.append((self.pos, self.pos + len(s), label))
        return self.t(s)

    def text(self) -> str:
        return "".join(self.parts)


class Gen:
    def __init__(self, seed: int) -> None:
        self.rnd = random.Random(seed)
        Faker.seed(seed)
        self.fk = Faker("en_US")

    def org(self) -> str:
        return f"{self.rnd.choice(ORG_ROOTS)} {self.rnd.choice(ORG_SUFFIX)}"

    def person(self) -> str:
        return f"{self.fk.first_name()} {self.fk.last_name()}"

    def email(self, name: str | None = None) -> str:
        if name:
            first, last = name.lower().split()[0], name.lower().split()[-1]
            local = self.rnd.choice([f"{first}.{last}", f"{first[0]}{last}", f"{last}{self.rnd.randint(1, 99)}"])
        else:
            local = self.rnd.choice(["billing", "info", "support", "legal", "contact", "accounts"])
        return f"{local}@{self.rnd.choice(EMAIL_DOMAINS)}"

    def date(self, lo: int = 2015, hi: int = 2025) -> str:
        d = self.fk.date_object().replace(year=self.rnd.randint(lo, hi), day=1)
        d = d.replace(day=self.rnd.randint(1, 28))
        fmt = self.rnd.choice(["%Y-%m-%d", "%B %d, %Y", "%d %B %Y", "%d.%m.%Y"])
        return d.strftime(fmt).replace(" 0", " ").lstrip("0")

    def money(self, lo: int = 50, hi: int = 90000) -> str:
        cur = self.rnd.choice(CURRENCIES)
        return f"{cur}{self.rnd.randint(lo, hi):,}.{self.rnd.randint(0, 99):02d}"

    def skills(self, k: int) -> list[str]:
        return self.rnd.sample(SKILLS, k)

    # ---- document types -------------------------------------------------
    def cv(self) -> Builder:
        b, r = Builder(), self.rnd
        name = self.person()
        b.e(name, "PERSON").t("\n").t("Email: ").e(self.email(name), "EMAIL").t("\n\n")
        b.t(r.choice(["Professional Summary", "Profile"])).t("\n")
        b.t(f"{r.choice(['Motivated', 'Detail-oriented', 'Results-driven'])} {r.choice(ROLES).lower()} "
            f"with {r.randint(2, 15)} years of experience building production systems.\n\n")
        b.t("Experience\n")
        for _ in range(r.randint(2, 3)):
            b.t(f"{r.choice(ROLES)} at ").e(self.org(), "ORG").t(" (").e(self.date(2012, 2020), "DATE")
            b.t(" - ").e(self.date(2020, 2025), "DATE").t(")\n")
            b.t(r.choice(["Led delivery of internal tooling and mentored junior colleagues.",
                          "Shipped data pipelines and monitoring dashboards.",
                          "Owned the migration of legacy services to containers."])).t("\n")
        b.t("\nEducation\n")
        b.t(f"{r.choice(['BSc', 'MSc'])} in {r.choice(['Computer Science', 'Statistics', 'Mathematics'])}, ")
        b.e(f"University of {self.fk.city()}", "ORG").t(", ").e(self.date(2005, 2018), "DATE").t("\n\n")
        b.t("Skills: ")
        sk = self.skills(r.randint(4, 8))
        for i, s in enumerate(sk):
            b.e(s, "SKILL").t(", " if i < len(sk) - 1 else "\n")
        return b

    def invoice(self) -> Builder:
        b, r = Builder(), self.rnd
        b.t("INVOICE\n")
        b.t(r.choice(["Invoice No: ", "Invoice #", "Invoice Number: "]))
        b.e(f"INV-{r.randint(2020, 2025)}-{r.randint(0, 99999):05d}", "INVOICE_NO").t("\n")
        b.t("Issue Date: ").e(self.date(2022, 2025), "DATE").t("\n")
        b.t("Due Date: ").e(self.date(2022, 2025), "DATE").t("\n\n")
        b.t("From: ").e(self.org(), "ORG").t("\n")
        b.t("Bill To: ").e(self.org(), "ORG").t("\n")
        b.t("Contact: ").e(self.email(), "EMAIL").t("\n\n")
        b.t("Description / Amount\n")
        for _ in range(r.randint(2, 4)):
            b.t(f"{r.choice(['Consulting services', 'Software license', 'Cloud hosting', 'Support plan', 'Hardware'])} "
                f"x{r.randint(1, 20)}   ").e(self.money(), "MONEY").t("\n")
        b.t("\nTotal Due: ").e(self.money(1000, 120000), "MONEY").t("\n")
        b.t(r.choice(["Payment terms: net 30 days.", "Please pay by bank transfer.", "Thank you for your business."]))
        b.t("\n")
        return b

    def contract(self) -> Builder:
        b, r = Builder(), self.rnd
        b.t(f"{r.choice(['SERVICES', 'CONSULTING', 'LICENSE', 'SUPPLY'])} AGREEMENT\n\n")
        b.t("This Agreement is entered into as of ").e(self.date(2022, 2025), "DATE")
        b.t(" between ").e(self.org(), "ORG").t(' ("Provider") and ').e(self.org(), "ORG").t(' ("Client").\n\n')
        b.t("1. Term. This Agreement remains in force for ").t(f"{r.randint(6, 36)} months and renews unless terminated.\n")
        b.t("2. Fees. The Client shall pay the Provider ").e(self.money(5000, 500000), "MONEY")
        b.t(r.choice([" per year", " in total", " per quarter"])).t(" in accordance with the payment schedule.\n")
        b.t("3. Confidentiality. Each party shall keep the other party's information confidential.\n")
        b.t("4. Governing Law. This Agreement is governed by the laws of "
            f"{r.choice(['England and Wales', 'the State of Delaware', 'Germany'])}.\n")
        b.t("5. Notices. Legal notices shall be sent to ").e(self.email(), "EMAIL").t(".\n\n")
        b.t("Signed:\n")
        b.e(self.person(), "PERSON").t(", Provider\n")
        b.e(self.person(), "PERSON").t(", Client\n")
        return b

    def tech_spec(self) -> Builder:
        b, r = Builder(), self.rnd
        b.t("TECHNICAL SPECIFICATION\n")
        b.t("Model: ").e(f"{r.choice(PART_PREFIX)}-{r.randint(100, 9999)}-{r.choice('ABCDEF')}", "PART_NO").t("\n")
        b.t("Manufacturer: ").e(self.org(), "ORG").t("\n")
        b.t("Revision date: ").e(self.date(2020, 2025), "DATE").t("\n\n")
        b.t("Specifications\n")
        pool = [("Operating voltage", f"{r.choice([5, 12, 24, 48])} V"),
                ("Power consumption", f"{r.randint(2, 400)} W"),
                ("Operating temperature", f"-{r.randint(10, 40)} to {r.randint(60, 105)} C"),
                ("Weight", f"{r.randint(1, 90) / 10:.1f} kg"),
                ("Ingress protection", f"IP{r.choice([54, 65, 67, 68])}"),
                ("Interface", r.choice(["RS-485", "CAN bus", "Ethernet", "USB-C"])),
                ("Warranty", f"{r.randint(1, 5)} years")]
        for k, v in r.sample(pool, r.randint(4, 6)):
            b.t(f"{k}: {v}\n")
        b.t("\nList price: ").e(self.money(20, 9000), "MONEY").t("\n")
        b.t("Support: ").e(self.email(), "EMAIL").t("\n")
        return b


def to_sentences(b: Builder) -> list[dict]:
    """Per-line token/BIO sequences from character spans."""
    text, out, off = b.text(), [], 0
    for line in text.split("\n"):
        toks, labels, prev = [], [], None
        for tok, s, e in tokenize(line):
            gs, ge = s + off, e + off
            lab = "O"
            for ss, se, name in b.spans:
                if gs >= ss and ge <= se:
                    lab = ("I-" if prev == (ss, se) else "B-") + name
                    prev = (ss, se)
                    break
            else:
                prev = None
            toks.append(tok)
            labels.append(lab)
        if toks:
            out.append({"tokens": toks, "labels": labels})
        off += len(line) + 1
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-per-type", type=int, default=200)
    ap.add_argument("--out", default="data/generated")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    g = Gen(args.seed)
    out = Path(args.out)
    (out / "txt").mkdir(parents=True, exist_ok=True)
    records = []
    for dt in DOC_TYPES:
        for i in range(args.n_per_type):
            b = getattr(g, dt)()
            doc_id = f"{dt}_{i:04d}"
            (out / "txt" / f"{doc_id}.txt").write_text(b.text(), encoding="utf-8")
            records.append({"id": doc_id, "doc_type": dt, "text": b.text(), "sentences": to_sentences(b)})
    g.rnd.shuffle(records)
    with (out / "docs.jsonl").open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"wrote {len(records)} documents to {out}")


if __name__ == "__main__":
    main()
