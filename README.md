# Info-Flow Dynamics (IFD)

**Info-Flow Dynamics (IFD)** is a minimal, control-theoretic experimental platform for studying
*stability and failure modes* in information-driven systems under **partial observability**.

This repository does **not** model the world.
It models the interaction between:

- what a system can observe,
- how it is forced to act,
- and how structural damage accumulates over time.

All experiments are **CPU-only**, **toy-scale**, and **fully reproducible**.

---

## Motivation

Many failures in modern AI systems are described using terms such as:

- reward hacking,
- Goodhart’s Law,
- hallucination,
- over-refusal,
- or “fake alignment”.

IFD approaches these phenomena from a **systems engineering perspective**:

> When a system is forced to act under insufficient information,
> long-term structural degradation becomes inevitable.

The goal of this project is **not optimization**, but **mechanism isolation**:
to identify minimal conditions under which stability fails or survives.

---

## Design Principle

### Design Law #1  
**Information Insufficiency Is a Valid Control Output**

When observation quality is insufficient,
*refusing to act* or *requesting more information*
is a **stable control behavior**, not a failure.

This principle is treated as a **physical constraint**, not a value judgment.

---

## Minimal Mechanism Demonstrations

All experiments below share the same core dynamics and differ **only** in policy or governance structure.

---

### Experiment — Cavitation Under Forced Action

**Question** What happens if a system is forced to act regardless of information quality?

**Setup**

Two extreme policies under identical conditions:

- **FORCE**: always act, ignore information quality  
- **REST**: never act, allow full recovery

**Result**

- FORCE → structural health collapses to zero  
- REST → structural health remains intact

<p align="center">
  <img src="figures/cavitation_forced_vs_rest.png" width="600">
</p>

**Interpretation**

Forced action under uncertainty causes irreversible structural damage,
analogous to cavitation in physical systems.

This establishes a **hard boundary condition** for safe control.

---

### Experiment — Governance Transparency vs Stability

**Question** Does opacity in governance rules affect long-term system stability?

**Setup**

Two governance regimes:

- **Transparent**: fixed, predictable intervention threshold  
- **Opaque**: threshold includes unobservable stochastic variation

Both operate under the same environmental uncertainty.

**Result**

- Transparent governance yields **higher mean health** and **lower variance**
- Opaque governance accelerates degradation and increases instability

<p align="center">
  <img src="figures/governance_transparency.png" width="600">
</p>

**Interpretation**

Opacity introduces control noise.
Even when average constraints are similar, unpredictability prevents stable convergence.

This effect appears as **accelerated wear** and **higher variance**, not immediate collapse.

---

## What This Project Is NOT

To avoid misuse or over-interpretation, the scope is strictly limited.

- ❌ Not a model of human cognition or consciousness  
- ❌ Not a social, political, or economic theory  
- ❌ Not a normative alignment proposal  
- ❌ Not a training method or production-ready system  

IFD is an **engineering testbed**, nothing more.

---

## Why Toy Models Are Enough

These experiments deliberately avoid scale.

- The goal is **mechanism clarity**, not performance.
- Failure modes appear at minimal scale.
- CPU-only execution allows fast falsification.

If a hypothesis does **not** survive toy-scale testing,
it should not be trusted at scale.

---

## Reproducibility

```bash
python experiments/cavitation_forced_vs_rest.py
python experiments/governance_transparency.py

```

Each script:

* runs in seconds,
* saves figures automatically,
* produces deterministic summary statistics.

---

## Status

* **Stage**: Proof-of-Concept / Research Prototype
* **Audience**: Systems engineers, AI safety researchers, control theorists
* **License**: Research use

---

## Contact

Developed by **Jiaquan He**
Independent Researcher
Focus: system stability, failure analysis, and control under uncertainty


