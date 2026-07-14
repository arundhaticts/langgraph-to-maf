# Framework Knowledge Store (Tier 3 context)

Each subfolder is a framework knowledge pack, injected into the Tier 3 LLM prompt
when the rules cannot resolve a construct. Adding a framework here (docs +
examples + vocabulary) is enough to make it a Tier 3 target -- no core pipeline
changes (Section 4.1 / Section 16 of the build plan).

```
frameworks/
├── maf/              # Microsoft Agent Framework (initial target)
│   ├── docs.md
│   ├── examples/
│   └── vocabulary.json
├── crewai/
├── autogen/
└── semantic_kernel/
```

Tier 1 / Tier 2 do NOT read this store -- they use the target adapters and
templates. Only Tier 3 retrieves from here.
