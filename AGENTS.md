# AGENTS.md

Guidance for AI agents working in **RecurrentLayers.jl**.

## Build, Test, Format

Run commands from the repository root.

- Full test suite: `julia --project -e 'using Pkg; Pkg.test()'`
- Format code and docstrings: `julia -e 'using JuliaFormatter; format(".")'`
- If `JuliaFormatter` is not installed globally, use a temporary environment:
  `julia -e 'using Pkg; Pkg.activate(; temp=true); Pkg.add("JuliaFormatter"); using JuliaFormatter; format(".")'`

`JuliaFormatter` is not a project dependency. Do not add it to `Project.toml`.
Do not add any package to `Project.toml` without asking the maintainer first.

## Definition of Done

A code change is complete only when:

1. Relevant tests pass, preferably the full `Pkg.test()` command above for
   behavior changes.
2. Formatting has been run with `JuliaFormatter`.
3. Public API changes are registered in `src/RecurrentLayers.jl`, documented
   under `docs/src/api/`, and reflected in the README feature table when
   applicable.
4. Generated files, unrelated manifests, and unrelated user edits are untouched.
5. Any skipped verification is reported with the reason.

## Boundaries

- Do not edit `docs/build/`; it is generated.
- Do not commit, push, branch, or rewrite git history unless explicitly asked.
- Do not add dependencies or test-only packages to `Project.toml` without asking
  first.
- Do not make unrelated `Manifest.toml` changes.
- Do not remove or rewrite existing user changes to solve conflicts; work with
  them or ask when the conflict cannot be resolved safely.
- Do not add comments that merely restate the code. Add comments only when they
  clarify non-obvious behavior, equations, compatibility constraints, or
  licensing.
- Preserve the Apache-2.0 header in `src/cells/nas_cell.jl`.

## What This Package Is

RecurrentLayers.jl extends [Flux.jl](https://github.com/FluxML/Flux.jl) with
recurrent layers not available in base deep learning libraries (30+ cells plus
higher-level wrappers). Sibling projects keep parity: `LuxRecurrentLayers.jl`
(Lux) and `torchrecurrent` (PyTorch). Keep naming and behavior consistent with
them where reasonable.

## Layout

```
RecurrentLayers/
├── src/
│   ├── RecurrentLayers.jl   # module root: imports, exports, includes,
│   │                        #   and the rlayers/rcells fallback loop
│   ├── generics.jl          # abstract types, default initialstates,
│   │                        #   scan-based layer forward passes, _integration_fn
│   ├── base_functions.jl    # shared kernels: dense_proj, add_bias!,
│   │                        #   add/mul_projections, _ind_rec
│   ├── cells/               # one file per cell family
│   │   ├── sgrn_cell.jl     #   template for Cell + Layer pairs
│   │   └── ...              #   30+ others: atr, ligru, mclstm, ...
│   └── wrappers/
│       ├── fastslow.jl
│       ├── multiplicative.jl
│       └── stackedrnn.jl
├── test/
│   ├── runtests.jl          # includes the files below via SafeTestsets
│   ├── qa.jl                # Aqua + JET quality checks
│   ├── test_cells.jl
│   ├── test_layers.jl
│   ├── test_cell_wrappers.jl
│   └── test_layer_wrappers.jl
├── docs/
│   ├── make.jl
│   ├── pages.jl
│   └── src/
│       ├── refs.bib         # bibliography for [Key](@cite) docstring refs
│       └── api/
│           ├── cells/       # one Markdown stub per cell type
│           ├── layers/      # one Markdown stub per layer type
│           └── wrappers/
├── Project.toml             # deps, [compat], and the test [targets]
└── README.md
```

Every new public name must be registered in `src/RecurrentLayers.jl`: exports,
`include`, and the `rlayers`/`rcells` tuples.

## Adding A Cell

Use `src/cells/sgrn_cell.jl` as the template. Each file defines a **Cell**
(single time step) and a **Layer** (whole sequence) pair.

1. Add a `@doc raw"""..."""` block above each type with `# Arguments`,
   `# Keyword arguments`, `# Equations` in LaTeX, `# Forward`, and `# Returns`.
   The first line links the paper with a `[Key](@cite)` reference.
2. Define `struct XCell{...} <: AbstractRecurrentCell`, or
   `AbstractDoubleRecurrentCell` for two-state cells, holding `weight_ih`,
   `weight_hh`, `bias_ih`, `bias_hh`, and any extra fields.
3. Add `@layer XCell`.
4. Add a constructor taking `(input_size => hidden_size)::Pair{<:Int,<:Int}`.
   Use keyword args such as `init_kernel`, `init_recurrent_kernel`, `bias`,
   `recurrent_bias`, `independent_recurrence`, and `integration_mode`. Build
   weights with `init_kernel`/`_indrec_matrix`, biases with `create_bias`, and
   integration with `_integration_fn`.
5. Implement `(cell::XCell)(inp, state)` returning `(output, new_state)`. Start
   with `_size_check`, project via `dense_proj`, combine via the stored
   `integration_fn`, and use `sigmoid_fast`/`tanh_fast`.
6. Add `initialstates(cell::XCell)` only when it differs from the generic
   default.
7. Add `Base.show` for the cell.
8. Add the matching `struct X{S,M} <: AbstractRecurrentLayer{S}`,
   `@layer :noexpand X`, constructor forwarding `kwargs...` to the cell and
   threading `return_state`, `functor`, and `Base.show`.

Also update:

- `src/RecurrentLayers.jl`: `include`, `export`, `rlayers`, and `rcells`.
- `docs/src/api/cells/` and `docs/src/api/layers/`: Markdown stubs.
- `docs/src/refs.bib`: bibliography entry.
- Tests and the README feature table.

## Conventions

- Follow SciML style via `.JuliaFormatter.toml`: 4-space indent, 92-column
  margin, `whitespace_in_kwargs=false`, `separate_kwargs_with_semicolon=true`,
  `always_for_in=true`, and `format_docstrings=true`.
- Prefer shared helpers in `base_functions.jl` and `generics.jl` over new
  projection or state-handling logic.
- `initialstates` is the canonical public way to get zero state
  (`@compat(public, initialstates)`).
- Keep docstrings in the established structure because they are rendered into
  the docs.
- Keep naming and behavior aligned with `LuxRecurrentLayers.jl` and
  `torchrecurrent` where reasonable.

## Compatibility

- Julia >= 1.10; Flux >= 0.16.1, NNlib, Functors, and Compat.
- `src/cells/nas_cell.jl` is Apache-2.0 licensed; everything else is MIT.
