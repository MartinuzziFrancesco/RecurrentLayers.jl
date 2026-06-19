# AGENTS.md

Guidance for AI agents working in **RecurrentLayers.jl**.

## What this package is

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
│   │   ├── sgrn_cell.jl     #   (template: Cell + Layer pair)
│   │   └── ...              #   30+ others (atr, ligru, mclstm, ...)
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

Every new public name must be registered in `src/RecurrentLayers.jl` (export
lists + `include` + the `rlayers`/`rcells` tuples). Don't edit `docs/build/`
(generated).

## Anatomy of a cell file

Use an existing cell such as `src/cells/sgrn_cell.jl` as the template. Each file
defines a **Cell** (single time step) and a **Layer** (whole sequence) pair:

1. A `@doc raw"""..."""` block above each type with `# Arguments`,
   `# Keyword arguments`, `# Equations` (LaTeX), `# Forward`, and `# Returns`.
   The first line links the paper via a `[Key](@cite)` reference.
2. `struct XCell{...} <: AbstractRecurrentCell` (or
   `AbstractDoubleRecurrentCell` for two-state cells) holding
   `weight_ih`, `weight_hh`, `bias_ih`, `bias_hh`, and any extra fields
   (e.g. `integration_fn`).
3. `@layer XCell`.
4. A constructor taking `(input_size => hidden_size)::Pair{<:Int,<:Int}` with
   keyword args (`init_kernel`, `init_recurrent_kernel`, `bias`,
   `recurrent_bias`, `independent_recurrence`, `integration_mode`, ...). Build
   weights with `init_kernel`/`_indrec_matrix`, biases with `create_bias`,
   integration with `_integration_fn`.
5. The forward `(cell::XCell)(inp, state)` returning `(output, new_state)`.
   Start with `_size_check`, project via `dense_proj`, combine via the stored
   `integration_fn`, use `sigmoid_fast`/`tanh_fast`.
6. `initialstates(cell::XCell)` if it differs from the generic default.
7. `Base.show` for the cell.
8. The matching `struct X{S,M} <: AbstractRecurrentLayer{S}`, `@layer :noexpand X`,
   constructor (forwards `kwargs...` to the cell, threads `return_state`),
   `functor`, and `Base.show`.

When adding a new cell, also:
- add the file to the `include` list in `src/RecurrentLayers.jl`;
- add both `XCell` and `X` to the two `export` lines;
- add `:X`/`:XCell` to the `rlayers`/`rcells` tuples (unless the layer needs a
  hand-written constructor, like the LSTM-family double-state cells);
- add doc stubs under `docs/src/api/cells/` and `docs/src/api/layers/`, and a
  bib entry in `docs/src/refs.bib`;
- add tests and a row in the README feature table.

## Conventions

- **Code style:** SciML style via JuliaFormatter (`.JuliaFormatter.toml`):
  4-space indent, 92-column margin, `whitespace_in_kwargs=false`,
  `separate_kwargs_with_semicolon=true`, `always_for_in=true`. A FormatCheck CI
  job enforces this — run the formatter before finishing (see below).
- Prefer the shared helpers in `base_functions.jl`/`generics.jl` over reinventing
  projections or state handling.
- `initialstates` is the canonical way to get zero state; it is the public
  re-export surface (`@compat(public, initialstates)`).
- Keep docstrings in the established structure; they are rendered into the docs
  and `format_docstrings=true` is on.

## Commands

Run from the repo root.

```julia
# Run the full test suite
julia --project -e 'using Pkg; Pkg.test()'
```

The test target pulls in `Aqua`, `JET`, `SafeTestsets`, and `Test`
(see `[targets]` in `Project.toml`). `qa.jl` runs the Aqua + JET quality checks.

**Formatting** (must pass FormatCheck CI). `JuliaFormatter` is *not* a project
dependency and must not be added to `Project.toml`. Run it from your default
(global) environment, where it should be installed once:

```bash
# Preferred: from the global environment
julia -e 'using JuliaFormatter; format(".")'
```

If it isn't installed globally, either `julia -e 'using Pkg; Pkg.add("JuliaFormatter")'`
once, or run it from a throwaway temporary environment without touching the
project:

```bash
julia -e 'using Pkg; Pkg.activate(; temp=true); Pkg.add("JuliaFormatter"); using JuliaFormatter; format(".")'
```

## Compatibility

- Julia ≥ 1.10; Flux ≥ 0.16.1, NNlib, Functors, Compat (see `[compat]`).
- `src/cells/nas_cell.jl` is Apache-2.0 licensed (reimplementation of
  TensorFlow's NASCell); everything else is MIT. Preserve the file header.

## Scope notes for agents

- Commit or push only when explicitly asked; branch off `main` first if needed.
- Don't edit files under `docs/build/` — they are generated.
- Keep `Manifest.toml` changes out of unrelated PRs.
