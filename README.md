# loupe

This is a type and shape analyzer for a very small Ruby subset. It is primarily
documented in [this Rails at Scale blog post][blog].

[blog]: https://railsatscale.com/2025-02-14-interprocedural-sparse-conditional-type-propagation/

## How to build

`cargo build --release`

## How to run

`cargo run --release`

The `main` function generates two big torture test programs pseudo-randomly and
then analyzes them. Then it parses a third program from source and analyzes it.
It prints some stats and the parsed IR to standard out.

## How to test

`cargo test`
