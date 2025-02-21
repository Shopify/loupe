# loupe

This is a type and shape analyzer based on an extension to [Constant
Propagation with Conditional Branches][sccp] (PDF) for a very small Ruby
subset. It is primarily documented in [this Rails at Scale blog post][blog].

[sccp]: https://www.cs.utexas.edu/~lin/cs380c/wegman.pdf
[blog]: https://railsatscale.com/2025-02-14-interprocedural-sparse-conditional-type-propagation/

It is just for show alongside the blog post and does not accept contributions.

## How to build

`cargo build --release`

## How to run

`cargo run --release`

The `main` function generates two big torture test programs pseudo-randomly and
then analyzes them. Then it parses a third program from source and analyzes it.
It prints some stats and the parsed IR to standard out.

## How to test

`cargo test`

It's a pain to write IR test cases by hand, so we wrote a little simplified
Ruby-esque parser. Unlike many other parsers, it does not go to an AST but
instead straight to the SSA IR.

```rust
    let sample_program = "
class Point
  attr_accessor :x
  attr_accessor :y
  def initialize(x, y)
    @x = x
    @y = y
  end
end

def main()
  p = Point.new(3, 4)
  return p.x + p.y
end
";
    let lexer = Lexer::new(sample_program);
    let mut parser = Parser::from_lexer(lexer);
    parser.parse_program();
    let prog = parser.prog;
    let result = sctp(&prog);
```

That lets us write test cases in a form similar to the above without actually
hand-writing a class data structure definition, two functions, a binary add,
etc. We can still write `assert`s on the generated data structures and their
discovered types without doing everything by hand.
