# A big static analysis

It's 4pm. Do you know where your variable is pointing?

```ruby
def load_all(files)
  files.each {|file|
    file.load()
  }
end
```

It's hard to tell just looking at the code. What type is `files`? We assume it
has an `each` method, but many classes do. What type is each `file` within
`files`? Apparently they all have a `load` method.

Adding type annotations helps... a little.

```ruby
# thing.rbs
def launch : (files: Array[File]) -> void
```

It looks like we have full knowledge about what each thing is but we actually
don't. Ruby, like many other languages, has this thing called inheritance which
means that type signatures like `Array` and `File` mean an instance of that
class... or an instance of a subclass of that class.

Additionally, type checkers such as Sorbet (for example) have features such as
`T.unsafe` and `T.untyped` which make it possible to lie to the type checker.
This unfortunately renders the type system *unsound*, which is dicy footing for
something we would like to use in program optimization.

This means that we have to take things into our own hands and track the types
ourselves.

In this post, we show an interprocedural type analysis over a vaguely Ruby-like
language. Such analysis could be used for program optimization by a
sufficiently advanced compiler. This is not something Shopify is working on but
we are sharing this post and attached analysis code because we think you will
find it interesting.

## Static analysis

Let's start from the top. We'll go over some examples and then continue on into
code and some benchmarks.

Do you know what type this program returns?

```ruby
1
```

That's right, it's `Integer[1]`. Not only is an `Integer`, but we have
additional information about its value available at analysis time. That will
come in handy later.

What about this variable? What type is `a`?

```ruby
a = 1
```

Not a trick question, at least not yet. It's still `Integer[1]`. But what if we
assign to it twice?

```ruby
a = 1
a = "hello"
```

Ah. Tricky. Things get a little complicated. If we split our program into
segments based on logical execution "time", we can say that `a` starts off as
`Integer[1]` and then becomes `String["hello"]`. This is not super pleasant
because it means that when analyzing the code, you have to carry around some
notion of "time" state in your analysis. It would be much nicer if instead
something rewrote the input code to look more like this:

```ruby
a0 = 1
a1 = "hello"
```

Then we could easily tell the two variables apart at any time because they have
different names. This is where static single assignment (SSA) form comes in.
Automatically converting your input program to SSA introduces some complexity
but gives you the guarantee that every variable is assigned exactly once. This
is why we analyze SSA instead of some other form of intermediate representation
(IR). Assume for the rest of this post that we are working with SSA.

Let's continue with our analysis.

What types do the variables have in the below program?

```ruby
a = 1
b = 2
c = a + b
```

We know `a` and `b` because they are constants, so can we constant-fold `a+b`
into `3`? Kind of. Sort of. In Ruby, without global knowledge that someone has
not and will not patch the `Integer` class or do a variety of other nasty
things, no.

But let's pretend for the duration of this post that we live in a world where
it's not possible to redefine the meaning of integer addition (remember, we're
looking at a Ruby-like language with different semantics but similar syntax).
In that case, it is absolutely possible to fold those constants. So `c` has
type `Integer[3]`.
