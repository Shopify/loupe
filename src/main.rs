#![allow(unused_imports)]
#![allow(dead_code)]
#![allow(unused_variables)]
#![allow(unused_mut)]

use std::collections::HashMap;
use rand::prelude::*;

#[derive(Clone, PartialEq, Eq, Debug)]
pub struct Class {
    name: String,

    // List of fields
    fields: Vec<String>,

    // List of methods
    methods: HashMap<String, FunId>,

    // Constructor method
    ctor: FunId,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug, Hash)]
pub struct ClassId(usize);

// Function id
#[derive(Default, Clone, Copy, PartialEq, Eq, Debug, Hash)]
pub struct FunId(usize);

// TODO: try this optimization later when we have a performance baseline
/*
pub enum TypeSet {
    Empty,
    Atom(ClassId),
    Union(set),
}
*/

pub struct TypeSet {
    types: std::collections::HashSet<ClassId>,
}

impl TypeSet {
    pub fn empty() -> TypeSet {
        TypeSet {
            types: std::collections::HashSet::new(),
        }
    }

    pub fn single(ty: ClassId) -> TypeSet {
        TypeSet {
            types: std::collections::HashSet::from([ty]),
        }
    }

    pub fn union(self: &Self, other: &TypeSet) -> TypeSet {
        TypeSet {
            types: self.types.union(&other.types).map(|ty| *ty).collect(),
        }
    }

    pub fn intersection(self: &Self, other: &TypeSet) -> TypeSet {
        TypeSet {
            types: self
                .types
                .intersection(&other.types)
                .map(|ty| *ty)
                .collect(),
        }
    }
}

#[derive(Clone, PartialEq, Eq, Debug)]
pub enum Value {
    Nil,
    Int(i64),
    Str(String),
    Name(String),
    Object(Class), // TODO: ClassId
}

impl std::fmt::Display for Value {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Value::Nil => write!(f, "Nil"),
            Value::Int(val) => write!(f, "{val}"),
            Value::Str(val) => write!(f, "{val:?}"),
            Value::Name(val) => write!(f, "{val}"),
            Value::Object(_) => write!(f, "<Object>"),
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct InsnId(usize);

impl std::fmt::Display for InsnId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "v{}", self.0)
    }
}

#[derive(Clone, PartialEq, Eq, Debug)]
pub enum Opnd {
    Arg(u32),

    // Constant
    Const(Value),

    // Output of a previous insn in a block
    // that dominates this one
    InsnOut(InsnId),
}

impl std::fmt::Display for Opnd {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Opnd::Const(val) => write!(f, "(Const {val})"),
            Opnd::InsnOut(insn_id) => write!(f, "{insn_id}"),
            Opnd::Arg(idx) => write!(f, "arg{idx}"),
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct BlockId(usize);

impl std::fmt::Display for BlockId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "bb{}", self.0)
    }
}

#[derive(Debug)]
pub enum Insn {
    // Send may need to branch to a return block (call continuation)
    // Send has an output
    // How do we handle phi nodes, branches following calls?
    // May be simpler if followed by a branch
    Send(Opnd, Vec<Opnd>),

    Return(Opnd),
    NewInstance(Opnd, Vec<Opnd>),
    IvarSet(Opnd, Opnd, Opnd),
    IvarGet(Opnd, Opnd),
    IsFixnum(Opnd),
    FixnumAdd(Opnd, Opnd),
    FixnumLt(Opnd, Opnd),

    // ?
    //RefineType(Opnd, Type)

    // Maxime says: for branches we're going to need
    // to supply block argumens for each target
    // we may also want to make IfTrue a one-sided branch for simplicity?
    // do we care about having only one final branch at the end of blocks?
    IfTrue(Opnd, BlockId, BlockId),
    Jump(BlockId),
}

impl Insn {
    pub fn is_terminator(self) -> bool {
        use Insn::*;
        match self {
            Return(_) | IfTrue(_, _, _) | Jump(_) => true,
            _ => false,
        }
    }
}

fn fmt_args(f: &mut std::fmt::Formatter<'_>, args: &Vec<Opnd>) -> std::fmt::Result {
    for arg in args {
        write!(f, ", {arg}")?;
    }
    Ok(())
}

impl std::fmt::Display for Insn {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Insn::Send(receiver, args) => {
                write!(f, "Send {receiver}")?;
                fmt_args(f, args)
            }
            Insn::Return(opnd) => {
                write!(f, "Return {opnd}")
            }
            Insn::NewInstance(cls, args) => {
                write!(f, "NewInstance {cls}")?;
                fmt_args(f, args)
            }
            Insn::IvarSet(receiver, attr, value) => {
                write!(f, "IvarSet {receiver}, {attr}, {value}")
            }
            Insn::IvarGet(receiver, attr) => {
                write!(f, "IvarGet {receiver}, {attr}")
            }
            Insn::IsFixnum(opnd) => {
                write!(f, "IsFixnum {opnd}")
            }
            Insn::FixnumAdd(left, right) => {
                write!(f, "FixnumAdd {left}, {right}")
            }
            Insn::FixnumLt(left, right) => {
                write!(f, "FixnumLt {left}, {right}")
            }
            Insn::IfTrue(cond, conseq, alt) => {
                write!(f, "IfTrue {cond} then {conseq} else {alt}")
            }
            Insn::Jump(block_id) => {
                write!(f, "Jump {block_id}")
            }
        }
    }
}

#[derive(Debug)]
pub struct Block {
    params: Vec<Opnd>,
    insns: Vec<InsnId>,
}

impl Block {
    pub fn empty() -> Block {
        Block {
            params: vec![],
            insns: vec![],
        }
    }
}

#[derive(Debug)]
pub struct Function {
    entrypoint: BlockId,

    // Permanent home of every instruction; grows without bound
    insns: Vec<Insn>,

    // Permanent home of every block; grows without bound
    blocks: Vec<Block>,

    // We may need to keep track of callers/successors of instructions
    // to implement an SCCP-like algorithm?
    // We need to be able to "push" type information forward when a type
    // changes

    // List of caller instructions
    //callers
}

impl Function {
    pub fn new() -> Function {
        let entry = Block::empty();
        Function {
            entrypoint: BlockId(0),
            insns: vec![],
            blocks: vec![entry],
        }
    }

    pub fn add_insn(&mut self, insn: Insn) -> InsnId {
        let result = InsnId(self.insns.len());
        self.insns.push(insn);
        result
    }

    pub fn insn_at(&self, insn_id: InsnId) -> &Insn {
        &self.insns[insn_id.0]
    }

    pub fn alloc_block(&mut self) -> BlockId {
        let result = BlockId(self.blocks.len());
        self.blocks.push(Block::empty());
        result
    }

    pub fn push(&mut self, block: BlockId, insn: Insn) -> InsnId {
        let result = self.add_insn(insn);
        self.blocks[block.0].insns.push(result);
        result
    }
}

struct DisplayBlock<'a> {
    function: &'a Function,
    block: &'a Block,
    indent: usize,
}

impl<'a> std::fmt::Display for DisplayBlock<'a> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let indent = self.indent;
        for insn_id in self.block.insns.iter() {
            let insn = self.function.insn_at(*insn_id);
            // TODO(max): Figure out how to get `indent' worth of spaces
            write!(f, "  {insn_id:<indent$} = {insn}\n")?;
        }
        Ok(())
    }
}

impl std::fmt::Display for Function {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        for (idx, block) in self.blocks.iter().enumerate() {
            let display_block = DisplayBlock {
                function: self,
                block: &block,
                indent: 2,
            };
            write!(f, "bb {idx} {{\n{display_block}}}\n")?;
        }
        Ok(())
    }
}

fn sample_function() -> Function {
    let mut result = Function::new();
    let add = result.push(
        result.entrypoint,
        Insn::FixnumAdd(Opnd::Const(Value::Int(3)), Opnd::Const(Value::Int(4))),
    );
    let lt = result.push(
        result.entrypoint,
        Insn::FixnumLt(Opnd::InsnOut(add), Opnd::Const(Value::Int(8))),
    );
    let conseq = result.alloc_block();
    let alt = result.alloc_block();
    let ift = result.push(
        result.entrypoint,
        Insn::IfTrue(Opnd::InsnOut(lt), conseq, alt),
    );
    result.push(
        conseq,
        Insn::Return(Opnd::Const(Value::Str("hello".into()))),
    );
    result.push(alt, Insn::Return(Opnd::Const(Value::Int(2))));
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn it_works() {
        let fun = sample_function();
        ()
    }
}

// TODO: do we want a struct to represent programs?
#[derive(Default, Debug)]
struct Program {
    // Permanent home of every class; grows without bound
    // Maps ClassId to class objects
    classes: Vec<Class>,

    // Permanent home of every function
    funs: Vec<Function>,

    // Main/entry function
    main: FunId,
}

impl Program {
    // TODO: pre-register types for things like Nil, Integer, etc?

    // Register a class
    pub fn reg_class(&mut self, ty: Class) -> ClassId {
        let result = ClassId(self.classes.len());
        self.classes.push(ty);
        result
    }
}

// TODO: we need a function to generate a big graph/program that's going to be
// a torture test
//
// Something like 10-20K classes and 20K methods that randomly call each other
// We want the size of it to approximate the size of our production apps
//
// IIRC the average size of a Ruby method is something like 8 bytecode instructions
// So we can generate many simple methods/functions
fn gen_torture_test(num_classes: usize, num_methods: usize) -> Function {
    let mut rng = rand::thread_rng();
    //println!("Random usize: {}", rng.gen::<usize>());
    //println!("Integer: {}", rng.gen_range(0..10));



    // TODO: start by generating a large number of random functions.
    // We'll worry about classes after
    //let fun_ids = Vec::new();
    for i in 0..num_methods {
    }




    todo!();
}

fn main() {
    let function = sample_function();
    println!("{function}");
}
