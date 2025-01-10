#![allow(dead_code)]
#![allow(unused_variables)]

// use std::fmt;

#[derive(Clone, PartialEq, Eq, Debug)]
pub struct Type {
    name: String,
    fields: Vec<String>,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug, Hash)]
pub struct TypeId(usize);

pub struct TypeSet {
    types: std::collections::HashSet<TypeId>,
}

impl TypeSet {
    pub fn empty() -> TypeSet {
        TypeSet {
            types: std::collections::HashSet::new(),
        }
    }

    pub fn single(ty: TypeId) -> TypeSet {
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
    Object(Type),
}

#[derive(Clone, PartialEq, Eq, Debug)]
pub struct InsnId(usize);

#[derive(Clone, PartialEq, Eq, Debug)]
pub enum Opnd {
    // Block argument
    Arg(u32),

    // Constant
    Const(Value),

    // Output of a previous insn in a block
    // that dominates this one
    InsnOut { idx: InsnId },
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct BlockId(usize);

#[derive(Debug)]
pub enum Insn {
    Send(Opnd, Vec<Opnd>),
    Return(Opnd),
    NewInstance(Opnd, Vec<Opnd>),
    IvarSet(Opnd, Opnd, Opnd),
    IvarGet(Opnd, Opnd),
    IsFixnum(Opnd),
    FixnumAdd(Opnd, Opnd),
    FixnumLt(Opnd, Opnd),

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
pub struct CFG {
    entrypoint: BlockId,
    // Permanent home of every instruction; grows without bound
    insns: Vec<Insn>,
    // Permanent home of every block; grows without bound
    blocks: Vec<Block>,
    // Permanent home of every type; grows without bound
    types: Vec<Type>,
}

impl CFG {
    pub fn new() -> CFG {
        let entry = Block::empty();
        CFG {
            entrypoint: BlockId(0),
            insns: vec![],
            blocks: vec![entry],
            types: vec![],
        }
    }

    pub fn add_insn(&mut self, insn: Insn) -> InsnId {
        let result = InsnId(self.insns.len());
        self.insns.push(insn);
        result
    }

    pub fn add_type(&mut self, ty: Type) -> TypeId {
        let result = TypeId(self.types.len());
        self.types.push(ty);
        result
    }

    pub fn alloc_block(&mut self) -> BlockId {
        let result = BlockId(self.blocks.len());
        self.blocks.push(Block::empty());
        result
    }

    pub fn push(&mut self, block: BlockId, insn: Insn) {
        let insn_id = self.add_insn(insn);
        self.blocks[block.0].insns.push(insn_id)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_cfg() -> CFG {
        let mut result = CFG::new();
        result.push(result.entrypoint, Insn::Return(Opnd::Const(Value::Int(5))));
        result
    }

    #[test]
    fn it_works() {
        let cfg = sample_cfg();
        ()
    }
}



// TODO: we need a function to generate a big graph/program that's going to be
// a torture test
// Something like 10-20K classes and 20K methods that randomly call each other
// We want the size of it to approximate the size of our production apps
fn gen_torture_test(num_classes: usize, num_methods: usize) -> CFG
{
    todo!();
}




fn main() {
    fn sample_cfg() -> CFG {
        let mut result = CFG::new();
        result.push(result.entrypoint, Insn::Return(Opnd::Const(Value::Int(5))));
        result
    }
    let cfg = sample_cfg();
    println!("{:?}", cfg);
}
