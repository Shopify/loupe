#![allow(dead_code)]
#![allow(unused_variables)]

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

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct InsnId(usize);

impl std::fmt::Display for InsnId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "v{}", self.0)
    }
}

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

impl std::fmt::Display for Insn {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Insn")
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

    pub fn insn_at(&self, insn_id: InsnId) -> &Insn {
        &self.insns[insn_id.0]
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

struct DisplayBlock<'a> {
    cfg: &'a CFG,
    block: &'a Block,
    indent: usize,
}

impl<'a> std::fmt::Display for DisplayBlock<'a> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let indent = self.indent;
        for insn_id in self.block.insns.iter() {
            let insn = self.cfg.insn_at(*insn_id);
            // TODO(max): Figure out how to get `indent' worth of spaces
            write!(f, "  {insn_id:<indent$} = {insn}\n")?;
        }
        Ok(())
    }
}

impl std::fmt::Display for CFG {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        for (idx, block) in self.blocks.iter().enumerate() {
            let display_block = DisplayBlock { cfg: self, block: &block, indent: 2 };
            write!(f, "bb {idx} {{\n{display_block}}}")?;
        }
        Ok(())
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
    println!("{cfg}");
}
