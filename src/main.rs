#![allow(unused_imports)]
#![allow(dead_code)]
#![allow(unused_variables)]
#![allow(unused_mut)]

use std::collections::{HashMap, HashSet};

// Wait until we have interprocedural analysis working before introducing classes
/*
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct ClassDesc {
    name: String,

    // List of fields
    fields: Vec<String>,

    // List of methods
    methods: HashMap<String, FunId>,

    // Ignore for now
    // Constructor method
    //ctor: FunId,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug, Hash)]
pub struct ClassId(usize);

impl std::fmt::Display for ClassId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Class@{}", self.0)
    }
}
*/

pub type FunId = usize;
pub type BlockId = usize;
pub type InsnId = usize;

// Type: Int, Nil, Class
#[derive(Debug)]
enum Type
{
    Bottom,
    Top,
}

// Home of our interprocedural CFG
#[derive(Default, Debug)]
struct Program
{
    funs: Vec<Function>,

    blocks: Vec<Block>,

    insns: Vec<Insn>,

    // Main/entry function
    main: FunId,
}

#[derive(Debug)]
struct Function
{
    entry_block: BlockId,

    // These are the blocks this function can return to
    // If we add a block to this, we need to update propagation from return block(s)
    cont_blocks: Vec<BlockId>,

    // We don't need an explicit list of blocks
}

#[derive(Default, Debug)]
struct Block
{
    // Indicates that this block has been marked as reachable/executable
    reachable: bool,

    // Do we need to keep the phi nodes separate?

    insns: Vec<Insn>,
}

#[derive(Debug)]
struct Insn
{
    op: InsnOp,

    // Output type of this instruction
    t: Type,

    // TODO:
    // List of uses/successors, needed for SCCP
    uses: Vec<InsnId>
}

#[derive(Clone, PartialEq, Eq, Hash, Debug)]
pub enum Value {
    Nil,
    Int(i64),
    Fun(FunId),
}

#[derive(Clone, PartialEq, Eq, Debug)]
pub enum Opnd {
    // Constant
    Const(Value),

    // Output of a previous insn in a block
    // that dominates this one
    InsnOut(InsnId),
}

#[derive(Debug)]
enum InsnOp
{
    Phi { ins: Vec<(BlockId, Opnd)> },
    Add { v0: Opnd, v1: Opnd },

    // Start with a direct send (no dynamic lookup)
    // to get the basics of the analysis working
    SendDirect { target: FunId },

    // Continuation blocks are the blocks we can return to
    Return { val: Opnd },

    // Wait until we have basic interprocedural analysis working
    //GetIvar,
    //SetIvar,

    IfTrue { val: Opnd, then_block: BlockId, else_block: BlockId  },
    Jump { target: BlockId }
}




// Sparse conditionall type propagation
fn sctp()
{
    enum ListItem
    {
        Insn(InsnId),
        Block(BlockId),
    }

    // Work list of instructions or blocks
    let mut worklist: Vec<ListItem> = Vec::new();

    // While the work list is not empty
    while worklist.len() > 0
    {
        let item = worklist.pop().unwrap();

        match item {
            ListItem::Block(id) => {
                // - Evaluate block's condition (if any) using current lattice values
                // - Update outgoing edges' executable status
                // - If any edge status changed, add destination blocks to worklist



            }

            ListItem::Insn(id) => {
                // - Evaluate RHS using current lattice values
                // - Compute new lattice value for LHS
                // - If LHS value changed:
                //  * Update variable's lattice value
                //  * Add all uses of variable to worklist
                //  * Add blocks containing uses to worklist




            }
        }
    }
}





fn main()
{
    // TODO:
    // 1. Start with intraprocedural analysis
    // 2. Get interprocedural analysis working with direct send (no classes, no native methods)
    // 3. Once confident that interprocedural analysis is working, then add classes and objects








}
