#![allow(unused_imports)]
#![allow(dead_code)]
#![allow(unused_variables)]
#![allow(unused_mut)]

use std::collections::{HashMap, HashSet};
use std::cmp::{min, max};

pub struct LCG {
    state: u64,
    // Parameters from "Numerical Recipes"
    // The modulus m is 2^64
    a: u64, // multiplier
    c: u64, // increment
}

impl LCG {
    pub fn new(seed: u64) -> Self {
        LCG {
            state: seed,
            a: 6364136223846793005,
            c: 1442695040888963407,
        }
    }

    // Avoid using the lower bits as they are less random
    pub fn next(&mut self) -> u64 {
        self.state = self.state.wrapping_mul(self.a).wrapping_add(self.c);
        self.state
    }

    // Get only the most significant 32 bits which are more random
    pub fn next_u32(&mut self) -> u32 {
        (self.next() >> 32) as u32
    }

    // Generate a random boolean
    pub fn next_bool(&mut self) -> bool {
        (self.next() & (1 << 60)) != 0
    }

    // Choose a random index in [0, max[
    pub fn next_idx(&mut self, max: usize) -> usize {
        (self.next_u32() as usize) % max
    }

    // Uniformly distributed usize in [min, max]
    pub fn rand_usize(&mut self, min: usize, max: usize) -> usize {
        assert!(max >= min);
        min + (self.next_u32() as usize) % (1 + max - min)
    }

    // Returns true with a specific percentage of probability
    pub fn pct_prob(&mut self, percent: usize) -> bool {
        let idx = self.next_idx(100);
        idx < percent
    }

    // Pick a random element from a slice
    pub fn choice<'a, T>(&mut self, slice: &'a [T]) -> &'a T {
        assert!(!slice.is_empty());
        &slice[self.next_idx(slice.len())]
    }
}

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
fn sctp(prog: &mut Program)
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

// TODO: port this to Rust
fn random_dag(rng: &mut LCG, num_nodes: usize, min_parents: usize, max_parents: usize) -> Vec<Vec<usize>>
{
    let mut callees: Vec<Vec<FunId>> = Vec::new();

    // For each node except the root
    // Node 0 is the root node and has no incoming edge
    for node_idx in 1..num_nodes
    {
        // Choose random number of parents
        let num_parents = rng.rand_usize(min_parents, min(max_parents, node_idx));

        // For each parent
        for i in 0..num_parents {
            // Select a random previous node as the parent
            let p_idx = rng.rand_usize(0, node_idx - 1);

            // Add thos node to the callees of the parent
            if p_idx >= callees.len() {
                callees.resize(p_idx + 1, Vec::default());
            }
            callees[p_idx].push(node_idx);
        }
    }

    callees
}

fn gen_torture_test(num_funs: usize) -> Program
{
    let mut rng = LCG::new(1337);

    let callees = random_dag(&mut rng, num_funs, 1, 10);

    let mut prog = Program::default();

    // The first function is the root node of the graph
    prog.main = 0;

    // For each function to be generated
    for fun_id in 0..num_funs {
        // List of callees for this function
        let callees = &callees[fun_id];

        // TODO: we need a list of callees for each function





        // TODO:
        //let f_id = prog.reg_fun()
        //assert!(f_id = fun_id);
    }






    prog
}

fn main()
{
    // TODO:
    // 1. Start with intraprocedural analysis
    // 2. Get interprocedural analysis working with direct send (no classes, no native methods)
    // 3. Once confident that interprocedural analysis is working, then add classes and objects





    let mut prog = gen_torture_test(200);
    //sctp(&mut prog);
}
